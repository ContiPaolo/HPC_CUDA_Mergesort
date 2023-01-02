/* For any size |A| + |B| = d sufficiently smaller than the global memory, write
TWO kernels that merge A and B using various blocks.
The first kernel "pathBig_k" finds the merge path and
the second one "mergeBig_k" merges A and B. */

/*Il s'agit en effet de la seule facon (=unico modo) d'utiliser la shared memory si on veut
travailler avec plusieurs blocs. Car sinon, on se retrouverait à indexer la shared
avec l'indice de thread global
- Le premier kernel (pathBig_k) construit le merge path en MEMOIRE GLOBALE et
  le second (mergeBig_k) se base sur le chemin précedemment construit pour faire
  le merge. Dans ce dernier kernel, on pourra alors CHARGER LE MERGE PATH EN SHARED MEMORY.
Vous pouvez regarder l'Algorithme 1 dans l'article de réféerence qui détaille comment implémenter
ces 2 kernels. */

#include<stdio.h>
#include<math.h>
#include<cassert>
#include<time.h>
#include"support-functions.hpp"
#define testCUDA(error) (testCUDA(error,__FILE__,__LINE__))

#define NTPB 3
#define NSM 4
#define Z 4

//#################### KERNEL - find MERGE PATH in the PARTITION POINTS ##############################
//X_path -> orizontal (>) <-> "B"-direction
//Y_path -> vertical (v) <-> "A"-direction
__global__ void MergePath_partition(int *A_gpu, int dim_A, int *B_gpu, int dim_B, int *X_gpu, int *Y_gpu){
//Divide the work among the multiple SPs (Stream Processors):
//"idx" is the partition index: it goes from 0 to #partitions (=#SPs)
  int idx = min(blockIdx.x * (dim_A+dim_B) / NSM, dim_A+dim_B);
  int K_x, K_y, P_x, P_y, Q_x, Q_y;
  int off_set;

    if(idx>dim_A){
      K_x = idx - dim_A; //(K_x, K_y): low point of diagonal
      K_y = dim_A;
      P_x = dim_A;      //(P_x,P_y): high point of diagonal
      P_y = idx - dim_A;
    }else{
      K_x = 0;
      K_y = idx;
      P_x = idx;
      P_y = 0;
    }
    while(1){
      off_set = (abs(K_y-P_y)/2);
      Q_x = K_x + off_set;
      Q_y = K_y - off_set;
      if (Q_y>=0 && Q_x<=dim_B && (Q_y == dim_A || Q_x==0 || A_gpu[Q_y] > B_gpu[Q_x-1]) ){
      // A[Q_y]>B[Q_x-1] -> the "chemin" goes 'over Q' or 'pass through Q'
          if (Q_x == dim_B || Q_y == 0 || A_gpu[Q_y-1] <= B_gpu[Q_x]){
          // A[Q_y-1]<=B[Q_x] -> the "chemin" goes 'under Q' or 'pass through Q' -> Q is on the "chemin"
                X_gpu[idx] = Q_x;
                Y_gpu[idx] = Q_y;
                break;
          }else{
              K_x = Q_x + 1;
              K_y = Q_y - 1;
          }
      }else{
        P_x = Q_x - 1;
        P_y = Q_y + 1;
      }
    }
//  }
  if(blockIdx.x==0 && threadIdx.x==0){
    X_gpu[dim_A+dim_B] = dim_B;
    Y_gpu[dim_A+dim_B] = dim_A;
  }
}


//######################### KERNEL - BIG PATH #################################
__global__ void pathBig_k(int *X_gpu, int *Y_gpu, const int *A_gpu, const int *B_gpu, int dim_A, int dim_B){
  int idx_start = min(blockIdx.x * (dim_A+dim_B) / NSM, dim_A + dim_B);
  int idx_end = min((blockIdx.x+1) * (dim_A+dim_B) / NSM, dim_A + dim_B);
  int x_start = X_gpu[idx_start];
  int y_start = Y_gpu[idx_start];
//  int x_end = X_gpu[idx_end];

  __shared__ int X_sh[Z], Y_sh[Z];
  __shared__ int A_sh[Z], B_sh[Z];
  int idx_sh = threadIdx.x;
  int K_x, K_y, P_x, P_y, Q_x, Q_y;
  int off_set;

 while(idx_start + idx_sh <= idx_end){
  if(idx_sh + y_start < dim_A)
    A_sh[idx_sh] = A_gpu[idx_sh + y_start];
  if(idx_sh + x_start<= dim_B)
    B_sh[idx_sh] = B_gpu[idx_sh + x_start];
  //  __syncthreads();
    if(idx_sh + y_start + 1 > dim_A){
      K_x = x_start + (idx_sh + y_start + 1 - dim_A);
      K_y = dim_A;
      } else{
        K_x = x_start;
        K_y = idx_sh + y_start + 1;
        }
    if(idx_sh + x_start + 1 > dim_B){
      P_x = dim_B;
      P_y = y_start + (idx_sh + x_start + 1 - dim_B);
      } else{
        P_x = idx_sh + x_start + 1;
        P_y = y_start;
      }

      while(1){
      off_set = (abs(K_y-P_y)/2);
      Q_x = K_x + off_set;
      Q_y = K_y - off_set;
      if ((Q_y>=y_start && Q_y>=0) && (Q_x<=(Z+x_start)<=dim_B) && ((Q_y == (Z+y_start) || Q_y == dim_A) || (Q_x==x_start || Q_x==0) || A_sh[Q_y - y_start] > B_sh[Q_x -1 - x_start]) ){
      //  __syncthreads();
      // A[Q_y>B[Q_x-1] -> the "chemin" goes 'over Q' or 'pass through Q'
          if ((Q_x == (Z+x_start) || Q_x == dim_B) || (Q_y == y_start || Q_y == 0) || A_sh[Q_y -1 - y_start] <= B_sh[Q_x - x_start]){
          //  __syncthreads();
          // A[Q_y-1]<=B[Q_x] -> the "chemin" goes 'under Q' or 'pass through Q' -> Q is on the "chemin"
                X_sh[idx_sh] = Q_x;
                Y_sh[idx_sh] = Q_y;
              //  __syncthreads();
                break;
          }else{
              K_x = Q_x + 1;
              K_y = Q_y - 1;
          }
      }else{
        P_x = Q_x - 1;
        P_y = Q_y + 1;
      }
    }

    x_start = X_sh[Z-1];
    y_start = Y_sh[Z-1];
    X_gpu[idx_start + idx_sh +1] = X_sh[idx_sh];
    Y_gpu[idx_start + idx_sh +1] = Y_sh[idx_sh];
    ////__syncthreads();
    idx_start += Z; //update index !!!
  }
}

//######################### KERNEL to MERGE  ##################################
__global__ void mergeBig_k(int *X_gpu, int *Y_gpu, int *A_gpu, int *B_gpu, int *M_gpu, int dim_A, int dim_B){

  int idx_start = min(blockIdx.x * (dim_A+dim_B) / NSM, dim_A + dim_B);
  int idx_end = min((blockIdx.x+1) * (dim_A+dim_B) / NSM, dim_A + dim_B);
  int x_start = X_gpu[idx_start];
  int y_start = Y_gpu[idx_start];
//  int x_end = X_gpu[idx_end];

  __shared__ int X_sh[Z], Y_sh[Z];
  __shared__ int A_sh[Z-1], B_sh[Z-1];
  int idx_sh = threadIdx.x;

 while(idx_start + idx_sh <= idx_end){ //!!!
   if(idx_sh < Z-1 && idx_sh + y_start < dim_A){
     A_sh[idx_sh] = A_gpu[idx_sh + y_start];
   }
   if(idx_sh < Z-1 && idx_sh + x_start<= dim_B){
     B_sh[idx_sh] = B_gpu[idx_sh + x_start];
   }
    X_sh[idx_sh] = X_gpu[idx_start + idx_sh];
    Y_sh[idx_sh] = Y_gpu[idx_start + idx_sh];
//    printf("X[%i] = %i, Y[%i] = %i", idx_sh, X_sh[idx_sh], idx_sh,Y_sh[idx_sh]);
    //__syncthreads();
    if( idx_sh>0 && X_sh[idx_sh]>X_sh[idx_sh-1])
      M_gpu[idx_start + idx_sh -1] = B_sh[X_sh[idx_sh-1] - x_start];
        //M[idx_start + idx_sh] = B[idx_sh + x_start];
    else if(idx_sh > 0)
      M_gpu[idx_start + idx_sh -1] = A_sh[Y_sh[idx_sh-1] - y_start];
    //__syncthreads();
    x_start = X_sh[Z-1];
    y_start = Y_sh[Z-1];
    idx_start += Z-1; //update index !!!
  }
}
//########################### WRAPPER FIND PATH ###################################
void wrapper_FindPartition(int *a, int dim_A, int *b, int dim_B, int *m, int * x_path, int * y_path){
  int *A_GPU, *B_GPU, *M_GPU, *X_GPU, *Y_GPU;

  int count;
  cudaDeviceProp prop;
  testCUDA(cudaGetDeviceCount(&count));
  testCUDA(cudaGetDeviceProperties(&prop, count-1));
  printf("Global memory size in octet (bytes): %ld \n", prop.totalGlobalMem); //ld: long double
  printf("Max dimension of the two input arrays: %ld \n", prop.totalGlobalMem/(sizeof(int)*8)); //399716352
  printf("Shared memory size per block: %ld \n", prop.sharedMemPerBlock);
  printf("Max dimension of the int array in the shared memory: %ld \n", prop.sharedMemPerBlock/(sizeof(int)*8)); // 1536
  printf("Number of multiprocessors: %i \n", prop.multiProcessorCount); //NSM
  printf("Maximum number of thread per block: %li \n", prop.maxThreadsPerBlock);
  //3*2^14 -> 3*2^11

  float TimerV;
  cudaEvent_t start, stop;
  testCUDA(cudaEventCreate(&start));
  testCUDA(cudaEventCreate(&stop));
  testCUDA(cudaEventRecord(start,0));

  testCUDA(cudaMalloc(&A_GPU, dim_A*sizeof(int)));
  testCUDA(cudaMalloc(&B_GPU, dim_B*sizeof(int)));
  testCUDA(cudaMalloc(&X_GPU, (dim_A+dim_B+1)*sizeof(int)));
  testCUDA(cudaMalloc(&Y_GPU, (dim_A+dim_B+1)*sizeof(int)));
  testCUDA(cudaMalloc(&M_GPU, (dim_A+dim_B)*sizeof(int)));
  //Copying the values form one ProcUni to the other ProcUnit
  testCUDA(cudaMemcpy(A_GPU, a, dim_A*sizeof(int), cudaMemcpyHostToDevice));
  testCUDA(cudaMemcpy(B_GPU, b, dim_B*sizeof(int), cudaMemcpyHostToDevice));

  printf("Max number of blocks avaiable: %li \n",prop.maxGridSize[0] );

  MergePath_partition<<<NSM,1>>>(A_GPU, dim_A, B_GPU, dim_B, X_GPU, Y_GPU);

  pathBig_k<<<NSM,Z>>>(X_GPU, Y_GPU, A_GPU, B_GPU, dim_A, dim_B);

  mergeBig_k<<<NSM,Z>>>(X_GPU, Y_GPU, A_GPU, B_GPU, M_GPU, dim_A, dim_B);
  //Copying the value from one ProcUnit ot the ohter ProcUnit

  testCUDA(cudaMemcpy(x_path, X_GPU,(dim_A+dim_B+1)*sizeof(int), cudaMemcpyDeviceToHost));
  testCUDA(cudaMemcpy(y_path, Y_GPU,(dim_A+dim_B+1)*sizeof(int), cudaMemcpyDeviceToHost));
  testCUDA(cudaMemcpy(m, M_GPU, (dim_A+dim_B)*sizeof(int), cudaMemcpyDeviceToHost));

  testCUDA(cudaEventRecord(stop,0));
  testCUDA(cudaEventSynchronize(stop));
  testCUDA(cudaEventElapsedTime(&TimerV, start, stop));
  printf("Execution time of FIND PATH: %f ms\n", TimerV);

  //Freeing the GPU MEMORY
  testCUDA(cudaFree(A_GPU));
  testCUDA(cudaFree(B_GPU));
  testCUDA(cudaFree(M_GPU));
  testCUDA(cudaFree(X_GPU));
  testCUDA(cudaFree(Y_GPU));
  testCUDA(cudaEventDestroy(start));
  testCUDA(cudaEventDestroy(stop));
}


//############################# MAIN ###################################
int main(){
  //INITIALIZATION:
// int A[] = {1,2,5,6,6,9,11,15,16};
// int B[] = {4,7,8,10,12,13,14};
//  int dim_A = 9;
//  int dim_B = 7;
  int *A, *B, *X_path, *Y_path, *M;
  int d = 25;
  srand (time(NULL));
  int dim_A = rand() % d; // !!!
  int dim_B = d - dim_A;
  A = (int*)malloc(dim_A*sizeof(int));
  B = (int*)malloc(dim_B*sizeof(int));
  //#diagonals == #elements chemin: |A|+|B|+1
  X_path = (int*)malloc((dim_A+dim_B+1)*sizeof(int)); // B <-> x
  Y_path = (int*)malloc((dim_A+dim_B+1)*sizeof(int)); // A <-> -y
  generate_random(A,dim_A,0,100);
  mergesort(A,0,dim_A-1);
  generate_random(B,dim_B,0,100);
  mergesort(B,0,dim_B-1);
  //assert((dim_A+dim_B)<=1024); //Check: |M| <= 1024
  M = (int*)malloc((dim_A+dim_B)*sizeof(int));

  // Necessary condition: |A| > |B| !!!
//  if(dim_A<dim_B)
  //  exchange(A, B, dim_A, dim_B);

  printf("A: \n");
  print_array(A,dim_A);

  printf("B: \n");
  print_array(B,dim_B);

  wrapper_FindPartition(A, dim_A, B, dim_B, M, X_path, Y_path);

  printf("M: \n");
  print_array(M, dim_A+dim_B);


  printf("Chemin: \n");
  for(int i=0; i<(dim_A+dim_B+1);++i)
    printf("X[%i]=%i Y[%i]=%i \n",i,X_path[i],i,Y_path[i]);


  free(A);
  free(B);
  free(M);
  free(X_path);
  free(Y_path);
}
