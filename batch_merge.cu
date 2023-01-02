/* In this part, we assume we have a large NUMBER N(>= 10^3) of arrays {A_i}(1<=i<=N) and {B_i}(1<=i<=N)
with |A_i|+|B_i|<=1024 for each i. Using some changes on "mergeSmall_k" we would like to write
"mergeSmallBatch_k" that merges two by two, for each i, A_i and B_i.
Given a fixed common size d<=1024, "mergeSmallBatch_k" is launched using the syntax:
  mergeSmallBatch_k<<<numBlocks, threadsPerBlock>>>(...);
with "threadsPerBlock" is MULTIPLE of "d" but smaller than 1024
and "numBlocks" is an arbirtrary sufficiently big number.

(1) Explain why the indices:
- int tidx = threadIdx.x%d;
    "tidx": since if d<=512 (1024/2 = 512 minimum multiple) we can mergesort more than one array
            per block (exactly m arrays: m = 1024/d). "tidx" allows us to enumerate from 1 to d, the elements of each of pair {A_i},{B_i} with (1<=i<=m) in the block.
- int Qt = (threadIdx.x-tidx)/d;
    "Qt": local numeration of arrays in a block. "Qt" goes from 1 to m
- gbx = Qt + blockIdx.x * (blockDim.x/d);
    "gbx": global numeration of arrays. "gbx" goes from 1 to N
are important int eh definition of "mergeSmallBatch_k"

(2) Write the kernel "mergeSmallBatch_k" that batch merges two by two {A_i} and {B_i} (1<=i<=N)
Give the execution time with respect to d = 4,8,...,1024 */

#include<stdio.h>
#include<math.h>
#include<cassert>
#include<time.h>
#include"support-functions.hpp"
#include<algorithm>
#include<fstream> //to export data
#define testCUDA(error) (testCUDA(error,__FILE__,__LINE__))

#define N 1000

//############################### KERNEL ##################################
__global__ void mergeSmallBatch_k(int **A, int *dim_A, int **B, int *dim_B, int **M, int d){
  int tidx = threadIdx.x % d; //enumeration of elements each array of the block -> 0 : d-1
  int Qt = (threadIdx.x-tidx)/d; //enumeration of the arrays in the block
  int gbx = Qt + blockIdx.x * (blockDim.x/d); //global enumeration of arrays

  int idx = threadIdx.x;
  int K_x, K_y, P_x, P_y, Q_x, Q_y;
  int off_set;
  if(gbx < N){
    if(tidx>dim_A[gbx]){
      K_x = tidx - dim_A[gbx]; //(K_x, K_y): low point of diagonal
      K_y = dim_A[gbx];
      P_x = dim_A[gbx]; //(P_x,P_y): high point of diagonal
      P_y = tidx - dim_A[gbx];
    }else{
      K_x = 0;
      K_y = tidx;
      P_x = tidx;
      P_y = 0;
    }
    while(1){
      off_set = (abs(K_y-P_y)/2); //integer
      //distance on y == distance on x, because diagonal
      Q_x = K_x + off_set;
      Q_y = K_y - off_set;
      //offset is an "int" (integer), so it's rounded to the smaller integer -> Q will be closer to K
      if (Q_y>=0 && Q_x<=dim_B[gbx] && (Q_y == dim_A[gbx] || Q_x==0 || A[gbx][Q_y] > B[gbx][Q_x-1]) ){
        // (Q_y>=0 and Q_x<=B) -> Q is in the grid
        // Q_y=|A| -> Q is on the down border
        // Q_x=0   -> Q is on the left border
        // A[Q_y>B[Q_x-1] -> the "chemin" goes 'over Q' or 'pass through Q'
          if (Q_x == dim_B[gbx] || Q_y == 0 || A[gbx][Q_y-1] <= B[gbx][Q_x]){
            // Q_x=|B| -> Q is on the right border
            // Q_y=0   -> Q in on the up border
            // A[Q_y-1]<=B[Q_x] -> the "chemin" goes 'under Q' or 'pass through Q'
              if (Q_y<dim_A[gbx] && (Q_x == dim_B[gbx] || A[gbx][Q_y]<= B[gbx][Q_x]) )
              //if Q is not on the down border (= if the "chemin" can go down) and it MUST go down
                M[gbx][tidx] = A[gbx][Q_y]; //if it can't go down
              else
                M[gbx][tidx] = B[gbx][Q_x]; //the "chemin" goes right (in B direction)s

              break;
          }else{ //if the chemin is over Q but not under Q
              K_x = Q_x + 1; //move Q up by moving K up (updating Q_x to remain on diagonal)
              K_y = Q_y - 1;
          }
      }else{ //if the chemin is under Q
        P_x = Q_x - 1; //move Q down by moving P down (updating Q_x to remain on diagonal)
        P_y = Q_y + 1;
      }
    }
  }
}


//############################### WRAPPER ##################################
void wrapper(int **A, int *dim_A, int **B, int *dim_B, int **M, int dd, float *duration, int k){
  //A, B, M: array of N array -> A[i]: array of N elements
  //dim_A, dim_B: array of N integers -> dim_A[i]: dimension of A_i
  int **A_GPU, **B_GPU, **M_GPU;
  int *dim_A_GPU, *dim_B_GPU;

  int count;
  cudaDeviceProp prop;
  testCUDA(cudaGetDeviceCount(&count));
  testCUDA(cudaGetDeviceProperties(&prop, count-1));


  //########## INITIALIZATION ##########
  float TimerV;
  cudaEvent_t start, stop;
  testCUDA(cudaEventCreate(&start));
  testCUDA(cudaEventCreate(&stop));
  testCUDA(cudaEventRecord(start,0));

  testCUDA(cudaMalloc(&dim_A_GPU, N*sizeof(int)));
  testCUDA(cudaMalloc(&dim_B_GPU, N*sizeof(int)));
  testCUDA(cudaMemcpy(dim_A_GPU, dim_A, N*sizeof(int), cudaMemcpyHostToDevice));
  testCUDA(cudaMemcpy(dim_B_GPU, dim_B, N*sizeof(int), cudaMemcpyHostToDevice));

  /*Procedure to allocate on the DEVICE an array of arrays:
  (1)Allocate the POINTERS to a HOST memory
  (2)Allocate DEVICE memory for EACH ARRAY and store its pointer in the host memory
  (3)Allocate DEVICE memory for storing the pointers
  (4)Then copy the host memory to the device memory */

  //(1) Allocate the ARRAY of POINTERS to HOST memory
    // in order to store "A", "B", "M" into "A_GPU", "B_GPU", "M_GPU"
    int ** A_host= (int**)malloc(N*sizeof(int *));
    int ** B_host = (int**)malloc(N*sizeof(int *));
    int ** M_host = (int**)malloc(N*sizeof(int *));
    // in order to copy from "M_ GPU" to "M"
    int ** M_host_pointers = (int**)malloc(N*sizeof(int *));//It is Necessary to keep a copy of the pointers in the host, in order to use cudaMemcpy -> cudaMemcpy(Pointer on CPU of a GPU object, pointer on CPU of a CPU object)

  //(2) Allocate DEVICE memory for EACH ARRAY and store its pointer in the host memory
    for(int i=0; i<N; ++i){
      testCUDA(cudaMalloc(&A_host[i], dim_A[i]*sizeof(int)));
      testCUDA(cudaMalloc(&B_host[i], dim_B[i]*sizeof(int)));
      testCUDA(cudaMalloc(&M_host[i], dd*sizeof(int)));
      M_host_pointers[i] = M_host[i]; //I keep in the host a copy of the pointers
    }
    for(int i=0; i<N; ++i){
      testCUDA(cudaMemcpy(A_host[i], A[i], dim_A[i]*sizeof(int), cudaMemcpyHostToDevice));
      testCUDA(cudaMemcpy(B_host[i], B[i], dim_B[i]*sizeof(int), cudaMemcpyHostToDevice));
    }

  //(3)Allocate DEVICE memory for storing the pointers
    testCUDA(cudaMalloc(&A_GPU, N*sizeof(int *)));
    testCUDA(cudaMalloc(&B_GPU, N*sizeof(int *)));
    testCUDA(cudaMalloc(&M_GPU, N*sizeof(int *)));

  //(4)Then copy the host memory to the device memory
    testCUDA(cudaMemcpy(A_GPU, A_host, N*sizeof(int *), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(B_GPU, B_host, N*sizeof(int *), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(M_GPU, M_host, N*sizeof(int *), cudaMemcpyHostToDevice));


  //########## CALLING THE KERNEL ##########
  printf("N: %i, d: %i \n",N, dd);
  int m = min(1024 / dd, N); //number of pair of arrays that each block can merge
  printf("Number of pair of arrays that each block merge: %i \n", m);
  int NB;// number of blocks needed
  if(N%m == 0)
    NB = N/m;
  else
    NB = N/m + 1;
  printf("Max number of blocks avaiable: %li \n",prop.maxGridSize[0] );
  printf("Max number of block required: %li \n", NB );
  printf("Number of thread per block: %li \n", dd*m);
  if( NB < prop.maxGridSize[0]){
    printf("No need for a while loop \n" );
    mergeSmallBatch_k<<<NB,dd*m>>>(A_GPU, dim_A_GPU, B_GPU, dim_B_GPU, M_GPU, dd);
  } else{
    printf("while looop is needed \n ");
    //Number of blocks: a number smaller than NB_MAX. I chose: NB == NTPB
    mergeSmallBatch_k<<<dd*m,dd*m>>>(A_GPU, dim_A_GPU, B_GPU, dim_B_GPU, M_GPU, dd);
  }

//Copying "M_GPU" into "M", thanks to the HOST pointers in "M_host_pointers"
 for(int i=0; i<N; ++i){
    testCUDA(cudaMemcpy(M[i], M_host_pointers[i], dd*sizeof(int), cudaMemcpyDeviceToHost));   //???
  }


  //##### FREE MEMORY & GET EXECUTION TIME #####
  for(int i=0; i<N; ++i){
    testCUDA(cudaFree(A_host[i]));
    testCUDA(cudaFree(B_host[i]));
    testCUDA(cudaFree(M_host[i]));
  }

  free(A_host);
  free(B_host);
  free(M_host);
  free(M_host_pointers);

  testCUDA(cudaFree(A_GPU));
  testCUDA(cudaFree(B_GPU));
  testCUDA(cudaFree(M_GPU));
  testCUDA(cudaFree(dim_A_GPU));
  testCUDA(cudaFree(dim_B_GPU));

  testCUDA(cudaEventRecord(stop,0));
  testCUDA(cudaEventSynchronize(stop));
  testCUDA(cudaEventElapsedTime(&TimerV, start, stop));
  printf("Execution time: %f ms\n \n", TimerV);
  duration[k] = TimerV;

  testCUDA(cudaEventDestroy(start));
  testCUDA(cudaEventDestroy(stop));
}

int main(){
  //#####  INITIALIZATION  #####
 int d[9];
 float duration[9];
  for(int i=0; i<9; ++i){
    d[i] = pow(2,i+2);
}

  int **A, **B, **M; // int **
  int *dim_A, *dim_B;
  A = (int**)malloc(N*sizeof(int *));
  B = (int**)malloc(N*sizeof(int *));
  M = (int**)malloc(N*sizeof(int *));
  dim_A = (int*)malloc(N*sizeof(int));
  dim_B = (int*)malloc(N*sizeof(int));

for(int k=0; k<9; ++k){
  srand (time(NULL));
  for(int i=0; i<N; ++i){
      dim_A[i] = rand() % d[k];
    dim_B[i] = d[k] - dim_A[i];
    if(dim_A[i]<dim_B[i]){
      std::swap(dim_A[i],dim_B[i]);
      std::swap(A[i],B[i]);
    }
    A[i] = (int*)malloc(dim_A[i]*sizeof(int));
    B[i] = (int*)malloc(dim_B[i]*sizeof(int));
    M[i] = (int*)malloc(d[k]*sizeof(int));
  //  M[i] = (int*)malloc(d*sizeof(int));
    generate_random(A[i],dim_A[i],0,100);
    mergesort(A[i],0,dim_A[i]-1);
    generate_random(B[i],dim_B[i],0,100);
    mergesort(B[i],0,dim_B[i]-1);
  }
/*
  printf("A\n");
  for(int i=0; i<N; ++i){
    print_array(A[i],dim_A[i]);
    printf("\n");
  }

  printf("B\n");
  for(int i=0; i<N; ++i){
    print_array(B[i],dim_B[i]);
    printf("\n");
  }
*/

//######## WRAPPER CALL #########
  wrapper(A, dim_A, B, dim_B, M, d[k], duration, k);

/*
  printf("M\n");
  for(int i=0; i<N; ++i){
    print_array(M[i],d[k]);
    printf("\n");
  }*/


//####### FREE THE MEMORY #######
  for(int i=0; i<N; ++i){
    free(A[i]);
    free(B[i]);
    free(M[i]);
  }

}
  free(A);
  free(B);
  free(M);
  free(dim_A);
  free(dim_B);


//######## EXPORT EXECUTION TIMES ########
FILE * fp;
fp = fopen("output.txt","w"); //writing
for(int i=0;i<9;i++){
    fprintf(fp, "%i\t%f\n", d[i], duration[i]);
  }
fclose(fp);

  return 0;
}
