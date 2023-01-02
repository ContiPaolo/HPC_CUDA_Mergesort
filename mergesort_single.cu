/*
(1) For |A| + |B| <= 1024 write a kernel "mergeSmall_k" that merges A and B
using only one block of threads
*/

/* To parallelize the algorithm, the grid has to be extended to the maximum size
equal to max(|A|,|B|)x max(|A|,|B|). We define K_0 and P_0  the
low point and the high point of the ascending diagonals Delta_k respectively.
On GPU, each thread k in [[0,|A|+|B|-1]] is responsible of ONE diagonal.
It finds the intersection of the merge path and the diagonal Delta_k with a
binary search described in Algorithm 2. */

#include<stdio.h>
#include<math.h> //abs
#include<cassert>
#include<fstream> //to export data
#include<time.h> //random numbers
#include<fstream>
#include"support-functions.hpp"
#define testCUDA(error) (testCUDA(error,__FILE__,__LINE__))

//####################### KERNEL #################################
__global__ void mergeSmall_k(int *A, int dim_A, int *B, int dim_B, int *M){
  int idx = threadIdx.x;
  int K_x, K_y, P_x, P_y, Q_x, Q_y;
  int off_set;
  if(idx < (dim_A + dim_B)){ // "<" par fusion
    if(idx>dim_A){
      K_x = idx - dim_A; //(K_x, K_y): low point of diagonal
      K_y = dim_A;
      P_x = dim_A; //(P_x,P_y): high point of diagonal
      P_y = idx - dim_A;
    }else{
      K_x = 0;
      K_y = idx;
      P_x = idx;
      P_y = 0;
    }
    while(1){
      off_set = (abs(K_y-P_y)/2); //integer
      //distance on y == distance on x, because diagonal
      Q_x = K_x + off_set;
      Q_y = K_y - off_set;
      //offset is an "int" (integer), so it's rounded to the smaller integer -> Q will be closer to K
      if (Q_y>=0 && Q_x<=dim_B && (Q_y == dim_A || Q_x==0 || A[Q_y] > B[Q_x-1]) ){
        // (Q_y>=0 and Q_x<=B) -> Q is in the grid
        // Q_y=|A| -> Q is on the down border
        // Q_x=0   -> Q is on the left border
        // A[Q_y>B[Q_x-1] -> the "chemin" goes 'over Q' or 'pass through Q'
          if (Q_x == dim_B || Q_y == 0 || A[Q_y-1] <= B[Q_x]){
            // Q_x=|B| -> Q is on the right border
            // Q_y=0   -> Q in on the up border
            // A[Q_y-1]<=B[Q_x] -> the "chemin" goes 'under Q' or 'pass through Q'
              if (Q_y<dim_A && (Q_x == dim_B || A[Q_y]<= B[Q_x]) )
              //if Q is not on the down border (= if the "chemin" can go down) and it MUST go down
                M[idx] = A[Q_y]; //if it can't go down
              else
                M[idx] = B[Q_x]; //the "chemin" goes right (in B direction)s

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

//########################### WRAPPER ###################################
void wrapper_mergeSmall(int *A, int dim_A, int *B, int dim_B, int *M, float & duration){
  int *A_GPU, *B_GPU, *M_GPU;

  float TimerV;
  cudaEvent_t start, stop;
  testCUDA(cudaEventCreate(&start));
  testCUDA(cudaEventCreate(&stop));
  testCUDA(cudaEventRecord(start,0));

  testCUDA(cudaMalloc(&A_GPU, dim_A*sizeof(int)));
  testCUDA(cudaMalloc(&B_GPU, dim_B*sizeof(int)));
  testCUDA(cudaMalloc(&M_GPU, (dim_A+dim_B)*sizeof(int)));
  //Copying the values form one ProcUni to the other ProcUnit
  testCUDA(cudaMemcpy(A_GPU, A, dim_A*sizeof(int), cudaMemcpyHostToDevice));
  testCUDA(cudaMemcpy(B_GPU, B, dim_B*sizeof(int), cudaMemcpyHostToDevice));

  //Launching the operation on the GPU
  mergeSmall_k<<<1,1024>>>(A_GPU, dim_A, B_GPU, dim_B, M_GPU);

  //Copying the value from one ProcUnit ot the ohter ProcUnit
  testCUDA(cudaMemcpy(M, M_GPU, (dim_A+dim_B)*sizeof(int), cudaMemcpyDeviceToHost));

  testCUDA(cudaEventRecord(stop,0));
  testCUDA(cudaEventSynchronize(stop));
  testCUDA(cudaEventElapsedTime(&TimerV, start, stop));
  printf("Execution time: %f ms\n", TimerV);
  duration = TimerV;

  //Freeing the GPU MEMORY
  testCUDA(cudaFree(A_GPU));
  testCUDA(cudaFree(B_GPU));
  testCUDA(cudaFree(M_GPU));
  testCUDA(cudaEventDestroy(start));
  testCUDA(cudaEventDestroy(stop));
}

//############################# MAIN ###################################
int main(){
  //INITIALIZATION of "A" and "B" (random) and of "M"
  // int A[] = {1,2,5,6,6,9,11,15,16};
  // int B[] = {4,7,8,10,12,13,14};
  // int dim_A = 9;
  // int dim_B = 7;
  int *A, *B, *M;
  int d[10];
  float duration[10];
  for(int i=0; i<10; ++i)
    d[i] = pow(2,i+1);

  for(int k=0; k<10; k++){
    srand (time(NULL));
    int dim_A = rand() % d[k];
    int dim_B = d[k] - dim_A;
    A = (int*)malloc(dim_A*sizeof(int));
    B = (int*)malloc(dim_B*sizeof(int));
    generate_random(A,dim_A,0,100);
    mergesort(A,0,dim_A-1);
    generate_random(B,dim_B,0,100);
    mergesort(B,0,dim_B-1);
    //assert((dim_A+dim_B)<=1024); //Check: |M| <= 1024
    M = (int*)malloc((dim_A+dim_B)*sizeof(int));

    //printf("A: \n");
    //print_array(A,dim_A);

    //printf("B: \n");
    //print_array(B,dim_B);

    // Necessary condition: |A| > |B|
    if(dim_A>=dim_B)
      wrapper_mergeSmall(A, dim_A, B, dim_B, M, duration[k]);
    else
      wrapper_mergeSmall(B, dim_B, A, dim_A, M, duration[k]);

    //printf("M: \n");
    //print_array(M, dim_A+dim_B);

    free(A);
    free(B);
    free(M);
  }

  FILE * fp;
  fp = fopen("output_global.txt","w"); //writing
  for(int i=0; i<10; i++){
      fprintf(fp, "%i\t%f\n", d[i], duration[i]);
    }
  fclose(fp);
    return 0;

}


