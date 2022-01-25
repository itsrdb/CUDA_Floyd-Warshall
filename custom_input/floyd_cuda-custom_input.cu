/*

 Command line arguments:
 n = the number of vertices into the graph
 w = the max weight between vertices
 p = the probability of generating edge
        
 */

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#define BLOCK_SIZE 16

struct timeval startwtime, endwtime;
double seq_time;
int n = 7, w = 6;
float a[7][7] = {0}, dist[7][7] = {0}, tesd[7][7], p;

//Declaring Functions
void hostAlloc();
void init();
void floydWarshall_serial();
void oneOneNo();
void initTest();
void prepare_a();
__global__ void floydWarshall_p1(float * dev_dist, size_t pitch, int en);
__global__ void floydWarshall_p2(float * dev_dist, size_t pitch, int en);

int main(int argc, char ** argv) {
  //check arguments
  int i, j;
  prepare_a();
  printf("Value of n : %d\n", n);
  printf("Value of max weight : %d\n", w);

  printf("\nPrinting Custom Input Matrix before Execution\n");
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if(a[i][j]>w){
        printf("X.0 ");
      }else{
        printf("%.1f ", a[i][j]);
      }
    }
    printf("\n");
  }
  printf("\n");

  //For Serial implementation
  gettimeofday( & startwtime, NULL);

  floydWarshall_serial();

  printf("Printing Distance Matrix after Execution\n");
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      printf("%.1f ", dist[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  gettimeofday( & endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 +
    endwtime.tv_sec - startwtime.tv_sec);
  printf("Execution time of Serial is : %f sec\n", seq_time);

  //Parallel begins
  prepare_a();
  initTest(); //This functions initalizes a temporary matrix required for calculation

  gettimeofday( & startwtime, NULL);

  oneOneNo();

  printf("\nPrinting Distance Matrix after Parallel Execution\n");
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      printf("%.1f ", tesd[i][j]);
    }
    printf("\n");
  }

  gettimeofday( & endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 +
    endwtime.tv_sec - startwtime.tv_sec);
  printf("\nExecution time of Parallel is : %f sec\n", seq_time);
}

void prepare_a(){
  float b[7][7] = {{0, 3, 6, 9, 9, 9, 9},
             {3, 0, 2, 1, 9, 9, 9},
             {6, 2, 0, 1, 4, 2, 9},
             {9, 1, 1, 0, 2, 9, 4},
             {9, 9, 4, 2, 0, 2, 1},
             {9, 9, 2, 9, 2, 0, 1},
             {9, 9, 9, 4, 1, 1, 0}};
  for(int i=0; i<7; i++){
    for(int j=0; j<7; j++){
      a[i][j] = b[i][j];
    }
  }
}

void floydWarshall_serial() {
  int i, j, k;
  float temp;
  init();
  //Main algorithm
  for (k = 0; k < n; k++) {
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        temp = dist[i][k] + dist[k][j];
        if (dist[i][j] > temp) {
          dist[i][j] = temp;
        }
      }
    }
  }
}

/**
initializing distance array with weight values
**/
void init() {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      dist[i][j] = a[i][j]; //simple weight initialization of dist array (over the w limit is infinity)
    }
    dist[i][i] = 0; //vertex from itself distance(weight) is 0
  }
}

/**
Host Function for executing 1 cell per thread without shared memory (host function)
**/
void oneOneNo() {
  //init dist
  init();

  float * dev_dist; //device memory dist
  size_t pitch;
  //memory allocation in device memory
  cudaMallocPitch( & dev_dist, & pitch, n * sizeof(float), n);
  //copy dist array to global memory at device
  cudaMemcpy2D(dev_dist, pitch, dist, n * sizeof(float), n * sizeof(float), n, cudaMemcpyHostToDevice);

  //call kernel
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE); //threads per block = BLOCK_SIZE^2
  dim3 numBlocks(n / threadsPerBlock.x, n / threadsPerBlock.y); //blocks per grid
  floydWarshall_p1 << < numBlocks, threadsPerBlock >>> (dev_dist, pitch, n); //call kernel

  cudaDeviceSynchronize();

  //get results from device to host memory
  cudaMemcpy2D(dist, n * sizeof(float), dev_dist, pitch, n * sizeof(float), n, cudaMemcpyDeviceToHost);
  //we have results (minimun weight path) in dist array 

  cudaFree(dev_dist);
}

/**
Kernel Function for executing 1 cell per thread without shared memory (host function)
**/
__global__ void floydWarshall_p1(float * dev_dist, size_t pitch, int en) {
  float temp, d1, d2, * row;
  int k;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < en && j < en) {
    for (k = 0; k < en; k++) {
      row = (float * )((char * ) dev_dist + i * pitch);
      d1 = row[k]; //=dist[i][k]
      row = (float * )((char * ) dev_dist + k * pitch);
      d2 = row[j]; //=dist[k][j]
      row = (float * )((char * ) dev_dist + i * pitch);
      temp = d1 + d2;
      if (row[j] > temp) {
        row[j] = temp; //=dist[i][j]
      }
    }
  }
  __syncthreads();
}

void initTest() {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      tesd[i][j] = dist[i][j];
    }
  }
}