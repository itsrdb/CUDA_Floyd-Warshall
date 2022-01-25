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

//global variables
struct timeval startwtime, endwtime;
double seq_time;
int n, w;
float ** a, ** dist, ** tesd, p;

//declare functions
void makeAdjacency();
void hostAlloc();
void init();
void floydWarshall_serial();
void oneOneNo();
void tester();
void initTest();
__global__ void floydWarshall_p1(float * dev_dist, size_t pitch, int en);
__global__ void floydWarshall_p2(float * dev_dist, size_t pitch, int en);

int main(int argc, char ** argv) {
  //check arguments  
  //clock_t begin = clock();
  if (argc != 4) {
    printf("non acceptable input error\n");
    exit(3); //error code 3 if arqs less or more
  }

  //n = 1 << atoi(argv[1])
  n = atoi(argv[1]);
  w = atoi(argv[2]);
  p = atof(argv[3]);

  printf("Number of Vertices n : %d\n", n);
  printf("Value of Max weight w : %d\n", w);
  printf("Value of p : %f\n", p);
  printf("\n");

  hostAlloc();

  makeAdjacency();

  //For serial implementation time calculation
  gettimeofday( & startwtime, NULL);

  floydWarshall_serial();

  gettimeofday( & endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 +
    endwtime.tv_sec - startwtime.tv_sec);
  printf("Serial execution took : %f  ", seq_time);

  initTest();

  gettimeofday( & startwtime, NULL);

  oneOneNo();

  gettimeofday( & endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 +
    endwtime.tv_sec - startwtime.tv_sec);
  printf("\nParallel execution took : %f\n", seq_time);

  free(tesd);
  free(dist);
  free(a);
}

/** 
Makes adjacency matrix a(1:n,1:n) where a edge is generated with 
probability p and random edge weights (0:w).
Instead of infity (if vertexes unconnected) we put a value over w
**/
void makeAdjacency() {
  int i, j;
  float ran;
  srand(time(NULL)); //initializing rand()

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      ran = ((float) rand()) / (float)(RAND_MAX); //random float [0,1]
      //check if i--->j vertexes conected
      if (ran > p) {
        //if not connected weight is out of the limit
        a[i][j] = w + 100;
      } else {
        ran = ((float) rand()) / (float)(RAND_MAX); //random float [0,1]
        a[i][j] = ran * w; //random float [0,w]
      }
    }
    //i-->i weight 0
    a[i][i] = 0;
  }
}

/**
Applies the Floy-Warshall algorithm into the graph
**/
void floydWarshall_serial() {
  int i, j, k;
  float temp;
  //init dist
  init();
  //main algorithm
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
Allocates memory for weight and distance arrays
**/
void hostAlloc() {
  int i;
  a = (float ** ) malloc(n * sizeof(float * ));
  dist = (float ** ) malloc(n * sizeof(float * ));
  tesd = (float ** ) malloc(n * sizeof(float * ));

  for (i = 0; i < n; i++) {
    a[i] = (float * ) malloc(n * sizeof(float));
    dist[i] = (float * ) malloc(n * sizeof(float));
    tesd[i] = (float * ) malloc(n * sizeof(float));
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
      d1 = row[k];
      row = (float * )((char * ) dev_dist + k * pitch);
      d2 = row[j];
      row = (float * )((char * ) dev_dist + i * pitch);
      temp = d1 + d2;
      if (row[j] > temp) {
        row[j] = temp;
      }
    }
  }
  __syncthreads();
}

/**
Initializes test array with distance values. It makes a copy of the
serial distance array for testing and validation
**/
void initTest() {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      tesd[i][j] = dist[i][j];
    }
  }
}

/**
It tests every cell of the parallel distance array 
with the serial one to test and validate results
**/
void tester() {
  int i, j, flag = 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (dist[i][j] != tesd[i][j]) {
        flag = 1;
        break;
      }
    }
    if (flag == 1) {
      printf("ALERT'''''''''''''different than serial'''''''''''''\n");
      break;
    }
  }
  if (flag == 0)
    printf("everything ok in test\n");
}