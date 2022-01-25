/*
 Apsp.c 
* This is a serial implementation of the Floyd-Warshall algorithm, for 
* finding shortest paths in a weighted graph with positive or negative 
* edge weights (without negative cycles). A single execution of the 
* algorithm can find the lengths (summed weights) of shortest paths 
* between all pairs of vertices.
* 
*  A random graph is generated, modeled by a NxN array of weights 
* between the graph vertices. The missing edges between vertices are 
* implemented as weights above the weight limit w.
 
 Command line arguments:
 n = the number of vertices into the graph
 w = the max weight between vertices
 p = the probability of generating edge
        
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

struct timeval startwtime, endwtime;
double seq_time;
int n = 7, w = 6;
float dist[7][7] = {0}, p;
float a[7][7] = {{0, 3, 6, 9, 9, 9, 9},
                {3, 0, 2, 1, 9, 9, 9},
                {6, 2, 0, 1, 4, 2, 9},
                {9, 1, 1, 0, 2, 9, 4},
                {9, 9, 4, 2, 0, 2, 1},
                {9, 9, 2, 9, 2, 0, 1},
                {9, 9, 9, 4, 1, 1, 0}};


void floydWarshall();

int main() {

  w = 7;

  printf("Value of n : %d\n", n);
  printf("Value of max weight : %d\n", w);
  printf("\nPrinting Input Matrix before Execution\n");
  int i, j; //indices
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if(a[i][j]>w){  //Write weights more than maximum as X and consider them infinity
        printf("X.0 ");
      }else{
        printf("%.1f ", a[i][j]);
      }
    }
    printf("\n");
  }

  gettimeofday( & startwtime, NULL);

  floydWarshall();
  printf("\nPrinting Distance Matrix\n");
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      printf("%.1f ", dist[i][j]);
    }
    printf("\n");
  }

  gettimeofday( & endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 +
    endwtime.tv_sec - startwtime.tv_sec);
  printf("\nExecution time is : %f sec\n", seq_time);

}

void floydWarshall() {

  int i, j, k;
  float temp;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      dist[i][j] = a[i][j];
    }
    dist[i][i] = 0; //vertex from itself distance(weight) is 0
  }
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