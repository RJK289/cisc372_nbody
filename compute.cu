#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

//Only Person Working on Project: Ryan Koller

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL

__global__ void compute_pairwise_kernel(vector3 *accels, vector3 *pos, double *mass, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n * n) return;
	int i = idx / n;
	int j = idx % n;
	if (i == j) {
		FILL_VECTOR(accels[idx],0,0,0);
	}
	else{
		vector3 distance;
		int k;
		for (k=0;k<3;k++) distance[k]=pos[i][k]-pos[j][k];
		double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
		double magnitude=sqrt(magnitude_sq);
		double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
		FILL_VECTOR(accels[idx],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
	}
}

__global__ void sum_rows_update_kernel(vector3 *accels, vector3 *vel, vector3 *pos, int n) { 
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	if (i >= n) return;
	int j,k;
	vector3 accel_sum={0,0,0};
	for (j=0;j<n;j++){
		int idx = i * n + j;
		for (k=0;k<3;k++)
			accel_sum[k]+=accels[idx][k];
	}
	//compute the new velocity based on the acceleration and time interval
	//compute the new position based on the velocity and time interval
	for (k=0;k<3;k++){
		vel[i][k]+=accel_sum[k]*INTERVAL;
		pos[i][k]+=vel[i][k]*INTERVAL;
	}
}

void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int n = NUMENTITIES;
	vector3 *d_hPos;
	vector3 *d_hVel;
	double  *d_mass;
	vector3 *d_accels;

	cudaMalloc((void**)&d_hPos, sizeof(vector3)*n); 
	cudaMalloc((void**)&d_hVel, sizeof(vector3)*n); 
	cudaMalloc((void**)&d_mass, sizeof(double)*n);
	cudaMalloc((void**)&d_accels, sizeof(vector3)*n*n); 

	cudaMemcpy(d_hPos, hPos, sizeof(vector3)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hVel, hVel, sizeof(vector3)*n, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_mass, mass, sizeof(double)*n, cudaMemcpyHostToDevice);  

	//first compute the pairwise accelerations.  Effect is on the first argument.
	int totalPairs = n * n;
	int blockSize1 = 256;
	int gridSize1 = (totalPairs + blockSize1 - 1) / blockSize1;
	compute_pairwise_kernel<<<gridSize1, blockSize1>>>(d_accels, d_hPos, d_mass, n);

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	int blockSize2 = 256;
	int gridSize2 = (n + blockSize2 - 1) / blockSize2;
	sum_rows_update_kernel<<<gridSize2, blockSize2>>>(d_accels, d_hVel, d_hPos, n);

	cudaMemcpy(hPos, d_hPos, sizeof(vector3)*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, d_hVel, sizeof(vector3)*n, cudaMemcpyDeviceToHost);

	cudaFree(d_accels);
	cudaFree(d_mass);
	cudaFree(d_hPos);
	cudaFree(d_hVel);
}
