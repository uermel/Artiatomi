
#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>


extern "C"
__global__ void computeEigenImages(int numberOfVoxels, int numberOfEigenImages, int particle, int numberOfParticles, const float* __restrict__ ccMatrix, const float* __restrict__ volIn, float* eigenImages)
{
	const int voxel = blockIdx.x * blockDim.x + threadIdx.x;
	const int eigenImage = blockIdx.y * blockDim.y + threadIdx.y;

	if (voxel >= numberOfVoxels || eigenImage >= numberOfEigenImages)
		return;

	float ev = ccMatrix[particle + (numberOfEigenImages - 1 - eigenImage) * numberOfParticles]; //eigenvectors are in inverse order (small to large eigen value)
	float vo = volIn[voxel];
	eigenImages[eigenImage * numberOfVoxels + voxel] += ev * vo;
}

