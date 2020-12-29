//Includes for IntelliSense 
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif


#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>


extern "C"
__global__ void subKernel(const float* __restrict__ value, float* dataInOut, int size)
{
	const int voxel = blockIdx.x * blockDim.x + threadIdx.x;
	if (voxel >= size)
		return;

	dataInOut[voxel] -= *value;
}

extern "C"
__global__ void subdivKernel(const float* __restrict__ value, float* dataInOut, int size, float div)
{
	const int voxel = blockIdx.x * blockDim.x + threadIdx.x;
	if (voxel >= size)
		return;

	dataInOut[voxel] -= *value / div;
}
