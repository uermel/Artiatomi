//  Copyright (c) 2018, Michael Kunz and Frangakis Lab, BMLS,
//  Goethe University, Frankfurt am Main.
//  All rights reserved.
//  http://kunzmi.github.io/Artiatomi
//  
//  This file is part of the Artiatomi package.
//  
//  Artiatomi is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  Artiatomi is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with Artiatomi. If not, see <http://www.gnu.org/licenses/>.
//  
////////////////////////////////////////////////////////////////////////



#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

extern volatile __shared__ float s_data[];


__device__ float* GetPixel(int x, int y, int z, int pitch, int height, const float* __restrict__ img)
{
	float* row = (float*)((unsigned char*)img + (size_t)z * (size_t)pitch * (size_t)height + (size_t)y * (size_t)pitch);
	return row + (size_t)x;
}

extern "C"
__global__ void convolveX(const float* __restrict__ d_Src, float* d_Dst, int width, int height, int depth, int pitch, int filterRadius)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	//init shared mem to 0:
	s_data[threadIdx.x] = 0;
	s_data[threadIdx.x + filterRadius] = 0;

	__syncthreads();

	if (x < filterRadius || x >= width - filterRadius) return;
	if (y < filterRadius || y >= height - filterRadius) return;
	if (z < filterRadius || z >= depth - filterRadius) return;

	//load main data to sharedMem:
	s_data[filterRadius + threadIdx.x] = *GetPixel(x, y, z, pitch, height, d_Src);

	//load left filter overlap
	if (threadIdx.x < filterRadius)
		s_data[threadIdx.x] = *GetPixel(x - filterRadius, y, z, pitch, height, d_Src);

	//load right filter overlap
	if (threadIdx.x >= blockDim.x - filterRadius)
		s_data[threadIdx.x + filterRadius + filterRadius] = *GetPixel(x + filterRadius, y, z, pitch, height, d_Src);

	__syncthreads();

	float ergPixel = 0;
	for (int i = -filterRadius; i <= filterRadius; i++)
	{
		ergPixel += s_data[threadIdx.x + filterRadius + i];
	}

	*GetPixel(x, y, z, pitch, height, d_Dst) = ergPixel;
}

extern "C"
__global__ void convolveY(const float* __restrict__ d_Src, float* d_Dst, int width, int height, int depth, int pitch, int filterRadius)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	//init shared mem to 0:
	s_data[threadIdx.y] = 0;
	s_data[threadIdx.y + filterRadius] = 0;

	__syncthreads();

	if (x < filterRadius || x >= width - filterRadius) return;
	if (y < filterRadius || y >= height - filterRadius) return;
	if (z < filterRadius || z >= depth - filterRadius) return;

	//load main data to sharedMem:
	s_data[filterRadius + threadIdx.y] = *GetPixel(x, y, z, pitch, height, d_Src);

	//load left filter overlap
	if (threadIdx.y < filterRadius)
		s_data[threadIdx.y] = *GetPixel(x, y - filterRadius, z, pitch, height, d_Src);

	//load right filter overlap
	if (threadIdx.y >= blockDim.y - filterRadius)
		s_data[threadIdx.y + filterRadius + filterRadius] = *GetPixel(x, y + filterRadius, z, pitch, height, d_Src);

	__syncthreads();

	float ergPixel = 0;
	for (int i = -filterRadius; i <= filterRadius; i++)
	{
		ergPixel += s_data[threadIdx.y + filterRadius + i];
	}

	*GetPixel(x, y, z, pitch, height, d_Dst) = ergPixel;
}

extern "C"
__global__ void convolveZ(const float* __restrict__ d_Src, float* d_Dst, int width, int height, int depth, int pitch, int filterRadius)
{
	int x = blockIdx.z * blockDim.z + threadIdx.z;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.x * blockDim.x + threadIdx.x;

	//init shared mem to 0:
	s_data[threadIdx.x] = 0;
	s_data[threadIdx.x + filterRadius] = 0;

	__syncthreads();

	if (x < filterRadius || x >= width - filterRadius) return;
	if (y < filterRadius || y >= height - filterRadius) return;
	if (z < filterRadius || z >= depth - filterRadius) return;

	//load main data to sharedMem:
	s_data[filterRadius + threadIdx.x] = *GetPixel(x, y, z, pitch, height, d_Src);

	//load left filter overlap
	if (threadIdx.x < filterRadius)
		s_data[threadIdx.x] = *GetPixel(x, y, z - filterRadius, pitch, height, d_Src);

	//load right filter overlap
	if (threadIdx.x >= blockDim.x - filterRadius)
		s_data[threadIdx.x + filterRadius + filterRadius] = *GetPixel(x, y, z + filterRadius, pitch, height, d_Src);

	__syncthreads();

	float ergPixel = 0;
	for (int i = -filterRadius; i <= filterRadius; i++)
	{
		ergPixel += s_data[threadIdx.x + filterRadius + i];
	}

	*GetPixel(x, y, z, pitch, height, d_Dst) = ergPixel;
}



extern "C"
__global__ void ComputeDistanceForShift(const float* __restrict__ imgIn, int pitchIn, float* dist, int pitchWeight, int width, int height, int depth, int shiftX, int shiftY, int shiftZ)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= width) return;
	if (y >= height) return;
	if (z >= depth) return;


	float distance = 66000; //some huge number for out of range pixels (255*255 would be 65.025)
	
	if (x + shiftX >= 0 && x + shiftX < width &&
		y + shiftY >= 0 && y + shiftY < height &&
		z + shiftZ >= 0 && z + shiftZ < depth)
	{
		float pixel = *GetPixel(x, y, z, pitchIn, height, imgIn);
		float shifted = *GetPixel(x + shiftX, y + shiftY, z + shiftZ, pitchIn, height, imgIn);
		distance = (pixel - shifted) * (pixel - shifted) * 255.0f * 255.0f; //convert from normailzed float to pseudo 8bit, so that old parameters fit...
	}

	*GetPixel(x, y, z, pitchWeight, height, dist) = distance;
}

extern "C"
__global__ void AddWeightedPixel(const float* __restrict__ imgIn, int pitchIn, float* imgOut, int pitchOut, float* weight, int pitchWeight, float* weightMax, float* weightSum, int width, int height, int depth, int shiftX, int shiftY, int shiftZ, float sigma, float filterParam, float patchSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= width) return;
	if (y >= height) return;
	if (z >= depth) return;



	if (x + shiftX >= 0 && x + shiftX < width &&
		y + shiftY >= 0 && y + shiftY < height &&
		z + shiftZ >= 0 && z + shiftZ < depth)
	{
		float pixel = *GetPixel(x, y, z, pitchOut, height, imgOut);
		float w = *GetPixel(x, y, z, pitchWeight, height, weight); //this is the distance convolved with mask

		w -= (2.0f * (patchSize * patchSize * patchSize) * sigma * sigma);

		w = fmaxf(w, 0.0f);

		float fH = filterParam * sigma;
		float fH2 = fH * fH;
		w = w / fH2;
		w = __expf(-w);
				
		if (w > 0)
		{
			*GetPixel(x, y, z, pitchWeight, height, weightMax) = fmaxf(*GetPixel(x, y, z, pitchWeight, height, weightMax), w);
			*GetPixel(x, y, z, pitchWeight, height, weightSum) += w;
			float shifted = *GetPixel(x + shiftX, y + shiftY, z + shiftZ, pitchIn, height, imgIn);
			pixel += shifted * w;
			*GetPixel(x, y, z, pitchOut, height, imgOut) = pixel;
		}
	}
}
extern "C"
__global__ void AddWeightedPixelFinal(const float* __restrict__ imgIn, int pitchIn, float* imgOut, int pitchOut, float* weight, int pitchWeight, float* weightSum, int width, int height, int depth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= width) return;
	if (y >= height) return;
	if (z >= depth) return;


	
	float pixel = *GetPixel(x, y, z, pitchOut, height, imgOut);
	float w = *GetPixel(x, y, z, pitchWeight, height, weight); //this is the distance convolved with mask
	
	if (w > 0)
	{
		*GetPixel(x, y, z, pitchWeight, height, weightSum) += w;
		float shifted = *GetPixel(x, y, z, pitchIn, height, imgIn);
		pixel += shifted * w;
		*GetPixel(x, y, z, pitchOut, height, imgOut) = pixel;
	}
	
}

extern "C"
__global__ void ComputeFinalPixel(const float* __restrict__ imgIn, int pitchIn, float* imgOut, int pitchOut, float* weight, int pitchWeight, int width, int height, int depth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= width) return;
	if (y >= height) return;
	if (z >= depth) return;


	float w = *GetPixel(x, y, z, pitchWeight, height, weight);

	float pixel;
	if (w > 0.000001f)
	{
		pixel = *GetPixel(x, y, z, pitchOut, height, imgOut);
		pixel /= w;
	}
	else
	{
		pixel = *GetPixel(x, y, z, pitchIn, height, imgIn);
	}

	*GetPixel(x, y, z, pitchOut, height, imgOut) = pixel;
}