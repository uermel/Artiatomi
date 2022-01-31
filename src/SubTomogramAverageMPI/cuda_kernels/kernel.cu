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


// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator       T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
	This version adds multiple elements per thread sequentially.  This reduces the overall
	cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
	(Brent's Theorem optimization)

	Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
	In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
	If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <unsigned int blockSize>
__global__ void
reduce(float *g_idata, float *g_odata, unsigned int n)
{
	float *sdata = SharedMemory<float>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		mySum += g_idata[i+blockSize];

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			sdata[tid] = mySum = mySum + sdata[tid +  64];
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float *smem = sdata;

		if (blockSize >=  64)
		{
			smem[tid] = mySum = mySum + smem[tid + 32];
		}

		if (blockSize >=  32)
		{
			smem[tid] = mySum = mySum + smem[tid + 16];
		}

		if (blockSize >=  16)
		{
			smem[tid] = mySum = mySum + smem[tid +  8];
		}

		if (blockSize >=   8)
		{
			smem[tid] = mySum = mySum + smem[tid +  4];
		}

		if (blockSize >=   4)
		{
			smem[tid] = mySum = mySum + smem[tid +  2];
		}

		if (blockSize >=   2)
		{
			smem[tid] = mySum = mySum + smem[tid +  1];
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}




////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////

void reduce(int size, int threads, int blocks, int whichKernel, float *d_idata, float *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads)
	{
		case 512:
			reduce<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 256:
			reduce<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 128:
			reduce<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 64:
			reduce< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 32:
			reduce< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 16:
			reduce< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  8:
			reduce<  8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  4:
			reduce<  4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  2:
			reduce<  2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  1:
			reduce<  1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
				
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned int blockSize>
__global__ void
reduceCplx(float2 *g_idata, float *g_odata, unsigned int n)
{
	float *sdata = SharedMemory<float>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i].x;

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		mySum += g_idata[i+blockSize].x;

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			sdata[tid] = mySum = mySum + sdata[tid +  64];
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float *smem = sdata;

		if (blockSize >=  64)
		{
			smem[tid] = mySum = mySum + smem[tid + 32];
		}

		if (blockSize >=  32)
		{
			smem[tid] = mySum = mySum + smem[tid + 16];
		}

		if (blockSize >=  16)
		{
			smem[tid] = mySum = mySum + smem[tid +  8];
		}

		if (blockSize >=   8)
		{
			smem[tid] = mySum = mySum + smem[tid +  4];
		}

		if (blockSize >=   4)
		{
			smem[tid] = mySum = mySum + smem[tid +  2];
		}

		if (blockSize >=   2)
		{
			smem[tid] = mySum = mySum + smem[tid +  1];
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////

void reduceCplx(int size, int threads, int blocks, int whichKernel, float2 *d_idata, float *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads)
	{
		case 512:
			reduceCplx<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 256:
			reduceCplx<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 128:
			reduceCplx<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 64:
			reduceCplx< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 32:
			reduceCplx< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 16:
			reduceCplx< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  8:
			reduceCplx<  8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  4:
			reduceCplx<  4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  2:
			reduceCplx<  2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  1:
			reduceCplx<  1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
				
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned int blockSize>
__global__ void
maskedReduceCplx(float2 *g_idata, float* g_maskdata, float *g_odata, unsigned int n)
{
    float *sdata = SharedMemory<float>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i].x * g_maskdata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        mySum += g_idata[i+blockSize].x * g_maskdata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////

void maskedReduceCplx(int size, int threads, int blocks, int whichKernel, float2 *d_idata, float* d_maskdata, float *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    switch (threads)
    {
        case 512:
            maskedReduceCplx<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_maskdata, d_odata, size); break;
        case 256:
            maskedReduceCplx<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_maskdata, d_odata, size); break;
        case 128:
            maskedReduceCplx<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_maskdata, d_odata, size); break;
        case 64:
            maskedReduceCplx< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_maskdata, d_odata, size); break;
        case 32:
            maskedReduceCplx< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_maskdata, d_odata, size); break;
        case 16:
            maskedReduceCplx< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_maskdata, d_odata, size); break;
        case  8:
            maskedReduceCplx<  8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_maskdata, d_odata, size); break;
        case  4:
            maskedReduceCplx<  4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_maskdata, d_odata, size); break;
        case  2:
            maskedReduceCplx<  2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_maskdata, d_odata, size); break;
        case  1:
            maskedReduceCplx<  1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_maskdata, d_odata, size); break;

    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned int blockSize>
__global__ void
reduceSqrCplx(float2 *g_idata, float *g_odata, unsigned int n)
{
	float *sdata = SharedMemory<float>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i].x * g_idata[i].x;

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		mySum += g_idata[i+blockSize].x * g_idata[i+blockSize].x;

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			sdata[tid] = mySum = mySum + sdata[tid +  64];
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float *smem = sdata;

		if (blockSize >=  64)
		{
			smem[tid] = mySum = mySum + smem[tid + 32];
		}

		if (blockSize >=  32)
		{
			smem[tid] = mySum = mySum + smem[tid + 16];
		}

		if (blockSize >=  16)
		{
			smem[tid] = mySum = mySum + smem[tid +  8];
		}

		if (blockSize >=   8)
		{
			smem[tid] = mySum = mySum + smem[tid +  4];
		}

		if (blockSize >=   4)
		{
			smem[tid] = mySum = mySum + smem[tid +  2];
		}

		if (blockSize >=   2)
		{
			smem[tid] = mySum = mySum + smem[tid +  1];
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////

void reduceSqrCplx(int size, int threads, int blocks, int whichKernel, float2 *d_idata, float *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads)
	{
		case 512:
			reduceSqrCplx<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 256:
			reduceSqrCplx<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 128:
			reduceSqrCplx<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 64:
			reduceSqrCplx< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 32:
			reduceSqrCplx< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 16:
			reduceSqrCplx< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  8:
			reduceSqrCplx<  8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  4:
			reduceSqrCplx<  4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  2:
			reduceSqrCplx<  2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  1:
			reduceSqrCplx<  1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
				
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
	This version adds multiple elements per thread sequentially.  This reduces the overall
	cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
	(Brent's Theorem optimization)

	Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
	In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
	If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <unsigned int blockSize>
__global__ void
maxIndex(float *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)
{
	//float *sdata = SharedMemory<float>();
	//int *sindex = (int*)sdata + blockDim.x;
	int *sindex = SharedMemory<int>();
	float *sdata = (float*)sindex + blockDim.x;

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float myMax = -FLT_MAX;
	int myIndex = -1;
	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		if (g_idata[i] > myMax)
		{
			myMax = g_idata[i];
			myIndex = readIndex ? index[i] : i;
			//mySum += g_idata[i];
		}
		
		if (g_idata[i+blockSize] > myMax)
		{
			myMax = g_idata[i+blockSize];
			myIndex = readIndex ? index[i+blockSize] : i+blockSize;
			//mySum += g_idata[i+blockSize];
		}
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = myMax;
	sindex[tid] = myIndex;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			if (sdata[tid + 256] > myMax)
			{
				sdata[tid] = myMax = sdata[tid + 256];
				sindex[tid] = myIndex = sindex[tid + 256];
				//sdata[tid] = mySum = mySum + sdata[tid + 256];
			}
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			if (sdata[tid + 128] > myMax)
			{
				sdata[tid] = myMax = sdata[tid + 128];
				sindex[tid] = myIndex = sindex[tid + 128];
				//sdata[tid] = mySum = mySum + sdata[tid + 128];
			}
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			if (sdata[tid +  64] > myMax)
			{
				sdata[tid] = myMax = sdata[tid + 64];
				sindex[tid] = myIndex = sindex[tid + 64];
				//sdata[tid] = mySum = mySum + sdata[tid +  64];
			}
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float *smem = sdata;
		volatile int* smemindex = sindex;

		if (blockSize >=  64)
		{
			if (smem[tid + 32] > myMax)
			{
				smem[tid] = myMax = smem[tid + 32];
				smemindex[tid] = myIndex = smemindex[tid + 32];
				//smem[tid] = mySum = mySum + smem[tid + 32];
			}
		}

		if (blockSize >=  32)
		{
			if (smem[tid + 16] > myMax)
			{
				smem[tid] = myMax = smem[tid + 16];
				smemindex[tid] = myIndex = smemindex[tid + 16];
				//smem[tid] = mySum = mySum + smem[tid + 16];
			}
		}

		if (blockSize >=  16)
		{
			if (smem[tid + 8] > myMax)
			{
				smem[tid] = myMax = smem[tid + 8];
				smemindex[tid] = myIndex = smemindex[tid + 8];
				//smem[tid] = mySum = mySum + smem[tid + 8];
			}
		}

		if (blockSize >=   8)
		{
			if (smem[tid + 4] > myMax)
			{
				smem[tid] = myMax = smem[tid + 4];
				smemindex[tid] = myIndex = smemindex[tid + 4];
				//smem[tid] = mySum = mySum + smem[tid + 4];
			}
		}

		if (blockSize >=   4)
		{
			if (smem[tid + 2] > myMax)
			{
				smem[tid] = myMax = smem[tid + 2];
				smemindex[tid] = myIndex = smemindex[tid + 2];
				//smem[tid] = mySum = mySum + smem[tid + 2];
			}
		}

		if (blockSize >=   2)
		{
			if (smem[tid + 1] > myMax)
			{
				smem[tid] = myMax = smem[tid + 1];
				smemindex[tid] = myIndex = smemindex[tid + 1];
				//smem[tid] = mySum = mySum + smem[tid + 1];
			}
		}
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0];
		index[blockIdx.x] = sindex[0];
	}
}




////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////

void maxIndex(int size, int threads, int blocks, int whichKernel, float *d_idata, float *d_odata, int *index)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 4 * threads * sizeof(float) : 2 * threads * sizeof(float);

	switch (threads)
	{
		case 512:
			maxIndex<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case 256:
			maxIndex<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case 128:
			maxIndex<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case 64:
			maxIndex< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case 32:
			maxIndex< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case 16:
			maxIndex< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case  8:
			maxIndex<  8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case  4:
			maxIndex<  4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case  2:
			maxIndex<  2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case  1:
			maxIndex<  1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
				
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned int blockSize>
__global__ void
maxIndexCplx(float2 *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)
{
	//float *sdata = SharedMemory<float>();
	//int *sindex = (int*)sdata + blockDim.x;
	int *sindex = SharedMemory<int>();
	float *sdata = (float*)sindex + blockDim.x;

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float myMax = -FLT_MAX;
	int myIndex = -1;
	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		if (g_idata[i].x > myMax)
		{
			myMax = g_idata[i].x;
			myIndex = readIndex ? index[i] : i;
			//mySum += g_idata[i];
		}
		
		if (g_idata[i+blockSize].x > myMax)
		{
			myMax = g_idata[i+blockSize].x;
			myIndex = readIndex ? index[i+blockSize] : i+blockSize;
			//mySum += g_idata[i+blockSize];
		}
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = myMax;
	sindex[tid] = myIndex;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			if (sdata[tid + 256] > myMax)
			{
				sdata[tid] = myMax = sdata[tid + 256];
				sindex[tid] = myIndex = sindex[tid + 256];
				//sdata[tid] = mySum = mySum + sdata[tid + 256];
			}
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			if (sdata[tid + 128] > myMax)
			{
				sdata[tid] = myMax = sdata[tid + 128];
				sindex[tid] = myIndex = sindex[tid + 128];
				//sdata[tid] = mySum = mySum + sdata[tid + 128];
			}
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			if (sdata[tid +  64] > myMax)
			{
				sdata[tid] = myMax = sdata[tid + 64];
				sindex[tid] = myIndex = sindex[tid + 64];
				//sdata[tid] = mySum = mySum + sdata[tid +  64];
			}
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float *smem = sdata;
		volatile int* smemindex = sindex;

		if (blockSize >=  64)
		{
			if (smem[tid + 32] > myMax)
			{
				smem[tid] = myMax = smem[tid + 32];
				smemindex[tid] = myIndex = smemindex[tid + 32];
				//smem[tid] = mySum = mySum + smem[tid + 32];
			}
		}

		if (blockSize >=  32)
		{
			if (smem[tid + 16] > myMax)
			{
				smem[tid] = myMax = smem[tid + 16];
				smemindex[tid] = myIndex = smemindex[tid + 16];
				//smem[tid] = mySum = mySum + smem[tid + 16];
			}
		}

		if (blockSize >=  16)
		{
			if (smem[tid + 8] > myMax)
			{
				smem[tid] = myMax = smem[tid + 8];
				smemindex[tid] = myIndex = smemindex[tid + 8];
				//smem[tid] = mySum = mySum + smem[tid + 8];
			}
		}

		if (blockSize >=   8)
		{
			if (smem[tid + 4] > myMax)
			{
				smem[tid] = myMax = smem[tid + 4];
				smemindex[tid] = myIndex = smemindex[tid + 4];
				//smem[tid] = mySum = mySum + smem[tid + 4];
			}
		}

		if (blockSize >=   4)
		{
			if (smem[tid + 2] > myMax)
			{
				smem[tid] = myMax = smem[tid + 2];
				smemindex[tid] = myIndex = smemindex[tid + 2];
				//smem[tid] = mySum = mySum + smem[tid + 2];
			}
		}

		if (blockSize >=   2)
		{
			if (smem[tid + 1] > myMax)
			{
				smem[tid] = myMax = smem[tid + 1];
				smemindex[tid] = myIndex = smemindex[tid + 1];
				//smem[tid] = mySum = mySum + smem[tid + 1];
			}
		}
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0];
		index[blockIdx.x] = sindex[0];
	}
}




////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////

void maxIndexCplx(int size, int threads, int blocks, int whichKernel, float2 *d_idata, float *d_odata, int *index)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 4 * threads * sizeof(float) : 2 * threads * sizeof(float);

	switch (threads)
	{
		case 512:
			maxIndexCplx<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case 256:
			maxIndexCplx<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case 128:
			maxIndexCplx<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case 64:
			maxIndexCplx< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case 32:
			maxIndexCplx< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case 16:
			maxIndexCplx< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case  8:
			maxIndexCplx<  8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case  4:
			maxIndexCplx<  4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case  2:
			maxIndexCplx<  2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
		case  1:
			maxIndexCplx<  1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, index, size, true); break;
				
	}
}