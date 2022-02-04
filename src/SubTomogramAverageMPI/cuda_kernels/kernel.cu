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
__device__ void
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

// Declaration of templated functions
extern "C" __global__ void
reduce_512(float *g_idata, float *g_odata, unsigned int n) {reduce<512>(g_idata, g_odata, n);}
extern "C" __global__ void
reduce_256(float *g_idata, float *g_odata, unsigned int n) {reduce<256>(g_idata, g_odata, n);}
extern "C" __global__ void
reduce_128(float *g_idata, float *g_odata, unsigned int n) {reduce<128>(g_idata, g_odata, n);}
extern "C" __global__ void
reduce_64(float *g_idata, float *g_odata, unsigned int n)  {reduce< 64>(g_idata, g_odata, n);}
extern "C" __global__ void
reduce_32(float *g_idata, float *g_odata, unsigned int n)  {reduce< 32>(g_idata, g_odata, n);}
extern "C" __global__ void
reduce_16(float *g_idata, float *g_odata, unsigned int n)  {reduce< 16>(g_idata, g_odata, n);}
extern "C" __global__ void
reduce_8(float *g_idata, float *g_odata, unsigned int n)   {reduce<  8>(g_idata, g_odata, n);}
extern "C" __global__ void
reduce_4(float *g_idata, float *g_odata, unsigned int n)   {reduce<  4>(g_idata, g_odata, n);}
extern "C" __global__ void
reduce_2(float *g_idata, float *g_odata, unsigned int n)   {reduce<  2>(g_idata, g_odata, n);}
extern "C" __global__ void
reduce_1(float *g_idata, float *g_odata, unsigned int n)   {reduce<  1>(g_idata, g_odata, n);}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned int blockSize>
__device__ void
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

// Declaration of templated functions
extern "C" __global__ void
reduceCplx_512(float2 *g_idata, float *g_odata, unsigned int n) {reduceCplx<512>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceCplx_256(float2 *g_idata, float *g_odata, unsigned int n) {reduceCplx<256>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceCplx_128(float2 *g_idata, float *g_odata, unsigned int n) {reduceCplx<128>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceCplx_64(float2 *g_idata, float *g_odata, unsigned int n)  {reduceCplx< 64>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceCplx_32(float2 *g_idata, float *g_odata, unsigned int n)  {reduceCplx< 32>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceCplx_16(float2 *g_idata, float *g_odata, unsigned int n)  {reduceCplx< 16>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceCplx_8(float2 *g_idata, float *g_odata, unsigned int n)   {reduceCplx<  8>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceCplx_4(float2 *g_idata, float *g_odata, unsigned int n)   {reduceCplx<  4>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceCplx_2(float2 *g_idata, float *g_odata, unsigned int n)   {reduceCplx<  2>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceCplx_1(float2 *g_idata, float *g_odata, unsigned int n)   {reduceCplx<  1>(g_idata, g_odata, n);}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned int blockSize>
__device__ void
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

// Declaration of templated functions
extern "C" __global__ void
maskedReduceCplx_512(float2 *g_idata, float* g_maskdata, float *g_odata, unsigned int n) {maskedReduceCplx<512>(g_idata, g_maskdata, g_odata, n);}
extern "C" __global__ void
maskedReduceCplx_256(float2 *g_idata, float* g_maskdata, float *g_odata, unsigned int n) {maskedReduceCplx<256>(g_idata, g_maskdata, g_odata, n);}
extern "C" __global__ void
maskedReduceCplx_128(float2 *g_idata, float* g_maskdata, float *g_odata, unsigned int n) {maskedReduceCplx<128>(g_idata, g_maskdata, g_odata, n);}
extern "C" __global__ void
maskedReduceCplx_64(float2 *g_idata, float* g_maskdata, float *g_odata, unsigned int n)  {maskedReduceCplx< 64>(g_idata, g_maskdata, g_odata, n);}
extern "C" __global__ void
maskedReduceCplx_32(float2 *g_idata, float* g_maskdata, float *g_odata, unsigned int n)  {maskedReduceCplx< 32>(g_idata, g_maskdata, g_odata, n);}
extern "C" __global__ void
maskedReduceCplx_16(float2 *g_idata, float* g_maskdata, float *g_odata, unsigned int n)  {maskedReduceCplx< 16>(g_idata, g_maskdata, g_odata, n);}
extern "C" __global__ void
maskedReduceCplx_8(float2 *g_idata, float* g_maskdata, float *g_odata, unsigned int n)   {maskedReduceCplx<  8>(g_idata, g_maskdata, g_odata, n);}
extern "C" __global__ void
maskedReduceCplx_4(float2 *g_idata, float* g_maskdata, float *g_odata, unsigned int n)   {maskedReduceCplx<  4>(g_idata, g_maskdata, g_odata, n);}
extern "C" __global__ void
maskedReduceCplx_2(float2 *g_idata, float* g_maskdata, float *g_odata, unsigned int n)   {maskedReduceCplx<  2>(g_idata, g_maskdata, g_odata, n);}
extern "C" __global__ void
maskedReduceCplx_1(float2 *g_idata, float* g_maskdata, float *g_odata, unsigned int n)   {maskedReduceCplx<  1>(g_idata, g_maskdata, g_odata, n);}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned int blockSize>
__device__ void
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

// Declaration of templated functions
extern "C" __global__ void
reduceSqrCplx_512(float2 *g_idata, float *g_odata, unsigned int n) {reduceSqrCplx<512>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceSqrCplx_256(float2 *g_idata, float *g_odata, unsigned int n) {reduceSqrCplx<256>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceSqrCplx_128(float2 *g_idata, float *g_odata, unsigned int n) {reduceSqrCplx<128>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceSqrCplx_64(float2 *g_idata, float *g_odata, unsigned int n)  {reduceSqrCplx< 64>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceSqrCplx_32(float2 *g_idata, float *g_odata, unsigned int n)  {reduceSqrCplx< 32>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceSqrCplx_16(float2 *g_idata, float *g_odata, unsigned int n)  {reduceSqrCplx< 16>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceSqrCplx_8(float2 *g_idata, float *g_odata, unsigned int n)   {reduceSqrCplx<  8>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceSqrCplx_4(float2 *g_idata, float *g_odata, unsigned int n)   {reduceSqrCplx<  4>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceSqrCplx_2(float2 *g_idata, float *g_odata, unsigned int n)   {reduceSqrCplx<  2>(g_idata, g_odata, n);}
extern "C" __global__ void
reduceSqrCplx_1(float2 *g_idata, float *g_odata, unsigned int n)   {reduceSqrCplx<  1>(g_idata, g_odata, n);}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned int blockSize>
__device__ void
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

// Declaration of templated functions
extern "C" __global__ void
maxIndex_512(float *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex) {maxIndex<512>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndex_256(float *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex) {maxIndex<256>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndex_128(float *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex) {maxIndex<128>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndex_64(float *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)  {maxIndex< 64>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndex_32(float *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)  {maxIndex< 32>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndex_16(float *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)  {maxIndex< 16>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndex_8(float *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)   {maxIndex<  8>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndex_4(float *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)   {maxIndex<  4>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndex_2(float *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)   {maxIndex<  2>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndex_1(float *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)   {maxIndex<  1>(g_idata, g_odata, index, n, readIndex);}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned int blockSize>
__device__ void
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

// Declaration of templated functions
extern "C" __global__ void
maxIndexCplx_512(float2 *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex) {maxIndexCplx<512>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndexCplx_256(float2 *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex) {maxIndexCplx<256>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndexCplx_128(float2 *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex) {maxIndexCplx<128>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndexCplx_64(float2 *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)  {maxIndexCplx< 64>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndexCplx_32(float2 *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)  {maxIndexCplx< 32>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndexCplx_16(float2 *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)  {maxIndexCplx< 16>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndexCplx_8(float2 *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)   {maxIndexCplx<  8>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndexCplx_4(float2 *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)   {maxIndexCplx<  4>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndexCplx_2(float2 *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)   {maxIndexCplx<  2>(g_idata, g_odata, index, n, readIndex);}
extern "C" __global__ void
maxIndexCplx_1(float2 *g_idata, float *g_odata, int* index, unsigned int n, bool readIndex)   {maxIndexCplx<  1>(g_idata, g_odata, index, n, readIndex);}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned int blockSize>
__device__ void
maxIndexMasked(float *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)
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
		if (g_idata[i] > myMax && g_maskdata[i] > 0)
		{
			myMax = g_idata[i];
			myIndex = readIndex ? index[i] : i;
			//mySum += g_idata[i];
		}
		
		if (g_idata[i+blockSize] > myMax && g_maskdata[i+blockSize] > 0)
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

// Declaration of templated functions
extern "C" __global__ void
maxIndexMasked_512(float *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex) {maxIndexMasked<512>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMasked_256(float *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex) {maxIndexMasked<256>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMasked_128(float *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex) {maxIndexMasked<128>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMasked_64(float *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)  {maxIndexMasked< 64>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMasked_32(float *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)  {maxIndexMasked< 32>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMasked_16(float *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)  {maxIndexMasked< 16>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMasked_8(float *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)   {maxIndexMasked<  8>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMasked_4(float *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)   {maxIndexMasked<  4>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMasked_2(float *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)   {maxIndexMasked<  2>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMasked_1(float *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)   {maxIndexMasked<  1>(g_idata, g_odata, g_maskdata, index, n, readIndex);}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned int blockSize>
__device__ void
maxIndexMaskedCplx(float2 *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)
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
		if (g_idata[i].x > myMax && g_maskdata[i] > 0)
		{
			myMax = g_idata[i].x;
			myIndex = readIndex ? index[i] : i;
			//mySum += g_idata[i];
		}
		
		if (g_idata[i+blockSize].x > myMax && g_maskdata[i+blockSize] > 0)
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

// Declaration of templated functions
extern "C" __global__ void
maxIndexMaskedCplx_512(float2 *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex) {maxIndexMaskedCplx<512>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMaskedCplx_256(float2 *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex) {maxIndexMaskedCplx<256>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMaskedCplx_128(float2 *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex) {maxIndexMaskedCplx<128>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMaskedCplx_64(float2 *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)  {maxIndexMaskedCplx< 64>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMaskedCplx_32(float2 *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)  {maxIndexMaskedCplx< 32>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMaskedCplx_16(float2 *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)  {maxIndexMaskedCplx< 16>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMaskedCplx_8(float2 *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)   {maxIndexMaskedCplx<  8>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMaskedCplx_4(float2 *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)   {maxIndexMaskedCplx<  4>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMaskedCplx_2(float2 *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)   {maxIndexMaskedCplx<  2>(g_idata, g_odata, g_maskdata, index, n, readIndex);}
extern "C" __global__ void
maxIndexMaskedCplx_1(float2 *g_idata, float *g_odata, float *g_maskdata, int* index, unsigned int n, bool readIndex)   {maxIndexMaskedCplx<  1>(g_idata, g_odata, g_maskdata, index, n, readIndex);}