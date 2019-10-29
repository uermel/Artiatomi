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


#include "CudaReducer.h"

CudaReducer::CudaReducer(int aVoxelCount, CUstream aStream, CudaContext* context)
	: voxelCount(aVoxelCount), stream(aStream), ctx(context)
{
	CUmodule cuMod = ctx->LoadModule("kernel.ptx_bu");
		
	sum512 = new CudaKernel("_Z6reduceILj512EEvPfS0_j", cuMod);
	sum256 = new CudaKernel("_Z6reduceILj256EEvPfS0_j", cuMod);
	sum128 = new CudaKernel("_Z6reduceILj128EEvPfS0_j", cuMod);
	sum64  = new CudaKernel("_Z6reduceILj64EEvPfS0_j",  cuMod);
	sum32  = new CudaKernel("_Z6reduceILj32EEvPfS0_j",  cuMod);
	sum16  = new CudaKernel("_Z6reduceILj16EEvPfS0_j",  cuMod);
	sum8   = new CudaKernel("_Z6reduceILj8EEvPfS0_j",   cuMod);
	sum4   = new CudaKernel("_Z6reduceILj4EEvPfS0_j",   cuMod);
	sum2   = new CudaKernel("_Z6reduceILj2EEvPfS0_j",   cuMod);
	sum1   = new CudaKernel("_Z6reduceILj1EEvPfS0_j",   cuMod);

		
	sumCplx512 = new CudaKernel("_Z10reduceCplxILj512EEvP6float2Pfj", cuMod);
	sumCplx256 = new CudaKernel("_Z10reduceCplxILj256EEvP6float2Pfj", cuMod);
	sumCplx128 = new CudaKernel("_Z10reduceCplxILj128EEvP6float2Pfj", cuMod);
	sumCplx64  = new CudaKernel("_Z10reduceCplxILj64EEvP6float2Pfj",  cuMod);
	sumCplx32  = new CudaKernel("_Z10reduceCplxILj32EEvP6float2Pfj",  cuMod);
	sumCplx16  = new CudaKernel("_Z10reduceCplxILj16EEvP6float2Pfj",  cuMod);
	sumCplx8   = new CudaKernel("_Z10reduceCplxILj8EEvP6float2Pfj",   cuMod);
	sumCplx4   = new CudaKernel("_Z10reduceCplxILj4EEvP6float2Pfj",   cuMod);
	sumCplx2   = new CudaKernel("_Z10reduceCplxILj2EEvP6float2Pfj",   cuMod);
	sumCplx1   = new CudaKernel("_Z10reduceCplxILj1EEvP6float2Pfj",   cuMod);

		
	sumSqrCplx512 = new CudaKernel("_Z13reduceSqrCplxILj512EEvP6float2Pfj", cuMod);
	sumSqrCplx256 = new CudaKernel("_Z13reduceSqrCplxILj256EEvP6float2Pfj", cuMod);
	sumSqrCplx128 = new CudaKernel("_Z13reduceSqrCplxILj128EEvP6float2Pfj", cuMod);
	sumSqrCplx64  = new CudaKernel("_Z13reduceSqrCplxILj64EEvP6float2Pfj",  cuMod);
	sumSqrCplx32  = new CudaKernel("_Z13reduceSqrCplxILj32EEvP6float2Pfj",  cuMod);
	sumSqrCplx16  = new CudaKernel("_Z13reduceSqrCplxILj16EEvP6float2Pfj",  cuMod);
	sumSqrCplx8   = new CudaKernel("_Z13reduceSqrCplxILj8EEvP6float2Pfj",   cuMod);
	sumSqrCplx4   = new CudaKernel("_Z13reduceSqrCplxILj4EEvP6float2Pfj",   cuMod);
	sumSqrCplx2   = new CudaKernel("_Z13reduceSqrCplxILj2EEvP6float2Pfj",   cuMod);
	sumSqrCplx1   = new CudaKernel("_Z13reduceSqrCplxILj1EEvP6float2Pfj",   cuMod);

	maxIndex512 = new CudaKernel("_Z8maxIndexILj512EEvPfS0_Pijb", cuMod);
	maxIndex256 = new CudaKernel("_Z8maxIndexILj256EEvPfS0_Pijb", cuMod);
	maxIndex128 = new CudaKernel("_Z8maxIndexILj128EEvPfS0_Pijb", cuMod);
	maxIndex64  = new CudaKernel("_Z8maxIndexILj64EEvPfS0_Pijb",  cuMod);
	maxIndex32  = new CudaKernel("_Z8maxIndexILj32EEvPfS0_Pijb",  cuMod);
	maxIndex16  = new CudaKernel("_Z8maxIndexILj16EEvPfS0_Pijb",  cuMod);
	maxIndex8   = new CudaKernel("_Z8maxIndexILj8EEvPfS0_Pijb",   cuMod);
	maxIndex4   = new CudaKernel("_Z8maxIndexILj4EEvPfS0_Pijb",   cuMod);
	maxIndex2   = new CudaKernel("_Z8maxIndexILj2EEvPfS0_Pijb",   cuMod);
	maxIndex1   = new CudaKernel("_Z8maxIndexILj1EEvPfS0_Pijb",   cuMod);

	maxIndexCplx512 = new CudaKernel("_Z12maxIndexCplxILj512EEvP6float2PfPijb", cuMod);
	maxIndexCplx256 = new CudaKernel("_Z12maxIndexCplxILj256EEvP6float2PfPijb", cuMod);
	maxIndexCplx128 = new CudaKernel("_Z12maxIndexCplxILj128EEvP6float2PfPijb", cuMod);
	maxIndexCplx64  = new CudaKernel("_Z12maxIndexCplxILj64EEvP6float2PfPijb",  cuMod);
	maxIndexCplx32  = new CudaKernel("_Z12maxIndexCplxILj32EEvP6float2PfPijb",  cuMod);
	maxIndexCplx16  = new CudaKernel("_Z12maxIndexCplxILj16EEvP6float2PfPijb",  cuMod);
	maxIndexCplx8   = new CudaKernel("_Z12maxIndexCplxILj8EEvP6float2PfPijb",   cuMod);
	maxIndexCplx4   = new CudaKernel("_Z12maxIndexCplxILj4EEvP6float2PfPijb",   cuMod);
	maxIndexCplx2   = new CudaKernel("_Z12maxIndexCplxILj2EEvP6float2PfPijb",   cuMod);
	maxIndexCplx1   = new CudaKernel("_Z12maxIndexCplxILj1EEvP6float2PfPijb",   cuMod);


}

void CudaReducer::MaxIndex(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, CudaDeviceVariable& d_index)
{
	int blocks;
	int threads;
	float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

	
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runMaxIndexKernel(voxelCount, blocks, threads, d_idata, d_odata, d_index, false);

    // sum partial block sums on GPU
    int s=blocks;

    while (s > 1)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, blocks, threads);

        runMaxIndexKernel(s, blocks, threads, d_odata, d_odata, d_index, true);
		s = (s + (threads*2-1)) / (threads*2);
    }

    if (s > 1)
    {
		printf("Oops, not a power of 2?\n");
		//      // copy result from device to host
			//d_odata.CopyDeviceToHost(h_odata, s * sizeof(float));

		//      for (int i=0; i < s; i++)
		//      {
		//          gpu_result += h_odata[i];
		//      }

		//      needReadBack = false;
    }
    


    if (needReadBack)
    {
        // copy final sum from device to host
		//d_odata.CopyDeviceToHost(&gpu_result, sizeof(float));
    }

}

void CudaReducer::MaxIndexCplx(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, CudaDeviceVariable& d_index)
{
	int blocks;
	int threads;
	float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

	
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runMaxIndexCplxKernel(voxelCount, blocks, threads, d_idata, d_odata, d_index, false);

    // sum partial block sums on GPU
    int s=blocks;

    while (s > 1)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, blocks, threads);

        runMaxIndexKernel(s, blocks, threads, d_odata, d_odata, d_index, true);
		s = (s + (threads*2-1)) / (threads*2);
    }

    if (s > 1)
    {
		printf("Oops, not a power of 2?\n");
		//      // copy result from device to host
			//d_odata.CopyDeviceToHost(h_odata, s * sizeof(float));

		//      for (int i=0; i < s; i++)
		//      {
		//          gpu_result += h_odata[i];
		//      }

		//      needReadBack = false;
    }
    


    if (needReadBack)
    {
        // copy final sum from device to host
		//d_odata.CopyDeviceToHost(&gpu_result, sizeof(float));
    }

}

void CudaReducer::Sum(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	int blocks;
	int threads;
	float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

	
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runSumKernel(voxelCount, blocks, threads, d_idata, d_odata);

    // sum partial block sums on GPU
    int s=blocks;

    while (s > 1)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, blocks, threads);

        runSumKernel(s, blocks, threads, d_odata, d_odata);
		s = (s + (threads*2-1)) / (threads*2);
    }

    if (s > 1)
    {
		printf("Oops, not a power of 2?\n");
		//      // copy result from device to host
			//d_odata.CopyDeviceToHost(h_odata, s * sizeof(float));

		//      for (int i=0; i < s; i++)
		//      {
		//          gpu_result += h_odata[i];
		//      }

		//      needReadBack = false;
    }
    


    if (needReadBack)
    {
        // copy final sum from device to host
		//d_odata.CopyDeviceToHost(&gpu_result, sizeof(float));
    }

}

void CudaReducer::SumSqrCplx(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	int blocks;
	int threads;
	float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

	
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runSumSqrCplxKernel(voxelCount, blocks, threads, d_idata, d_odata);

    // sum partial block sums on GPU
    int s=blocks;

    while (s > 1)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, blocks, threads);

        runSumKernel(s, blocks, threads, d_odata, d_odata);
		s = (s + (threads*2-1)) / (threads*2);
    }

    if (s > 1)
    {
		printf("Oops, not a power of 2?\n");
		//      // copy result from device to host
			//d_odata.CopyDeviceToHost(h_odata, s * sizeof(float));

		//      for (int i=0; i < s; i++)
		//      {
		//          gpu_result += h_odata[i];
		//      }

		//      needReadBack = false;
    }
    


    if (needReadBack)
    {
        // copy final sum from device to host
		//d_odata.CopyDeviceToHost(&gpu_result, sizeof(float));
    }

}

void CudaReducer::SumCplx(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	int blocks;
	int threads;
	float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

	
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runSumCplxKernel(voxelCount, blocks, threads, d_idata, d_odata);

    // sum partial block sums on GPU
    int s=blocks;

    while (s > 1)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, blocks, threads);

        runSumKernel(s, blocks, threads, d_odata, d_odata);
		s = (s + (threads*2-1)) / (threads*2);
    }

    if (s > 1)
    {
		printf("Oops, not a power of 2?\n");
		//      // copy result from device to host
			//d_odata.CopyDeviceToHost(h_odata, s * sizeof(float));

		//      for (int i=0; i < s; i++)
		//      {
		//          gpu_result += h_odata[i];
		//      }

		//      needReadBack = false;
    }
    


    if (needReadBack)
    {
        // copy final sum from device to host
		//d_odata.CopyDeviceToHost(&gpu_result, sizeof(float));
    }

}



void CudaReducer::runMaxIndexKernel(int size, int blocks, int threads, CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, CudaDeviceVariable& d_index, bool readIndex)
{
	CudaKernel* kernel;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 4 * threads * sizeof(float) : 2 * threads * sizeof(float);

	switch (threads)
    {
        case 512:
            kernel = maxIndex512; break;
        case 256:
            kernel = maxIndex256; break;
        case 128:
            kernel = maxIndex128; break;
        case 64:
            kernel = maxIndex64; break;
        case 32:
            kernel = maxIndex32; break;
        case 16:
            kernel = maxIndex16; break;
        case  8:
            kernel = maxIndex8; break;
        case  4:
            kernel = maxIndex4; break;
        case  2:
            kernel = maxIndex2; break;
        case  1:
            kernel = maxIndex1; break;
    }
	
    CUdeviceptr in_dptr = d_idata.GetDevicePtr();
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();
    CUdeviceptr index_dptr = d_index.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[5];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &index_dptr;
    arglist[3] = &n;
	arglist[4] = &readIndex;

    cudaSafeCall(cuLaunchKernel(kernel->GetCUfunction(), dimGrid.x, dimGrid.y,
		dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, smemSize, stream, arglist,NULL));

    delete[] arglist;
}

void CudaReducer::runMaxIndexCplxKernel(int size, int blocks, int threads, CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, CudaDeviceVariable& d_index, bool readIndex)
{
	CudaKernel* kernel;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 4 * threads * sizeof(float) : 2 * threads * sizeof(float);

	switch (threads)
    {
        case 512:
            kernel = maxIndexCplx512; break;
        case 256:
            kernel = maxIndexCplx256; break;
        case 128:
            kernel = maxIndexCplx128; break;
        case 64:
            kernel = maxIndexCplx64; break;
        case 32:
            kernel = maxIndexCplx32; break;
        case 16:
            kernel = maxIndexCplx16; break;
        case  8:
            kernel = maxIndexCplx8; break;
        case  4:
            kernel = maxIndexCplx4; break;
        case  2:
            kernel = maxIndexCplx2; break;
        case  1:
            kernel = maxIndexCplx1; break;
    }
	
    CUdeviceptr in_dptr = d_idata.GetDevicePtr();
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();
    CUdeviceptr index_dptr = d_index.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[5];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &index_dptr;
    arglist[3] = &n;
	arglist[4] = &readIndex;

    cudaSafeCall(cuLaunchKernel(kernel->GetCUfunction(), dimGrid.x, dimGrid.y,
		dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, smemSize, stream, arglist,NULL));

    delete[] arglist;
}

void CudaReducer::runSumKernel(int size, int blocks, int threads, CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	CudaKernel* kernel;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads)
    {
        case 512:
            kernel = sum512; break;
        case 256:
            kernel = sum256; break;
        case 128:
            kernel = sum128; break;
        case 64:
            kernel = sum64; break;
        case 32:
            kernel = sum32; break;
        case 16:
            kernel = sum16; break;
        case  8:
            kernel = sum8; break;
        case  4:
            kernel = sum4; break;
        case  2:
            kernel = sum2; break;
        case  1:
            kernel = sum1; break;
    }
	
    CUdeviceptr in_dptr = d_idata.GetDevicePtr();
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[3];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &n;

    //float ms;

    //CUevent eventStart;
    //CUevent eventEnd;
    //CUstream stream = 0;
    //cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    //cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    //cudaSafeCall(cuStreamQuery(stream));
    //cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(kernel->GetCUfunction(), dimGrid.x, dimGrid.y,
		dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, smemSize, stream, arglist,NULL));

    //cudaSafeCall(cuCtxSynchronize());

    //cudaSafeCall(cuStreamQuery(stream));
    //cudaSafeCall(cuEventRecord(eventEnd, stream));
    //cudaSafeCall(cuEventSynchronize(eventEnd));
    //cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    delete[] arglist;
}

void CudaReducer::runSumSqrCplxKernel(int size, int blocks, int threads, CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	CudaKernel* kernel;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads)
    {
        case 512:
            kernel = sumSqrCplx512; break;
        case 256:
            kernel = sumSqrCplx256; break;
        case 128:
            kernel = sumSqrCplx128; break;
        case 64:
            kernel = sumSqrCplx64; break;
        case 32:
            kernel = sumSqrCplx32; break;
        case 16:
            kernel = sumSqrCplx16; break;
        case  8:
            kernel = sumSqrCplx8; break;
        case  4:
            kernel = sumSqrCplx4; break;
        case  2:
            kernel = sumSqrCplx2; break;
        case  1:
            kernel = sumSqrCplx1; break;
    }
	
    CUdeviceptr in_dptr = d_idata.GetDevicePtr();
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[3];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &n;

    //float ms;

    //CUevent eventStart;
    //CUevent eventEnd;
    //CUstream stream = 0;
    //cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    //cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    //cudaSafeCall(cuStreamQuery(stream));
    //cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(kernel->GetCUfunction(), dimGrid.x, dimGrid.y,
		dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, smemSize, stream, arglist,NULL));

    //cudaSafeCall(cuCtxSynchronize());

    //cudaSafeCall(cuStreamQuery(stream));
    //cudaSafeCall(cuEventRecord(eventEnd, stream));
    //cudaSafeCall(cuEventSynchronize(eventEnd));
    //cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    delete[] arglist;
}

void CudaReducer::runSumCplxKernel(int size, int blocks, int threads, CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	CudaKernel* kernel;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads)
    {
        case 512:
            kernel = sumCplx512; break;
        case 256:
            kernel = sumCplx256; break;
        case 128:
            kernel = sumCplx128; break;
        case 64:
            kernel = sumCplx64; break;
        case 32:
            kernel = sumCplx32; break;
        case 16:
            kernel = sumCplx16; break;
        case  8:
            kernel = sumCplx8; break;
        case  4:
            kernel = sumCplx4; break;
        case  2:
            kernel = sumCplx2; break;
        case  1:
            kernel = sumCplx1; break;
    }
	
    CUdeviceptr in_dptr = d_idata.GetDevicePtr();
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[3];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &n;

    //float ms;

    //CUevent eventStart;
    //CUevent eventEnd;
    //CUstream stream = 0;
    //cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    //cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    //cudaSafeCall(cuStreamQuery(stream));
    //cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(kernel->GetCUfunction(), dimGrid.x, dimGrid.y,
		dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, smemSize, stream, arglist,NULL));

    //cudaSafeCall(cuCtxSynchronize());

    //cudaSafeCall(cuStreamQuery(stream));
    //cudaSafeCall(cuEventRecord(eventEnd, stream));
    //cudaSafeCall(cuEventSynchronize(eventEnd));
    //cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    delete[] arglist;
}

int CudaReducer::GetOutBufferSize()
{
	int numBlocks = 0;
	int temp = 0;
	getNumBlocksAndThreads(voxelCount, numBlocks, temp);
	return numBlocks;
}

void CudaReducer::getNumBlocksAndThreads(int n, int &blocks, int &threads)
{    
    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    

    if (blocks > 2147483647) //Maximum of GTX Titan
    {
        printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, 2147483647, threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    blocks = min(maxBlocks, blocks);
}


unsigned int CudaReducer::nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

