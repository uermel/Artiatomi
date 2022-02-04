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
#include "CudaKernelBinaries.h"

CudaReducer::CudaReducer(int aVoxelCount, CUstream aStream, CudaContext* context)
	: voxelCount(aVoxelCount), stream(aStream), ctx(context)
{
	// CUmodule cuMod = ctx->LoadModule("kernel.ptx_bu");
    CUmodule cuMod = ctx->LoadModulePTX(SubTomogramAverageKernel, 0, false, false);
		
	sum512 = new CudaKernel("reduce_512", cuMod);
	sum256 = new CudaKernel("reduce_256", cuMod);
	sum128 = new CudaKernel("reduce_128", cuMod);
	sum64  = new CudaKernel("reduce_64",  cuMod);
	sum32  = new CudaKernel("reduce_32",  cuMod);
	sum16  = new CudaKernel("reduce_16",  cuMod);
	sum8   = new CudaKernel("reduce_8",   cuMod);
	sum4   = new CudaKernel("reduce_4",   cuMod);
	sum2   = new CudaKernel("reduce_2",   cuMod);
	sum1   = new CudaKernel("reduce_1",   cuMod);

		
	sumCplx512 = new CudaKernel("reduceCplx_512", cuMod);
	sumCplx256 = new CudaKernel("reduceCplx_256", cuMod);
	sumCplx128 = new CudaKernel("reduceCplx_128", cuMod);
	sumCplx64  = new CudaKernel("reduceCplx_64",  cuMod);
	sumCplx32  = new CudaKernel("reduceCplx_32",  cuMod);
	sumCplx16  = new CudaKernel("reduceCplx_16",  cuMod);
	sumCplx8   = new CudaKernel("reduceCplx_8",   cuMod);
	sumCplx4   = new CudaKernel("reduceCplx_4",   cuMod);
	sumCplx2   = new CudaKernel("reduceCplx_2",   cuMod);
	sumCplx1   = new CudaKernel("reduceCplx_1",   cuMod);

    maskedSumCplx512 = new CudaKernel("maskedReduceCplx_512", cuMod);
    maskedSumCplx256 = new CudaKernel("maskedReduceCplx_256", cuMod);
    maskedSumCplx128 = new CudaKernel("maskedReduceCplx_128", cuMod);
    maskedSumCplx64  = new CudaKernel("maskedReduceCplx_64",  cuMod);
    maskedSumCplx32  = new CudaKernel("maskedReduceCplx_32",  cuMod);
    maskedSumCplx16  = new CudaKernel("maskedReduceCplx_16",  cuMod);
    maskedSumCplx8   = new CudaKernel("maskedReduceCplx_8",   cuMod);
    maskedSumCplx4   = new CudaKernel("maskedReduceCplx_4",   cuMod);
    maskedSumCplx2   = new CudaKernel("maskedReduceCplx_2",   cuMod);
    maskedSumCplx1   = new CudaKernel("maskedReduceCplx_1",   cuMod);

	sumSqrCplx512 = new CudaKernel("reduceSqrCplx_512", cuMod);
	sumSqrCplx256 = new CudaKernel("reduceSqrCplx_256", cuMod);
	sumSqrCplx128 = new CudaKernel("reduceSqrCplx_128", cuMod);
	sumSqrCplx64  = new CudaKernel("reduceSqrCplx_64",  cuMod);
	sumSqrCplx32  = new CudaKernel("reduceSqrCplx_32",  cuMod);
	sumSqrCplx16  = new CudaKernel("reduceSqrCplx_16",  cuMod);
	sumSqrCplx8   = new CudaKernel("reduceSqrCplx_8",   cuMod);
	sumSqrCplx4   = new CudaKernel("reduceSqrCplx_4",   cuMod);
	sumSqrCplx2   = new CudaKernel("reduceSqrCplx_2",   cuMod);
	sumSqrCplx1   = new CudaKernel("reduceSqrCplx_1",   cuMod);

    maxIndex512 = new CudaKernel("maxIndex_512", cuMod);
    maxIndex256 = new CudaKernel("maxIndex_256", cuMod);
    maxIndex128 = new CudaKernel("maxIndex_128", cuMod);
    maxIndex64  = new CudaKernel("maxIndex_64",  cuMod);
    maxIndex32  = new CudaKernel("maxIndex_32",  cuMod);
    maxIndex16  = new CudaKernel("maxIndex_16",  cuMod);
    maxIndex8   = new CudaKernel("maxIndex_8",   cuMod);
    maxIndex4   = new CudaKernel("maxIndex_4",   cuMod);
    maxIndex2   = new CudaKernel("maxIndex_2",   cuMod);
    maxIndex1   = new CudaKernel("maxIndex_1",   cuMod);

    maxIndexCplx512 = new CudaKernel("maxIndexCplx_512", cuMod);
    maxIndexCplx256 = new CudaKernel("maxIndexCplx_256", cuMod);
    maxIndexCplx128 = new CudaKernel("maxIndexCplx_128", cuMod);
    maxIndexCplx64  = new CudaKernel("maxIndexCplx_64",  cuMod);
    maxIndexCplx32  = new CudaKernel("maxIndexCplx_32",  cuMod);
    maxIndexCplx16  = new CudaKernel("maxIndexCplx_16",  cuMod);
    maxIndexCplx8   = new CudaKernel("maxIndexCplx_8",   cuMod);
    maxIndexCplx4   = new CudaKernel("maxIndexCplx_4",   cuMod);
    maxIndexCplx2   = new CudaKernel("maxIndexCplx_2",   cuMod);
    maxIndexCplx1   = new CudaKernel("maxIndexCplx_1",   cuMod);

	maxIndexMasked512 = new CudaKernel("maxIndexMasked_512", cuMod);
	maxIndexMasked256 = new CudaKernel("maxIndexMasked_256", cuMod);
	maxIndexMasked128 = new CudaKernel("maxIndexMasked_128", cuMod);
	maxIndexMasked64  = new CudaKernel("maxIndexMasked_64",  cuMod);
	maxIndexMasked32  = new CudaKernel("maxIndexMasked_32",  cuMod);
	maxIndexMasked16  = new CudaKernel("maxIndexMasked_16",  cuMod);
	maxIndexMasked8   = new CudaKernel("maxIndexMasked_8",   cuMod);
	maxIndexMasked4   = new CudaKernel("maxIndexMasked_4",   cuMod);
	maxIndexMasked2   = new CudaKernel("maxIndexMasked_2",   cuMod);
	maxIndexMasked1   = new CudaKernel("maxIndexMasked_1",   cuMod);

	maxIndexMaskedCplx512 = new CudaKernel("maxIndexMaskedCplx_512", cuMod);
	maxIndexMaskedCplx256 = new CudaKernel("maxIndexMaskedCplx_256", cuMod);
	maxIndexMaskedCplx128 = new CudaKernel("maxIndexMaskedCplx_128", cuMod);
	maxIndexMaskedCplx64  = new CudaKernel("maxIndexMaskedCplx_64",  cuMod);
	maxIndexMaskedCplx32  = new CudaKernel("maxIndexMaskedCplx_32",  cuMod);
	maxIndexMaskedCplx16  = new CudaKernel("maxIndexMaskedCplx_16",  cuMod);
	maxIndexMaskedCplx8   = new CudaKernel("maxIndexMaskedCplx_8",   cuMod);
	maxIndexMaskedCplx4   = new CudaKernel("maxIndexMaskedCplx_4",   cuMod);
	maxIndexMaskedCplx2   = new CudaKernel("maxIndexMaskedCplx_2",   cuMod);
	maxIndexMaskedCplx1   = new CudaKernel("maxIndexMaskedCplx_1",   cuMod);


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


void CudaReducer::MaxIndexMasked(CudaDeviceVariable& d_idata,
                                 CudaDeviceVariable& d_odata,
                                 CudaDeviceVariable& d_maskdata,
                                 CudaDeviceVariable& d_index)
{
	int blocks;
	int threads;
	float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

	
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runMaxIndexMaskedKernel(voxelCount, blocks, threads, d_idata, d_odata, d_maskdata, d_index, false);

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

void CudaReducer::MaxIndexMaskedCplx(CudaDeviceVariable& d_idata,
                                     CudaDeviceVariable& d_odata,
                                     CudaDeviceVariable& d_maskdata,
                                     CudaDeviceVariable& d_index)
{
	int blocks;
	int threads;
	float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

	
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runMaxIndexMaskedCplxKernel(voxelCount, blocks, threads, d_idata, d_odata, d_maskdata, d_index, false);

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

void CudaReducer::MaskedSumCplx(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_mask, CudaDeviceVariable& d_odata)
{
    int blocks;
    int threads;
    float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;


    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runMaskedSumCplxKernel(voxelCount, blocks, threads, d_idata, d_mask, d_odata);

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

void CudaReducer::runMaxIndexMaskedKernel(int size, int blocks, int threads,
                                    CudaDeviceVariable& d_idata,
                                    CudaDeviceVariable& d_odata,
                                    CudaDeviceVariable& d_maskdata,
                                    CudaDeviceVariable& d_index,
                                    bool readIndex)
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
            kernel = maxIndexMasked512; break;
        case 256:
            kernel = maxIndexMasked256; break;
        case 128:
            kernel = maxIndexMasked128; break;
        case 64:
            kernel = maxIndexMasked64; break;
        case 32:
            kernel = maxIndexMasked32; break;
        case 16:
            kernel = maxIndexMasked16; break;
        case  8:
            kernel = maxIndexMasked8; break;
        case  4:
            kernel = maxIndexMasked4; break;
        case  2:
            kernel = maxIndexMasked2; break;
        case  1:
            kernel = maxIndexMasked1; break;
    }
	
    CUdeviceptr in_dptr = d_idata.GetDevicePtr();
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();
    CUdeviceptr mask_dptr = d_maskdata.GetDevicePtr();
    CUdeviceptr index_dptr = d_index.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[6];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &mask_dptr;
    arglist[3] = &index_dptr;
    arglist[4] = &n;
	arglist[5] = &readIndex;

    cudaSafeCall(cuLaunchKernel(kernel->GetCUfunction(), dimGrid.x, dimGrid.y,
		dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, smemSize, stream, arglist,NULL));

    delete[] arglist;
}

void CudaReducer::runMaxIndexMaskedCplxKernel(int size, int blocks, int threads,
                                              CudaDeviceVariable& d_idata,
                                              CudaDeviceVariable& d_odata,
                                              CudaDeviceVariable& d_maskdata,
                                              CudaDeviceVariable& d_index,
                                              bool readIndex)
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
            kernel = maxIndexMaskedCplx512; break;
        case 256:
            kernel = maxIndexMaskedCplx256; break;
        case 128:
            kernel = maxIndexMaskedCplx128; break;
        case 64:
            kernel = maxIndexMaskedCplx64; break;
        case 32:
            kernel = maxIndexMaskedCplx32; break;
        case 16:
            kernel = maxIndexMaskedCplx16; break;
        case  8:
            kernel = maxIndexMaskedCplx8; break;
        case  4:
            kernel = maxIndexMaskedCplx4; break;
        case  2:
            kernel = maxIndexMaskedCplx2; break;
        case  1:
            kernel = maxIndexMaskedCplx1; break;
    }
	
    CUdeviceptr in_dptr = d_idata.GetDevicePtr();
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();
    CUdeviceptr mask_dptr = d_maskdata.GetDevicePtr();
    CUdeviceptr index_dptr = d_index.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[6];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &mask_dptr;
    arglist[3] = &index_dptr;
    arglist[4] = &n;
	arglist[5] = &readIndex;

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

void CudaReducer::runMaskedSumCplxKernel(int size,
                                         int blocks,
                                         int threads,
                                         CudaDeviceVariable& d_idata,
                                         CudaDeviceVariable& d_mask,
                                         CudaDeviceVariable& d_odata)
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
            kernel = maskedSumCplx512; break;
        case 256:
            kernel = maskedSumCplx256; break;
        case 128:
            kernel = maskedSumCplx128; break;
        case 64:
            kernel = maskedSumCplx64; break;
        case 32:
            kernel = maskedSumCplx32; break;
        case 16:
            kernel = maskedSumCplx16; break;
        case  8:
            kernel = maskedSumCplx8; break;
        case  4:
            kernel = maskedSumCplx4; break;
        case  2:
            kernel = maskedSumCplx2; break;
        case  1:
            kernel = maskedSumCplx1; break;
    }

    CUdeviceptr in_dptr = d_idata.GetDevicePtr();
    CUdeviceptr mask_dptr = d_mask.GetDevicePtr();
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[4];

    arglist[0] = &in_dptr;
    arglist[1] = &mask_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &n;

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

