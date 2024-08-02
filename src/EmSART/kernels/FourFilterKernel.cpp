//
// Created by uermel on 10/1/21.
//

#include "FourFilterKernel.h"

using namespace Cuda;

FourFilterKernel::FourFilterKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("fourierFilter", aModule, aGridDim, aBlockDim, 0)
{

}

FourFilterKernel::FourFilterKernel(CUmodule aModule)
        : CudaKernel("fourierFilter", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float FourFilterKernel::operator()(Cuda::CudaDeviceVariable& img,
                                   size_t stride,
                                   uint2 pixelcount,
                                   float2 asymCorrFac,
                                   float lp, float hp, float lps, float hps)
{
    CUdeviceptr img_dptr = img.GetDevicePtr();

    void** arglist = (void**)new void*[8];

    arglist[0] = &img_dptr;
    arglist[1] = &stride;
    arglist[2] = &pixelcount;
    arglist[3] = &asymCorrFac;
    arglist[4] = &lp;
    arglist[5] = &hp;
    arglist[6] = &lps;
    arglist[7] = &hps;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

    cudaSafeCall(cuCtxSynchronize());

    cudaSafeCall(cuEventRecord(eventEnd, stream));
    cudaSafeCall(cuEventSynchronize(eventEnd));
    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    cudaSafeCall(cuEventDestroy(eventStart));
    cudaSafeCall(cuEventDestroy(eventEnd));

    delete[] arglist;
    return ms;
}