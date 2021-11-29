//
// Created by uermel on 10/1/21.
//

#include "FindPeakKernel.h"

using namespace Cuda;

FindPeakKernel::FindPeakKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("findPeak", aModule, aGridDim, aBlockDim, 0)
{

}

FindPeakKernel::FindPeakKernel(CUmodule aModule)
        : CudaKernel("findPeak", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float FindPeakKernel::operator()(Cuda::CudaDeviceVariable& img1, size_t stride, Cuda::CudaPitchedDeviceVariable& mask, int pixelcount, float maxThreashold)
{
    CUdeviceptr img_dptr1 = img1.GetDevicePtr();
    CUdeviceptr mask_dptr = mask.GetDevicePtr();
    size_t pitchMask = mask.GetPitch();

    void** arglist = (void**)new void*[6];

    arglist[0] = &img_dptr1;
    arglist[1] = &stride;
    arglist[2] = &mask_dptr;
    arglist[3] = &pitchMask;
    arglist[4] = &pixelcount;
    arglist[5] = &maxThreashold;

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