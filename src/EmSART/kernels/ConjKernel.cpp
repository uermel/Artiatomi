//
// Created by uermel on 10/1/21.
//

#include "ConjKernel.h"

using namespace Cuda;

ConjKernel::ConjKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("conjMul", aModule, aGridDim, aBlockDim, 0)
{

}

ConjKernel::ConjKernel(CUmodule aModule)
        : CudaKernel("conjMul", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float ConjKernel::operator()(Cuda::CudaDeviceVariable& img1, Cuda::CudaPitchedDeviceVariable& img2, size_t stride, int pixelcount)
{
    CUdeviceptr img_dptr1 = img1.GetDevicePtr();
    CUdeviceptr img_dptr2 = img2.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &img_dptr1;
    arglist[1] = &img_dptr2;
    arglist[2] = &stride;
    arglist[3] = &pixelcount;

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