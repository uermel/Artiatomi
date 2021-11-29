//
// Created by uermel on 10/1/21.
//

#include "DoseWeightingKernel.h"

using namespace Cuda;

DoseWeightingKernel::DoseWeightingKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("doseWeighting", aModule, aGridDim, aBlockDim, 0)
{

}

DoseWeightingKernel::DoseWeightingKernel(CUmodule aModule)
        : CudaKernel("doseWeighting", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float DoseWeightingKernel::operator()(Cuda::CudaDeviceVariable& img, size_t stride, int pixelcount, float dose, float pixelSizeInA)
{
    CUdeviceptr img_dptr = img.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &img_dptr;
    arglist[1] = &stride;
    arglist[2] = &pixelcount;
    arglist[3] = &dose;
    arglist[4] = &pixelSizeInA;

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