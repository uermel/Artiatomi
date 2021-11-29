//
// Created by uermel on 10/1/21.
//

#include "DimBordersKernel.h"

using namespace Cuda;

DimBordersKernel::DimBordersKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("dimBorders", aModule, aGridDim, aBlockDim, 0)
{

}

DimBordersKernel::DimBordersKernel(CUmodule aModule)
        : CudaKernel("dimBorders", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float DimBordersKernel::operator()(CudaPitchedDeviceVariable& image, float4 crop, float4 cropDim)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    int proj_x = (int)image.GetWidth();
    int proj_y = (int)image.GetHeight();
    size_t stride = image.GetPitch();

    void** arglist = (void**)new void* [6];

    arglist[0] = &proj_x;
    arglist[1] = &proj_y;
    arglist[2] = &stride;
    arglist[3] = &image_dptr;
    arglist[4] = &crop;
    arglist[5] = &cropDim;

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