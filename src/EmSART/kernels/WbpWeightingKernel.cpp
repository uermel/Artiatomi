//
// Created by uermel on 10/1/21.
//

#include "WbpWeightingKernel.h"

using namespace Cuda;

WbpWeightingKernel::WbpWeightingKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("wbpWeightingNew", aModule, aGridDim, aBlockDim, 0)
{

}

WbpWeightingKernel::WbpWeightingKernel(CUmodule aModule)
        : CudaKernel("wbpWeightingNew", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float WbpWeightingKernel::operator()(CudaDeviceVariable& img,
                                     size_t stride,
                                     uint2 imDim,
                                     float2 asymCorrFac,
                                     FilterMethod fm,
                                     int projectionCount,
                                     float thickness,
                                     Matrix<double>& Mproj,
                                     CudaDeviceVariable& Mdet)
{
    CUdeviceptr img_dptr = img.GetDevicePtr();
    CUdeviceptr Mdet_dptr = Mdet.GetDevicePtr();
    float3x3 Mp = MatrixTo3x3(Mproj);

    void** arglist = (void**)new void*[9];

    arglist[0] = &img_dptr;
    arglist[1] = &stride;
    arglist[2] = &imDim;
    arglist[3] = &asymCorrFac;
    arglist[4] = &fm;
    arglist[5] = &projectionCount;
    arglist[6] = &thickness;
    arglist[7] = &Mp;
    arglist[8] = &Mdet_dptr;

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