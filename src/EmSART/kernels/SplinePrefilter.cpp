//
// Created by uermel on 10/4/21.
//

#include "SplinePrefilter.h"

using namespace Cuda;


uint PowTwoDivider(uint n)
{
    if (n == 0) return 0;
    uint divider = 1;
    while ((n & divider) == 0) divider <<= 1;
    return divider;
}

SplinePrefilter2DX::SplinePrefilter2DX(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients2DX", aModule, aGridDim, aBlockDim, 0)
{

}

SplinePrefilter2DX::SplinePrefilter2DX(CUmodule aModule)
        : CudaKernel("SamplesToCoefficients2DX", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float SplinePrefilter2DX::operator()(Cuda::CudaPitchedDeviceVariable& image, int width, int height)
{
    // Block/Grid
    SetBlockDimensions(min(PowTwoDivider(height), 64), 1, 1);
    SetGridDimensions(height / mBlockDim.x);

    CUdeviceptr image_dptr = image.GetDevicePtr();
    uint p = image.GetPitch();
    uint w = (uint) width;
    uint h = (uint) height;

    void** arglist = (void**)new void*[4];

    arglist[0] = &image_dptr;
    arglist[1] = &p;
    arglist[2] = &w;
    arglist[3] = &h;

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

SplinePrefilter2DY::SplinePrefilter2DY(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients2DY", aModule, aGridDim, aBlockDim, 0)
{

}

SplinePrefilter2DY::SplinePrefilter2DY(CUmodule aModule)
        : CudaKernel("SamplesToCoefficients2DY", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float SplinePrefilter2DY::operator()(Cuda::CudaPitchedDeviceVariable& image, int width, int height)
{
    // Block/Grid
    SetBlockDimensions(min(PowTwoDivider(width), 64), 1, 1);
    SetGridDimensions(width / mBlockDim.x);

    CUdeviceptr image_dptr = image.GetDevicePtr();
    uint p = image.GetPitch();
    uint w = (uint) width;
    uint h = (uint) height;

    void** arglist = (void**)new void*[4];

    arglist[0] = &image_dptr;
    arglist[1] = &p;
    arglist[2] = &w;
    arglist[3] = &h;

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