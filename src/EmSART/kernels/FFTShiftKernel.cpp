//
// Created by uermel on 2/7/22.
//

#include "FFTShiftKernel.h"

using namespace Cuda;

FFTShiftKernel::FFTShiftKernel(CUmodule aModule)
        : CudaKernel("fftshift_real", aModule)
{

}

float FFTShiftKernel::operator()(int size,
                                 Cuda::CudaDeviceVariable& image_in,
                                 Cuda::CudaPitchedDeviceVariable& image_out,
                                 float scaleFactor)
{
    CUdeviceptr in_dptr = image_in.GetDevicePtr();
    CUdeviceptr out_dptr = image_out.GetDevicePtr();
    size_t stride_in = size * sizeof(float);
    size_t stride_out = image_out.GetPitch();

    void** arglist = (void**)new void*[6];

    arglist[0] = &size;
    arglist[1] = &stride_in;
    arglist[2] = &stride_out;
    arglist[3] = &in_dptr;
    arglist[4] = &out_dptr;
    arglist[5] = &scaleFactor;

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