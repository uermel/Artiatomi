//
// Created by uermel on 10/2/21.
//

#include "SphericalMaskKernel.h"

using namespace Cuda;

SphericalMaskKernel::SphericalMaskKernel(CUmodule aModule, int aSize)
        : CudaKernel("sphericalMask3D", aModule, make_dim3((aSize + 7) / 8, (aSize + 7) / 8, (aSize + 7) / 8), make_dim3(8, 8, 8), 0),
          size(aSize)
{

}

float SphericalMaskKernel::operator()(Cuda::CudaDeviceVariable & aVolOut, float radius)
{
    CUdeviceptr out_dptr = aVolOut.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &size;
    arglist[1] = &radius;
    arglist[2] = &out_dptr;

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
