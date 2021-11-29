//
// Created by uermel on 10/2/21.
//

#include "ApplyMaskKernel.h"

using namespace Cuda;

ApplyMaskKernel::ApplyMaskKernel(CUmodule aModule, int aSize)
        : CudaKernel("applyMask", aModule, make_dim3((aSize + 7) / 8, (aSize + 7) / 8, (aSize + 7) / 8), make_dim3(8, 8, 8), 0),
          size(aSize)
{

}

float ApplyMaskKernel::operator()(Cuda::CudaSurfaceObject3D& volume, Cuda::CudaDeviceVariable& mask, Cuda::CudaDeviceVariable& tempStore, int3 volmin, int3 volmax, int3 dimMask, int3 radiusMask, int3 centerInVol)
{
    CUdeviceptr d_volume = volume.GetSurfObject();
    CUdeviceptr d_mask = mask.GetDevicePtr();
    CUdeviceptr d_tempStore = tempStore.GetDevicePtr();

    void** arglist = (void**)new void*[8];

    arglist[0] = &d_volume;
    arglist[1] = &d_mask;
    arglist[2] = &d_tempStore;
    arglist[3] = &volmin;
    arglist[4] = &volmax;
    arglist[5] = &dimMask;
    arglist[6] = &radiusMask;
    arglist[7] = &centerInVol;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    //printf("mGridDim x:%d y:%d z:%d\n", mGridDim.x, mGridDim.y, mGridDim.z);
    //printf("mBlockDim x:%d y:%d z:%d\n", mBlockDim.x, mBlockDim.y, mBlockDim.z);
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