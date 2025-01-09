//
// Created by uermel on 11/19/21.
//

#include "FPOrthoKernel.h"

using namespace Cuda;

FPOrthoKernel::FPOrthoKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("fpOrthoProject", aModule, aGridDim, aBlockDim, 0)
{

}

FPOrthoKernel::FPOrthoKernel(CUmodule aModule)
        : CudaKernel("fpOrthoProject", aModule, make_dim3(1, 1, 1), make_dim3(2, 2, 2), 0)
{

}

float FPOrthoKernel::operator()(uint2 projDim,
                                uint3 volDim,
                                ctfImageConstants imageConstants,
                                float4x4 systemMatrix,
                                CudaDeviceVariable& projection,
                                CudaSurfaceObject3D& volume,
                                int2 minmaxSlice)
{
    CUdeviceptr projPtr = projection.GetDevicePtr();
    CUsurfObject volObj = volume.GetSurfObject();

    int gridX = (int)ceilf((float)volDim.x/2);
    int gridY = (int)ceilf((float)volDim.y/2);
    int gridZ = (int)ceilf((float)volDim.z/2);

    SetGridDimensions(make_dim3(gridX, gridY, gridZ));
    SetBlockDimensions(make_dim3(2, 2, 2));

    void** arglist = (void**)new void*[7];

    arglist[0] = &projDim;
    arglist[1] = &volDim;
    arglist[2] = &imageConstants;
    arglist[3] = &systemMatrix;
    arglist[4] = &projPtr;
    arglist[5] = &volObj;
    arglist[6] = &minmaxSlice;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100000000);
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


FPOrthoSSKernel::FPOrthoSSKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("fpOrthoProjectSS", aModule, aGridDim, aBlockDim, 0)
{

}

FPOrthoSSKernel::FPOrthoSSKernel(CUmodule aModule)
        : CudaKernel("fpOrthoProjectSS", aModule, make_dim3(1, 1, 1), make_dim3(2, 2, 2), 0)
{

}

float FPOrthoSSKernel::operator()(uint2 projDim,
                                uint3 volDim,
                                ctfImageConstants imageConstants,
                                float4x4 systemMatrix,
                                CudaDeviceVariable& projection,
                                CudaSurfaceObject3D& volume)
{
    CUdeviceptr projPtr = projection.GetDevicePtr();
    CUsurfObject volObj = volume.GetSurfObject();

    int gridX = (int)ceilf((float)volDim.x/2);
    int gridY = (int)ceilf((float)volDim.y/2);
    int gridZ = (int)ceilf((float)volDim.z/2);

    SetGridDimensions(make_dim3(gridX, gridY, gridZ));
    SetBlockDimensions(make_dim3(2, 2, 2));

    void** arglist = (void**)new void*[6];

    arglist[0] = &projDim;
    arglist[1] = &volDim;
    arglist[2] = &imageConstants;
    arglist[3] = &systemMatrix;
    arglist[4] = &projPtr;
    arglist[5] = &volObj;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100000000);
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

FPOrthoOVKernel::FPOrthoOVKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("fpOrthoProjectOV", aModule, aGridDim, aBlockDim, 0)
{

}

FPOrthoOVKernel::FPOrthoOVKernel(CUmodule aModule)
        : CudaKernel("fpOrthoProjectOV", aModule, make_dim3(1, 1, 1), make_dim3(2, 2, 2), 0)
{

}

float FPOrthoOVKernel::operator()(uint2 projDim,
                                  uint3 volDim,
                                  ctfImageConstants imageConstants,
                                  float4x4 systemMatrix,
                                  CudaDeviceVariable& projection,
                                  CudaSurfaceObject3D& volume,
                                  int2 minmaxSlice,
                                  CudaTextureObject3D& overlapVolume,
                                  float4x4 childToParent)
{
    CUdeviceptr projPtr = projection.GetDevicePtr();
    CUsurfObject volObj = volume.GetSurfObject();
    CUtexObject ovlObj = overlapVolume.GetTexObject();

    int gridX = (int)ceilf((float)volDim.x/2);
    int gridY = (int)ceilf((float)volDim.y/2);
    int gridZ = (int)ceilf((float)volDim.z/2);

    SetGridDimensions(make_dim3(gridX, gridY, gridZ));
    SetBlockDimensions(make_dim3(2, 2, 2));

    void** arglist = (void**)new void*[9];

    arglist[0] = &projDim;
    arglist[1] = &volDim;
    arglist[2] = &imageConstants;
    arglist[3] = &systemMatrix;
    arglist[4] = &projPtr;
    arglist[5] = &volObj;
    arglist[6] = &minmaxSlice;
    arglist[7] = &ovlObj;
    arglist[8] = &childToParent;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100000000);
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