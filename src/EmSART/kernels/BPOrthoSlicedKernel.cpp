//
// Created by uermel on 11/17/21.
//

#include "BPOrthoSlicedKernel.h"

using namespace Cuda;

BPOrthoSlicedKernel::BPOrthoSlicedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("bpOrthoOverwrite", aModule, aGridDim, aBlockDim, 0)
{

}

BPOrthoSlicedKernel::BPOrthoSlicedKernel(CUmodule aModule)
        : CudaKernel("bpOrthoOverwrite", aModule, make_dim3(1, 1, 1), make_dim3(4, 4, 4), 0)
{

}

void BPOrthoSlicedKernel::setSlices(vector<Cuda::CudaTextureObject2D *> textures, int sliceNumber)
{
    slicenum = sliceNumber;
    d_textures.Alloc(slicenum * sizeof(CUtexObject));
    auto h_textures = new CUsurfObject[slicenum];

    for (int slice=0; slice < slicenum; slice++){
        h_textures[slice] = textures[slice]->GetTexObject();
    }

    d_textures.CopyHostToDevice(h_textures, slicenum * sizeof(CUsurfObject));

    delete[] h_textures;
}

float BPOrthoSlicedKernel::operator()(uint2 projDim,
                                      uint3 volDim,
                                      float lambda,
                                      ctfImageConstants imageConstants,
                                      float4x4 systemMatrix,
                                      CudaSurfaceObject3D& volume,
                                      int2 minmaxSlice)
{
    CUtexObject texPtr = d_textures.GetDevicePtr();
    CUsurfObject volObj = volume.GetSurfObject();

    int gridX = (int)ceilf((float)volDim.x/2.f);
    int gridY = (int)ceilf((float)volDim.y/2.f);
    int gridZ = (int)ceilf((float)volDim.z/2.f);

    SetGridDimensions(make_dim3(gridX, gridY, gridZ));
    SetBlockDimensions(make_dim3(2, 2, 2));

    void** arglist = (void**)new void*[8];

    arglist[0] = &projDim;
    arglist[1] = &volDim;
    arglist[2] = &lambda;
    arglist[3] = &imageConstants;
    arglist[4] = &systemMatrix;
    arglist[5] = &texPtr;
    arglist[6] = &volObj;
    arglist[7] = &minmaxSlice;

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

BPOrthoSlicedAddKernel::BPOrthoSlicedAddKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("bpOrthoAdd", aModule, aGridDim, aBlockDim, 0)
{

}

BPOrthoSlicedAddKernel::BPOrthoSlicedAddKernel(CUmodule aModule)
        : CudaKernel("bpOrthoAdd", aModule, make_dim3(1, 1, 1), make_dim3(4, 4, 4), 0)
{

}

void BPOrthoSlicedAddKernel::setSlices(vector<Cuda::CudaTextureObject2D *> textures, int sliceNumber)
{
    slicenum = sliceNumber;
    d_textures.Alloc(slicenum * sizeof(CUtexObject));
    auto h_textures = new CUsurfObject[slicenum];

    for (int slice=0; slice < slicenum; slice++){
        h_textures[slice] = textures[slice]->GetTexObject();
    }

    d_textures.CopyHostToDevice(h_textures, slicenum * sizeof(CUsurfObject));

    delete[] h_textures;
}

float BPOrthoSlicedAddKernel::operator()(uint2 projDim,
                                         uint3 volDim,
                                         float lambda,
                                         ctfImageConstants imageConstants,
                                         float4x4 systemMatrix,
                                         CudaSurfaceObject3D& volume,
                                         int2 minmaxSlice)
{
    CUtexObject texPtr = d_textures.GetDevicePtr();
    CUsurfObject volObj = volume.GetSurfObject();

    int gridX = (int)ceilf((float)volDim.x/2.f);
    int gridY = (int)ceilf((float)volDim.y/2.f);
    int gridZ = (int)ceilf((float)volDim.z/2.f);

    SetGridDimensions(make_dim3(gridX, gridY, gridZ));
    SetBlockDimensions(make_dim3(2, 2, 2));

    void** arglist = (void**)new void*[8];

    arglist[0] = &projDim;
    arglist[1] = &volDim;
    arglist[2] = &lambda;
    arglist[3] = &imageConstants;
    arglist[4] = &systemMatrix;
    arglist[5] = &texPtr;
    arglist[6] = &volObj;
    arglist[7] = &minmaxSlice;

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


BPOrthoSlicedAddSSKernel::BPOrthoSlicedAddSSKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("bpOrthoAddSS", aModule, aGridDim, aBlockDim, 0)
{

}

BPOrthoSlicedAddSSKernel::BPOrthoSlicedAddSSKernel(CUmodule aModule)
        : CudaKernel("bpOrthoAddSS", aModule, make_dim3(1, 1, 1), make_dim3(4, 4, 4), 0)
{

}

void BPOrthoSlicedAddSSKernel::setSlices(vector<Cuda::CudaTextureObject2D *> textures, int sliceNumber)
{
    slicenum = sliceNumber;
    d_textures.Alloc(slicenum * sizeof(CUtexObject));
    auto h_textures = new CUsurfObject[slicenum];

    for (int slice=0; slice < slicenum; slice++){
        h_textures[slice] = textures[slice]->GetTexObject();
    }

    d_textures.CopyHostToDevice(h_textures, slicenum * sizeof(CUsurfObject));

    delete[] h_textures;
}

float BPOrthoSlicedAddSSKernel::operator()(uint2 projDim,
                                           uint3 volDim,
                                           float lambda,
                                           ctfImageConstants imageConstants,
                                           float4x4 systemMatrix,
                                           CudaSurfaceObject3D& volume)
{
    CUtexObject texPtr = d_textures.GetDevicePtr();
    CUsurfObject volObj = volume.GetSurfObject();

    int gridX = (int)ceilf((float)volDim.x/2.f);
    int gridY = (int)ceilf((float)volDim.y/2.f);
    int gridZ = (int)ceilf((float)volDim.z/2.f);

    SetGridDimensions(make_dim3(gridX, gridY, gridZ));
    SetBlockDimensions(make_dim3(2, 2, 2));

    void** arglist = (void**)new void*[7];

    arglist[0] = &projDim;
    arglist[1] = &volDim;
    arglist[2] = &lambda;
    arglist[3] = &imageConstants;
    arglist[4] = &systemMatrix;
    arglist[5] = &texPtr;
    arglist[6] = &volObj;

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
