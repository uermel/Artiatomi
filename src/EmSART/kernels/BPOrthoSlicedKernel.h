//
// Created by uermel on 11/17/21.
//

#ifndef ARTIATOMI_BPORTHOSLICEDKERNEL_H
#define ARTIATOMI_BPORTHOSLICEDKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class BPOrthoSlicedKernel : public Cuda::CudaKernel
{
public:
    int slicenum;
    Cuda::CudaDeviceVariable d_textures;

    explicit BPOrthoSlicedKernel(CUmodule aModule);
    BPOrthoSlicedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    void setSlices(vector<Cuda::CudaTextureObject2D*> textures, int sliceNumber);

    float operator()(uint2 projDim,
                     uint3 volDim,
                     float lambda,
                     ctfImageConstants imageConstants,
                     float4x4 systemMatrix,
                     Cuda::CudaSurfaceObject3D& volume,
                     int2 minmaxSlice);
};

class BPOrthoSlicedAddKernel : public Cuda::CudaKernel
{
public:
    int slicenum;
    Cuda::CudaDeviceVariable d_textures;

    explicit BPOrthoSlicedAddKernel(CUmodule aModule);
    BPOrthoSlicedAddKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    void setSlices(vector<Cuda::CudaTextureObject2D*> textures, int sliceNumber);

    float operator()(uint2 projDim,
                     uint3 volDim,
                     float lambda,
                     ctfImageConstants imageConstants,
                     float4x4 systemMatrix,
                     Cuda::CudaSurfaceObject3D& volume,
                     int2 minmaxSlice);
};

class BPOrthoSlicedAddSSKernel : public Cuda::CudaKernel
{
public:
    int slicenum;
    Cuda::CudaDeviceVariable d_textures;

    explicit BPOrthoSlicedAddSSKernel(CUmodule aModule);
    BPOrthoSlicedAddSSKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    void setSlices(vector<Cuda::CudaTextureObject2D*> textures, int sliceNumber);

    float operator()(uint2 projDim,
                     uint3 volDim,
                     float lambda,
                     ctfImageConstants imageConstants,
                     float4x4 systemMatrix,
                     Cuda::CudaSurfaceObject3D& volume);
};

#endif //ARTIATOMI_BPORTHOSLICEDKERNEL_H
