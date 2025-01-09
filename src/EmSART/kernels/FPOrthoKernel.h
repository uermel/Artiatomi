//
// Created by uermel on 11/19/21.
//

#ifndef ARTIATOMI_FPORTHOKERNEL_H
#define ARTIATOMI_FPORTHOKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class FPOrthoKernel : public Cuda::CudaKernel
{
public:
    explicit FPOrthoKernel(CUmodule aModule);
    FPOrthoKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(uint2 projDim,
                     uint3 volDim,
                     ctfImageConstants imageConstants,
                     float4x4 systemMatrix,
                     Cuda::CudaDeviceVariable& projection,
                     Cuda::CudaSurfaceObject3D& volume,
                     int2 minmaxSlice);
};

class FPOrthoSSKernel : public Cuda::CudaKernel
{
public:
    explicit FPOrthoSSKernel(CUmodule aModule);
    FPOrthoSSKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(uint2 projDim,
                     uint3 volDim,
                     ctfImageConstants imageConstants,
                     float4x4 systemMatrix,
                     Cuda::CudaDeviceVariable& projection,
                     Cuda::CudaSurfaceObject3D& volume);
};

class FPOrthoOVKernel : public Cuda::CudaKernel
{
public:
    explicit FPOrthoOVKernel(CUmodule aModule);
    FPOrthoOVKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(uint2 projDim,
                     uint3 volDim,
                     ctfImageConstants imageConstants,
                     float4x4 systemMatrix,
                     Cuda::CudaDeviceVariable& projection,
                     Cuda::CudaSurfaceObject3D& volume,
                     int2 minmaxSlice,
                     Cuda::CudaTextureObject3D& overlapVolume,
                     float4x4 childToParent);
};


#endif //ARTIATOMI_FPORTHOKERNEL_H
