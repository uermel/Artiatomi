//
// Created by uermel on 10/5/21.
//

#ifndef ARTIATOMI_CUBICRESAMPLEKERNEL_H
#define ARTIATOMI_CUBICRESAMPLEKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class CubicResampleKernel2D : public Cuda::CudaKernel
{
public:
    CubicResampleKernel2D(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    CubicResampleKernel2D(CUmodule aModule);

    float operator()(Cuda::CudaTextureObject2D& inimage,
                     Cuda::CudaPitchedDeviceVariable& outimage);

    float operator()(Cuda::CudaTextureObject2D& inimage,
                     Cuda::CudaDeviceVariable& outimage,
                     int width, int height, int z);
};

class CubicResampleKernel3D : public Cuda::CudaKernel
{
public:
    CubicResampleKernel3D(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    CubicResampleKernel3D(CUmodule aModule);

    float operator()(Cuda::CudaTextureObject3D& involume,
                     Cuda::CudaSurfaceObject3D& outvolume,
                     Volume<float>* vol,
                     bool addToExisting);
};

class Add3DKernel : public Cuda::CudaKernel
{
public:
    explicit Add3DKernel(CUmodule aModule);
    Add3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaSurfaceObject3D& involume,
                     Cuda::CudaSurfaceObject3D& outvolume,
                     uint3 volDim,
                     float scaleFactor = 1.f);
};

class Set3DKernel : public Cuda::CudaKernel
{
public:
    explicit Set3DKernel(CUmodule aModule);
    Set3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaSurfaceObject3D& inoutvolume,
                     float value,
                     uint3 voldim);
};

class Add3DMaskedKernel : public Cuda::CudaKernel
{
public:
    explicit Add3DMaskedKernel(CUmodule aModule);
    Add3DMaskedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaSurfaceObject3D& involume,
                     Cuda::CudaSurfaceObject3D& outvolume,
                     Cuda::CudaSurfaceObject3D& maskvolume,
                     uint3 volDim,
                     float scaleFactor = 1.f);
};

class Mask3DKernel : public Cuda::CudaKernel
{
public:
    explicit Mask3DKernel(CUmodule aModule);
    Mask3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaSurfaceObject3D& involume,
                     Cuda::CudaSurfaceObject3D& outvolume,
                     Cuda::CudaSurfaceObject3D& maskvolume,
                     uint3 volDim);
};

class Div3DMaskKernel : public Cuda::CudaKernel
{
public:
    explicit Div3DMaskKernel(CUmodule aModule);
    Div3DMaskKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaSurfaceObject3D& involume,
                     Cuda::CudaSurfaceObject3D& outvolume,
                     Cuda::CudaSurfaceObject3D& maskvolume,
                     float divVal,
                     uint3 volDim);
};


class Mask3DTransformKernel : public Cuda::CudaKernel
{
public:
    explicit Mask3DTransformKernel(CUmodule aModule);
    Mask3DTransformKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaSurfaceObject3D& volume,
                     Cuda::CudaTextureObject3D& maskTex,
                     float4x4 transformMatrix,
                     uint3 volDim,
                     uint3 offset);
};

class Add3DTransformKernel : public Cuda::CudaKernel
{
public:
    explicit Add3DTransformKernel(CUmodule aModule);
    Add3DTransformKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaSurfaceObject3D& volume,
                     Cuda::CudaTextureObject3D& maskTex,
                     float4x4 transformMatrix,
                     uint3 volDim,
                     uint3 offset);
};

class Norm3DOverlapKernel : public Cuda::CudaKernel
{
public:
    explicit Norm3DOverlapKernel(CUmodule aModule);
    Norm3DOverlapKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaSurfaceObject3D& volume,
                     uint3 volDim);
};


class Multiplicity3DKernel : public Cuda::CudaKernel
{
public:
    explicit Multiplicity3DKernel(CUmodule aModule);
    Multiplicity3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaDeviceVariable& outvolume,
                     uint3 voldim,
                     float3x3 Msys);
};

class MultNorm1DKernel : public Cuda::CudaKernel
{
public:
    explicit MultNorm1DKernel(CUmodule aModule);
    MultNorm1DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaDeviceVariable& line,
                     Cuda::CudaDeviceVariable& lineMult,
                     uint length,
                     float threshold);
};

class MultNorm1DcompKernel : public Cuda::CudaKernel
{
public:
    explicit MultNorm1DcompKernel(CUmodule aModule);
    MultNorm1DcompKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaDeviceVariable& lineFFT,
                     Cuda::CudaDeviceVariable& lineMult,
                     uint length,
                     float threshold);
};


class MultNorm3DKernel : public Cuda::CudaKernel
{
public:
    explicit MultNorm3DKernel(CUmodule aModule);
    MultNorm3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaDeviceVariable& vol,
                     Cuda::CudaDeviceVariable& volMult,
                     uint3 volDim,
                     float threshold);
};

class MultNorm3DcompKernel : public Cuda::CudaKernel
{
public:
    explicit MultNorm3DcompKernel(CUmodule aModule);
    MultNorm3DcompKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaDeviceVariable& volFFT,
                     Cuda::CudaDeviceVariable& volMult,
                     uint3 volDim,
                     float threshold);
};


#endif //ARTIATOMI_CUBICRESAMPLEKERNEL_H
