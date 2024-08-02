//
// Created by uermel on 2/7/22.
//

#ifndef ARTIATOMI_RADIALSUMKERNEL_H
#define ARTIATOMI_RADIALSUMKERNEL_H


#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class RadialSumAbsKernel : public Cuda::CudaKernel
{
public:
    explicit RadialSumAbsKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& image,
                     Cuda::CudaDeviceVariable& sum,
                     Cuda::CudaDeviceVariable& multiplicity,
                     uint2 fftDim,
                     float2 imDim,
                     float2 asymCorrFac,
                     float scaleFactor);
};

class RadialSum3DKernel : public Cuda::CudaKernel
{
public:
    explicit RadialSum3DKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& image,
                     Cuda::CudaDeviceVariable& sum,
                     Cuda::CudaDeviceVariable& multiplicity,
                     uint3 fftDim,
                     uint3 volDim,
                     float3 asymCorrFac,
                     float scaleFactor,
                     int maxShell);
};


class RadialSumAbs3DKernel : public Cuda::CudaKernel
{
public:
    explicit RadialSumAbs3DKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& volume,
                     Cuda::CudaDeviceVariable& sum,
                     Cuda::CudaDeviceVariable& multiplicity,
                     uint3 fftDim,
                     uint3 volDim,
                     float3 asymCorrFac,
                     float scaleFactor,
                     int maxShell);
};


class FSC3DKernel : public Cuda::CudaKernel
{
public:
    explicit FSC3DKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& image_1,
                     Cuda::CudaDeviceVariable& image_2,
                     Cuda::CudaDeviceVariable& amp1_sum,
                     Cuda::CudaDeviceVariable& amp2_sum,
                     Cuda::CudaDeviceVariable& ampdiff_sum,
                     Cuda::CudaDeviceVariable& multiplicity,
                     uint3 fftDim,
                     uint3 volDim,
                     float3 asymCorrFac,
                     float scaleFactor,
                     int maxShell);
};

class FSCNorm1DKernel : public Cuda::CudaKernel
{
public:
    explicit FSCNorm1DKernel(CUmodule aModule);
    FSCNorm1DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaDeviceVariable& amp1_sum,
                     Cuda::CudaDeviceVariable& amp2_sum,
                     Cuda::CudaDeviceVariable& ampd_sum,
                     Cuda::CudaDeviceVariable& lineMult,
                     uint length,
                     float threshold = 1.f);
};

#endif //ARTIATOMI_RADIALSUMKERNEL_H
