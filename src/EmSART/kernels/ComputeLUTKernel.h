//
// Created by uermel on 2/7/22.
//

#ifndef ARTIATOMI_COMPUTELUTKERNEL_H
#define ARTIATOMI_COMPUTELUTKERNEL_H


#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class ComputeLUTKernel : public Cuda::CudaKernel
{
public:
    ComputeLUTKernel(CUmodule aModule);

    float operator()(int pixelcount,
                     float freqStepSize,
                     float2 Xi_x,
                     float2 Xi_y,
                     float2 Xi_z,
                     float3 nu,
                     Cuda::CudaDeviceVariable& outIm);
};

class BoxSplineDualBKernel : public Cuda::CudaKernel
{
public:
    BoxSplineDualBKernel(CUmodule aModule);

    float operator()(int2 pixelcount,
                     float2 freqStepSize,
                     float2 Xi_x,
                     float2 Xi_y,
                     float2 Xi_z,
                     float3 nu,
                     float thickness,
                     Cuda::CudaDeviceVariable& outIm);
};

class BoxSplineDualBReflKernel : public Cuda::CudaKernel
{
public:
    BoxSplineDualBReflKernel(CUmodule aModule);

    float operator()(int2 pixelcount,
                     float2 freqStepSize,
                     float2 Xi_x,
                     float2 Xi_y,
                     float2 Xi_z,
                     float3 nu,
                     Cuda::CudaDeviceVariable& outIm);
};


#endif //ARTIATOMI_COMPUTELUTKERNEL_H
