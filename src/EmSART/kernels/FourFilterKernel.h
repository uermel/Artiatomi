//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_FOURFILTERKERNEL_H
#define ARTIATOMI_FOURFILTERKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class FourFilterKernel : public Cuda::CudaKernel
{
public:
    FourFilterKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    FourFilterKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& img,
                     size_t stride,
                     uint2 pixelcount,
                     float2 asymCorrFac,
                     float lp, float hp, float lps, float hps);
};


#endif //ARTIATOMI_FOURFILTERKERNEL_H
