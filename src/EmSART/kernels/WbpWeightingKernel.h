//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_WBPWEIGHTINGKERNEL_H
#define ARTIATOMI_WBPWEIGHTINGKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

enum FilterMethod
{
    FM_RAMP,
    FM_EXACT,
    FM_CONTRAST2,
    FM_CONTRAST10,
    FM_CONTRAST30
};


class WbpWeightingKernel : public Cuda::CudaKernel
{
public:
    WbpWeightingKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    WbpWeightingKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& img,
                     size_t stride,
                     uint2 imDim,
                     float2 asymCorrFac,
                     FilterMethod fm,
                     int projectionCount,
                     float thickness,
                     Matrix<double>& Mproj,
                     Cuda::CudaDeviceVariable& Mdet);
};


#endif //ARTIATOMI_WBPWEIGHTINGKERNEL_H
