//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_DIMBORDERSKERNEL_H
#define ARTIATOMI_DIMBORDERSKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class DimBordersKernel : public Cuda::CudaKernel
{
public:
    DimBordersKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    DimBordersKernel(CUmodule aModule);

    float operator()(Cuda::CudaPitchedDeviceVariable& image, float4 crop, float4 cropDim);
};

#endif //ARTIATOMI_DIMBORDERSKERNEL_H
