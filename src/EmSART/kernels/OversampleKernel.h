//
// Created by uermel on 10/8/21.
//

#ifndef ARTIATOMI_OVERSAMPLEKERNEL_H
#define ARTIATOMI_OVERSAMPLEKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class OversampleKernel : public Cuda::CudaKernel
{
public:
    OversampleKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    OversampleKernel(CUmodule aModule);

    float operator()(int x,
                     int y,
                     int maxOverSample,
                     Cuda::CudaTextureObject2D& projection,
                     Cuda::CudaPitchedDeviceVariable& outprojection);
};


#endif //ARTIATOMI_OVERSAMPLEKERNEL_H
