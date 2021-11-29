//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_CONJKERNEL_H
#define ARTIATOMI_CONJKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class ConjKernel : public Cuda::CudaKernel
{
public:
    ConjKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    ConjKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& img1, Cuda::CudaPitchedDeviceVariable& img2, size_t stride, int pixelcount);
};

#endif //ARTIATOMI_CONJKERNEL_H
