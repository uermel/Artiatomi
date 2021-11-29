//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_PCKERNEL_H
#define ARTIATOMI_PCKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class PCKernel : public Cuda::CudaKernel
{
public:
    PCKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    PCKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& img1, Cuda::CudaPitchedDeviceVariable& img2, size_t stride, int pixelcount);
};


#endif //ARTIATOMI_PCKERNEL_H
