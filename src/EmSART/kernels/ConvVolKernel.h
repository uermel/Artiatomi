//
// Created by uermel on 10/2/21.
//

#ifndef ARTIATOMI_CONVVOLKERNEL_H
#define ARTIATOMI_CONVVOLKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class ConvVolKernel : public Cuda::CudaKernel
{
public:
    ConvVolKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    ConvVolKernel(CUmodule aModule);

    float operator()(Cuda::CudaPitchedDeviceVariable& img, Cuda::CudaSurfaceObject3D& surf, unsigned int z);
};


#endif //ARTIATOMI_CONVVOLKERNEL_H
