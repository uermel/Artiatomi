//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_SUBEKERNEL_H
#define ARTIATOMI_SUBEKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class SubEKernel : public Cuda::CudaKernel
{
public:
    SubEKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    SubEKernel(CUmodule aModule);

    float operator()(Cuda::CudaPitchedDeviceVariable& real_raw, Cuda::CudaPitchedDeviceVariable& error, Cuda::CudaPitchedDeviceVariable& vol_distance_map);
};


#endif //ARTIATOMI_SUBEKERNEL_H
