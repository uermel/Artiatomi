//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_COMPKERNEL_H
#define ARTIATOMI_COMPKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class CompKernel : public Cuda::CudaKernel
{
public:
    CompKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    CompKernel(CUmodule aModule);

    float operator()(Cuda::CudaPitchedDeviceVariable& real_raw,
                     Cuda::CudaPitchedDeviceVariable& virtual_raw,
                     Cuda::CudaPitchedDeviceVariable& vol_distance_map,
                     float realLength,
                     float4 crop,
                     float4 cropDim,
                     float projValScale);
};

#endif //ARTIATOMI_COMPKERNEL_H
