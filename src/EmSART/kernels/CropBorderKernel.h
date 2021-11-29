//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_CROPBORDERKERNEL_H
#define ARTIATOMI_CROPBORDERKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class CropBorderKernel : public Cuda::CudaKernel
{
public:
    CropBorderKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    CropBorderKernel(CUmodule aModule);

    float operator()(Cuda::CudaPitchedDeviceVariable& image, float2 cutLength, float2 dimLength, int2 p1, int2 p2, int2 p3, int2 p4);
};


#endif //ARTIATOMI_CROPBORDERKERNEL_H
