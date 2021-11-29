//
// Created by uermel on 11/23/21.
//

#ifndef ARTIATOMI_CROPSLICESKERNEL_H
#define ARTIATOMI_CROPSLICESKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class CropSlicesKernel : public Cuda::CudaKernel
{
public:
    CropSlicesKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    CropSlicesKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& image,
                     int proj_x,
                     int proj_y,
                     int sliceNumber,
                     float2 cutLength,
                     float2 dimLength,
                     int2 p1,
                     int2 p2,
                     int2 p3,
                     int2 p4);
};


#endif //ARTIATOMI_CROPSLICESKERNEL_H
