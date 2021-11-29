//
// Created by uermel on 11/22/21.
//

#ifndef ARTIATOMI_RECTSLICETOSQRSLICEKERNEL_H
#define ARTIATOMI_RECTSLICETOSQRSLICEKERNEL_H


#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class RectSliceToSqrSliceKernel : public Cuda::CudaKernel
{
public:
    RectSliceToSqrSliceKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    RectSliceToSqrSliceKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& aIn,
                     int proj_x,
                     int proj_y,
                     int maxsize,
                     Cuda::CudaDeviceVariable& aOut,
                     int borderSizeX,
                     int borderSizeY,
                     bool mirrorY,
                     bool fillZero,
                     int sliceNumber);
};

#endif //ARTIATOMI_RECTSLICETOSQRSLICEKERNEL_H
