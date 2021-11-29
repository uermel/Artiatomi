//
// Created by uermel on 11/22/21.
//

#ifndef ARTIATOMI_SQRSLICETORECTKERNEL_H
#define ARTIATOMI_SQRSLICETORECTKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class SqrSliceToRectKernel : public Cuda::CudaKernel
{
public:
    SqrSliceToRectKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    SqrSliceToRectKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& aIn,
                     int maxsize,
                     Cuda::CudaPitchedDeviceVariable& aOut,
                     int borderSizeX,
                     int borderSizeY,
                     bool mirrorY,
                     bool fillZero,
                     int sliceNumber);
};

#endif //ARTIATOMI_SQRSLICETORECTKERNEL_H
