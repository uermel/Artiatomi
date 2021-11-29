//
// Created by uermel on 11/15/21.
//

#ifndef ARTIATOMI_RECTTOSQRSLICE_H
#define ARTIATOMI_RECTTOSQRSLICE_H


#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class RectToSqrSlice : public Cuda::CudaKernel
{
public:
    RectToSqrSlice(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    RectToSqrSlice(CUmodule aModule);

    float operator()(Cuda::CudaPitchedDeviceVariable& aIn,
                     int maxsize,
                     Cuda::CudaDeviceVariable& aOut,
                     int borderSizeX,
                     int borderSizeY,
                     bool mirrorY,
                     bool fillZero,
                     int sliceNumber);
};


#endif //ARTIATOMI_RECTTOSQRSLICE_H
