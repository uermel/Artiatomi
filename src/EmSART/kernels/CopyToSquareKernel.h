//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_COPYTOSQUAREKERNEL_H
#define ARTIATOMI_COPYTOSQUAREKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class CopyToSquareKernel : public Cuda::CudaKernel
{
public:
    CopyToSquareKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    CopyToSquareKernel(CUmodule aModule);

    float operator()(Cuda::CudaPitchedDeviceVariable& aIn, int maxsize, Cuda::CudaDeviceVariable& aOut, int borderSizeX, int borderSizeY, bool mirrorY, bool fillZero);
};


#endif //ARTIATOMI_COPYTOSQUAREKERNEL_H
