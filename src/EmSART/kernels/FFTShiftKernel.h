//
// Created by uermel on 2/7/22.
//

#ifndef ARTIATOMI_FFTSHIFTKERNEL_H
#define ARTIATOMI_FFTSHIFTKERNEL_H


#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class FFTShiftKernel : public Cuda::CudaKernel
{
public:
    FFTShiftKernel(CUmodule aModule);

    float operator()(int size,
                     Cuda::CudaDeviceVariable& image_in,
                     Cuda::CudaPitchedDeviceVariable& image_out,
                     float scaleFactor = 1);
};

#endif //ARTIATOMI_FFTSHIFTKERNEL_H
