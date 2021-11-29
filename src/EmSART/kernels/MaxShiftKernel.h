//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_MAXSHIFTKERNEL_H
#define ARTIATOMI_MAXSHIFTKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class MaxShiftKernel : public Cuda::CudaKernel
{
public:
    MaxShiftKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    MaxShiftKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& img1, size_t stride, int pixelcount, int maxShift);
};

#endif //ARTIATOMI_MAXSHIFTKERNEL_H
