//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_MAXSHIFTWEIGHTEDKERNEL_H
#define ARTIATOMI_MAXSHIFTWEIGHTEDKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class MaxShiftWeightedKernel : public Cuda::CudaKernel
{
public:
    MaxShiftWeightedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    MaxShiftWeightedKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& img1, size_t stride, int pixelcount, int maxShift);
};



#endif //ARTIATOMI_MAXSHIFTWEIGHTEDKERNEL_H
