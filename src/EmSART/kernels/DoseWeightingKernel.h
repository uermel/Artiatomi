//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_DOSEWEIGHTINGKERNEL_H
#define ARTIATOMI_DOSEWEIGHTINGKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class DoseWeightingKernel : public Cuda::CudaKernel
{
public:
    DoseWeightingKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    DoseWeightingKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& img, size_t stride, int pixelcount, float dose, float pixelSizeInA);
};



#endif //ARTIATOMI_DOSEWEIGHTINGKERNEL_H
