//
// Created by uermel on 10/10/23.
//

#ifndef ARTIATOMI_FREQSAMPLEKERNEL_H
#define ARTIATOMI_FREQSAMPLEKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class FreqSampleKernel : public Cuda::CudaKernel
{
public:
    explicit FreqSampleKernel(CUmodule aModule);

    float operator()(Cuda::CudaDeviceVariable& dst,
                     Cuda::CudaTextureObject1D& src,
                     uint dstLength,
                     uint dstNyq,
                     uint srcLength,
                     float voxelSizeDst,
                     float voxelSizeSrc,
                     SF_EXTRAP_MODE mode = SF_EXTRAP_TEX,
                     float extrapVal = 0.f);
};

#endif //ARTIATOMI_FREQSAMPLEKERNEL_H
