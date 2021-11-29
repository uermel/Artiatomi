//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_FINDPEAKKERNEL_H
#define ARTIATOMI_FINDPEAKKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class FindPeakKernel : public Cuda::CudaKernel
{
public:
    FindPeakKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    FindPeakKernel(CUmodule aModule);
    //findPeak(float* img, size_t stride, char* maskInv, size_t strideMask, int pixelcount, float maxThreshold)
    float operator()(Cuda::CudaDeviceVariable& img1, size_t stride, Cuda::CudaPitchedDeviceVariable& mask, int pixelcount, float maxThreshold);
};


#endif //ARTIATOMI_FINDPEAKKERNEL_H
