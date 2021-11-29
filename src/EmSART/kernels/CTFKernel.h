//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_CTFKERNEL_H
#define ARTIATOMI_CTFKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class CTFKernel : public Cuda::CudaKernel
{
public:
    CTFKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    CTFKernel(CUmodule aModule);

    //float operator()(CudaPitchedDeviceVariable& ctf, float defocus, bool absolute)
    float operator()(Cuda::CudaDeviceVariable& ctf,
                     float defocusMin,
                     float defocusMax,
                     float angle,
                     bool applyForFP,
                     bool phaseFlipOnly,
                     float WienerFilterNoiseLevel,
                     size_t stride,
                     float4 betaFac);
};

void SetConstantValues(CTFKernel& kernel, Projection& proj, int index, float cs, float voltage);


#endif //ARTIATOMI_CTFKERNEL_H
