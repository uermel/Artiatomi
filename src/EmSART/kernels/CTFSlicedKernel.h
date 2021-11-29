//
// Created by uermel on 11/15/21.
//

#ifndef ARTIATOMI_CTFSLICEDKERNEL_H
#define ARTIATOMI_CTFSLICEDKERNEL_H


#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class CTFSlicedKernel : public Cuda::CudaKernel
{
private:
    Cuda::CudaDeviceVariable d_offsets;
    float* h_offsets;

public:
    CTFSlicedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    CTFSlicedKernel(CUmodule aModule);

    void AllocOffsets(int size);

    float operator()(Cuda::CudaDeviceVariable& ctf,
                     int ctf_x,
                     int ctf_y,
                     int sliceNumber,
                     float defocusMin,
                     float defocusMax,
                     vector<float>& offsets,
                     float angle,
                     bool applyForFP,
                     bool phaseFlipOnly,
                     float WienerFilterNoiseLevel,
                     size_t stride,
                     float4 betaFac);
};

void SetConstantValues(CTFSlicedKernel& kernel, Projection& proj, int index, float cs, float voltage);


#endif //ARTIATOMI_CTFSLICEDKERNEL_H
