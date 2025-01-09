//
// Created by uermel on 11/19/21.
//

#ifndef ARTIATOMI_FPDISTORTHOKERNEL_H
#define ARTIATOMI_FPDISTORTHOKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class FPDistOrthoKernel : public Cuda::CudaKernel
{
public:
    explicit FPDistOrthoKernel(CUmodule aModule);
    FPDistOrthoKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(uint2 projDim,
                     uint3 volDim,
                     ctfImageConstants imageConstants,
                     float4x4 systemMatrix,
                     Cuda::CudaDeviceVariable& distanceMap);
};

void SetConstantValues(FPDistOrthoKernel& kernel,
                       Volume<float>& vol,
                       Projection& proj,
                       int index,
                       int subVol,
                       Matrix<float>& m,
                       Matrix<float>& mInv,
                       float LUTstepinv,
                       float LUTcenter,
                       int support,
                       int maxOverSample,
                       int sliceNumber,
                       float entry,
                       float sliceThickness);

#endif //ARTIATOMI_FPDISTORTHOKERNEL_H
