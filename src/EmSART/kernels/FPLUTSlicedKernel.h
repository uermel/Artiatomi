//
// Created by uermel on 11/19/21.
//

#ifndef ARTIATOMI_FPLUTSLICEDKERNEL_H
#define ARTIATOMI_FPLUTSLICEDKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class FPLUTSlicedKernel : public Cuda::CudaKernel
{
public:
    FPLUTSlicedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    FPLUTSlicedKernel(CUmodule aModule);

    void MatrixVector3Mul(float4x4 M, float3& v, float2& erg);

    int2 computeSupport(Volume<float> &vol,
                        Projection &proj,
                        int index,
                        int3 voxelBlockDim,
                        int supportsize,
                        int maxOversample);

    float operator()(int proj_x,
                     int proj_y,
                     float lambda,
                     float maxOverSample,
                     Cuda::CudaDeviceVariable& projection,
                     Cuda::CudaTextureObject2D& LUT,
                     Cuda::CudaSurfaceObject3D& volume,
                     float tmin,
                     float tmax,
                     Volume<float>* vol);
};

void SetConstantValues(FPLUTSlicedKernel& kernel,
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

#endif //ARTIATOMI_FPLUTSLICEDKERNEL_H
