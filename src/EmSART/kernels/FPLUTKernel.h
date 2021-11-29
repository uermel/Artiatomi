//
// Created by uermel on 9/30/21.
//

#ifndef ARTIATOMI_FPLUTKERNEL_H
#define ARTIATOMI_FPLUTKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class FPLUTKernel : public Cuda::CudaKernel
{
public:
    FPLUTKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    FPLUTKernel(CUmodule aModule);

    float operator()(int x,
                     int y,
                     Cuda::CudaPitchedDeviceVariable& projection,
                     Cuda::CudaPitchedDeviceVariable& distMap,
                     Cuda::CudaTextureObject2D& LUT,
                     Cuda::CudaSurfaceObject3D& volume,
                     float tmin,
                     float tmax,
                     float support,
                     float LUTstepinv,
                     float LUTcenter);

    float operator()(int x,
                     int y,
                     Cuda::CudaPitchedDeviceVariable& projection,
                     Cuda::CudaPitchedDeviceVariable& distMap,
                     Cuda::CudaTextureObject2D& LUT,
                     Cuda::CudaSurfaceObject3D& volume,
                     float tmin,
                     float tmax,
                     float support,
                     float LUTstepinv,
                     float LUTcenter,
                     int2 roiMin,
                     int2 roiMax);
};

void SetConstantValues(FPLUTKernel& kernel, Volume<float>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv);

#endif //ARTIATOMI_FPLUTKERNEL_H
