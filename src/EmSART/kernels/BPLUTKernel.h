//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_BPLUTKERNEL_H
#define ARTIATOMI_BPLUTKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class BPLUTKernel : public Cuda::CudaKernel
{
public:
    BPLUTKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    BPLUTKernel(CUmodule aModule);

    float operator()(int proj_x,
                     int proj_y,
                     float lambda,
                     float maxOverSample,
                     Cuda::CudaPitchedDeviceVariable& projection,
                     Cuda::CudaTextureObject2D& LUT,
                     Cuda::CudaSurfaceObject3D& volume,
                     float tmin,
                     float tmax,
                     float support,
                     float LUTstepinv,
                     float LUTcenter);
};

void SetConstantValues(BPLUTKernel& kernel,
                       Volume<float>& vol,
                       Projection& proj,
                       int index, int subVol,
                       Matrix<float>& m,
                       Matrix<float>& mInv);

#endif //ARTIATOMI_BPLUTKERNEL_H
