//
// Created by uermel on 11/1/21.
//

#ifndef ARTIATOMI_BPLUTBWKERNEL_H
#define ARTIATOMI_BPLUTBWKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class BPLUTBWKernel : public Cuda::CudaKernel
{
public:
    BPLUTBWKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    BPLUTBWKernel(CUmodule aModule);

    float operator()(int proj_x,
                     int proj_y,
                     float lambda,
                     float maxOverSample,
                     //Cuda::CudaSurfaceObject2D& projection,
                     Cuda::CudaPitchedDeviceVariable& projection,
                     Cuda::CudaTextureObject2D& LUT,
                     Cuda::CudaSurfaceObject3D& volume,
                     float tmin,
                     float tmax,
                     float support,
                     float LUTstepinv,
                     float LUTcenter,
                     Volume<float>* vol);
};

void SetConstantValues(BPLUTBWKernel& kernel,
                       Volume<float>& vol,
                       Projection& proj,
                       int index, int subVol,
                       Matrix<float>& m,
                       Matrix<float>& mInv);

#endif //ARTIATOMI_BPLUTBWKERNEL_H
