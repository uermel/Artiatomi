//
// Created by uermel on 11/8/21.
//

#ifndef ARTIATOMI_BPLUTBLOCK_H
#define ARTIATOMI_BPLUTBLOCK_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class BPLUTBlockKernel : public Cuda::CudaKernel
{
public:
    BPLUTBlockKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    BPLUTBlockKernel(CUmodule aModule);

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

void SetConstantValues(BPLUTBlockKernel& kernel,
                       Volume<float>& vol,
                       Projection& proj,
                       int index, int subVol,
                       Matrix<float>& m,
                       Matrix<float>& mInv);


#endif //ARTIATOMI_BPLUTBLOCK_H
