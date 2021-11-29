//
// Created by uermel on 10/2/21.
//

#ifndef ARTIATOMI_CONVVOL3DKERNEL_H
#define ARTIATOMI_CONVVOL3DKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class ConvVol3DKernel : public Cuda::CudaKernel
{
public:
    ConvVol3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    ConvVol3DKernel(CUmodule aModule);

    float operator()(Cuda::CudaPitchedDeviceVariable& img, Cuda::CudaSurfaceObject3D& surf);
};


#endif //ARTIATOMI_CONVVOL3DKERNEL_H
