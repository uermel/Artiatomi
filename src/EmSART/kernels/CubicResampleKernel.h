//
// Created by uermel on 10/5/21.
//

#ifndef ARTIATOMI_CUBICRESAMPLEKERNEL_H
#define ARTIATOMI_CUBICRESAMPLEKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class CubicResampleKernel : public Cuda::CudaKernel
{
public:
    CubicResampleKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    CubicResampleKernel(CUmodule aModule);

    float operator()(Cuda::CudaTextureObject3D& involume,
                     Cuda::CudaSurfaceObject3D& outvolume,
                     Volume<float>* vol);
};



#endif //ARTIATOMI_CUBICRESAMPLEKERNEL_H
