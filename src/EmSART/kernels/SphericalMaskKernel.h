//
// Created by uermel on 10/2/21.
//

#ifndef ARTIATOMI_SPHERICALMASKKERNEL_H
#define ARTIATOMI_SPHERICALMASKKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class SphericalMaskKernel : public Cuda::CudaKernel
{
private:
    int size;

public:
    SphericalMaskKernel(CUmodule aModule, int aSize);

    float operator()(Cuda::CudaDeviceVariable& aVolOut, float radius);
};



#endif //ARTIATOMI_SPHERICALMASKKERNEL_H
