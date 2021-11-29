//
// Created by uermel on 10/2/21.
//

#ifndef ARTIATOMI_RESTOREVOLUMEKERNEL_H
#define ARTIATOMI_RESTOREVOLUMEKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class RestoreVolumeKernel : public Cuda::CudaKernel
{
private:
    int size;

public:
    RestoreVolumeKernel(CUmodule aModule, int aSize);

    float operator()(Cuda::CudaSurfaceObject3D& volume, Cuda::CudaDeviceVariable& tempStore, int3 volmin, int3 volmax, int3 dimMask, int3 radiusMask, int3 centerInVol);
};



#endif //ARTIATOMI_RESTOREVOLUMEKERNEL_H
