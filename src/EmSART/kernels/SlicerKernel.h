//
// Created by uermel on 9/30/21.
//

#ifndef ARTIATOMI_SLICERKERNEL_H
#define ARTIATOMI_SLICERKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class SlicerKernel : public Cuda::CudaKernel
{
public:
    SlicerKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    SlicerKernel(CUmodule aModule);

    float operator()(int x,
                     int y,
                     Cuda::CudaPitchedDeviceVariable& projection,
                     float tmin,
                     float tmax,
                     Cuda::CudaTextureObject3D& texObj);
    float operator()(int x,
                     int y,
                     Cuda::CudaPitchedDeviceVariable& projection,
                     float tmin,
                     float tmax,
                     Cuda::CudaTextureObject3D& texObj,
                     int2 roiMin,
                     int2 roiMax);
};

void SetConstantValues(SlicerKernel& kernel, Volume<float>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv);
void SetConstantValues(SlicerKernel& kernel, Volume<unsigned short>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv);

#endif //ARTIATOMI_SLICERKERNEL_H
