//
// Created by uermel on 9/30/21.
//

#ifndef ARTIATOMI_FPKERNEL_H
#define ARTIATOMI_FPKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class FPKernel : public Cuda::CudaKernel
{
public:
    FPKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    FPKernel(CUmodule aModule);

    float operator()(int x, int y, Cuda::CudaPitchedDeviceVariable& projection, Cuda::CudaPitchedDeviceVariable& distMap, Cuda::CudaTextureObject3D& texObj);
    float operator()(int x, int y, Cuda::CudaPitchedDeviceVariable& projection, Cuda::CudaPitchedDeviceVariable& distMap, Cuda::CudaTextureObject3D& texObj, int2 roiMin, int2 roiMax);
};

void SetConstantValues(FPKernel& kernel, Volume<float>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv);
void SetConstantValues(FPKernel& kernel, Volume<unsigned short>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv);

#endif //ARTIATOMI_FPKERNEL_H
