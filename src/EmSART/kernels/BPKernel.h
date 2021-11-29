//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_BPKERNEL_H
#define ARTIATOMI_BPKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class BPKernel : public Cuda::CudaKernel
{
public:
    BPKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim, bool fp16);
    BPKernel(CUmodule aModule, bool fp16);

    float operator()(int proj_x,
            int proj_y, float
            lambda,
            int maxOverSample,
            float maxOverSampleInv,
            Cuda::CudaTextureObject2D& img,
            Cuda::CudaSurfaceObject3D& surf,
            float distMin,
            float distMax);
};

void SetConstantValues(BPKernel& kernel,
                       Volume<unsigned short>& vol,
                       Projection& proj, int index,
                       int subVol,
                       Matrix<float>& m,
                       Matrix<float>& mInv);

void SetConstantValues(BPKernel& kernel,
                       Volume<float>& vol,
                       Projection& proj,
                       int index, int subVol,
                       Matrix<float>& m,
                       Matrix<float>& mInv);



#endif //ARTIATOMI_BPKERNEL_H
