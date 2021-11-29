//
// Created by uermel on 10/2/21.
//

#ifndef ARTIATOMI_ROTKERNEL_H
#define ARTIATOMI_ROTKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class RotKernel : public Cuda::CudaKernel
{
private:
    int size;
    Cuda::CudaTextureArray3D volTexArray;
    void computeRotMat(float phi, float psi, float theta, float rotMat[3][3]);

public:
    RotKernel(CUmodule aModule, int aSize);

    float operator()(Cuda::CudaDeviceVariable& aVolOut, float phi, float psi, float theta);
    void SetData(float* data);
};

#endif //ARTIATOMI_ROTKERNEL_H
