//
// Created by uermel on 10/4/21.
//

#ifndef ARTIATOMI_SPLINEPREFILTER_H
#define ARTIATOMI_SPLINEPREFILTER_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class SplinePrefilter2DX : public Cuda::CudaKernel
{
public:
    SplinePrefilter2DX(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter2DX(CUmodule aModule);

    float operator()(Cuda::CudaPitchedDeviceVariable& image, int width, int height);
};

class SplinePrefilter2DY : public Cuda::CudaKernel
{
public:
    SplinePrefilter2DY(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter2DY(CUmodule aModule);

    float operator()(Cuda::CudaPitchedDeviceVariable& image, int width, int height);
};

//class SplinePrefilter3DX : public Cuda::CudaKernel
//{
//public:
//    SplinePrefilter3DX(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
//    SplinePrefilter3DX(CUmodule aModule);
//
//    float operator()(Cuda::CudaDeviceVariable& img1, size_t stride, int width, int height, int depth);
//};
//
//class SplinePrefilter3DY : public Cuda::CudaKernel
//{
//public:
//    SplinePrefilter3DY(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
//    SplinePrefilter3DY(CUmodule aModule);
//
//    float operator()(Cuda::CudaDeviceVariable& img1, size_t stride, int width, int height, int depth);
//};
//
//class SplinePrefilter3DZ : public Cuda::CudaKernel
//{
//public:
//    SplinePrefilter3DZ(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
//    SplinePrefilter3DZ(CUmodule aModule);
//
//    float operator()(Cuda::CudaDeviceVariable& img1, size_t stride, int width, int height, int depth);
//};


#endif //ARTIATOMI_SPLINEPREFILTER_H