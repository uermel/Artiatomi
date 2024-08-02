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

/// 2DX, 2DY for linear memory, in place
class SplinePrefilter2DX : public Cuda::CudaKernel
{
public:
    SplinePrefilter2DX(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter2DX(CUmodule aModule, int aSplineDegree);

    float operator()(Cuda::CudaPitchedDeviceVariable& image, int width, int height);

private:
    int NbPoles;
    float4 Poles;
    int4 Horizon;
    float Lambda;
};

class SplinePrefilter2DY : public Cuda::CudaKernel
{
public:
    SplinePrefilter2DY(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter2DY(CUmodule aModule, int aSplineDegree);

    float operator()(Cuda::CudaPitchedDeviceVariable& image, int width, int height);

private:
    int NbPoles;
    float4 Poles;
    int4 Horizon;
    float Lambda;
};

/// 2DX, 2DY for CUDA array to CUDA array, in place AND out of place
class SplinePrefilter2DXSurf : public Cuda::CudaKernel
{
public:
    SplinePrefilter2DXSurf(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter2DXSurf(CUmodule aModule, int aSplineDegree);

    float operator()(Cuda::CudaSurfaceObject2D& image_in,
                     Cuda::CudaSurfaceObject2D& image_out,
                     int width, int height);

private:
    int NbPoles;
    float4 Poles;
    int4 Horizon;
    float Lambda;
};

class SplinePrefilter2DYSurf : public Cuda::CudaKernel
{
public:
    SplinePrefilter2DYSurf(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter2DYSurf(CUmodule aModule, int aSplineDegree);

    float operator()(Cuda::CudaSurfaceObject2D& image_in,
                     Cuda::CudaSurfaceObject2D& image_out,
                     int width, int height);

private:
    int NbPoles;
    float4 Poles;
    int4 Horizon;
    float Lambda;
};

/// 2DX, 2DY for linear memory to CUDA array, out of place
class SplinePrefilter2DXPtr2Surf : public Cuda::CudaKernel
{
public:
    SplinePrefilter2DXPtr2Surf(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter2DXPtr2Surf(CUmodule aModule, int aSplineDegree);

    float operator()(Cuda::CudaDeviceVariable& image_in,
                     Cuda::CudaSurfaceObject2D& image_out,
                     int width, int height, int z = 0);

private:
    int NbPoles;
    float4 Poles;
    int4 Horizon;
    float Lambda;
};

class SplinePrefilter2DYPtr2Surf : public Cuda::CudaKernel
{
public:
    SplinePrefilter2DYPtr2Surf(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter2DYPtr2Surf(CUmodule aModule, int aSplineDegree);

    float operator()(Cuda::CudaDeviceVariable& image_in,
                     Cuda::CudaSurfaceObject2D& image_out,
                     int width, int height, int z = 0);

private:
    int NbPoles;
    float4 Poles;
    int4 Horizon;
    float Lambda;
};

/// 3DX, 3DY, 3DZ for linear memory, in place
class SplinePrefilter3DX : public Cuda::CudaKernel
{
public:
    SplinePrefilter3DX(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter3DX(CUmodule aModule, int aSplineDegree);

    float operator()(Cuda::CudaDeviceVariable& img1, int width, int height, int depth);

private:
    int NbPoles;
    float4 Poles;
    int4 Horizon;
    float Lambda;
};

class SplinePrefilter3DY : public Cuda::CudaKernel
{
public:
    SplinePrefilter3DY(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter3DY(CUmodule aModule, int aSplineDegree);

    float operator()(Cuda::CudaDeviceVariable& img1, int width, int height, int depth);

private:
    int NbPoles;
    float4 Poles;
    int4 Horizon;
    float Lambda;
};

class SplinePrefilter3DZ : public Cuda::CudaKernel
{
public:
    SplinePrefilter3DZ(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter3DZ(CUmodule aModule, int aSplineDegree);

    float operator()(Cuda::CudaDeviceVariable& img1, int width, int height, int depth);

private:
    int NbPoles;
    float4 Poles;
    int4 Horizon;
    float Lambda;
};

/// 3DX, 3DY, 3DZ for CUDA array to CUDA array, in place AND out of place
class SplinePrefilter3DXSurf : public Cuda::CudaKernel
{
public:
    SplinePrefilter3DXSurf(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter3DXSurf(CUmodule aModule, int aSplineDegree);

    float operator()(Cuda::CudaSurfaceObject3D& volume_in,
                     Cuda::CudaSurfaceObject3D& volume_out,
                     int width, int height, int depth);

private:
    int NbPoles;
    float4 Poles;
    int4 Horizon;
    float Lambda;
};

class SplinePrefilter3DYSurf : public Cuda::CudaKernel
{
public:
    SplinePrefilter3DYSurf(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter3DYSurf(CUmodule aModule, int aSplineDegree);

    float operator()(Cuda::CudaSurfaceObject3D& volume_in,
                     Cuda::CudaSurfaceObject3D& volume_out,
                     int width, int height, int depth);

private:
    int NbPoles;
    float4 Poles;
    int4 Horizon;
    float Lambda;
};

class SplinePrefilter3DZSurf : public Cuda::CudaKernel
{
public:
    SplinePrefilter3DZSurf(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim);
    SplinePrefilter3DZSurf(CUmodule aModule, int aSplineDegree);

    float operator()(Cuda::CudaSurfaceObject3D& volume_in,
                     Cuda::CudaSurfaceObject3D& volume_out,
                     int width, int height, int depth);

private:
    int NbPoles;
    float4 Poles;
    int4 Horizon;
    float Lambda;
};


#endif //ARTIATOMI_SPLINEPREFILTER_H