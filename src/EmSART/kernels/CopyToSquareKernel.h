//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_COPYTOSQUAREKERNEL_H
#define ARTIATOMI_COPYTOSQUAREKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class CopyToSquareKernel : public Cuda::CudaKernel
{
public:
    explicit CopyToSquareKernel(CUmodule aModule);
    CopyToSquareKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);


    float operator()(Cuda::CudaPitchedDeviceVariable& aIn, int maxsize, Cuda::CudaDeviceVariable& aOut, int borderSizeX, int borderSizeY, bool mirrorY, bool fillZero);
};

class CopyToPitchedKernel : public Cuda::CudaKernel
{
public:
    explicit CopyToPitchedKernel(CUmodule aModule);
    CopyToPitchedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);


    float operator()(Cuda::CudaDeviceVariable& aIn,
                     Cuda::CudaPitchedDeviceVariable& aOut,
                     uint2 imDim);
};

class CopyFromPitchedKernel : public Cuda::CudaKernel
{
public:
    explicit CopyFromPitchedKernel(CUmodule aModule);
    CopyFromPitchedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);


    float operator()(Cuda::CudaPitchedDeviceVariable& aIn,
                     Cuda::CudaDeviceVariable& aOut,
                     uint2 imDim);
};

class AddToPitchedKernel : public Cuda::CudaKernel
{
public:
    explicit AddToPitchedKernel(CUmodule aModule);
    AddToPitchedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);


    float operator()(Cuda::CudaDeviceVariable& aIn,
                     Cuda::CudaPitchedDeviceVariable& aOut,
                     uint2 imDim);
};

class SlicesToArraysKernel : public Cuda::CudaKernel
{
public:
    int slicenum = 0;
    Cuda::CudaDeviceVariable d_surfaces;

    explicit SlicesToArraysKernel(CUmodule aModule);
    SlicesToArraysKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);


    void setSlices(vector<Cuda::CudaSurfaceObject2D*> surfs, int sliceNumber);

    float operator()(Cuda::CudaDeviceVariable& stack, uint2 imDim);
};

class MaskedSlicesToArraysKernel : public Cuda::CudaKernel
{
public:
    int slicenum = 0;
    Cuda::CudaDeviceVariable d_surfaces;

    explicit MaskedSlicesToArraysKernel(CUmodule aModule);
    MaskedSlicesToArraysKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    void setSlices(vector<Cuda::CudaSurfaceObject2D*> surfs, int sliceNumber);

    float operator()(Cuda::CudaDeviceVariable& stack,
                     Cuda::CudaPitchedDeviceVariable& mask,
                     float maskScale,
                     uint2 imDim);
};



#endif //ARTIATOMI_COPYTOSQUAREKERNEL_H
