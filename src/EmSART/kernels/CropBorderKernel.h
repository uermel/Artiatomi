//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_CROPBORDERKERNEL_H
#define ARTIATOMI_CROPBORDERKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class CropBorderKernel : public Cuda::CudaKernel
{
private:
    float2* corners_h;
    float* norm_h;

public:
    explicit CropBorderKernel(CUmodule aModule);
    CropBorderKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    ~CropBorderKernel();

    void CopyCornersToDevice(vector<float2>& aCorners, vector<float>& aNorm);

    float operator()(Cuda::CudaPitchedDeviceVariable& image,
                     uint2 projDim,
                     float2 cutLength,
                     float2 dimLength,
                     vector<float2>& corners,
                     vector<float>& norm);

    float operator()(Cuda::CudaDeviceVariable& image,
                     uint2 projDim,
                     float2 cutLength,
                     float2 dimLength,
                     vector<float2>& corners,
                     vector<float>& norm);
};


class CropBorderInvKernel : public Cuda::CudaKernel
{
private:
    float2* corners_h;
    float* norm_h;

public:
    explicit CropBorderInvKernel(CUmodule aModule);
    CropBorderInvKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    ~CropBorderInvKernel();

    void CopyCornersToDevice(vector<float2>& aCorners, vector<float>& aNorm);

    float operator()(Cuda::CudaPitchedDeviceVariable& image,
                     uint2 projDim,
                     float2 cutLength,
                     float2 dimLength,
                     vector<float2>& corners,
                     vector<float>& norm);

    float operator()(Cuda::CudaDeviceVariable& image,
                     uint2 projDim,
                     float2 cutLength,
                     float2 dimLength,
                     vector<float2>& corners,
                     vector<float>& norm);
};



//class CropBorderInvKernel : public Cuda::CudaKernel
//{
//public:
//    CropBorderInvKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
//    CropBorderInvKernel(CUmodule aModule);
//
//    float operator()(Cuda::CudaPitchedDeviceVariable& image,
//                     uint2 projDim,
//                     float2 cutLength,
//                     float2 dimLength,
//                     int2 p1, int2 p2, int2 p3, int2 p4);
//    float operator()(Cuda::CudaDeviceVariable& image,
//                     uint2 projDim,
//                     float2 cutLength,
//                     float2 dimLength,
//                     int2 p1, int2 p2, int2 p3, int2 p4);
//
//};


#endif //ARTIATOMI_CROPBORDERKERNEL_H
