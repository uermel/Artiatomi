//
// Created by uermel on 11/23/21.
//

#ifndef ARTIATOMI_CROPSLICESKERNEL_H
#define ARTIATOMI_CROPSLICESKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class CropSlicesKernel : public Cuda::CudaKernel
{
private:
    float2* corners_h;
    float* norm_h;

public:
    explicit CropSlicesKernel(CUmodule aModule);
    CropSlicesKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    ~CropSlicesKernel();

    void CopyCornersToDevice(vector<float2>& aCorners, vector<float>& aNorm);

    float operator()(Cuda::CudaDeviceVariable& image,
                     uint2 projDim,
                     uint sliceNumber,
                     float2 cutLength,
                     float2 dimLength,
                     vector<float2>& corners,
                     vector<float>& norm);
};

class CropSlicesInvKernel : public Cuda::CudaKernel
{
private:
    float2* corners_h;
    float* norm_h;

public:
    explicit CropSlicesInvKernel(CUmodule aModule);
    CropSlicesInvKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    ~CropSlicesInvKernel();

    void CopyCornersToDevice(vector<float2>& aCorners, vector<float>& aNorm);

    float operator()(Cuda::CudaDeviceVariable& image,
                     uint2 projDim,
                     uint sliceNumber,
                     float2 cutLength,
                     float2 dimLength,
                     vector<float2>& corners,
                     vector<float>& norm);
};

//class CropSlicesInvKernel : public Cuda::CudaKernel
//{
//public:
//    CropSlicesInvKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
//    CropSlicesInvKernel(CUmodule aModule);
//
//    float operator()(Cuda::CudaDeviceVariable& image,
//                     int proj_x,
//                     int proj_y,
//                     int sliceNumber,
//                     float2 cutLength,
//                     float2 dimLength,
//                     int2 p1,
//                     int2 p2,
//                     int2 p3,
//                     int2 p4);
//};


#endif //ARTIATOMI_CROPSLICESKERNEL_H
