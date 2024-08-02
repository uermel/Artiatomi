//
// Created by uermel on 10/1/21.
//

#ifndef ARTIATOMI_COMPKERNEL_H
#define ARTIATOMI_COMPKERNEL_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class CompKernel : public Cuda::CudaKernel
{
private:
    float2* corners_h;
    float* norm_h;

public:
    explicit CompKernel(CUmodule aModule);
    CompKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    ~CompKernel();

    void CopyCornersToDevice(vector<float2>& aCorners, vector<float>& aNorm);

    float operator()(Cuda::CudaPitchedDeviceVariable& real_proj,
                     Cuda::CudaPitchedDeviceVariable& fwd_proj,
                     Cuda::CudaPitchedDeviceVariable& dist_proj,
                     uint2 projDim, float2 cutLength, float2 dimLength,
                     float realLength,
                     vector<float2>& corners,
                     vector<float>& norm,
                     float voxelSize);
};

class CompSpecialKernel : public Cuda::CudaKernel
{
private:
    float2* corners_h;
    float* norm_h;

public:
    explicit CompSpecialKernel(CUmodule aModule);
    CompSpecialKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    ~CompSpecialKernel();

    void CopyCornersToDevice(vector<float2>& aCorners, vector<float>& aNorm);

    float operator()(Cuda::CudaPitchedDeviceVariable& real_proj,
                     Cuda::CudaPitchedDeviceVariable& fwd_proj_parent,
                     Cuda::CudaPitchedDeviceVariable& fwd_proj_child,
                     Cuda::CudaPitchedDeviceVariable& dist_proj_parent,
                     Cuda::CudaPitchedDeviceVariable& dist_proj_child,
                     uint2 projDim, float2 cutLength, float2 dimLength,
                     float maxDistParent,
                     float maxDistChild,
                     vector<float2>& corners,
                     vector<float>& norm,
                     float voxelSizeParent,
                     float voxelSizeChild);
};

#endif //ARTIATOMI_COMPKERNEL_H
