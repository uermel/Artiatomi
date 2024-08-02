//
// Created by uermel on 10/1/21.
//

#include "CompKernel.h"

using namespace Cuda;


CompKernel::CompKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("compare", aModule, aGridDim, aBlockDim, 0),
          corners_h(new float2[16]), norm_h(new float[16])
{

}

CompKernel::CompKernel(CUmodule aModule)
        : CudaKernel("compare", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          corners_h(new float2[16]), norm_h(new float[16])
{

}

CompKernel::~CompKernel()
{
    delete[] corners_h;
    delete[] norm_h;
}

void CompKernel::CopyCornersToDevice(vector<float2>& aCorners, vector<float>& aNorm)
{
    for (int i=0; i<aCorners.size(); i++){
        corners_h[i] = aCorners[i];
    }
    SetConstantValue("c_polygon", corners_h);

    for (int i=0; i<aNorm.size(); i++){
        norm_h[i] = aNorm[i];
    }
    SetConstantValue("c_polynorm", norm_h);
}


float CompKernel::operator()(CudaPitchedDeviceVariable& real_proj,
                             CudaPitchedDeviceVariable& fwd_proj,
                             CudaPitchedDeviceVariable& dist_proj,
                             uint2 projDim, float2 cutLength, float2 dimLength,
                             float realLength,
                             vector<float2>& corners,
                             vector<float>& norm,
                             float voxelSize)
{
    CUdeviceptr real_proj_dptr = real_proj.GetDevicePtr();
    CUdeviceptr fwd_proj_dptr = fwd_proj.GetDevicePtr();
    CUdeviceptr vol_distance_map_dptr = dist_proj.GetDevicePtr();
    size_t stride = real_proj.GetPitch();
    uint cornerCount = corners.size();

    CopyCornersToDevice(corners, norm);

    void** arglist = (void**)new void*[10];

    arglist[0] = &projDim;
    arglist[1] = &stride;
    arglist[2] = &fwd_proj_dptr;
    arglist[3] = &cutLength.x;
    arglist[4] = &dimLength.x;
    arglist[5] = &cornerCount;
    arglist[6] = &real_proj_dptr;
    arglist[7] = &vol_distance_map_dptr;
    arglist[8] = &realLength;
    arglist[9] = &voxelSize;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

    cudaSafeCall(cuCtxSynchronize());

    cudaSafeCall(cuEventRecord(eventEnd, stream));
    cudaSafeCall(cuEventSynchronize(eventEnd));
    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    cudaSafeCall(cuEventDestroy(eventStart));
    cudaSafeCall(cuEventDestroy(eventEnd));

    delete[] arglist;
    return ms;
}


CompSpecialKernel::CompSpecialKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("compareSpecial", aModule, aGridDim, aBlockDim, 0),
          corners_h(new float2[16]), norm_h(new float[16])
{

}

CompSpecialKernel::CompSpecialKernel(CUmodule aModule)
        : CudaKernel("compareSpecial", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          corners_h(new float2[16]), norm_h(new float[16])
{

}

CompSpecialKernel::~CompSpecialKernel()
{
    delete[] corners_h;
    delete[] norm_h;
}

void CompSpecialKernel::CopyCornersToDevice(vector<float2>& aCorners, vector<float>& aNorm)
{
    for (int i=0; i<aCorners.size(); i++){
        corners_h[i] = aCorners[i];
    }
    SetConstantValue("c_polygon", corners_h);

    for (int i=0; i<aNorm.size(); i++){
        norm_h[i] = aNorm[i];
    }
    SetConstantValue("c_polynorm", norm_h);
}


float CompSpecialKernel::operator()(Cuda::CudaPitchedDeviceVariable& real_proj,
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
                                    float voxelSizeChild)
{
    CUdeviceptr real_proj_dptr = real_proj.GetDevicePtr();
    CUdeviceptr fwd_parent_dptr = fwd_proj_parent.GetDevicePtr();
    CUdeviceptr fwd_child_dptr = fwd_proj_child.GetDevicePtr();
    CUdeviceptr dist_parent_dptr = dist_proj_parent.GetDevicePtr();
    CUdeviceptr dist_child_dptr = dist_proj_child.GetDevicePtr();
    size_t stride = real_proj.GetPitch();
    uint cornerCount = corners.size();

    CopyCornersToDevice(corners, norm);

    void** arglist = (void**)new void*[14];

    arglist[0] = &projDim;
    arglist[1] = &stride;
    arglist[2] = &fwd_parent_dptr;
    arglist[3] = &fwd_child_dptr;
    arglist[4] = &cutLength.x;
    arglist[5] = &dimLength.x;
    arglist[6] = &cornerCount;
    arglist[7] = &real_proj_dptr;
    arglist[8] = &dist_parent_dptr;
    arglist[9] = &dist_child_dptr;
    arglist[10] = &maxDistParent;
    arglist[11] = &maxDistChild;
    arglist[12] = &voxelSizeParent;
    arglist[13] = &voxelSizeChild;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

    cudaSafeCall(cuCtxSynchronize());

    cudaSafeCall(cuEventRecord(eventEnd, stream));
    cudaSafeCall(cuEventSynchronize(eventEnd));
    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    cudaSafeCall(cuEventDestroy(eventStart));
    cudaSafeCall(cuEventDestroy(eventEnd));

    delete[] arglist;
    return ms;
}