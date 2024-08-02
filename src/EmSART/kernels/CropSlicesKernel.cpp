//
// Created by uermel on 11/23/21.
//

#include "CropSlicesKernel.h"

using namespace Cuda;

CropSlicesKernel::CropSlicesKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("cropBorderSlices", aModule, aGridDim, aBlockDim, 0),
          corners_h(new float2[16]), norm_h(new float[16])
{

}

CropSlicesKernel::CropSlicesKernel(CUmodule aModule)
        : CudaKernel("cropBorderSlices", aModule, make_dim3(1, 1, 1), make_dim3(4, 4, 4), 0),
          corners_h(new float2[16]), norm_h(new float[16])
{

}

CropSlicesKernel::~CropSlicesKernel()
{
    delete[] corners_h;
    delete[] norm_h;
}

void CropSlicesKernel::CopyCornersToDevice(vector<float2>& aCorners, vector<float>& aNorm)
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

float CropSlicesKernel::operator()(CudaDeviceVariable& image,
                                   uint2 projDim,
                                   uint sliceNumber,
                                   float2 cutLength,
                                   float2 dimLength,
                                   vector<float2>& corners,
                                   vector<float>& norm)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    uint cornerCount = corners.size();

    CopyCornersToDevice(corners, norm);

    void** arglist = (void**)new void*[6];

    arglist[0] = &projDim;
    arglist[1] = &sliceNumber;
    arglist[2] = &image_dptr;
    arglist[3] = &cutLength.x;
    arglist[4] = &dimLength.x;
    arglist[5] = &cornerCount;

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

CropSlicesInvKernel::CropSlicesInvKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("cropBorderSlicesInv", aModule, aGridDim, aBlockDim, 0),
          corners_h(new float2[16]), norm_h(new float[16])
{

}

CropSlicesInvKernel::CropSlicesInvKernel(CUmodule aModule)
        : CudaKernel("cropBorderSlicesInv", aModule, make_dim3(1, 1, 1), make_dim3(4, 4, 4), 0),
          corners_h(new float2[16]), norm_h(new float[16])
{

}

CropSlicesInvKernel::~CropSlicesInvKernel()
{
    delete[] corners_h;
    delete[] norm_h;
}

void CropSlicesInvKernel::CopyCornersToDevice(vector<float2>& aCorners, vector<float>& aNorm)
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

float CropSlicesInvKernel::operator()(CudaDeviceVariable& image,
                                   uint2 projDim,
                                   uint sliceNumber,
                                   float2 cutLength,
                                   float2 dimLength,
                                   vector<float2>& corners,
                                   vector<float>& norm)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    uint cornerCount = corners.size();

    CopyCornersToDevice(corners, norm);

    void** arglist = (void**)new void*[6];

    arglist[0] = &projDim;
    arglist[1] = &sliceNumber;
    arglist[2] = &image_dptr;
    arglist[3] = &cutLength.x;
    arglist[4] = &dimLength.x;
    arglist[5] = &cornerCount;

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
