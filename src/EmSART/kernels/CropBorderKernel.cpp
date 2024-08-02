//
// Created by uermel on 10/1/21.
//

#include "CropBorderKernel.h"

using namespace Cuda;

CropBorderKernel::CropBorderKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("cropBorder", aModule, aGridDim, aBlockDim, 0),
        corners_h(new float2[16]), norm_h(new float[16])
{

}

CropBorderKernel::CropBorderKernel(CUmodule aModule)
        : CudaKernel("cropBorder", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
        corners_h(new float2[16]), norm_h(new float[16])
{

}

CropBorderKernel::~CropBorderKernel()
{
    delete[] corners_h;
    delete[] norm_h;
}

void CropBorderKernel::CopyCornersToDevice(vector<float2>& aCorners, vector<float>& aNorm)
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

float CropBorderKernel::operator()(CudaPitchedDeviceVariable& image,
                                   uint2 projDim,
                                   float2 cutLength,
                                   float2 dimLength,
                                   vector<float2>& corners,
                                   vector<float>& norm)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    size_t stride = image.GetPitch();
    uint cornerCount = corners.size();

    CopyCornersToDevice(corners, norm);

    void** arglist = (void**)new void*[6];

    arglist[0] = &projDim;
    arglist[1] = &stride;
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

float CropBorderKernel::operator()(CudaDeviceVariable& image,
                                   uint2 projDim,
                                   float2 cutLength,
                                   float2 dimLength,
                                   vector<float2>& corners,
                                   vector<float>& norm)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    size_t stride = projDim.x * sizeof(float);
    uint cornerCount = corners.size();

    CopyCornersToDevice(corners, norm);

    void** arglist = (void**)new void*[6];

    arglist[0] = &projDim;
    arglist[1] = &stride;
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

CropBorderInvKernel::CropBorderInvKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("cropBorderInv", aModule, aGridDim, aBlockDim, 0),
          corners_h(new float2[16]), norm_h(new float[16])
{

}

CropBorderInvKernel::CropBorderInvKernel(CUmodule aModule)
        : CudaKernel("cropBorderInv", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          corners_h(new float2[16]), norm_h(new float[16])
{

}

CropBorderInvKernel::~CropBorderInvKernel()
{
    delete[] corners_h;
    delete[] norm_h;
}

void CropBorderInvKernel::CopyCornersToDevice(vector<float2>& aCorners, vector<float>& aNorm)
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

float CropBorderInvKernel::operator()(CudaPitchedDeviceVariable& image,
                                      uint2 projDim,
                                      float2 cutLength,
                                      float2 dimLength,
                                      vector<float2>& corners,
                                      vector<float>& norm)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    size_t stride = image.GetPitch();
    uint cornerCount = corners.size();

    CopyCornersToDevice(corners, norm);

    void** arglist = (void**)new void*[6];

    arglist[0] = &projDim;
    arglist[1] = &stride;
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

float CropBorderInvKernel::operator()(CudaDeviceVariable& image,
                                      uint2 projDim,
                                      float2 cutLength,
                                      float2 dimLength,
                                      vector<float2>& corners,
                                      vector<float>& norm)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    size_t stride = projDim.x * sizeof(float);
    uint cornerCount = corners.size();

    CopyCornersToDevice(corners, norm);

    void** arglist = (void**)new void*[6];

    arglist[0] = &projDim;
    arglist[1] = &stride;
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



//CropBorderInvKernel::CropBorderInvKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
//        : CudaKernel("cropBorderInv", aModule, aGridDim, aBlockDim, 0)
//{
//
//}
//
//CropBorderInvKernel::CropBorderInvKernel(CUmodule aModule)
//        : CudaKernel("cropBorderInv", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
//{
//
//}
//
//float CropBorderInvKernel::operator()(CudaPitchedDeviceVariable& image, uint2 projDim, float2 cutLength, float2 dimLength, int2 p1, int2 p2, int2 p3, int2 p4)
//{
//    CUdeviceptr image_dptr = image.GetDevicePtr();
//    int proj_x = (int)projDim.x;
//    int proj_y = (int)projDim.y;
//    size_t stride = image.GetPitch();
//
//    void** arglist = (void**)new void*[10];
//
//    arglist[0] = &proj_x;
//    arglist[1] = &proj_y;
//    arglist[2] = &stride;
//    arglist[3] = &image_dptr;
//    arglist[4] = &cutLength;
//    arglist[5] = &dimLength;
//    arglist[6] = &p1;
//    arglist[7] = &p2;
//    arglist[8] = &p3;
//    arglist[9] = &p4;
//
//    float ms;
//
//    CUevent eventStart;
//    CUevent eventEnd;
//    CUstream stream = 0;
//    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
//    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));
//
//    cudaSafeCall(cuEventRecord(eventStart, stream));
//    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));
//
//    cudaSafeCall(cuCtxSynchronize());
//
//    cudaSafeCall(cuEventRecord(eventEnd, stream));
//    cudaSafeCall(cuEventSynchronize(eventEnd));
//    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));
//
//    cudaSafeCall(cuEventDestroy(eventStart));
//    cudaSafeCall(cuEventDestroy(eventEnd));
//
//    delete[] arglist;
//    return ms;
//}
//
//float CropBorderInvKernel::operator()(CudaDeviceVariable& image, uint2 projDim, float2 cutLength, float2 dimLength, int2 p1, int2 p2, int2 p3, int2 p4)
//{
//    CUdeviceptr image_dptr = image.GetDevicePtr();
//    int proj_x = (int)projDim.x;
//    int proj_y = (int)projDim.y;
//    size_t stride = projDim.x * sizeof(float);
//
//    float2 pf1, pf2, pf3, pf4;
//    pf1.x = (float)p1.x;
//    pf1.y = (float)p1.y;
//
//    pf2.x = (float)p2.x;
//    pf2.y = (float)p2.y;
//
//    pf3.x = (float)p3.x;
//    pf3.y = (float)p3.y;
//
//    pf4.x = (float)p4.x;
//    pf4.y = (float)p4.y;
//
//    void** arglist = (void**)new void*[10];
//
//    arglist[0] = &proj_x;
//    arglist[1] = &proj_y;
//    arglist[2] = &stride;
//    arglist[3] = &image_dptr;
//    arglist[4] = &cutLength;
//    arglist[5] = &dimLength;
//    arglist[6] = &pf1;
//    arglist[7] = &pf2;
//    arglist[8] = &pf3;
//    arglist[9] = &pf4;
//
//    float ms;
//
//    CUevent eventStart;
//    CUevent eventEnd;
//    CUstream stream = 0;
//    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
//    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));
//
//    cudaSafeCall(cuEventRecord(eventStart, stream));
//    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));
//
//    cudaSafeCall(cuCtxSynchronize());
//
//    cudaSafeCall(cuEventRecord(eventEnd, stream));
//    cudaSafeCall(cuEventSynchronize(eventEnd));
//    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));
//
//    cudaSafeCall(cuEventDestroy(eventStart));
//    cudaSafeCall(cuEventDestroy(eventEnd));
//
//    delete[] arglist;
//    return ms;
//}