//
// Created by uermel on 2/7/22.
//

#include "ComputeLUTKernel.h"


using namespace Cuda;

ComputeLUTKernel::ComputeLUTKernel(CUmodule aModule)
        : CudaKernel("computeLUT", aModule)
{

}

float ComputeLUTKernel::operator()(int pixelcount,
                                   float freqStepSize,
                                   float2 Xi_x,
                                   float2 Xi_y,
                                   float2 Xi_z,
                                   float3 nu,
                                   CudaDeviceVariable& outIm)
{
    CUdeviceptr outIm_dptr = outIm.GetDevicePtr();
    //size_t stride = (pixelcount / 2 + 1) * sizeof(float2);
    size_t stride = (pixelcount / 2 + 1) * sizeof(float2);

    void** arglist = (void**)new void*[8];

    arglist[0] = &pixelcount;
    arglist[1] = &freqStepSize;
    arglist[2] = &Xi_x;
    arglist[3] = &Xi_y;
    arglist[4] = &Xi_z;
    arglist[5] = &nu;
    arglist[6] = &outIm_dptr;
    arglist[7] = &stride;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    //printf("mGridDim x:%d y:%d z:%d\n", mGridDim.x, mGridDim.y, mGridDim.z);
    //printf("mBlockDim x:%d y:%d z:%d\n", mBlockDim.x, mBlockDim.y, mBlockDim.z);
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

BoxSplineDualBKernel::BoxSplineDualBKernel(CUmodule aModule)
        : CudaKernel("computeBoxSplineDualB", aModule)
{

}

float BoxSplineDualBKernel::operator()(int2 pixelcount,
                                       float2 freqStepSize,
                                       float2 Xi_x,
                                       float2 Xi_y,
                                       float2 Xi_z,
                                       float3 nu,
                                       float thickness,
                                       CudaDeviceVariable& outIm)
{
    CUdeviceptr outIm_dptr = outIm.GetDevicePtr();

    void** arglist = (void**)new void*[8];

    arglist[0] = &pixelcount;
    arglist[1] = &freqStepSize;
    arglist[2] = &Xi_x;
    arglist[3] = &Xi_y;
    arglist[4] = &Xi_z;
    arglist[5] = &nu;
    arglist[6] = &thickness;
    arglist[7] = &outIm_dptr;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    //printf("mGridDim x:%d y:%d z:%d\n", mGridDim.x, mGridDim.y, mGridDim.z);
    //printf("mBlockDim x:%d y:%d z:%d\n", mBlockDim.x, mBlockDim.y, mBlockDim.z);
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

BoxSplineDualBReflKernel::BoxSplineDualBReflKernel(CUmodule aModule)
        : CudaKernel("computeBoxSplineDualBRefl", aModule)
{

}

float BoxSplineDualBReflKernel::operator()(int2 pixelcount,
                                           float2 freqStepSize,
                                           float2 Xi_x,
                                           float2 Xi_y,
                                           float2 Xi_z,
                                           float3 nu,
                                           CudaDeviceVariable& outIm)
{
    CUdeviceptr outIm_dptr = outIm.GetDevicePtr();

    void** arglist = (void**)new void*[7];

    arglist[0] = &pixelcount;
    arglist[1] = &freqStepSize;
    arglist[2] = &Xi_x;
    arglist[3] = &Xi_y;
    arglist[4] = &Xi_z;
    arglist[5] = &nu;
    arglist[6] = &outIm_dptr;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    //printf("mGridDim x:%d y:%d z:%d\n", mGridDim.x, mGridDim.y, mGridDim.z);
    //printf("mBlockDim x:%d y:%d z:%d\n", mBlockDim.x, mBlockDim.y, mBlockDim.z);
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