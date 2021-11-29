//
// Created by uermel on 10/1/21.
//

#include "WbpWeightingKernel.h"

using namespace Cuda;

WbpWeightingKernel::WbpWeightingKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("wbpWeighting", aModule, aGridDim, aBlockDim, 0)
{

}

WbpWeightingKernel::WbpWeightingKernel(CUmodule aModule)
        : CudaKernel("wbpWeighting", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float WbpWeightingKernel::operator()(CudaDeviceVariable& img, size_t stride, unsigned int pixelcount, float psiAngle, FilterMethod fm, int proj_index, int projectionCount, float thickness, Cuda::CudaDeviceVariable& tiltAngles)
{
    CUdeviceptr img_dptr = img.GetDevicePtr();
    CUdeviceptr tiltAngles_dptr = tiltAngles.GetDevicePtr();
    float _angle = -psiAngle / 180.0f * (float)M_PI;

    void** arglist = (void**)new void*[9];

    arglist[0] = &img_dptr;
    arglist[1] = &stride;
    arglist[2] = &pixelcount;
    arglist[3] = &_angle;
    arglist[4] = &fm;
    arglist[5] = &proj_index;
    arglist[6] = &projectionCount;
    arglist[7] = &thickness;
    arglist[8] = &tiltAngles_dptr;

    float ms;

    //CudaDeviceVariable filter(pixelcount*pixelcount*sizeof(float));

    printf("pixelcount: %d \n", pixelcount);
    printf("stride: %ld \n", stride);

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