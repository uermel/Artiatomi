//
// Created by uermel on 11/23/21.
//

#include "CropSlicesKernel.h"

using namespace Cuda;

CropSlicesKernel::CropSlicesKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("cropBorderSlices", aModule, aGridDim, aBlockDim, 0)
{

}

CropSlicesKernel::CropSlicesKernel(CUmodule aModule)
        : CudaKernel("cropBorderSlices", aModule, make_dim3(1, 1, 1), make_dim3(4, 4, 4), 0)
{

}

float CropSlicesKernel::operator()(CudaDeviceVariable& image,
                                   int proj_x,
                                   int proj_y,
                                   int sliceNumber,
                                   float2 cutLength,
                                   float2 dimLength,
                                   int2 p1,
                                   int2 p2,
                                   int2 p3,
                                   int2 p4)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();

    void** arglist = (void**)new void*[10];

    arglist[0] = &proj_x;
    arglist[1] = &proj_y;
    arglist[2] = &sliceNumber;
    arglist[3] = &image_dptr;
    arglist[4] = &cutLength;
    arglist[5] = &dimLength;
    arglist[6] = &p1;
    arglist[7] = &p2;
    arglist[8] = &p3;
    arglist[9] = &p4;

    printf("\nCropKernel Grid: %d %d %d\n", mGridDim.x, mGridDim.y, mGridDim.z);
    printf("\nCropKernel Block: %d %d %d\n", mBlockDim.x, mBlockDim.y, mBlockDim.z);

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