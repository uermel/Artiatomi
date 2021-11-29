//
// Created by uermel on 11/17/21.
//

#include "CopyToRectSlicesKernel.h"

using namespace Cuda;


CopyToRectSlicesKernel::CopyToRectSlicesKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("makeRectSlices", aModule, aGridDim, aBlockDim, 0)
{

}

CopyToRectSlicesKernel::CopyToRectSlicesKernel(CUmodule aModule)
        : CudaKernel("makeRectSlices", aModule, make_dim3(1, 1, 1), make_dim3(4, 4, 4), 0)
{

}

float CopyToRectSlicesKernel::operator()(CudaDeviceVariable& aIn,
                                         int proj_x,
                                         int proj_y,
                                         int maxsize,
                                         CudaDeviceVariable& aOut,
                                         int borderSizeX,
                                         int borderSizeY,
                                         bool mirrorY,
                                         bool fillZero,
                                         int sliceNumber)
{
    CUdeviceptr in_dptr = aIn.GetDevicePtr();
    CUdeviceptr out_dptr = aOut.GetDevicePtr();
    int _maxsize = maxsize;
    int _borderSizeX = borderSizeX;
    int _borderSizeY = borderSizeY;
    bool _mirrorY = mirrorY;
    bool _fillZero = fillZero;
//    int proj_x = (int)aIn.GetWidth();
//    int proj_y = (int)aIn.GetHeight();
    //int stride = (int)aIn.GetPitch();
    int _sliceNumber = sliceNumber;


    void** arglist = (void**)new void*[10];

    arglist[0] = &proj_x;
    arglist[1] = &proj_y;
    arglist[2] = &_sliceNumber;
    arglist[3] = &_maxsize;
    arglist[4] = &in_dptr;
    arglist[5] = &out_dptr;
    arglist[6] = &_borderSizeX;
    arglist[7] = &_borderSizeY;
    arglist[8] = &_mirrorY;
    arglist[9] = &_fillZero;

    printf("\nmGridDim: %u %u %u\n", mGridDim.x, mGridDim.y, mGridDim.z);
    printf("\nmBlockDim: %u %u %u\n", mBlockDim.x, mBlockDim.y, mBlockDim.z);
    printf("sliceNumber: %i", sliceNumber);

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