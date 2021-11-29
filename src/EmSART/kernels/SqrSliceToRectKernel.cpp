//
// Created by uermel on 11/22/21.
//

#include "SqrSliceToRectKernel.h"

using namespace Cuda;


SqrSliceToRectKernel::SqrSliceToRectKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("squareSlices2rect", aModule, aGridDim, aBlockDim, 0)
{

}

SqrSliceToRectKernel::SqrSliceToRectKernel(CUmodule aModule)
        : CudaKernel("squareSlices2rect", aModule, make_dim3(1, 1, 1), make_dim3(4, 4, 4), 0)
{

}

float SqrSliceToRectKernel::operator()(CudaDeviceVariable& aIn,
                                            int maxsize,
                                            CudaPitchedDeviceVariable& aOut,
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
    int proj_x = (int)aOut.GetWidth();
    int proj_y = (int)aOut.GetHeight();
    int stride = (int)aOut.GetPitch();
    int _sliceNumber = sliceNumber;


    void** arglist = (void**)new void*[11];

    arglist[0] = &proj_x;
    arglist[1] = &proj_y;
    arglist[2] = &_sliceNumber;
    arglist[3] = &_maxsize;
    arglist[4] = &stride;
    arglist[5] = &in_dptr;
    arglist[6] = &out_dptr;
    arglist[7] = &_borderSizeX;
    arglist[8] = &_borderSizeY;
    arglist[9] = &_mirrorY;
    arglist[10] = &_fillZero;

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