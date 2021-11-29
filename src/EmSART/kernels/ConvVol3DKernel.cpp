//
// Created by uermel on 10/2/21.
//

#include "ConvVol3DKernel.h"

using namespace Cuda;

ConvVol3DKernel::ConvVol3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("convertVolume3DFP16ToFP32", aModule, aGridDim, aBlockDim, 0)
{

}

ConvVol3DKernel::ConvVol3DKernel(CUmodule aModule)
        : CudaKernel("convertVolume3DFP16ToFP32", aModule, make_dim3(1, 1, 1), make_dim3(8, 8, 8), 0)
{

}

float ConvVol3DKernel::operator()(Cuda::CudaPitchedDeviceVariable& img, Cuda::CudaSurfaceObject3D& surf)
{
    CUdeviceptr img_ptr = img.GetDevicePtr();
    int stride = (int)img.GetPitch();
    CUsurfObject surfObj = surf.GetSurfObject();

    void** arglist = (void**)new void*[3];

    arglist[0] = &img_ptr;
    arglist[1] = &stride;
    arglist[2] = &surfObj;

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