//
// Created by uermel on 10/1/21.
//

#include "CompKernel.h"

using namespace Cuda;


CompKernel::CompKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("compare", aModule, aGridDim, aBlockDim, 0)
{

}

CompKernel::CompKernel(CUmodule aModule)
        : CudaKernel("compare", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float CompKernel::operator()(CudaPitchedDeviceVariable& real_raw, CudaPitchedDeviceVariable& virtual_raw, CudaPitchedDeviceVariable& vol_distance_map, float realLength, float4 crop, float4 cropDim, float projValScale)
{
    CUdeviceptr real_raw_dptr = real_raw.GetDevicePtr();
    CUdeviceptr virtual_raw_dptr = virtual_raw.GetDevicePtr();
    CUdeviceptr vol_distance_map_dptr = vol_distance_map.GetDevicePtr();
    int proj_x = (int)real_raw.GetWidth();
    int proj_y = (int)real_raw.GetHeight();
    size_t stride = real_raw.GetPitch();
    //std::cout << stride << std::endl << std::flush;

    void** arglist = (void**)new void*[10];

    arglist[0] = &proj_x;
    arglist[1] = &proj_y;
    arglist[2] = &stride;
    arglist[3] = &real_raw_dptr;
    arglist[4] = &virtual_raw_dptr;
    arglist[5] = &vol_distance_map_dptr;
    arglist[6] = &realLength;
    arglist[7] = &crop;
    arglist[8] = &cropDim;
    arglist[9] = &projValScale;

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