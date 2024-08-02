//
// Created by uermel on 10/1/21.
//

#include "CopyToSquareKernel.h"

using namespace Cuda;


CopyToSquareKernel::CopyToSquareKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("makeSquare", aModule, aGridDim, aBlockDim, 0)
{

}

CopyToSquareKernel::CopyToSquareKernel(CUmodule aModule)
        : CudaKernel("makeSquare", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float CopyToSquareKernel::operator()(CudaPitchedDeviceVariable& aIn, int maxsize, CudaDeviceVariable& aOut, int borderSizeX, int borderSizeY, bool mirrorY, bool fillZero)
{
    CUdeviceptr in_dptr = aIn.GetDevicePtr();
    CUdeviceptr out_dptr = aOut.GetDevicePtr();
    int _maxsize = maxsize;
    int _borderSizeX = borderSizeX;
    int _borderSizeY = borderSizeY;
    bool _mirrorY = mirrorY;
    bool _fillZero = fillZero;
    int proj_x = (int)aIn.GetWidth();
    int proj_y = (int)aIn.GetHeight();
    int stride = (int)aIn.GetPitch();


    void** arglist = (void**)new void*[10];

    arglist[0] = &proj_x;
    arglist[1] = &proj_y;
    arglist[2] = &_maxsize;
    arglist[3] = &stride;
    arglist[4] = &in_dptr;
    arglist[5] = &out_dptr;
    arglist[6] = &_borderSizeX;
    arglist[7] = &_borderSizeY;
    arglist[8] = &_mirrorY;
    arglist[9] = &_fillZero;

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

CopyToPitchedKernel::CopyToPitchedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("copyToPitched", aModule, aGridDim, aBlockDim, 0)
{

}

CopyToPitchedKernel::CopyToPitchedKernel(CUmodule aModule)
        : CudaKernel("copyToPitched", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float CopyToPitchedKernel::operator()(CudaDeviceVariable& aIn,
                                      CudaPitchedDeviceVariable& aOut,
                                      uint2 imDim)
{
    CUdeviceptr in_dptr = aIn.GetDevicePtr();
    CUdeviceptr out_dptr = aOut.GetDevicePtr();
    size_t stride = aOut.GetPitch();

    void** arglist = (void**)new void*[4];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &stride;
    arglist[3] = &imDim;

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

CopyFromPitchedKernel::CopyFromPitchedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("copyFromPitched", aModule, aGridDim, aBlockDim, 0)
{

}

CopyFromPitchedKernel::CopyFromPitchedKernel(CUmodule aModule)
        : CudaKernel("copyFromPitched", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float CopyFromPitchedKernel::operator()(CudaPitchedDeviceVariable& aIn,
                                        CudaDeviceVariable& aOut,
                                        uint2 imDim)
{
    CUdeviceptr in_dptr = aIn.GetDevicePtr();
    CUdeviceptr out_dptr = aOut.GetDevicePtr();
    size_t stride = aIn.GetPitch();

    void** arglist = (void**)new void*[4];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &stride;
    arglist[3] = &imDim;

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

AddToPitchedKernel::AddToPitchedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("addToPitched", aModule, aGridDim, aBlockDim, 0)
{

}

AddToPitchedKernel::AddToPitchedKernel(CUmodule aModule)
        : CudaKernel("addToPitched", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float AddToPitchedKernel::operator()(CudaDeviceVariable& aIn,
                                     CudaPitchedDeviceVariable& aOut,
                                     uint2 imDim)
{
    CUdeviceptr in_dptr = aIn.GetDevicePtr();
    CUdeviceptr out_dptr = aOut.GetDevicePtr();
    size_t stride = aOut.GetPitch();

    void** arglist = (void**)new void*[4];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &stride;
    arglist[3] = &imDim;

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

SlicesToArraysKernel::SlicesToArraysKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("slicesToSurfs", aModule, aGridDim, aBlockDim, 0)
{

}

SlicesToArraysKernel::SlicesToArraysKernel(CUmodule aModule)
        : CudaKernel("slicesToSurfs", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

void SlicesToArraysKernel::setSlices(vector<Cuda::CudaSurfaceObject2D *> surfs, int sliceNumber)
{
    slicenum = sliceNumber;
    d_surfaces.Alloc(slicenum * sizeof(CUsurfObject));
    auto h_surfaces = new CUsurfObject[slicenum];

    for (int slice=0; slice < slicenum; slice++){
        h_surfaces[slice] = surfs[slice]->GetSurfObject();
    }

    d_surfaces.CopyHostToDevice(h_surfaces, slicenum * sizeof(CUsurfObject));

    delete[] h_surfaces;
}

float SlicesToArraysKernel::operator()(CudaDeviceVariable& stack, uint2 imDim)
{
    CUdeviceptr in_dptr = stack.GetDevicePtr();
    CUdeviceptr out_dptr = d_surfaces.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &imDim;
    arglist[3] = &slicenum;

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

MaskedSlicesToArraysKernel::MaskedSlicesToArraysKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("slicesToSurfsMasked", aModule, aGridDim, aBlockDim, 0)
{

}

MaskedSlicesToArraysKernel::MaskedSlicesToArraysKernel(CUmodule aModule)
        : CudaKernel("slicesToSurfsMasked", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

void MaskedSlicesToArraysKernel::setSlices(vector<Cuda::CudaSurfaceObject2D *> surfs, int sliceNumber)
{
    slicenum = sliceNumber;
    d_surfaces.Alloc(slicenum * sizeof(CUsurfObject));
    auto h_surfaces = new CUsurfObject[slicenum];

    for (int slice=0; slice < slicenum; slice++){
        h_surfaces[slice] = surfs[slice]->GetSurfObject();
    }

    d_surfaces.CopyHostToDevice(h_surfaces, slicenum * sizeof(CUsurfObject));

    delete[] h_surfaces;
}

float MaskedSlicesToArraysKernel::operator()(CudaDeviceVariable& stack,
                                             CudaPitchedDeviceVariable& mask,
                                             float maskScale,
                                             uint2 imDim)
{
    CUdeviceptr in_dptr = stack.GetDevicePtr();
    CUdeviceptr mask_dptr = mask.GetDevicePtr();
    size_t stride = mask.GetPitch();
    CUdeviceptr out_dptr = d_surfaces.GetDevicePtr();

    void** arglist = (void**)new void*[7];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &mask_dptr;
    arglist[3] = &stride;
    arglist[4] = &maskScale;
    arglist[5] = &imDim;
    arglist[6] = &slicenum;

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