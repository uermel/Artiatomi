//
// Created by uermel on 9/30/21.
//

#include "FPLUTKernel.h"

using namespace Cuda;

FPLUTKernel::FPLUTKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : Cuda::CudaKernel("forwardProjectionLUT", aModule, aGridDim, aBlockDim, 0)
{

}

FPLUTKernel::FPLUTKernel(CUmodule aModule)
        : Cuda::CudaKernel("forwardProjectionLUT", aModule, make_dim3(1,1,1), make_dim3(4, 4, 4), 0)
{

}

float FPLUTKernel::operator()(int x,
                              int y,
                              Cuda::CudaPitchedDeviceVariable& projection,
                              Cuda::CudaPitchedDeviceVariable& distMap,
                              Cuda::CudaTextureObject2D& LUT,
                              Cuda::CudaSurfaceObject3D& volume,
                              float tmin,
                              float tmax,
                              float support,
                              float LUTstepinv,
                              float LUTcenter)
{
    return (*this)(x,
                   y,
                   projection,
                   distMap,
                   LUT,
                   volume,
                   tmin,
                   tmax,
                   support,
                   LUTstepinv,
                   LUTcenter,
                   make_int2(0, 0),
                   make_int2(x, y));
}

float FPLUTKernel::operator()(int x,
                              int y,
                              Cuda::CudaPitchedDeviceVariable& projection,
                              Cuda::CudaPitchedDeviceVariable& distMap,
                              Cuda::CudaTextureObject2D& LUT,
                              Cuda::CudaSurfaceObject3D& volume,
                              float tmin,
                              float tmax,
                              float support,
                              float LUTstepinv,
                              float LUTcenter,
                              int2 roiMin,
                              int2 roiMax)
{
    CUdeviceptr proj_dptr = projection.GetDevicePtr();
    size_t stride = projection.GetPitch();
    CUdeviceptr distmap_dptr = distMap.GetDevicePtr();
    CUtexObject LUTobj = LUT.GetTexObject();
    CUsurfObject VolObj = volume.GetSurfObject();
    float supporthalf = support * 0.5f;

    void** arglist = (void**)new void*[14];

    arglist[0] = &x;
    arglist[1] = &y;
    arglist[2] = &stride;
    arglist[3] = &proj_dptr;
    arglist[4] = &distmap_dptr;
    arglist[5] = &LUTobj;
    arglist[6] = &VolObj;
    arglist[7] = &tmin;
    arglist[8] = &tmax;
    arglist[9] = &roiMin;
    arglist[10] = &roiMax;
    arglist[11] = &supporthalf;
    arglist[12] = &LUTstepinv;
    arglist[13] = &LUTcenter;

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

void SetConstantValues(FPLUTKernel& kernel, Volume<float>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv)
{
    //printf("==============================================================\n");
    //printf("START Back Projection Constants FLOAT START\n");
    //printf("==============================================================\n");
    //Set constant values
    float3 temp = proj.GetNormalVector(index);
//    temp.x = -temp.x;
//    temp.y = -temp.y;
//    temp.z = -temp.z;
    kernel.SetConstantValue("c_projNorm", &temp);
    //printf("c_projNorm: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = proj.GetPosition(index);
    kernel.SetConstantValue("c_detektor", &temp);
    //printf("c_detektor: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = proj.GetPixelUPitch(index);
    kernel.SetConstantValue("c_uPitch", &temp);
    //printf("c_uPitch: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = proj.GetPixelVPitch(index);
    kernel.SetConstantValue("c_vPitch", &temp);
    //printf("c_vPitch: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetSubVolumeBBoxRcp(subVol);
    kernel.SetConstantValue("c_volumeBBoxRcp", &temp);
    //printf("c_volumeBBoxRcp: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetVolumeBBoxMin();
    kernel.SetConstantValue("c_bBoxMinComplete", &temp);
    //printf("c_bBoxMinComplete: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetVolumeBBoxMax();
    kernel.SetConstantValue("c_bBoxMaxComplete", &temp);
    //printf("c_bBoxMaxComplete: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetDimension();
    kernel.SetConstantValue("c_volumeDimComplete", &temp);
    //printf("c_volumeDimComplete: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetSubVolumeDimension(subVol);
    kernel.SetConstantValue("c_volumeDim", &temp);
    //printf("c_volumeDim: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetVoxelSize();
    kernel.SetConstantValue("c_voxelSize", &temp);
    //printf("c_voxelSize: %f, %f, %f\n", temp.x, temp.y, temp.z);

    float t = vol.GetSubVolumeZShift(subVol);
    kernel.SetConstantValue("c_zShiftForPartialVolume", &t);
    //printf("c_zShiftForPartialVolume: %f\n", t);

    int volquat = (int)vol.GetDimension().x / 4;
    kernel.SetConstantValue("c_volumeDim_x_quarter", &volquat);
    //printf("c_volumeDim_x_quarter: %i\n", volquat);

    temp = vol.GetSubVolumeBBoxMin(subVol);
    kernel.SetConstantValue("c_bBoxMin", &temp);
    //printf("c_bBoxMin: %f, %f, %f\n", temp.x, temp.y, temp.z);

    float matrix[16];
    proj.GetDetectorMatrix(index, matrix, 1);
    kernel.SetConstantValue("c_DetectorMatrix", matrix);
    //printf("c_DetectorMatrix:\n %f, %f, %f, %f,\n %f, %f, %f, %f,\n %f, %f, %f, %f,\n %f, %f, %f, %f,\n",
    //       matrix[0],matrix[1],matrix[2],matrix[3],
    //       matrix[4],matrix[5],matrix[6],matrix[7],
    //       matrix[8],matrix[9],matrix[10],matrix[11],
    //       matrix[12],matrix[13],matrix[14],matrix[15]);

    kernel.SetConstantValue("c_magAniso", m.GetData());
    //printf("c_projNorm: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_magAnisoInv", mInv.GetData());
    //printf("c_projNorm: %f, %f, %f\n", temp.x, temp.y, temp.z);

    //printf("==============================================================\n");
    //printf("END Back Projection Constants FLOAT END\n");
    //printf("==============================================================\n");
}