//
// Created by uermel on 9/30/21.
//

#include "SlicerKernel.h"
#include <CudaSurfaces.h>

using namespace Cuda;

SlicerKernel::SlicerKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("slicer", aModule, aGridDim, aBlockDim, 0)
{

}

SlicerKernel::SlicerKernel(CUmodule aModule)
        : CudaKernel("slicer", aModule, make_dim3(1, 1, 1), make_dim3(32, 8, 1), 0)
{

}

float SlicerKernel::operator()(int x, int y, CudaPitchedDeviceVariable& projection, float tmin, float tmax, Cuda::CudaTextureObject3D& texObj)
{
    return (*this)(x, y, projection, tmin, tmax, texObj, make_int2(0, 0), make_int2(x, y));
}

float SlicerKernel::operator()(int x, int y, CudaPitchedDeviceVariable& projection, float tmin, float tmax, Cuda::CudaTextureObject3D& texObj, int2 roiMin, int2 roiMax)
{
    CUdeviceptr proj_dptr = projection.GetDevicePtr();
    size_t stride = projection.GetPitch();
    CUtexObject tex = texObj.GetTexObject();

    void** arglist = (void**)new void*[9];

    arglist[0] = &x;
    arglist[1] = &y;
    arglist[2] = &stride;
    arglist[3] = &proj_dptr;
    arglist[4] = &tmin;
    arglist[5] = &tmax;
    arglist[6] = &tex;
    arglist[7] = &roiMin;
    arglist[8] = &roiMax;

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

void SetConstantValues(SlicerKernel& kernel, Volume<unsigned short>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv)
{
    printf("==============================================================\n");
    printf("START Forward Projection Constants SHORT START\n");
    printf("==============================================================\n");
    //Set constant values
    float3 temp = proj.GetNormalVector(index);
    printf("c_projNorm: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_projNorm", &temp);

    temp = proj.GetPosition(index);
    printf("c_detektor: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_detektor", &temp);

    temp = proj.GetPixelUPitch(index);
    printf("c_uPitch: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_uPitch", &temp);

    temp = proj.GetPixelVPitch(index);
    printf("c_vPitch: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_vPitch", &temp);

    temp = vol.GetSubVolumeBBoxRcp(subVol);
    printf("c_volumeBBoxRcp: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_volumeBBoxRcp", &temp);

    temp = vol.GetVolumeBBoxMin();
    printf("c_bBoxMinComplete: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_bBoxMinComplete", &temp);

    temp = vol.GetVolumeBBoxMax();
    printf("c_bBoxMaxComplete: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_bBoxMaxComplete", &temp);

    temp = vol.GetSubVolumeBBoxMin(subVol);
    printf("c_bBoxMin: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_bBoxMin", &temp);

    temp = vol.GetSubVolumeBBoxMax(subVol);
    printf("c_bBoxMax: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_bBoxMax", &temp);

    temp = vol.GetDimension();
    printf("c_volumeDimComplete: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_volumeDimComplete", &temp);

    temp = vol.GetSubVolumeDimension(subVol);
    printf("c_volumeDim: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_volumeDim", &temp);

    temp = vol.GetVoxelSize();
    printf("c_voxelSize: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_voxelSize", &temp);

    float t = 0;//vol.GetSubVolumeZShift(subVol);
    //printf("zShift: %f\n", t);
    kernel.SetConstantValue("c_zShiftForPartialVolume", &t);


    kernel.SetConstantValue("c_magAniso", m.GetData());
    kernel.SetConstantValue("c_magAnisoInv", mInv.GetData());

    printf("==============================================================\n");
    printf("END Forward Projection Constants END\n");
    printf("==============================================================\n");
}

void SetConstantValues(SlicerKernel& kernel, Volume<float>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv)
{
    //printf("==============================================================\n");
    //printf("START Forward Projection Constants FLOAT START\n");
    //printf("==============================================================\n");
    //Set constant values
    float3 temp = proj.GetNormalVector(index);
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

    temp = vol.GetSubVolumeBBoxMin(subVol);
    kernel.SetConstantValue("c_bBoxMin", &temp);
    //printf("c_bBoxMin: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetSubVolumeBBoxMax(subVol);
    kernel.SetConstantValue("c_bBoxMax", &temp);
    //printf("c_bBoxMax: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetDimension();
    kernel.SetConstantValue("c_volumeDimComplete", &temp);
    //printf("c_volumeDimComplete: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetSubVolumeDimension(subVol);
    kernel.SetConstantValue("c_volumeDim", &temp);
    //printf("c_volumeDim: %f, %f, %f\n", temp.x, temp.y, temp.z);

    temp = vol.GetVoxelSize();
    kernel.SetConstantValue("c_voxelSize", &temp);
    //printf("c_voxelSize: %f, %f, %f\n", temp.x, temp.y, temp.z);

    float t = 0;//vol.GetSubVolumeZShift(subVol);
    ////printf("zShift: %f\n", t);
    kernel.SetConstantValue("c_zShiftForPartialVolume", &t);

    kernel.SetConstantValue("c_magAniso", m.GetData());
    kernel.SetConstantValue("c_magAnisoInv", mInv.GetData());
    //printf("==============================================================\n");
    //printf("END Forward Projection Constants END\n");
    //printf("==============================================================\n");
}