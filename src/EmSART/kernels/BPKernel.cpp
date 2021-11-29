//
// Created by uermel on 10/1/21.
//

#include <CudaSurfaces.h>
#include "BPKernel.h"

using namespace Cuda;

BPKernel::BPKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim, bool fp16)
        : CudaKernel(fp16 ? "backProjectionFP16" : "backProjection", aModule, aGridDim, aBlockDim, 2 * aBlockDim.x * aBlockDim.y * aBlockDim.z * sizeof(float) * 4)
{

}

BPKernel::BPKernel(CUmodule aModule, bool fp16)
        : CudaKernel(fp16 ? "backProjectionFP16" : "backProjection", aModule, make_dim3(1, 1, 1), make_dim3(8, 16, 4), 2 * 8 * 16 * 4 * sizeof(float) * 4)
{

}

float BPKernel::operator()(int proj_x, int proj_y, float lambda, int maxOverSample, float maxOverSampleInv, Cuda::CudaTextureObject2D& img, Cuda::CudaSurfaceObject3D& surf, float distMin, float distMax)
{
    CUtexObject texObj = img.GetTexObject();
    CUsurfObject surfObj = surf.GetSurfObject();

    void** arglist = (void**)new void*[9];

    arglist[0] = &proj_x;
    arglist[1] = &proj_y;
    arglist[2] = &lambda;
    arglist[3] = &maxOverSample;
    arglist[4] = &maxOverSampleInv;
    arglist[5] = &texObj;
    arglist[6] = &surfObj;
    arglist[7] = &distMin;
    arglist[8] = &distMax;

    //printf("==============================================================\n");
    //printf("START Back Projection Parameters START\n");
    //printf("==============================================================\n");

    //printf("proj_x: %i\n", proj_x);
    //printf("proj_y: %i\n", proj_y);
    //printf("lambda: %f\n", lambda);
    //printf("maxOverSample: %i\n", maxOverSample);
    //printf("maxOverSampleInv: %f\n", maxOverSampleInv);

    //printf("==============================================================\n");
    //printf("END Back Projection Parameters END\n");
    //printf("==============================================================\n");

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

void SetConstantValues(BPKernel& kernel, Volume<unsigned short>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv)
{
    printf("==============================================================\n");
    printf("START Back Projection Constants SHORT START\n");
    printf("==============================================================\n");
    //Set constant values
    float3 temp = proj.GetNormalVector(index);
//    temp.x = -temp.x;
//    temp.y = -temp.y;
//    temp.z = -temp.z;
    kernel.SetConstantValue("c_projNorm", &temp);
    temp = proj.GetPosition(index);
    kernel.SetConstantValue("c_detektor", &temp);
    temp = proj.GetPixelUPitch(index);
    kernel.SetConstantValue("c_uPitch", &temp);
    temp = proj.GetPixelVPitch(index);
    kernel.SetConstantValue("c_vPitch", &temp);
    temp = vol.GetSubVolumeBBoxRcp(subVol);
    kernel.SetConstantValue("c_volumeBBoxRcp", &temp);
    temp = vol.GetVolumeBBoxMin();
    kernel.SetConstantValue("c_bBoxMinComplete", &temp);
    temp = vol.GetVolumeBBoxMax();
    kernel.SetConstantValue("c_bBoxMaxComplete", &temp);
    temp = vol.GetDimension();
    kernel.SetConstantValue("c_volumeDimComplete", &temp);
    temp = vol.GetSubVolumeDimension(subVol);
    kernel.SetConstantValue("c_volumeDim", &temp);
    temp = vol.GetVoxelSize();
    kernel.SetConstantValue("c_voxelSize", &temp);
    float t = vol.GetSubVolumeZShift(subVol);
    kernel.SetConstantValue("c_zShiftForPartialVolume", &t);
    int volquat = (int)vol.GetDimension().x / 4;
    kernel.SetConstantValue("c_volumeDim_x_quarter", &volquat);
    temp = vol.GetSubVolumeBBoxMin(subVol);
    kernel.SetConstantValue("c_bBoxMin", &temp);

    float matrix[16];
    proj.GetDetectorMatrix(index, matrix, 1);
    kernel.SetConstantValue("c_DetectorMatrix", matrix);

    kernel.SetConstantValue("c_magAniso", m.GetData());
    kernel.SetConstantValue("c_magAnisoInv", mInv.GetData());

    printf("==============================================================\n");
    printf("END Back Projection Constants SHORT END\n");
    printf("==============================================================\n");
}

void SetConstantValues(BPKernel& kernel, Volume<float>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv)
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