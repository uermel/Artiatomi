//
// Created by uermel on 10/2/21.
//

#include "RotKernel.h"

using namespace Cuda;

void RotKernel::computeRotMat(float phi, float psi, float theta, float rotMat[3][3])
{
    int i, j;
    float sinphi, sinpsi, sintheta;	/* sin of rotation angles */
    float cosphi, cospsi, costheta;	/* cos of rotation angles */


    float angles[] = { 0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330 };
    float angle_cos[16];
    float angle_sin[16];

    angle_cos[0] = 1.0f;
    angle_cos[1] = sqrt(3.0f) / 2.0f;
    angle_cos[2] = sqrt(2.0f) / 2.0f;
    angle_cos[3] = 0.5f;
    angle_cos[4] = 0.0f;
    angle_cos[5] = -0.5f;
    angle_cos[6] = -sqrt(2.0f) / 2.0f;
    angle_cos[7] = -sqrt(3.0f) / 2.0f;
    angle_cos[8] = -1.0f;
    angle_cos[9] = -sqrt(3.0f) / 2.0f;
    angle_cos[10] = -sqrt(2.0f) / 2.0f;
    angle_cos[11] = -0.5f;
    angle_cos[12] = 0.0f;
    angle_cos[13] = 0.5f;
    angle_cos[14] = sqrt(2.0f) / 2.0f;
    angle_cos[15] = sqrt(3.0f) / 2.0f;
    angle_sin[0] = 0.0f;
    angle_sin[1] = 0.5f;
    angle_sin[2] = sqrt(2.0f) / 2.0f;
    angle_sin[3] = sqrt(3.0f) / 2.0f;
    angle_sin[4] = 1.0f;
    angle_sin[5] = sqrt(3.0f) / 2.0f;
    angle_sin[6] = sqrt(2.0f) / 2.0f;
    angle_sin[7] = 0.5f;
    angle_sin[8] = 0.0f;
    angle_sin[9] = -0.5f;
    angle_sin[10] = -sqrt(2.0f) / 2.0f;
    angle_sin[11] = -sqrt(3.0f) / 2.0f;
    angle_sin[12] = -1.0f;
    angle_sin[13] = -sqrt(3.0f) / 2.0f;
    angle_sin[14] = -sqrt(2.0f) / 2.0f;
    angle_sin[15] = -0.5f;

    for (i = 0, j = 0; i<16; i++)
        if (angles[i] == phi)
        {
            cosphi = angle_cos[i];
            sinphi = angle_sin[i];
            j = 1;
        }

    if (j < 1)
    {
        phi = phi * (float)M_PI / 180.0f;
        cosphi = cos(phi);
        sinphi = sin(phi);
    }

    for (i = 0, j = 0; i<16; i++)
        if (angles[i] == psi)
        {
            cospsi = angle_cos[i];
            sinpsi = angle_sin[i];
            j = 1;
        }

    if (j < 1)
    {
        psi = psi * (float)M_PI / 180.0f;
        cospsi = cos(psi);
        sinpsi = sin(psi);
    }

    for (i = 0, j = 0; i<16; i++)
        if (angles[i] == theta)
        {
            costheta = angle_cos[i];
            sintheta = angle_sin[i];
            j = 1;
        }

    if (j < 1)
    {
        theta = theta * (float)M_PI / 180.0f;
        costheta = cos(theta);
        sintheta = sin(theta);
    }

    /* calculation of rotation matrix */

    rotMat[0][0] = cospsi*cosphi - costheta*sinpsi*sinphi;
    rotMat[1][0] = sinpsi*cosphi + costheta*cospsi*sinphi;
    rotMat[2][0] = sintheta*sinphi;
    rotMat[0][1] = -cospsi*sinphi - costheta*sinpsi*cosphi;
    rotMat[1][1] = -sinpsi*sinphi + costheta*cospsi*cosphi;
    rotMat[2][1] = sintheta*cosphi;
    rotMat[0][2] = sintheta*sinpsi;
    rotMat[1][2] = -sintheta*cospsi;
    rotMat[2][2] = costheta;
}

RotKernel::RotKernel(CUmodule aModule, int aSize)
        : CudaKernel("rot3d", aModule, make_dim3((aSize + 7) / 8, (aSize + 7) / 8, (aSize + 7) / 8), make_dim3(8, 8, 8), 0),
          size(aSize),
          volTexArray(this, "texVol", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0,
                      CU_AD_FORMAT_FLOAT, aSize, aSize, aSize, 1)
{

}

float RotKernel::operator()(Cuda::CudaDeviceVariable & aVolOut, float phi, float psi, float theta)
{
    //make sure that the texture is properly bound to textref as other rotTools might have changed it!
    volTexArray.BindToTexRef();

    float rotMat[3][3];
    computeRotMat(phi, psi, theta, rotMat);
    CUdeviceptr out_dptr = aVolOut.GetDevicePtr();

    float3 rotMat0 = make_float3(rotMat[0][0], rotMat[0][1], rotMat[0][2]);
    float3 rotMat1 = make_float3(rotMat[1][0], rotMat[1][1], rotMat[1][2]);
    float3 rotMat2 = make_float3(rotMat[2][0], rotMat[2][1], rotMat[2][2]);

    void** arglist = (void**)new void*[5];

    arglist[0] = &size;
    arglist[1] = &rotMat0;
    arglist[2] = &rotMat1;
    arglist[3] = &rotMat2;
    arglist[4] = &out_dptr;

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

void RotKernel::SetData(float* data)
{
    volTexArray.GetArray()->CopyFromHostToArray(data);
}