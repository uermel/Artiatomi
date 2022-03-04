//  Copyright (c) 2018, Michael Kunz and Frangakis Lab, BMLS,
//  Goethe University, Frankfurt am Main.
//  All rights reserved.
//  http://kunzmi.github.io/Artiatomi
//  
//  This file is part of the Artiatomi package.
//  
//  Artiatomi is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  Artiatomi is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with Artiatomi. If not, see <http://www.gnu.org/licenses/>.
//  
////////////////////////////////////////////////////////////////////////


#include "CudaRot.h"
#include "CudaKernelBinaries.h"

CudaRot::CudaRot(int aVolSize, CUstream aStream, CudaContext* context, bool linearInterpolation)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1),
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize),
	  shiftTex(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 1, 0),
	  dataTex(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 1, 0),
	  dataTexCplx(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 2, 0),
	  oldphi(0), oldpsi(0), oldtheta(0)
{
	CUmodule cuMod = ctx->LoadModulePTX(SubTomogramAverageBasicKernel, 0, false, false);
	// CUmodule cuMod = ctx->LoadModule("basicKernels.ptx");

	shift = new CudaKernel("shift", cuMod);
	rotVol = new CudaKernel("rot3d", cuMod);
	shiftRotVol = new CudaKernel("shiftRot3d", cuMod);
	rotVolCplx = new CudaKernel("rot3dCplx", cuMod);

	CUfilter_mode filter = linearInterpolation ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;

	CudaTextureArray3D shiftTex2(rotVol, "texShift", CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_WRAP,
		CU_TR_ADDRESS_MODE_WRAP, CU_TR_FILTER_MODE_LINEAR, CU_TRSF_NORMALIZED_COORDINATES, &shiftTex);
	CudaTextureArray3D tex(rotVol, "texVol", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
		CU_TR_ADDRESS_MODE_CLAMP, filter, 0, &dataTex);
	CudaTextureArray3D texCplx(rotVolCplx, "texVolCplx", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
		CU_TR_ADDRESS_MODE_CLAMP, filter, 0, &dataTexCplx);

}


void CudaRot::SetTexture(CudaDeviceVariable& d_idata)
{
	dataTex.CopyFromDeviceToArray(d_idata);
}

void CudaRot::SetTextureShift(CudaDeviceVariable& d_idata)
{
	shiftTex.CopyFromDeviceToArray(d_idata);
}

void CudaRot::SetTextureCplx(CudaDeviceVariable& d_idata)
{
	dataTexCplx.CopyFromDeviceToArray(d_idata);
}

void CudaRot::Rot(CudaDeviceVariable& d_odata, float phi, float psi, float theta)
{
	float rotMat1[9];
	float rotMat2[9];
	float rotMat[9];
	computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
	computeRotMat(phi, psi, theta, rotMat2);
	multiplyRotMatrix(rotMat1, rotMat2, rotMat);

	runRotKernel(d_odata, rotMat);
}

void CudaRot::ShiftRot(CudaDeviceVariable& d_odata, float3 shiftVal, float phi, float psi, float theta)
{
    float rotMat1[9];
    float rotMat2[9];
    float rotMat[9];
    computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
    computeRotMat(phi, psi, theta, rotMat2);
    multiplyRotMatrix(rotMat1, rotMat2, rotMat);

    runShiftRotKernel(d_odata, shiftVal, rotMat);
}

void CudaRot::Shift(CudaDeviceVariable& d_odata, float3 shiftVal)
{
	runShiftKernel(d_odata, shiftVal);
}

void CudaRot::RotCplx(CudaDeviceVariable& d_odata, float phi, float psi, float theta)
{
	float rotMat1[9];
	float rotMat2[9];
	float rotMat[9];
	computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
	computeRotMat(phi, psi, theta, rotMat2);
	multiplyRotMatrix(rotMat1, rotMat2, rotMat);

	runRotCplxKernel(d_odata, rotMat);
}

void CudaRot::computeRotMat(float phi, float psi, float the, float rotMat[9])
{
    float sinphi, sinpsi, sinthe;	/* sin of rotation angles */
    float cosphi, cospsi, costhe;	/* cos of rotation angles */

    sinphi = sin(phi * (float)M_PI/180.f);
    sinpsi = sin(psi * (float)M_PI/180.f);
    sinthe = sin(the * (float)M_PI/180.f);

    cosphi = cos(phi * (float)M_PI/180.f);
    cospsi = cos(psi * (float)M_PI/180.f);
    costhe = cos(the * (float)M_PI/180.f);

    /* calculation of rotation matrix */
    // [ 0 1 2
    //   3 4 5
    //   6 7 8 ]
    // This is the matrix of the actual forward rotation     // rot3dc.c from TOM
    rotMat[0] = cosphi * cospsi - costhe * sinphi * sinpsi;  // rm00 = cospsi*cosphi-costheta*sinpsi*sinphi;
    rotMat[1] = -cospsi * sinphi - cosphi * costhe * sinpsi; // rm01 =-cospsi*sinphi-costheta*sinpsi*cosphi;
    rotMat[2] = sinpsi * sinthe;                             // rm02 = sintheta*sinpsi;
    rotMat[3] = cosphi * sinpsi + cospsi * costhe * sinphi;  // rm10 = sinpsi*cosphi+costheta*cospsi*sinphi;
    rotMat[4] = cosphi * cospsi * costhe - sinphi * sinpsi;  // rm11 =-sinpsi*sinphi+costheta*cospsi*cosphi;
    rotMat[5] = -cospsi * sinthe;                            // rm12 =-sintheta*cospsi;
    rotMat[6] = sinphi * sinthe;                             // rm20 = sintheta*sinphi;
    rotMat[7] = cosphi * sinthe;                             // rm21 = sintheta*cosphi;
    rotMat[8] = costhe;                                      // rm22 = costheta;
}

void CudaRot::multiplyRotMatrix(const float B[9], const float A[9], float out[9])
{
    // Implements Matrix rotation out = B * A (matlab convention)
    out[0] = A[0]*B[0] + A[3]*B[1] + A[6]*B[2];
    out[1] = A[1]*B[0] + A[4]*B[1] + A[7]*B[2];
    out[2] = A[2]*B[0] + A[5]*B[1] + A[8]*B[2];
    out[3] = A[0]*B[3] + A[3]*B[4] + A[6]*B[5];
    out[4] = A[1]*B[3] + A[4]*B[4] + A[7]*B[5];
    out[5] = A[2]*B[3] + A[5]*B[4] + A[8]*B[5];
    out[6] = A[0]*B[6] + A[3]*B[7] + A[6]*B[8];
    out[7] = A[1]*B[6] + A[4]*B[7] + A[7]*B[8];
    out[8] = A[2]*B[6] + A[5]*B[7] + A[8]*B[8];
}

void CudaRot::runRotKernel(CudaDeviceVariable& d_odata, float rotMat[9])
{
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    // Transposed Matrix for rotation
    // [ 0 1 2          [ 0 3 6         [ rotMat0
    //   3 4 5    --->    1 4 7   --->    rotMat1
    //   6 7 8 ]          2 5 8 ]         rotMat2 ]
    // x_rot = rotMat0.x * x + rotMat0.y * y + rotMat0.z * z
    // y_rot = rotMat1.x * x + rotMat1.y * y + rotMat1.z * z
    // z_rot = rotMat2.x * x + rotMat2.y * y + rotMat2.z * z
    float3 rotMat0 = make_float3(rotMat[0], rotMat[3], rotMat[6]);
    float3 rotMat1 = make_float3(rotMat[1], rotMat[4], rotMat[7]);
    float3 rotMat2 = make_float3(rotMat[2], rotMat[5], rotMat[8]);

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &rotMat0;
    arglist[2] = &rotMat1;
    arglist[3] = &rotMat2;
    arglist[4] = &out_dptr;

    cudaSafeCall(cuLaunchKernel(rotVol->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}

void CudaRot::runShiftRotKernel(CudaDeviceVariable& d_odata, float3 shiftVal, float rotMat[9])
{
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    // Transposed Matrix for rotation
    // [ 0 1 2          [ 0 3 6         [ rotMat0
    //   3 4 5    --->    1 4 7   --->    rotMat1
    //   6 7 8 ]          2 5 8 ]         rotMat2 ]
    // x_rot = rotMat0.x * x + rotMat0.y * y + rotMat0.z * z
    // y_rot = rotMat1.x * x + rotMat1.y * y + rotMat1.z * z
    // z_rot = rotMat2.x * x + rotMat2.y * y + rotMat2.z * z
    float3 rotMat0 = make_float3(rotMat[0], rotMat[3], rotMat[6]);
    float3 rotMat1 = make_float3(rotMat[1], rotMat[4], rotMat[7]);
    float3 rotMat2 = make_float3(rotMat[2], rotMat[5], rotMat[8]);

    void** arglist = (void**)new void*[6];

    arglist[0] = &volSize;
    arglist[1] = &shiftVal;
    arglist[2] = &rotMat0;
    arglist[3] = &rotMat1;
    arglist[4] = &rotMat2;
    arglist[5] = &out_dptr;

    cudaSafeCall(cuLaunchKernel(shiftRotVol->GetCUfunction(), gridSize.x, gridSize.y,
                                gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}

void CudaRot::runShiftKernel(CudaDeviceVariable& d_odata, float3 shiftVal)
{
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &out_dptr;
    arglist[2] = &shiftVal;

    cudaSafeCall(cuLaunchKernel(shift->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}

void CudaRot::runRotCplxKernel(CudaDeviceVariable& d_odata, float rotMat[9])
{
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    // Transposed Matrix for rotation
    // [ 0 1 2          [ 0 3 6         [ rotMat0
    //   3 4 5    --->    1 4 7   --->    rotMat1
    //   6 7 8 ]          2 5 8 ]         rotMat2 ]
    // x_rot = rotMat0.x * x + rotMat0.y * y + rotMat0.z * z
    // y_rot = rotMat1.x * x + rotMat1.y * y + rotMat1.z * z
    // z_rot = rotMat2.x * x + rotMat2.y * y + rotMat2.z * z
    float3 rotMat0 = make_float3(rotMat[0], rotMat[3], rotMat[6]);
    float3 rotMat1 = make_float3(rotMat[1], rotMat[4], rotMat[7]);
    float3 rotMat2 = make_float3(rotMat[2], rotMat[5], rotMat[8]);

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &rotMat0;
    arglist[2] = &rotMat1;
    arglist[3] = &rotMat2;
    arglist[4] = &out_dptr;

    cudaSafeCall(cuLaunchKernel(rotVolCplx->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}

void CudaRot::SetOldAngles(float aPhi, float aPsi, float aTheta)
{
	oldphi = aPhi;
	oldpsi = aPsi;
	oldtheta = aTheta;
}
