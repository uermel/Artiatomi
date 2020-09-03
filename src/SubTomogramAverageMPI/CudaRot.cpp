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
	float rotMat1[3][3];
	float rotMat2[3][3];
	float rotMat[3][3];
	computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
	computeRotMat(phi, psi, theta, rotMat2);
	multiplyRotMatrix(rotMat2, rotMat1, rotMat);

	runRotKernel(d_odata, rotMat);
}

void CudaRot::ShiftRot(CudaDeviceVariable& d_odata, float3 shiftVal, float phi, float psi, float theta)
{
    float rotMat1[3][3];
    float rotMat2[3][3];
    float rotMat[3][3];
    computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
    computeRotMat(phi, psi, theta, rotMat2);
    multiplyRotMatrix(rotMat2, rotMat1, rotMat);

    runShiftRotKernel(d_odata, shiftVal, rotMat);
}

void CudaRot::Shift(CudaDeviceVariable& d_odata, float3 shiftVal)
{
	runShiftKernel(d_odata, shiftVal);
}

void CudaRot::RotCplx(CudaDeviceVariable& d_odata, float phi, float psi, float theta)
{
	float rotMat1[3][3];
	float rotMat2[3][3];
	float rotMat[3][3];
	computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
	computeRotMat(phi, psi, theta, rotMat2);
	multiplyRotMatrix(rotMat2, rotMat1, rotMat);

	runRotCplxKernel(d_odata, rotMat);
}

void CudaRot::computeRotMat(float phi, float psi, float theta, float rotMat[3][3])
{
	int i, j;
	float sinphi, sinpsi, sintheta;	/* sin of rotation angles */
	float cosphi, cospsi, costheta;	/* cos of rotation angles */


	float angles[] = { 0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330 };
	float angle_cos[16];
	float angle_sin[16];

	angle_cos[0]=1.0f;
	angle_cos[1]=sqrt(3.0f)/2.0f;
	angle_cos[2]=sqrt(2.0f)/2.0f;
	angle_cos[3]=0.5f;
	angle_cos[4]=0.0f;
	angle_cos[5]=-0.5f;
	angle_cos[6]=-sqrt(2.0f)/2.0f;
	angle_cos[7]=-sqrt(3.0f)/2.0f;
	angle_cos[8]=-1.0f;
	angle_cos[9]=-sqrt(3.0f)/2.0f;
	angle_cos[10]=-sqrt(2.0f)/2.0f;
	angle_cos[11]=-0.5f;
	angle_cos[12]=0.0f;
	angle_cos[13]=0.5f;
	angle_cos[14]=sqrt(2.0f)/2.0f;
	angle_cos[15]=sqrt(3.0f)/2.0f;
	angle_sin[0]=0.0f;
	angle_sin[1]=0.5f;
	angle_sin[2]=sqrt(2.0f)/2.0f;
	angle_sin[3]=sqrt(3.0f)/2.0f;
	angle_sin[4]=1.0f;
	angle_sin[5]=sqrt(3.0f)/2.0f;
	angle_sin[6]=sqrt(2.0f)/2.0f;
	angle_sin[7]=0.5f;
	angle_sin[8]=0.0f;
	angle_sin[9]=-0.5f;
	angle_sin[10]=-sqrt(2.0f)/2.0f;
	angle_sin[11]=-sqrt(3.0f)/2.0f;
	angle_sin[12]=-1.0f;
	angle_sin[13]=-sqrt(3.0f)/2.0f;
	angle_sin[14]=-sqrt(2.0f)/2.0f;
	angle_sin[15]=-0.5f;

	for (i=0, j=0 ; i<16; i++)
		if (angles[i] == phi )
		{
		   cosphi = angle_cos[i];
		   sinphi = angle_sin[i];
		   j = 1;
		}

	if (j < 1)
	{
	   phi = phi * (float)M_PI / 180.0f;
	   cosphi=cos(phi);
	   sinphi=sin(phi);
	}

	for (i=0, j=0 ; i<16; i++)
		if (angles[i] == psi )
		{
		   cospsi = angle_cos[i];
		   sinpsi = angle_sin[i];
		   j = 1;
		}

	if (j < 1)
	{
		psi = psi * (float)M_PI / 180.0f;
	   cospsi=cos(psi);
	   sinpsi=sin(psi);
	}

	for (i=0, j=0 ; i<16; i++)
		if (angles[i] == theta )
		{
		   costheta = angle_cos[i];
		   sintheta = angle_sin[i];
		   j = 1;
		}

	if (j < 1)
	{
		theta = theta * (float)M_PI / 180.0f;
	   costheta=cos(theta);
	   sintheta=sin(theta);
	}

	/* calculation of rotation matrix */

	rotMat[0][0] = cospsi*cosphi-costheta*sinpsi*sinphi;
	rotMat[1][0] = sinpsi*cosphi+costheta*cospsi*sinphi;
	rotMat[2][0] = sintheta*sinphi;
	rotMat[0][1] = -cospsi*sinphi-costheta*sinpsi*cosphi;
	rotMat[1][1] = -sinpsi*sinphi+costheta*cospsi*cosphi;
	rotMat[2][1] = sintheta*cosphi;
	rotMat[0][2] = sintheta*sinpsi;
	rotMat[1][2] = -sintheta*cospsi;
	rotMat[2][2] = costheta;
}

void CudaRot::multiplyRotMatrix(float m1[3][3], float m2[3][3], float out[3][3])
{
	out[0][0] = m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1] + m1[2][0] * m2[0][2];
    out[1][0] = m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1] + m1[2][0] * m2[1][2];
    out[2][0] = m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1] + m1[2][0] * m2[2][2];
    out[0][1] = m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1] + m1[2][1] * m2[0][2];
    out[1][1] = m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1] + m1[2][1] * m2[1][2];
    out[2][1] = m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1] + m1[2][1] * m2[2][2];
    out[0][2] = m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1] + m1[2][2] * m2[0][2];
    out[1][2] = m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1] + m1[2][2] * m2[1][2];
    out[2][2] = m1[0][2] * m2[2][0] + m1[1][2] * m2[2][1] + m1[2][2] * m2[2][2];

}

void CudaRot::runRotKernel(CudaDeviceVariable& d_odata, float rotMat[3][3])
{
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();

	float3 rotMat0 = make_float3(rotMat[0][0], rotMat[0][1], rotMat[0][2]);
	float3 rotMat1 = make_float3(rotMat[1][0], rotMat[1][1], rotMat[1][2]);
	float3 rotMat2 = make_float3(rotMat[2][0], rotMat[2][1], rotMat[2][2]);

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

void CudaRot::runShiftRotKernel(CudaDeviceVariable& d_odata, float3 shiftVal, float rotMat[3][3])
{
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    float3 rotMat0 = make_float3(rotMat[0][0], rotMat[0][1], rotMat[0][2]);
    float3 rotMat1 = make_float3(rotMat[1][0], rotMat[1][1], rotMat[1][2]);
    float3 rotMat2 = make_float3(rotMat[2][0], rotMat[2][1], rotMat[2][2]);

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

void CudaRot::runRotCplxKernel(CudaDeviceVariable& d_odata, float rotMat[3][3])
{
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

	float3 rotMat0 = make_float3(rotMat[0][0], rotMat[0][1], rotMat[0][2]);
	float3 rotMat1 = make_float3(rotMat[1][0], rotMat[1][1], rotMat[1][2]);
	float3 rotMat2 = make_float3(rotMat[2][0], rotMat[2][1], rotMat[2][2]);

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
