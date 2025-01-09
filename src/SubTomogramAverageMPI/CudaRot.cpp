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

uint PowTwoDivider(uint n)
{
    if (n == 0) return 0;
    uint divider = 1;
    while ((n & divider) == 0) divider <<= 1;
    return divider;
}

CudaRot::CudaRot(int aVolSize, CUstream aStream, CudaContext* context, cudarot_interp_mode interpolation)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1),
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize),
      md_dataArray(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 1, 0),
      md_dataTex(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &md_dataArray),
      md_tempData(),
	  oldphi(0), oldpsi(0), oldtheta(0),
      mInterpMode(interpolation)
{
	CUmodule cuMod = ctx->LoadModulePTX(SubTomogramAverageBasicKernel, 0, false, false);
    CUmodule cuModSplines = ctx->LoadModulePTX(SplinesKernel, 0, false, false);

	shift_linear = new CudaKernel("shift_linear", cuMod);
	rotVol_linear = new CudaKernel("rot3d_linear", cuMod);
	shiftRotVol_linear = new CudaKernel("shiftRot3d_linear", cuMod);
    shift_spline = new CudaKernel("shift_spline", cuMod);
    rotVol_spline = new CudaKernel("rot3d_spline", cuMod);
    shiftRotVol_spline = new CudaKernel("shiftRot3d_spline", cuMod);
    prefilter3DX = new CudaKernel("SamplesToCoefficients3DX", cuModSplines);
    prefilter3DY = new CudaKernel("SamplesToCoefficients3DY", cuModSplines);
    prefilter3DZ = new CudaKernel("SamplesToCoefficients3DZ", cuModSplines);

    if (mInterpMode == CR_INTERP_CUBIC)
    {
        md_tempData.Alloc(aVolSize*aVolSize*aVolSize*sizeof(float));
    }
}


void CudaRot::SetTexture(CudaDeviceVariable& d_idata)
{
    if (mInterpMode == CR_INTERP_LINEAR) {
        md_dataArray.CopyFromDeviceToArray(d_idata);
    }
    else if (mInterpMode == CR_INTERP_CUBIC)
    {
        md_tempData.CopyDeviceToDevice(d_idata);
        runPrefilterXKernel(md_tempData);
        runPrefilterYKernel(md_tempData);
        runPrefilterZKernel(md_tempData);
        md_dataArray.CopyFromDeviceToArray(md_tempData);
    }
}

void CudaRot::SetTextureShift(CudaDeviceVariable& d_idata)
{
	//shiftTex.CopyFromDeviceToArray(d_idata);
    SetTexture(d_idata);
}

//void CudaRot::SetTextureCplx(CudaDeviceVariable& d_idata)
//{
//	dataTexCplx.CopyFromDeviceToArray(d_idata);
//}

void CudaRot::Rot(CudaDeviceVariable& d_odata, float phi, float psi, float theta, bool print)
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

//void CudaRot::RotCplx(CudaDeviceVariable& d_odata, float phi, float psi, float theta)
//{
//	float rotMat1[3][3];
//	float rotMat2[3][3];
//	float rotMat[3][3];
//	computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
//	computeRotMat(phi, psi, theta, rotMat2);
//	multiplyRotMatrix(rotMat2, rotMat1, rotMat);
//
//	runRotCplxKernel(d_odata, rotMat);
//}

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
    CUtexObject in_tex = md_dataTex.GetTexObject();
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

//    printf("\nMat line 1: %f %f %f\n", rotMat0.x, rotMat0.y, rotMat0.z);
//    printf("Mat line 2: %f %f %f\n", rotMat1.x, rotMat1.y, rotMat1.z);
//    printf("Mat line 3: %f %f %f\n\n", rotMat2.x, rotMat2.y, rotMat2.z);

    void** arglist = (void**)new void*[6];

    arglist[0] = &volSize;
    arglist[1] = &in_tex;
    arglist[2] = &rotMat0;
    arglist[3] = &rotMat1;
    arglist[4] = &rotMat2;
    arglist[5] = &out_dptr;

    if (mInterpMode == CR_INTERP_LINEAR) {
        cudaSafeCall(cuLaunchKernel(rotVol_linear->GetCUfunction(),
                                    gridSize.x,
                                    gridSize.y,
                                    gridSize.z,
                                    blockSize.x,
                                    blockSize.y,
                                    blockSize.z, 0, stream, arglist, NULL));
    }
    else if (mInterpMode == CR_INTERP_CUBIC){
        cudaSafeCall(cuLaunchKernel(rotVol_spline->GetCUfunction(),
                                    gridSize.x,
                                    gridSize.y,
                                    gridSize.z,
                                    blockSize.x,
                                    blockSize.y,
                                    blockSize.z, 0, stream, arglist, NULL));
    }

    delete[] arglist;
}

void CudaRot::runShiftRotKernel(CudaDeviceVariable& d_odata, float3 shiftVal, float rotMat[9])
{
    CUtexObject in_tex = md_dataTex.GetTexObject();
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

    void** arglist = (void**)new void*[7];

    arglist[0] = &volSize;
    arglist[1] = &in_tex;
    arglist[2] = &shiftVal;
    arglist[3] = &rotMat0;
    arglist[4] = &rotMat1;
    arglist[5] = &rotMat2;
    arglist[6] = &out_dptr;

    if (mInterpMode == CR_INTERP_LINEAR) {
        cudaSafeCall(cuLaunchKernel(shiftRotVol_linear->GetCUfunction(),
                                    gridSize.x,
                                    gridSize.y,
                                    gridSize.z,
                                    blockSize.x,
                                    blockSize.y,
                                    blockSize.z, 0, stream, arglist, NULL));
    }
    else if (mInterpMode == CR_INTERP_CUBIC){
        cudaSafeCall(cuLaunchKernel(shiftRotVol_spline->GetCUfunction(),
                                    gridSize.x,
                                    gridSize.y,
                                    gridSize.z,
                                    blockSize.x,
                                    blockSize.y,
                                    blockSize.z, 0, stream, arglist, NULL));
    }

    delete[] arglist;
}

void CudaRot::runShiftKernel(CudaDeviceVariable& d_odata, float3 shiftVal)
{
    CUtexObject in_tex = md_dataTex.GetTexObject();
    CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &volSize;
    arglist[1] = &in_tex;
    arglist[2] = &out_dptr;
    arglist[3] = &shiftVal;

    if (mInterpMode == CR_INTERP_LINEAR) {
        cudaSafeCall(cuLaunchKernel(shift_linear->GetCUfunction(),
                                    gridSize.x,
                                    gridSize.y,
                                    gridSize.z,
                                    blockSize.x,
                                    blockSize.y,
                                    blockSize.z, 0, stream, arglist, NULL));
    }
    else if (mInterpMode == CR_INTERP_CUBIC){
        cudaSafeCall(cuLaunchKernel(shift_spline->GetCUfunction(),
                                    gridSize.x,
                                    gridSize.y,
                                    gridSize.z,
                                    blockSize.x,
                                    blockSize.y,
                                    blockSize.z, 0, stream, arglist, NULL));
    }

    delete[] arglist;
}

void CudaRot::runPrefilterXKernel(CudaDeviceVariable &d_iodata)
{
    CUdeviceptr inout_dptr = d_iodata.GetDevicePtr();

    uint pitch = volSize * sizeof(float);
    uint width = volSize;
    uint height = volSize;
    uint depth = volSize;

    void** arglist = (void**)new void*[5];

    arglist[0] = &inout_dptr;
    arglist[1] = &pitch;
    arglist[2] = &width;
    arglist[3] = &height;
    arglist[4] = &depth;

    // Block/Grid
    uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
    uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
    dim3 blockDim(dimX, dimY);
    dim3 gridDim(height / blockDim.x, depth / blockDim.y);

    cudaSafeCall(cuLaunchKernel(prefilter3DX->GetCUfunction(),
                                gridDim.x,
                                gridDim.y,
                                gridDim.z,
                                blockDim.x,
                                blockDim.y,
                                blockDim.z,
                                0, stream, arglist,NULL));

    delete[] arglist;
}

void CudaRot::runPrefilterYKernel(CudaDeviceVariable &d_iodata)
{
    CUdeviceptr inout_dptr = d_iodata.GetDevicePtr();

    uint pitch = volSize * sizeof(float);
    uint width = volSize;
    uint height = volSize;
    uint depth = volSize;

    void** arglist = (void**)new void*[5];

    arglist[0] = &inout_dptr;
    arglist[1] = &pitch;
    arglist[2] = &width;
    arglist[3] = &height;
    arglist[4] = &depth;

    // Block/Grid
    uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
    uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
    dim3 blockDim(dimX, dimY);
    dim3 gridDim(width / blockDim.x, depth / blockDim.y);

    cudaSafeCall(cuLaunchKernel(prefilter3DY->GetCUfunction(),
                                gridDim.x,
                                gridDim.y,
                                gridDim.z,
                                blockDim.x,
                                blockDim.y,
                                blockDim.z,
                                0, stream, arglist,NULL));

    delete[] arglist;
}

void CudaRot::runPrefilterZKernel(CudaDeviceVariable &d_iodata)
{
    CUdeviceptr inout_dptr = d_iodata.GetDevicePtr();

    uint pitch = volSize * sizeof(float);
    uint width = volSize;
    uint height = volSize;
    uint depth = volSize;

    void** arglist = (void**)new void*[5];

    arglist[0] = &inout_dptr;
    arglist[1] = &pitch;
    arglist[2] = &width;
    arglist[3] = &height;
    arglist[4] = &depth;

    // Block/Grid
    uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
    uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
    dim3 blockDim(dimX, dimY);
    dim3 gridDim(width / blockDim.x, height / blockDim.y);

    cudaSafeCall(cuLaunchKernel(prefilter3DZ->GetCUfunction(),
                                gridDim.x,
                                gridDim.y,
                                gridDim.z,
                                blockDim.x,
                                blockDim.y,
                                blockDim.z,
                                0, stream, arglist,NULL));

    delete[] arglist;
}
//void CudaRot::runRotCplxKernel(CudaDeviceVariable& d_odata, float rotMat[3][3])
//{
//	CUdeviceptr out_dptr = d_odata.GetDevicePtr();
//
//	float3 rotMat0 = make_float3(rotMat[0][0], rotMat[0][1], rotMat[0][2]);
//	float3 rotMat1 = make_float3(rotMat[1][0], rotMat[1][1], rotMat[1][2]);
//	float3 rotMat2 = make_float3(rotMat[2][0], rotMat[2][1], rotMat[2][2]);
//
//    void** arglist = (void**)new void*[5];
//
//    arglist[0] = &volSize;
//    arglist[1] = &rotMat0;
//    arglist[2] = &rotMat1;
//    arglist[3] = &rotMat2;
//    arglist[4] = &out_dptr;
//
//    cudaSafeCall(cuLaunchKernel(rotVolCplx->GetCUfunction(), gridSize.x, gridSize.y,
//		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));
//
//    delete[] arglist;
//}

void CudaRot::SetOldAngles(float aPhi, float aPsi, float aTheta)
{
	oldphi = aPhi;
	oldpsi = aPsi;
	oldtheta = aTheta;
}
