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


#ifndef CUDAROT_H
#define CUDAROT_H

#include "basics/default.h"
#include <cuda.h>
#include "cuda/CudaVariables.h"
#include "cuda/CudaKernel.h"
#include "cuda/CudaContext.h"
#include "cuda/CudaTextures.h"



using namespace Cuda;

class CudaRot
{
private:
	CudaKernel* rotVol;
	CudaKernel* shift;
	CudaKernel* rotVolCplx;

	CudaContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	float oldphi, oldpsi, oldtheta;

	CudaArray3D shiftTex;
	CudaArray3D dataTex;
	CudaArray3D dataTexCplx;

	CUstream stream;

	void runShiftKernel(CudaDeviceVariable& d_odata, float3 shiftVal);
	void runRotKernel(CudaDeviceVariable& d_odata, float rotMat[3][3]);
	void runRotCplxKernel(CudaDeviceVariable& d_odata, float rotMat[3][3]);

	void computeRotMat(float phi, float psi, float theta, float rotMat[3][3]);
	void multiplyRotMatrix(float m1[3][3], float m2[3][3], float out[3][3]);
public:

	CudaRot(int aVolSize, CUstream aStream, CudaContext* context, bool linearInterpolation);

	void SetTextureShift(CudaDeviceVariable& d_idata);
	void SetTexture(CudaDeviceVariable& d_idata);
	void SetTextureCplx(CudaDeviceVariable& d_idata);

	void Shift(CudaDeviceVariable& d_odata, float3 shiftVal);
	void Rot(CudaDeviceVariable& d_odata, float phi, float psi, float theta);
	void RotCplx(CudaDeviceVariable& d_odata, float phi, float psi, float theta);

	void SetOldAngles(float aPhi, float aPsi, float aTheta);
};

#endif //CUDAROT_H
