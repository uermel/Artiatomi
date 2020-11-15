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


#ifndef ROTATOR_H
#define ROTATOR_H

#include "../Basics/Default.h"
#include "RotationMatrix.h"
#include "RotationKernel.h"

#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaKernel.h>


class Rotator 
{
private:
	Cuda::CudaContext* _ctx;
	Cuda::CudaArray3D _array;
	Rot3dKernel* kernelRot3d;
	ShiftRot3dKernel* kernelShiftRot3d;
	ShiftKernel* kernelShift;

	int _size;

	float oldPhi;
	float oldPsi;
	float oldTheta;

public:
	Rotator(Cuda::CudaContext* aCtx, int aSize);

	~Rotator();

	void SetOldAngles(float aPhi, float aPsi, float aTheta);

	void Rotate(float phi, float psi, float theta, Cuda::CudaDeviceVariable& vol);

	void Shift(float3 shift, Cuda::CudaDeviceVariable& vol);

	void ShiftRotate(float3 shift, float phi, float psi, float theta, Cuda::CudaDeviceVariable& vol);

	void ShiftRotateTwoStep(float3 shift, float phi, float psi, float theta, Cuda::CudaDeviceVariable& vol);

};

#endif