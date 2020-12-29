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


#include "Rotator.h"
#include "Kernels/RotationKernel.cu.h"

using namespace Cuda;

Rotator::Rotator(Cuda::CudaContext* aCtx, CUstream stream, int aSize) : 
	_ctx(aCtx),
	_array(CUarray_format::CU_AD_FORMAT_FLOAT, aSize, aSize, aSize, 1),
	_size(aSize),
	_texObjShift(NULL),
	_texObjRot(NULL),
	oldPhi(0),
	oldPsi(0),
	oldTheta(0),
	_stream(stream)
{
	CUmodule cuMod = aCtx->LoadModulePTX(SubTomogramRotationKernel, 0, false, false);

	kernelRot3d = new Rot3dKernel(cuMod);
	kernelShiftRot3d = new ShiftRot3dKernel(cuMod);
	kernelShift = new ShiftKernel(cuMod);

	_texObjShift = new Cuda::CudaTextureObject3D(CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_WRAP,
		CU_TR_ADDRESS_MODE_WRAP, CU_TR_FILTER_MODE_LINEAR, CU_TRSF_NORMALIZED_COORDINATES, &_array);

	_texObjRot = new Cuda::CudaTextureObject3D(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
		CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &_array);
}

Rotator::~Rotator()
{
	delete _texObjRot;
	delete _texObjShift;
	delete kernelRot3d;
	delete kernelShiftRot3d;
	delete kernelShift;
}

void Rotator::SetOldAngles(float aPhi, float aPsi, float aTheta)
{
	oldPhi = aPhi;
	oldPsi = aPsi;
	oldTheta = aTheta;
}

void Rotator::Rotate(float phi, float psi, float theta, Cuda::CudaDeviceVariable& vol)
{
	_array.CopyFromDeviceToArrayAsync(_stream, vol);
	//Cuda::CudaTextureObject3D texObj(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
	//	CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &_array);

	RotationMatrix rotMatOld(oldPhi, oldPsi, oldTheta);
	RotationMatrix rotMatNew(phi, psi, theta);
	RotationMatrix rotMat = rotMatNew * rotMatOld;

	float3 rotMat0 = make_float3(rotMat(0, 0), rotMat(0, 1), rotMat(0, 2));
	float3 rotMat1 = make_float3(rotMat(1, 0), rotMat(1, 1), rotMat(1, 2));
	float3 rotMat2 = make_float3(rotMat(2, 0), rotMat(2, 1), rotMat(2, 2));

	(*kernelRot3d)(_stream, _size, rotMat0, rotMat1, rotMat2, *_texObjRot, vol);
}

void Rotator::Shift(float3 shift, Cuda::CudaDeviceVariable& vol)
{
	_array.CopyFromDeviceToArrayAsync(_stream, vol);
	//Cuda::CudaTextureObject3D texObj(CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_WRAP,
	//	CU_TR_ADDRESS_MODE_WRAP, CU_TR_FILTER_MODE_LINEAR, CU_TRSF_NORMALIZED_COORDINATES, &_array);

	(*kernelShift)(_stream, _size, shift, *_texObjShift, vol);
}

void Rotator::ShiftRotate(float3 shift, float phi, float psi, float theta, Cuda::CudaDeviceVariable& vol)
{
	_array.CopyFromDeviceToArrayAsync(_stream, vol);
	//Cuda::CudaTextureObject3D texObj(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
	//	CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &_array);

	RotationMatrix rotMatOld(oldPhi, oldPsi, oldTheta);
	RotationMatrix rotMatNew(phi, psi, theta);
	RotationMatrix rotMat = rotMatNew * rotMatOld;

	float3 rotMat0 = make_float3(rotMat(0, 0), rotMat(0, 1), rotMat(0, 2));
	float3 rotMat1 = make_float3(rotMat(1, 0), rotMat(1, 1), rotMat(1, 2));
	float3 rotMat2 = make_float3(rotMat(2, 0), rotMat(2, 1), rotMat(2, 2));

	(*kernelShiftRot3d)(_stream, _size, shift, rotMat0, rotMat1, rotMat2, *_texObjRot, vol);
}

void Rotator::ShiftRotateTwoStep(float3 shift, float phi, float psi, float theta, Cuda::CudaDeviceVariable& vol)
{
	_array.CopyFromDeviceToArrayAsync(_stream, vol);
	//Cuda::CudaTextureObject3D texObjShift(CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_WRAP,
	//	CU_TR_ADDRESS_MODE_WRAP, CU_TR_FILTER_MODE_LINEAR, CU_TRSF_NORMALIZED_COORDINATES, &_array);

	float3 shiftInteger = make_float3(round(shift.x), round(shift.y), round(shift.z));
	shift.x = shift.x - shiftInteger.x;
	shift.y = shift.y - shiftInteger.y;
	shift.z = shift.z - shiftInteger.z;

	(*kernelShift)(_stream, _size, shiftInteger, *_texObjShift, vol);

	_array.CopyFromDeviceToArrayAsync(_stream, vol);

	//Cuda::CudaTextureObject3D texObjRot(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
	//	CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &_array);

	RotationMatrix rotMatOld(oldPhi, oldPsi, oldTheta);
	RotationMatrix rotMatNew(phi, psi, theta);
	RotationMatrix rotMat = rotMatNew * rotMatOld;

	float3 rotMat0 = make_float3(rotMat(0, 0), rotMat(0, 1), rotMat(0, 2));
	float3 rotMat1 = make_float3(rotMat(1, 0), rotMat(1, 1), rotMat(1, 2));
	float3 rotMat2 = make_float3(rotMat(2, 0), rotMat(2, 1), rotMat(2, 2));

	(*kernelShiftRot3d)(_stream, _size, shift, rotMat0, rotMat1, rotMat2, *_texObjRot, vol);
}

//
//void Rotator::Rotate(CUstream stream, float phi, float psi, float theta, Cuda::CudaDeviceVariable& vol)
//{
//	cudaSafeCall(cuStreamSynchronize(stream));
//	_array.CopyFromDeviceToArrayAsync(stream, vol);
//	cudaSafeCall(cuStreamSynchronize(stream));
//
//	RotationMatrix rotMatOld(oldPhi, oldPsi, oldTheta);
//	RotationMatrix rotMatNew(phi, psi, theta);
//	RotationMatrix rotMat = rotMatNew * rotMatOld;
//
//	float3 rotMat0 = make_float3(rotMat(0, 0), rotMat(0, 1), rotMat(0, 2));
//	float3 rotMat1 = make_float3(rotMat(1, 0), rotMat(1, 1), rotMat(1, 2));
//	float3 rotMat2 = make_float3(rotMat(2, 0), rotMat(2, 1), rotMat(2, 2));
//
//	(*kernelRot3d)(stream, _size, rotMat0, rotMat1, rotMat2, *_texObjRot, vol);
//}
//
//void Rotator::Shift(CUstream stream, float3 shift, Cuda::CudaDeviceVariable& vol)
//{
//	_array.CopyFromDeviceToArrayAsync(stream, vol);
//
//	(*kernelShift)(stream, _size, shift, *_texObjShift, vol);
//}
//
//void Rotator::ShiftRotate(CUstream stream, float3 shift, float phi, float psi, float theta, Cuda::CudaDeviceVariable& vol)
//{
//	_array.CopyFromDeviceToArrayAsync(stream, vol);
//
//	RotationMatrix rotMatOld(oldPhi, oldPsi, oldTheta);
//	RotationMatrix rotMatNew(phi, psi, theta);
//	RotationMatrix rotMat = rotMatNew * rotMatOld;
//
//	float3 rotMat0 = make_float3(rotMat(0, 0), rotMat(0, 1), rotMat(0, 2));
//	float3 rotMat1 = make_float3(rotMat(1, 0), rotMat(1, 1), rotMat(1, 2));
//	float3 rotMat2 = make_float3(rotMat(2, 0), rotMat(2, 1), rotMat(2, 2));
//
//	(*kernelShiftRot3d)(stream, _size, shift, rotMat0, rotMat1, rotMat2, *_texObjRot, vol);
//}
//
//void Rotator::ShiftRotateTwoStep(CUstream stream, float3 shift, float phi, float psi, float theta, Cuda::CudaDeviceVariable& vol)
//{
//	//cudaSafeCall(cuStreamSynchronize(stream));
//	_array.CopyFromDeviceToArrayAsync(stream, vol); 
//	//cudaSafeCall(cuStreamSynchronize(stream));
//
//	float3 shiftInteger = make_float3(round(shift.x), round(shift.y), round(shift.z));
//	shift.x = shift.x - shiftInteger.x;
//	shift.y = shift.y - shiftInteger.y;
//	shift.z = shift.z - shiftInteger.z;
//
//	(*kernelShift)(stream, _size, shiftInteger, *_texObjShift, vol);
//
//	//cudaSafeCall(cuStreamSynchronize(stream));
//	_array.CopyFromDeviceToArrayAsync(stream, vol);
//	//cudaSafeCall(cuStreamSynchronize(stream));
//
//	RotationMatrix rotMatOld(oldPhi, oldPsi, oldTheta);
//	RotationMatrix rotMatNew(phi, psi, theta);
//	RotationMatrix rotMat = rotMatNew * rotMatOld;
//
//	float3 rotMat0 = make_float3(rotMat(0, 0), rotMat(0, 1), rotMat(0, 2));
//	float3 rotMat1 = make_float3(rotMat(1, 0), rotMat(1, 1), rotMat(1, 2));
//	float3 rotMat2 = make_float3(rotMat(2, 0), rotMat(2, 1), rotMat(2, 2));
//
//	(*kernelShiftRot3d)(stream, _size, shift, rotMat0, rotMat1, rotMat2, *_texObjRot, vol);
//}