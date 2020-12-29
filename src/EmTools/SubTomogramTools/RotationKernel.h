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


#ifndef ROTATIONKERNEL_H
#define ROTATIONKERNEL_H

#include "../Basics/Default.h"

#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaKernel.h>
#include <cutil_math_.h>


class Rot3dKernel : public Cuda::CudaKernel
{
public:
	Rot3dKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	Rot3dKernel(CUmodule aModule);

	float operator()(int size, float3 rotMat0, float3 rotMat1, float3 rotMat2, Cuda::CudaTextureObject3D& inVol, Cuda::CudaDeviceVariable& outVol);
	void operator()(CUstream stream, int size, float3 rotMat0, float3 rotMat1, float3 rotMat2, Cuda::CudaTextureObject3D& inVol, Cuda::CudaDeviceVariable& outVol);
};

class ShiftRot3dKernel : public Cuda::CudaKernel
{
public:
	ShiftRot3dKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	ShiftRot3dKernel(CUmodule aModule);

	float operator()(int size, float3 shift, float3 rotMat0, float3 rotMat1, float3 rotMat2, Cuda::CudaTextureObject3D& inVol, Cuda::CudaDeviceVariable& outVol);
	void operator()(CUstream stream, int size, float3 shift, float3 rotMat0, float3 rotMat1, float3 rotMat2, Cuda::CudaTextureObject3D& inVol, Cuda::CudaDeviceVariable& outVol);
};

class ShiftKernel : public Cuda::CudaKernel
{
public:
	ShiftKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	ShiftKernel(CUmodule aModule);

	float operator()(int size, float3 shift, Cuda::CudaTextureObject3D& inVol, Cuda::CudaDeviceVariable& outVol);
	void operator()(CUstream stream, int size, float3 shift, Cuda::CudaTextureObject3D& inVol, Cuda::CudaDeviceVariable& outVol);
};

#endif