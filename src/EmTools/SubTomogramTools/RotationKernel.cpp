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

#include "RotationKernel.h"
using namespace Cuda;

Rot3dKernel::Rot3dKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("rot3d", aModule, aGridDim, aBlockDim, 0)
{

}

Rot3dKernel::Rot3dKernel(CUmodule aModule)
	: CudaKernel("rot3d", aModule, make_dim3(1, 1, 1), make_dim3(32, 16, 1), 0)
{

}

float Rot3dKernel::operator()(int size, float3 rotMat0, float3 rotMat1, float3 rotMat2, Cuda::CudaTextureObject3D& inVol, Cuda::CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(size, size, size);
	CUtexObject tex = inVol.GetTexObject();
	CUdeviceptr vol_dptr = outVol.GetDevicePtr();

	void* arglist[6];

	arglist[0] = &size;
	arglist[1] = &rotMat0;
	arglist[2] = &rotMat1;
	arglist[3] = &rotMat2;
	arglist[4] = &tex;
	arglist[5] = &vol_dptr;

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

	return ms;
}

void Rot3dKernel::operator()(CUstream stream, int size, float3 rotMat0, float3 rotMat1, float3 rotMat2, Cuda::CudaTextureObject3D& inVol, Cuda::CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(size, size, size);
	CUtexObject tex = inVol.GetTexObject();
	CUdeviceptr vol_dptr = outVol.GetDevicePtr();

	void* arglist[6];

	arglist[0] = &size;
	arglist[1] = &rotMat0;
	arglist[2] = &rotMat1;
	arglist[3] = &rotMat2;
	arglist[4] = &tex;
	arglist[5] = &vol_dptr;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));

}

ShiftRot3dKernel::ShiftRot3dKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("shiftRot3d", aModule, aGridDim, aBlockDim, 0)
{

}

ShiftRot3dKernel::ShiftRot3dKernel(CUmodule aModule)
	: CudaKernel("shiftRot3d", aModule, make_dim3(1, 1, 1), make_dim3(32, 16, 1), 0)
{

}

float ShiftRot3dKernel::operator()(int size, float3 shift, float3 rotMat0, float3 rotMat1, float3 rotMat2, Cuda::CudaTextureObject3D& inVol, Cuda::CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(size, size, size);
	CUtexObject tex = inVol.GetTexObject();
	CUdeviceptr vol_dptr = outVol.GetDevicePtr();

	void* arglist[7];

	arglist[0] = &size;
	arglist[1] = &shift;
	arglist[2] = &rotMat0;
	arglist[3] = &rotMat1;
	arglist[4] = &rotMat2;
	arglist[5] = &tex;
	arglist[6] = &vol_dptr;

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

	return ms;
}

void ShiftRot3dKernel::operator()(CUstream stream, int size, float3 shift, float3 rotMat0, float3 rotMat1, float3 rotMat2, Cuda::CudaTextureObject3D& inVol, Cuda::CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(size, size, size);
	CUtexObject tex = inVol.GetTexObject();
	CUdeviceptr vol_dptr = outVol.GetDevicePtr();

	void* arglist[7];

	arglist[0] = &size;
	arglist[1] = &shift;
	arglist[2] = &rotMat0;
	arglist[3] = &rotMat1;
	arglist[4] = &rotMat2;
	arglist[5] = &tex;
	arglist[6] = &vol_dptr;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}



ShiftKernel::ShiftKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("shift", aModule, aGridDim, aBlockDim, 0)
{

}

ShiftKernel::ShiftKernel(CUmodule aModule)
	: CudaKernel("shift", aModule, make_dim3(1, 1, 1), make_dim3(32, 16, 1), 0)
{

}

float ShiftKernel::operator()(int size, float3 shift, Cuda::CudaTextureObject3D& inVol, Cuda::CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(size, size, size);
	CUtexObject tex = inVol.GetTexObject();
	CUdeviceptr vol_dptr = outVol.GetDevicePtr();

	void* arglist[4];

	arglist[0] = &size;
	arglist[1] = &shift;
	arglist[2] = &tex;
	arglist[3] = &vol_dptr;

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

	return ms;
}

void ShiftKernel::operator()(CUstream stream, int size, float3 shift, Cuda::CudaTextureObject3D& inVol, Cuda::CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(size, size, size);
	CUtexObject tex = inVol.GetTexObject();
	CUdeviceptr vol_dptr = outVol.GetDevicePtr();

	void* arglist[4];

	arglist[0] = &size;
	arglist[1] = &shift;
	arglist[2] = &tex;
	arglist[3] = &vol_dptr;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}
