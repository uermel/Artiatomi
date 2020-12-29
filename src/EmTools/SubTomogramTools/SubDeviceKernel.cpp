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

#include "SubDeviceKernel.h"
using namespace Cuda;

SubDeviceKernel::SubDeviceKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("subKernel", aModule, aGridDim, aBlockDim, 0)
{

}

SubDeviceKernel::SubDeviceKernel(CUmodule aModule)
	: CudaKernel("subKernel", aModule, make_dim3(1, 1, 1), make_dim3(512, 1, 1), 0)
{

}

float SubDeviceKernel::operator()(Cuda::CudaDeviceVariable& value, Cuda::CudaDeviceVariable& dataInOut, int size)
{
	CudaKernel::SetComputeSize(size, 1, 1);
	CUdeviceptr value_dptr = value.GetDevicePtr();
	CUdeviceptr dataInOut_dptr = dataInOut.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &value_dptr;
	arglist[1] = &dataInOut_dptr;
	arglist[2] = &size;

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

void SubDeviceKernel::operator()(CUstream stream, Cuda::CudaDeviceVariable& value, Cuda::CudaDeviceVariable& dataInOut, int size)
{
	CudaKernel::SetComputeSize(size, 1, 1);
	CUdeviceptr value_dptr = value.GetDevicePtr();
	CUdeviceptr dataInOut_dptr = dataInOut.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &value_dptr;
	arglist[1] = &dataInOut_dptr;
	arglist[2] = &size;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}

SubDivDeviceKernel::SubDivDeviceKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("subdivKernel", aModule, aGridDim, aBlockDim, 0)
{

}

SubDivDeviceKernel::SubDivDeviceKernel(CUmodule aModule)
	: CudaKernel("subdivKernel", aModule, make_dim3(1, 1, 1), make_dim3(512, 1, 1), 0)
{

}

float SubDivDeviceKernel::operator()(Cuda::CudaDeviceVariable& value, Cuda::CudaDeviceVariable& dataInOut, int size, float div)
{
	CudaKernel::SetComputeSize(size, 1, 1);
	CUdeviceptr value_dptr = value.GetDevicePtr();
	CUdeviceptr dataInOut_dptr = dataInOut.GetDevicePtr();

	void* arglist[4];

	arglist[0] = &value_dptr;
	arglist[1] = &dataInOut_dptr;
	arglist[2] = &size;
	arglist[3] = &div;

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

void SubDivDeviceKernel::operator()(CUstream stream, Cuda::CudaDeviceVariable& value, Cuda::CudaDeviceVariable& dataInOut, int size, float div)
{
	CudaKernel::SetComputeSize(size, 1, 1);
	CUdeviceptr value_dptr = value.GetDevicePtr();
	CUdeviceptr dataInOut_dptr = dataInOut.GetDevicePtr();

	void* arglist[4];

	arglist[0] = &value_dptr;
	arglist[1] = &dataInOut_dptr;
	arglist[2] = &size;
	arglist[3] = &div;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}