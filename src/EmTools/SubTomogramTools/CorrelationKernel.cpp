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

#include "CorrelationKernel.h"
using namespace Cuda;

FftshiftRealKernel::FftshiftRealKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("fftshiftReal", aModule, aGridDim, aBlockDim, 0)
{

}

FftshiftRealKernel::FftshiftRealKernel(CUmodule aModule)
	: CudaKernel("fftshiftReal", aModule, make_dim3(1, 1, 1), make_dim3(32, 16, 1), 0)
{

}

float FftshiftRealKernel::operator()(int size, CudaDeviceVariable& inVol, CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(size, size, size);
	CUdeviceptr inVol_dptr = inVol.GetDevicePtr();
	CUdeviceptr outVol_dptr = outVol.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &size;
	arglist[1] = &inVol_dptr;
	arglist[2] = &outVol_dptr;

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

void FftshiftRealKernel::operator()(CUstream stream,int size, CudaDeviceVariable& inVol, CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(size, size, size);
	CUdeviceptr inVol_dptr = inVol.GetDevicePtr();
	CUdeviceptr outVol_dptr = outVol.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &size;
	arglist[1] = &inVol_dptr;
	arglist[2] = &outVol_dptr;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}

EnergynormKernel::EnergynormKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("energynorm", aModule, aGridDim, aBlockDim, 0)
{

}

EnergynormKernel::EnergynormKernel(CUmodule aModule)
	: CudaKernel("energynorm", aModule, make_dim3(1, 1, 1), make_dim3(32, 16, 1), 0)
{

}

float EnergynormKernel::operator()(int size, CudaDeviceVariable& particle, CudaDeviceVariable& partSqr, CudaDeviceVariable& cccMap, CudaDeviceVariable& energyRef, CudaDeviceVariable& nVox)
{
	CudaKernel::SetComputeSize(size, size, size);
	CUdeviceptr particle_dptr = particle.GetDevicePtr();
	CUdeviceptr partSqr_dptr = partSqr.GetDevicePtr();
	CUdeviceptr cccMap_dptr = cccMap.GetDevicePtr();
	CUdeviceptr energyRef_dptr = energyRef.GetDevicePtr();
	CUdeviceptr nVox_dptr = nVox.GetDevicePtr();

	void* arglist[6];

	arglist[0] = &size;
	arglist[1] = &particle_dptr;
	arglist[2] = &partSqr_dptr;
	arglist[3] = &cccMap_dptr;
	arglist[4] = &energyRef_dptr;
	arglist[5] = &nVox_dptr;

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

void EnergynormKernel::operator()(CUstream stream, int size, CudaDeviceVariable& particle, CudaDeviceVariable& partSqr, CudaDeviceVariable& cccMap, CudaDeviceVariable& energyRef, CudaDeviceVariable& nVox)
{
	CudaKernel::SetComputeSize(size, size, size);
	CUdeviceptr particle_dptr = particle.GetDevicePtr();
	CUdeviceptr partSqr_dptr = partSqr.GetDevicePtr();
	CUdeviceptr cccMap_dptr = cccMap.GetDevicePtr();
	CUdeviceptr energyRef_dptr = energyRef.GetDevicePtr();
	CUdeviceptr nVox_dptr = nVox.GetDevicePtr();

	void* arglist[6];

	arglist[0] = &size;
	arglist[1] = &particle_dptr;
	arglist[2] = &partSqr_dptr;
	arglist[3] = &cccMap_dptr;
	arglist[4] = &energyRef_dptr;
	arglist[5] = &nVox_dptr;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));

	cudaSafeCall(cuCtxSynchronize());
}


BinarizeKernel::BinarizeKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("binarize", aModule, aGridDim, aBlockDim, 0)
{

}

BinarizeKernel::BinarizeKernel(CUmodule aModule)
	: CudaKernel("binarize", aModule, make_dim3(1, 1, 1), make_dim3(32 * 16, 1, 1), 0)
{

}

float BinarizeKernel::operator()(int length, CudaDeviceVariable& inVol, CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(length);
	CUdeviceptr inVol_dptr = inVol.GetDevicePtr();
	CUdeviceptr outVol_dptr = outVol.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &length;
	arglist[1] = &inVol_dptr;
	arglist[2] = &outVol_dptr;

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

void BinarizeKernel::operator()(CUstream stream, int length, CudaDeviceVariable& inVol, CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(length);
	CUdeviceptr inVol_dptr = inVol.GetDevicePtr();
	CUdeviceptr outVol_dptr = outVol.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &length;
	arglist[1] = &inVol_dptr;
	arglist[2] = &outVol_dptr;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}


ConvKernel::ConvKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("conv", aModule, aGridDim, aBlockDim, 0)
{

}

ConvKernel::ConvKernel(CUmodule aModule)
	: CudaKernel("conv", aModule, make_dim3(1, 1, 1), make_dim3(32 * 16, 1, 1), 0)
{

}

float ConvKernel::operator()(int length, CudaDeviceVariable& inVol, CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(length);
	CUdeviceptr inVol_dptr = inVol.GetDevicePtr();
	CUdeviceptr outVol_dptr = outVol.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &length;
	arglist[1] = &inVol_dptr;
	arglist[2] = &outVol_dptr;

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

void ConvKernel::operator()(CUstream stream, int length, CudaDeviceVariable& inVol, CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(length);
	CUdeviceptr inVol_dptr = inVol.GetDevicePtr();
	CUdeviceptr outVol_dptr = outVol.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &length;
	arglist[1] = &inVol_dptr;
	arglist[2] = &outVol_dptr;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}


CorrelKernel::CorrelKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("correl", aModule, aGridDim, aBlockDim, 0)
{

}

CorrelKernel::CorrelKernel(CUmodule aModule)
	: CudaKernel("correl", aModule, make_dim3(1, 1, 1), make_dim3(32 * 16, 1, 1), 0)
{

}

float CorrelKernel::operator()(int length, CudaDeviceVariable& inVol, CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(length);
	CUdeviceptr inVol_dptr = inVol.GetDevicePtr();
	CUdeviceptr outVol_dptr = outVol.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &length;
	arglist[1] = &inVol_dptr;
	arglist[2] = &outVol_dptr;

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

void CorrelKernel::operator()(CUstream stream, int length, CudaDeviceVariable& inVol, CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(length);
	CUdeviceptr inVol_dptr = inVol.GetDevicePtr();
	CUdeviceptr outVol_dptr = outVol.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &length;
	arglist[1] = &inVol_dptr;
	arglist[2] = &outVol_dptr;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}


PhaseCorrelKernel::PhaseCorrelKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("phaseCorrel", aModule, aGridDim, aBlockDim, 0)
{

}

PhaseCorrelKernel::PhaseCorrelKernel(CUmodule aModule)
	: CudaKernel("phaseCorrel", aModule, make_dim3(1, 1, 1), make_dim3(32*16, 1, 1), 0)
{

}

float PhaseCorrelKernel::operator()(int length, CudaDeviceVariable& inVol, CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(length);
	CUdeviceptr inVol_dptr = inVol.GetDevicePtr();
	CUdeviceptr outVol_dptr = outVol.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &length;
	arglist[1] = &inVol_dptr;
	arglist[2] = &outVol_dptr;

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

void PhaseCorrelKernel::operator()(CUstream stream, int length, CudaDeviceVariable& inVol, CudaDeviceVariable& outVol)
{
	CudaKernel::SetComputeSize(length);
	CUdeviceptr inVol_dptr = inVol.GetDevicePtr();
	CUdeviceptr outVol_dptr = outVol.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &length;
	arglist[1] = &inVol_dptr;
	arglist[2] = &outVol_dptr;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}



BandpassFFTShiftKernel::BandpassFFTShiftKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("bandpassFFTShift", aModule, aGridDim, aBlockDim, 0)
{

}

BandpassFFTShiftKernel::BandpassFFTShiftKernel(CUmodule aModule)
	: CudaKernel("bandpassFFTShift", aModule, make_dim3(1, 1, 1), make_dim3(32, 16, 1), 0)
{

}

float BandpassFFTShiftKernel::operator()(int size, CudaDeviceVariable& vol, float rDown, float rUp, float smooth)
{
	CudaKernel::SetComputeSize((size / 2 + 1), size, size);
	CUdeviceptr vol_dptr = vol.GetDevicePtr();

	void* arglist[5];

	arglist[0] = &size;
	arglist[1] = &vol_dptr;
	arglist[2] = &rDown;
	arglist[3] = &rUp;
	arglist[4] = &smooth;

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

void BandpassFFTShiftKernel::operator()(CUstream stream, int size, CudaDeviceVariable& vol, float rDown, float rUp, float smooth)
{
	CudaKernel::SetComputeSize((size / 2 + 1), size, size);
	CUdeviceptr vol_dptr = vol.GetDevicePtr();

	void* arglist[5];

	arglist[0] = &size;
	arglist[1] = &vol_dptr;
	arglist[2] = &rDown;
	arglist[3] = &rUp;
	arglist[4] = &smooth;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}


MulRealCplxFFTShiftKernel::MulRealCplxFFTShiftKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("mulRealCplxFFTShift", aModule, aGridDim, aBlockDim, 0)
{

}

MulRealCplxFFTShiftKernel::MulRealCplxFFTShiftKernel(CUmodule aModule)
	: CudaKernel("mulRealCplxFFTShift", aModule, make_dim3(1, 1, 1), make_dim3(32, 16, 1), 0)
{

}

float MulRealCplxFFTShiftKernel::operator()(int size, CudaDeviceVariable& inRealVol, CudaDeviceVariable& inOutCplxVol)
{
	CudaKernel::SetComputeSize((size / 2 + 1), size, size);
	CUdeviceptr inRealVol_dptr = inRealVol.GetDevicePtr();
	CUdeviceptr inOutCplxVol_dptr = inOutCplxVol.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &size;
	arglist[1] = &inRealVol_dptr;
	arglist[2] = &inOutCplxVol_dptr;

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

void MulRealCplxFFTShiftKernel::operator()(CUstream stream, int size, CudaDeviceVariable& inRealVol, CudaDeviceVariable& inOutCplxVol)
{
	CudaKernel::SetComputeSize((size / 2 + 1), size, size);
	CUdeviceptr inRealVol_dptr = inRealVol.GetDevicePtr();
	CUdeviceptr inOutCplxVol_dptr = inOutCplxVol.GetDevicePtr();

	void* arglist[3];

	arglist[0] = &size;
	arglist[1] = &inRealVol_dptr;
	arglist[2] = &inOutCplxVol_dptr;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}


WedgeNormKernel::WedgeNormKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("wedgeNorm", aModule, aGridDim, aBlockDim, 0)
{

}

WedgeNormKernel::WedgeNormKernel(CUmodule aModule)
	: CudaKernel("wedgeNorm", aModule, make_dim3(1, 1, 1), make_dim3(32, 16, 1), 0)
{

}

float WedgeNormKernel::operator()(int size, CudaDeviceVariable& wedge, CudaDeviceVariable& part, CudaDeviceVariable& maxVal)
{
	CudaKernel::SetComputeSize((size / 2 + 1), size, size);
	CUdeviceptr wedge_dptr = wedge.GetDevicePtr();
	CUdeviceptr part_dptr = part.GetDevicePtr();

	void* arglist[4];

	arglist[0] = &size;
	arglist[1] = &wedge_dptr;
	arglist[2] = &part;
	arglist[3] = &maxVal;

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

void WedgeNormKernel::operator()(CUstream stream, int size, CudaDeviceVariable& wedge, CudaDeviceVariable& part, CudaDeviceVariable& maxVal)
{
	CudaKernel::SetComputeSize((size / 2 + 1), size, size);
	CUdeviceptr wedge_dptr = wedge.GetDevicePtr();
	CUdeviceptr part_dptr = part.GetDevicePtr();

	void* arglist[4];

	arglist[0] = &size;
	arglist[1] = &wedge_dptr;
	arglist[2] = &part;
	arglist[3] = &maxVal;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}



NormalizeKernel::NormalizeKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("normalize", aModule, aGridDim, aBlockDim, 0)
{

}

NormalizeKernel::NormalizeKernel(CUmodule aModule)
	: CudaKernel("normalize", aModule, make_dim3(1, 1, 1), make_dim3(32*16, 1, 1), 0)
{

}

float NormalizeKernel::operator()(int size, CudaDeviceVariable& data)
{
	CudaKernel::SetComputeSize(size, 1, 1);
	CUdeviceptr data_dptr = data.GetDevicePtr();

	void* arglist[2];

	arglist[0] = &size;
	arglist[1] = &data_dptr;

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

void NormalizeKernel::operator()(CUstream stream, int size, CudaDeviceVariable& data)
{
	CudaKernel::SetComputeSize(size, 1, 1);
	CUdeviceptr data_dptr = data.GetDevicePtr();

	void* arglist[2];

	arglist[0] = &size;
	arglist[1] = &data_dptr;

	cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, stream, arglist, NULL));
}
