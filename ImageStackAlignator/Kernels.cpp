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


#include "Kernels.h"

using namespace Cuda;

FourierFilterKernel::FourierFilterKernel(CUmodule aModule, dim3 aGriDim, dim3 aBlockDim) :
	Cuda::CudaKernel("fourierFilter", aModule, aGriDim, aBlockDim, 0)
{
}

FourierFilterKernel::FourierFilterKernel(CUmodule aModule) :
	Cuda::CudaKernel("fourierFilter", aModule, dim3{ 1, 1, 1 }, dim3{16, 16, 1}, 0)
{
}

float FourierFilterKernel::operator()(CudaDeviceVariable & img, size_t stride, int pixelcount, float lp, float hp, float lps, float hps)
{
	CUdeviceptr ptrImg = img.GetDevicePtr();

	void** arglist = (void**)new void*[7];
	arglist[0] = &ptrImg;
	arglist[1] = &stride;
	arglist[2] = &pixelcount;
	arglist[3] = &lp;
	arglist[4] = &hp;
	arglist[5] = &lps;
	arglist[6] = &hps;

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

	delete[] arglist;
	return ms;
}

float FourierFilterKernel::operator()(CudaPitchedDeviceVariable & img, int pixelcount, float lp, float hp, float lps, float hps)
{
	CUdeviceptr ptrImg = img.GetDevicePtr();
	size_t stride = img.GetPitch();

	void** arglist = (void**)new void*[7];
	arglist[0] = &ptrImg;
	arglist[1] = &stride;
	arglist[2] = &pixelcount;
	arglist[3] = &lp;
	arglist[4] = &hp;
	arglist[5] = &lps;
	arglist[6] = &hps;

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

	delete[] arglist;
	return ms;
}

float FourierFilterKernel::operator()(NPPImage_32fcC1 & img, float lp, float hp, float lps, float hps)
{
	CUdeviceptr ptrImg = img.GetDevicePointer();
	size_t stride = img.GetPitch();
	int pixelcount = img.GetWidthRoi();

	void** arglist = (void**)new void*[7];
	arglist[0] = &ptrImg;
	arglist[1] = &stride;
	arglist[2] = &pixelcount;
	arglist[3] = &lp;
	arglist[4] = &hp;
	arglist[5] = &lps;
	arglist[6] = &hps;

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

	delete[] arglist;
	return ms;
}

ConjMulKernel::ConjMulKernel(CUmodule aModule, dim3 aGriDim, dim3 aBlockDim) :
	Cuda::CudaKernel("conjMul", aModule, aGriDim, aBlockDim, 0)
{
}

ConjMulKernel::ConjMulKernel(CUmodule aModule) :
	Cuda::CudaKernel("conjMul", aModule, dim3{ 1, 1, 1 }, dim3{16, 16, 1}, 0)
{
}

float ConjMulKernel::operator()(CudaDeviceVariable & complxA, CudaDeviceVariable & complxB, size_t stride, int pixelcount)
{
	CUdeviceptr ptrImgA = complxA.GetDevicePtr();
	CUdeviceptr ptrImgB = complxB.GetDevicePtr();

	void** arglist = (void**)new void*[4];
	arglist[0] = &ptrImgA;
	arglist[1] = &ptrImgB;
	arglist[2] = &stride;
	arglist[3] = &pixelcount;

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

	delete[] arglist;
	return ms;
}

float ConjMulKernel::operator()(CudaPitchedDeviceVariable & complxA, CudaPitchedDeviceVariable & complxB, int pixelcount)
{
	CUdeviceptr ptrImgA = complxA.GetDevicePtr();
	CUdeviceptr ptrImgB = complxB.GetDevicePtr();
	size_t stride = complxA.GetPitch();

	void** arglist = (void**)new void*[4];
	arglist[0] = &ptrImgA;
	arglist[1] = &ptrImgB;
	arglist[2] = &stride;
	arglist[3] = &pixelcount;

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

	delete[] arglist;
	return ms;
}

float ConjMulKernel::operator()(NPPImage_32fcC1 & complxA, NPPImage_32fcC1 & complxB)
{
	CUdeviceptr ptrImgA = complxA.GetDevicePointer();
	CUdeviceptr ptrImgB = complxB.GetDevicePointer();
	size_t stride = complxA.GetPitch();
	int pixelcount = complxA.GetWidthRoi();

	void** arglist = (void**)new void*[4];
	arglist[0] = &ptrImgA;
	arglist[1] = &ptrImgB;
	arglist[2] = &stride;
	arglist[3] = &pixelcount;

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

	delete[] arglist;
	return ms;
}

MaxShiftKernel::MaxShiftKernel(CUmodule aModule, dim3 aGriDim, dim3 aBlockDim) :
	Cuda::CudaKernel("maxShift", aModule, aGriDim, aBlockDim, 0)
{
}

MaxShiftKernel::MaxShiftKernel(CUmodule aModule) :
	Cuda::CudaKernel("maxShift", aModule, dim3{ 1, 1, 1 }, dim3{16, 16, 1}, 0)
{
}

float MaxShiftKernel::operator()(CudaDeviceVariable & img, size_t stride, int pixelcount, int maxShift)
{
	CUdeviceptr ptrImg = img.GetDevicePtr();

	void** arglist = (void**)new void*[4];
	arglist[0] = &ptrImg;
	arglist[1] = &stride;
	arglist[2] = &pixelcount;
	arglist[3] = &maxShift;

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

	delete[] arglist;
	return ms;
}

float MaxShiftKernel::operator()(Cuda::CudaPitchedDeviceVariable & img, int pixelcount, int maxShift)
{
	CUdeviceptr ptrImg = img.GetDevicePtr();
	size_t stride = img.GetPitch();

	void** arglist = (void**)new void*[4];
	arglist[0] = &ptrImg;
	arglist[1] = &stride;
	arglist[2] = &pixelcount;
	arglist[3] = &maxShift;

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

	delete[] arglist;
	return ms;
}

float MaxShiftKernel::operator()(NPPImage_32fC1 & img, int maxShift)
{
	CUdeviceptr ptrImg = img.GetDevicePointerRoi();
	size_t stride = img.GetPitch();
	int pixelcount = img.GetWidthRoi();

	void** arglist = (void**)new void*[4];
	arglist[0] = &ptrImg;
	arglist[1] = &stride;
	arglist[2] = &pixelcount;
	arglist[3] = &maxShift;

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

	delete[] arglist;
	return ms;
}




SumRowKernel::SumRowKernel(CUmodule aModule, dim3 aGriDim, dim3 aBlockDim) :
	Cuda::CudaKernel("SumRow", aModule, aGriDim, aBlockDim, 0)
{
}

SumRowKernel::SumRowKernel(CUmodule aModule) :
	Cuda::CudaKernel("SumRow", aModule, dim3{ 1, 1, 1 }, dim3{ 256, 1, 1 }, 0)
{
}

float SumRowKernel::operator()(CudaDeviceVariable & img, size_t stride, int width, int height, Cuda::CudaDeviceVariable & sum)
{
	CUdeviceptr ptrImg = img.GetDevicePtr();
	CUdeviceptr ptrSum = sum.GetDevicePtr();

	void** arglist = (void**)new void*[5];
	arglist[0] = &ptrImg;
	arglist[1] = &stride;
	arglist[2] = &width;
	arglist[3] = &height;
	arglist[4] = &ptrSum;

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

	delete[] arglist;
	return ms;
}

float SumRowKernel::operator()(Cuda::CudaPitchedDeviceVariable & img, int width, int height, Cuda::CudaDeviceVariable & sum)
{
	CUdeviceptr ptrImg = img.GetDevicePtr();
	size_t stride = img.GetPitch();
	CUdeviceptr ptrSum = sum.GetDevicePtr();

	void** arglist = (void**)new void*[5];
	arglist[0] = &ptrImg;
	arglist[1] = &stride;
	arglist[2] = &width;
	arglist[3] = &height;
	arglist[4] = &ptrSum;

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

	delete[] arglist;
	return ms;
}

float SumRowKernel::operator()(NPPImage_32fC1 & img, Cuda::CudaDeviceVariable & sum)
{
	CUdeviceptr ptrImg = img.GetDevicePointerRoi();
	size_t stride = img.GetPitch();
	int width = img.GetWidthRoi();
	int height = img.GetHeightRoi();
	CUdeviceptr ptrSum = sum.GetDevicePtr();

	void** arglist = (void**)new void*[5];
	arglist[0] = &ptrImg;
	arglist[1] = &stride;
	arglist[2] = &width;
	arglist[3] = &height;
	arglist[4] = &ptrSum;

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

	delete[] arglist;
	return ms;
}


CreateMaskKernel::CreateMaskKernel(CUmodule aModule, dim3 aGriDim, dim3 aBlockDim) :
	Cuda::CudaKernel("CreateMask", aModule, aGriDim, aBlockDim, 0)
{
}

CreateMaskKernel::CreateMaskKernel(CUmodule aModule) :
	Cuda::CudaKernel("CreateMask", aModule, dim3{ 1, 1, 1 }, dim3{ 16, 16, 1 }, 0)
{
}

float CreateMaskKernel::operator()(CudaDeviceVariable & img, size_t stride, int width, int height, Cuda::CudaDeviceVariable & sum)
{
	CUdeviceptr ptrImg = img.GetDevicePtr();
	CUdeviceptr ptrSum = sum.GetDevicePtr();

	void** arglist = (void**)new void*[5];
	arglist[0] = &ptrImg;
	arglist[1] = &stride;
	arglist[2] = &width;
	arglist[3] = &height;
	arglist[4] = &ptrSum;

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

	delete[] arglist;
	return ms;
}

float CreateMaskKernel::operator()(Cuda::CudaPitchedDeviceVariable & img, int width, int height, Cuda::CudaDeviceVariable & sum)
{
	CUdeviceptr ptrImg = img.GetDevicePtr();
	size_t stride = img.GetPitch();
	CUdeviceptr ptrSum = sum.GetDevicePtr();

	void** arglist = (void**)new void*[5];
	arglist[0] = &ptrImg;
	arglist[1] = &stride;
	arglist[2] = &width;
	arglist[3] = &height;
	arglist[4] = &ptrSum;

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

	delete[] arglist;
	return ms;
}

float CreateMaskKernel::operator()(NPPImage_8uC1 & img, Cuda::CudaDeviceVariable & sum)
{
	CUdeviceptr ptrImg = img.GetDevicePointerRoi();
	size_t stride = img.GetPitch();
	int width = img.GetWidthRoi();
	int height = img.GetHeightRoi();
	CUdeviceptr ptrSum = sum.GetDevicePtr();

	void** arglist = (void**)new void*[5];
	arglist[0] = &ptrImg;
	arglist[1] = &stride;
	arglist[2] = &width;
	arglist[3] = &height;
	arglist[4] = &ptrSum;

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

	delete[] arglist;
	return ms;
}
