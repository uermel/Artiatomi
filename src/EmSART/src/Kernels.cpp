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


	FPKernel::FPKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: Cuda::CudaKernel("march", aModule, aGridDim, aBlockDim, 0)
	{

	}

	FPKernel::FPKernel(CUmodule aModule)
		: Cuda::CudaKernel("march", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
	{

	}

	float FPKernel::operator()(int x, int y, Cuda::CudaPitchedDeviceVariable& projection, Cuda::CudaPitchedDeviceVariable& distMap, Cuda::CudaTextureObject3D& texObj)
	{
		return (*this)(x, y, projection, distMap, texObj, make_int2(0, 0), make_int2(x, y));
	}

	float FPKernel::operator()(int x, int y, Cuda::CudaPitchedDeviceVariable& projection, Cuda::CudaPitchedDeviceVariable& distMap, Cuda::CudaTextureObject3D& texObj, int2 roiMin, int2 roiMax)
	{
		CUdeviceptr proj_dptr = projection.GetDevicePtr();
		CUdeviceptr vol_dptr = NULL;//volume.GetDevicePtr();
		CUdeviceptr distmap_dptr = distMap.GetDevicePtr();
		size_t stride = projection.GetPitch();
		CUtexObject tex = texObj.GetTexObject();

		void** arglist = (void**)new void*[8];

		arglist[0] = &x;
		arglist[1] = &y;
		arglist[2] = &stride;
		arglist[3] = &proj_dptr;
		arglist[4] = &distmap_dptr;
		arglist[5] = &tex;
		arglist[6] = &roiMin;
		arglist[7] = &roiMax;

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



	SlicerKernel::SlicerKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("slicer", aModule, aGridDim, aBlockDim, 0)
	{

	}

	SlicerKernel::SlicerKernel(CUmodule aModule)
		: CudaKernel("slicer", aModule, make_dim3(1, 1, 1), make_dim3(32, 8, 1), 0)
	{

	}

	float SlicerKernel::operator()(int x, int y, CudaPitchedDeviceVariable& projection, float tmin, float tmax, Cuda::CudaTextureObject3D& texObj)
	{
		return (*this)(x, y, projection, tmin, tmax, texObj, make_int2(0, 0), make_int2(x, y));
	}

	float SlicerKernel::operator()(int x, int y, CudaPitchedDeviceVariable& projection, float tmin, float tmax, Cuda::CudaTextureObject3D& texObj, int2 roiMin, int2 roiMax)
	{
		CUdeviceptr proj_dptr = projection.GetDevicePtr();
		size_t stride = projection.GetPitch();
		CUtexObject tex = texObj.GetTexObject();

		void** arglist = (void**)new void*[9];

		arglist[0] = &x;
		arglist[1] = &y;
		arglist[2] = &stride;
		arglist[3] = &proj_dptr;
		arglist[4] = &tmin;
		arglist[5] = &tmax;
		arglist[6] = &tex;
		arglist[7] = &roiMin;
		arglist[8] = &roiMax;

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



	VolTravLengthKernel::VolTravLengthKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("volTraversalLength", aModule, aGridDim, aBlockDim, 0)
	{

	}

	VolTravLengthKernel::VolTravLengthKernel(CUmodule aModule)
		: CudaKernel("volTraversalLength", aModule, make_dim3(1, 1, 1), make_dim3(32, 8, 1), 0)
	{

	}

	float VolTravLengthKernel::operator()(int x, int y, CudaPitchedDeviceVariable& distMap)
	{
		return (*this)(x, y, distMap, make_int2(0, 0), make_int2(x, y));
	}

	float VolTravLengthKernel::operator()(int x, int y, CudaPitchedDeviceVariable& distMap, int2 roiMin, int2 roiMax)
	{
		CUdeviceptr distMap_dptr = distMap.GetDevicePtr();
		size_t stride = distMap.GetPitch();

		void** arglist = (void**)new void*[6];

		arglist[0] = &x;
		arglist[1] = &y;
		arglist[2] = &stride;
		arglist[3] = &distMap_dptr;
		arglist[4] = &roiMin;
		arglist[5] = &roiMax;

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



	CompKernel::CompKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("compare", aModule, aGridDim, aBlockDim, 0)
	{

	}

	CompKernel::CompKernel(CUmodule aModule)
		: CudaKernel("compare", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
	{

	}

	float CompKernel::operator()(CudaPitchedDeviceVariable& real_raw, CudaPitchedDeviceVariable& virtual_raw, CudaPitchedDeviceVariable& vol_distance_map, float realLength, float4 crop, float4 cropDim, float projValScale)
	{
		CUdeviceptr real_raw_dptr = real_raw.GetDevicePtr();
		CUdeviceptr virtual_raw_dptr = virtual_raw.GetDevicePtr();
		CUdeviceptr vol_distance_map_dptr = vol_distance_map.GetDevicePtr();
		int proj_x = (int)real_raw.GetWidth();
		int proj_y = (int)real_raw.GetHeight();
		size_t stride = real_raw.GetPitch();

		void** arglist = (void**)new void*[10];

		arglist[0] = &proj_x;
		arglist[1] = &proj_y;
		arglist[2] = &stride;
		arglist[3] = &real_raw_dptr;
		arglist[4] = &virtual_raw_dptr;
		arglist[5] = &vol_distance_map_dptr;
		arglist[6] = &realLength;
		arglist[7] = &crop;
		arglist[8] = &cropDim;
		arglist[9] = &projValScale;

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



	CropBorderKernel::CropBorderKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("cropBorder", aModule, aGridDim, aBlockDim, 0)
	{

	}

	CropBorderKernel::CropBorderKernel(CUmodule aModule)
		: CudaKernel("cropBorder", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
	{

	}

	float CropBorderKernel::operator()(CudaPitchedDeviceVariable& image, float2 cutLength, float2 dimLength, int2 p1, int2 p2, int2 p3, int2 p4)
	{
		CUdeviceptr image_dptr = image.GetDevicePtr();
		int proj_x = (int)image.GetWidth();
		int proj_y = (int)image.GetHeight();
		size_t stride = image.GetPitch();

		void** arglist = (void**)new void*[10];

		arglist[0] = &proj_x;
		arglist[1] = &proj_y;
		arglist[2] = &stride;
		arglist[3] = &image_dptr;
		arglist[4] = &cutLength;
		arglist[5] = &dimLength;
		arglist[6] = &p1;
		arglist[7] = &p2;
		arglist[8] = &p3;
		arglist[9] = &p4;

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



	BPKernel::BPKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim, bool fp16)
		: CudaKernel(fp16 ? "backProjectionFP16" : "backProjection", aModule, aGridDim, aBlockDim, 2 * aBlockDim.x * aBlockDim.y * aBlockDim.z * sizeof(float) * 4)
	{
		
	}

	BPKernel::BPKernel(CUmodule aModule, bool fp16)
		: CudaKernel(fp16 ? "backProjectionFP16" : "backProjection", aModule, make_dim3(1, 1, 1), make_dim3(8, 16, 4), 2 * 8 * 16 * 4 * sizeof(float) * 4)
	{

	}

    float BPKernel::operator()(int proj_x, int proj_y, float lambda, int maxOverSample, float maxOverSampleInv, CudaPitchedDeviceVariable& img, float distMin, float distMax)
	{
		CUdeviceptr img_ptr = img.GetDevicePtr();
		int stride = (int)img.GetPitch();

		void** arglist = (void**)new void*[9];

		arglist[0] = &proj_x;
		arglist[1] = &proj_y;
		arglist[2] = &lambda;
		arglist[3] = &maxOverSample;
		arglist[4] = &maxOverSampleInv;
		arglist[5] = &img_ptr;
		arglist[6] = &stride;
		arglist[7] = &distMin;
		arglist[8] = &distMax;

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



	CTFKernel::CTFKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("ctf", aModule, aGridDim, aBlockDim, 0)
	{

	}

	CTFKernel::CTFKernel(CUmodule aModule)
		: CudaKernel("ctf", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
	{

	}


	float CTFKernel::operator()(CudaDeviceVariable& ctf, float defocusMin, float defocusMax, float angle, bool applyForFP, bool phaseFlipOnly, float WienerFilterNoiseLevel, size_t stride, float4 betaFac)
	{
		CUdeviceptr ctf_dptr = ctf.GetDevicePtr();
		//size_t stride = 2049 * sizeof(float2);
		float _defocusMin = defocusMin * 0.000000001f;
		float _defocusMax = defocusMax * 0.000000001f;
		float _angle = angle / 180.0f * (float)M_PI;

		void** arglist = (void**)new void*[9];

		arglist[0] = &ctf_dptr;
		arglist[1] = &stride;
		arglist[2] = &_defocusMin;
		arglist[3] = &_defocusMax;
		arglist[4] = &_angle;
		arglist[5] = &applyForFP;
		arglist[6] = &phaseFlipOnly;
		arglist[7] = &WienerFilterNoiseLevel;
		arglist[8] = &betaFac;

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


	WbpWeightingKernel::WbpWeightingKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("wbpWeighting", aModule, aGridDim, aBlockDim, 0)
	{

	}

	WbpWeightingKernel::WbpWeightingKernel(CUmodule aModule)
		: CudaKernel("wbpWeighting", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
	{

	}


	float WbpWeightingKernel::operator()(CudaDeviceVariable& img, size_t stride, unsigned int pixelcount, float psiAngle, FilterMethod fm)
	{
		CUdeviceptr img_dptr = img.GetDevicePtr();
		float _angle = -psiAngle / 180.0f * (float)M_PI;

		void** arglist = (void**)new void*[5];

		arglist[0] = &img_dptr;
		arglist[1] = &stride;
		arglist[2] = &pixelcount;
		arglist[3] = &_angle;
		arglist[4] = &fm;

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


	FourFilterKernel::FourFilterKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("fourierFilter", aModule, aGridDim, aBlockDim, 0)
	{

	}

	FourFilterKernel::FourFilterKernel(CUmodule aModule)
		: CudaKernel("fourierFilter", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
	{

	}


	float FourFilterKernel::operator()(Cuda::CudaDeviceVariable& img, size_t stride, int pixelcount, float lp, float hp, float lps, float hps)
	{
		CUdeviceptr img_dptr = img.GetDevicePtr();

		void** arglist = (void**)new void*[7];

		arglist[0] = &img_dptr;
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


	DoseWeightingKernel::DoseWeightingKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("doseWeighting", aModule, aGridDim, aBlockDim, 0)
	{

	}

	DoseWeightingKernel::DoseWeightingKernel(CUmodule aModule)
		: CudaKernel("doseWeighting", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
	{

	}


	float DoseWeightingKernel::operator()(Cuda::CudaDeviceVariable& img, size_t stride, int pixelcount, float dose, float pixelSizeInA)
	{
		CUdeviceptr img_dptr = img.GetDevicePtr();

		void** arglist = (void**)new void*[5];

		arglist[0] = &img_dptr;
		arglist[1] = &stride;
		arglist[2] = &pixelcount;
		arglist[3] = &dose;
		arglist[4] = &pixelSizeInA;

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


	ConjKernel::ConjKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("conjMul", aModule, aGridDim, aBlockDim, 0)
	{

	}

	ConjKernel::ConjKernel(CUmodule aModule)
		: CudaKernel("conjMul", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
	{

	}


	float ConjKernel::operator()(Cuda::CudaDeviceVariable& img1, Cuda::CudaPitchedDeviceVariable& img2, size_t stride, int pixelcount)
	{
		CUdeviceptr img_dptr1 = img1.GetDevicePtr();
		CUdeviceptr img_dptr2 = img2.GetDevicePtr();

		void** arglist = (void**)new void*[4];

		arglist[0] = &img_dptr1;
		arglist[1] = &img_dptr2;
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

	MaxShiftKernel::MaxShiftKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("maxShift", aModule, aGridDim, aBlockDim, 0)
	{

	}

	MaxShiftKernel::MaxShiftKernel(CUmodule aModule)
		: CudaKernel("maxShift", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
	{

	}


	float MaxShiftKernel::operator()(Cuda::CudaDeviceVariable& img1, size_t stride, int pixelcount, int maxShift)
	{
		CUdeviceptr img_dptr1 = img1.GetDevicePtr();

		void** arglist = (void**)new void*[4];

		arglist[0] = &img_dptr1;
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

	MaxShiftWeightedKernel::MaxShiftWeightedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("maxShiftWeighted", aModule, aGridDim, aBlockDim, 0)
	{

	}

	MaxShiftWeightedKernel::MaxShiftWeightedKernel(CUmodule aModule)
		: CudaKernel("maxShiftWeighted", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
	{

	}


	float MaxShiftWeightedKernel::operator()(Cuda::CudaDeviceVariable& img1, size_t stride, int pixelcount, int maxShift)
	{
		CUdeviceptr img_dptr1 = img1.GetDevicePtr();

		void** arglist = (void**)new void*[4];

		arglist[0] = &img_dptr1;
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

	FindPeakKernel::FindPeakKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("findPeak", aModule, aGridDim, aBlockDim, 0)
	{

	}

	FindPeakKernel::FindPeakKernel(CUmodule aModule)
		: CudaKernel("findPeak", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
	{

	}


	float FindPeakKernel::operator()(Cuda::CudaDeviceVariable& img1, size_t stride, Cuda::CudaPitchedDeviceVariable& mask, int pixelcount, float maxThreashold)
	{
		CUdeviceptr img_dptr1 = img1.GetDevicePtr();
		CUdeviceptr mask_dptr = mask.GetDevicePtr();
		size_t pitchMask = mask.GetPitch();

		void** arglist = (void**)new void*[6];

		arglist[0] = &img_dptr1;
		arglist[1] = &stride;
		arglist[2] = &mask_dptr;
		arglist[3] = &pitchMask;
		arglist[4] = &pixelcount;
		arglist[5] = &maxThreashold;

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



	CopyToSquareKernel::CopyToSquareKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("makeSquare", aModule, aGridDim, aBlockDim, 0)
	{

	}

	CopyToSquareKernel::CopyToSquareKernel(CUmodule aModule)
		: CudaKernel("makeSquare", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
	{

	}

	float CopyToSquareKernel::operator()(CudaPitchedDeviceVariable& aIn, int maxsize, CudaDeviceVariable& aOut, int borderSizeX, int borderSizeY, bool mirrorY, bool fillZero)
	{
		CUdeviceptr in_dptr = aIn.GetDevicePtr();
		CUdeviceptr out_dptr = aOut.GetDevicePtr();
		int _maxsize = maxsize;
		int _borderSizeX = borderSizeX;
		int _borderSizeY = borderSizeY;
		bool _mirrorY = mirrorY;
		bool _fillZero = fillZero;
		int proj_x = aIn.GetWidth();
		int proj_y = aIn.GetHeight();
		int stride = (int)aIn.GetPitch() / sizeof(float);

		//printf("\n\nStride: %ld, %ld\n", stride, stride / sizeof(float));

		void** arglist = (void**)new void*[10];

		arglist[0] = &proj_x;
		arglist[1] = &proj_y;
		arglist[2] = &_maxsize;
		arglist[3] = &stride;
		arglist[4] = &in_dptr;
		arglist[5] = &out_dptr;
		arglist[6] = &_borderSizeX;
		arglist[7] = &_borderSizeY;
		arglist[8] = &_mirrorY;
		arglist[9] = &_fillZero;

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



	SamplesToCoefficients2DX::SamplesToCoefficients2DX(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("SamplesToCoefficients2DX", aModule, aGridDim, aBlockDim, 0)
	{

	}

	float SamplesToCoefficients2DX::operator()(CudaPitchedDeviceVariable& image)
	{
		CUdeviceptr in_dptr = image.GetDevicePtr();
		uint _pitch = image.GetPitch();
		uint _width = image.GetWidth();
		uint _height = image.GetHeight();

		void** arglist = (void**)new void*[4];

		arglist[0] = &in_dptr;
		arglist[1] = &_pitch;
		arglist[2] = &_width;
		arglist[3] = &_height;

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


	SamplesToCoefficients2DY::SamplesToCoefficients2DY(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
		: CudaKernel("SamplesToCoefficients2DY", aModule, aGridDim, aBlockDim, 0)
	{

	}

	float SamplesToCoefficients2DY::operator()(CudaPitchedDeviceVariable& image)
	{
		CUdeviceptr in_dptr = image.GetDevicePtr();
		uint _pitch = image.GetPitch();
		uint _width = image.GetWidth();
		uint _height = image.GetHeight();

		void** arglist = (void**)new void*[4];

		arglist[0] = &in_dptr;
		arglist[1] = &_pitch;
		arglist[2] = &_width;
		arglist[3] = &_height;

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


	
void SetConstantValues(CudaKernel& kernel, Volume<unsigned short>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv)
{
    //Set constant values
    float3 temp = proj.GetNormalVector(index);
    kernel.SetConstantValue("c_projNorm", &temp);
    temp = proj.GetPosition(index);
    kernel.SetConstantValue("c_detektor", &temp);
    temp = proj.GetPixelUPitch(index);
    kernel.SetConstantValue("c_uPitch", &temp);
    temp = proj.GetPixelVPitch(index);
    kernel.SetConstantValue("c_vPitch", &temp);
    temp = vol.GetSubVolumeBBoxRcp(subVol);
    kernel.SetConstantValue("c_volumeBBoxRcp", &temp);
    temp = vol.GetVolumeBBoxMin();
    kernel.SetConstantValue("c_bBoxMinComplete", &temp);
    temp = vol.GetVolumeBBoxMax();
    kernel.SetConstantValue("c_bBoxMaxComplete", &temp);
    temp = vol.GetSubVolumeBBoxMin(subVol);
	//printf("c_bBoxMin: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_bBoxMin", &temp);
    temp = vol.GetSubVolumeBBoxMax(subVol);
	//printf("c_bBoxMax: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_bBoxMax", &temp);
    temp = vol.GetDimension();
    kernel.SetConstantValue("c_volumeDimComplete", &temp);
    temp = vol.GetSubVolumeDimension(subVol);
    kernel.SetConstantValue("c_volumeDim", &temp);
    temp = vol.GetVoxelSize();
    kernel.SetConstantValue("c_voxelSize", &temp);
    float t = 0;//vol.GetSubVolumeZShift(subVol);
	//printf("zShift: %f\n", t);
    kernel.SetConstantValue("c_zShiftForPartialVolume", &t);

	kernel.SetConstantValue("c_magAniso", m.GetData());
	kernel.SetConstantValue("c_magAnisoInv", mInv.GetData());
}
void SetConstantValues(CudaKernel& kernel, Volume<float>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv)
{
    //Set constant values
    float3 temp = proj.GetNormalVector(index);
    kernel.SetConstantValue("c_projNorm", &temp);
    temp = proj.GetPosition(index);
    kernel.SetConstantValue("c_detektor", &temp);
    temp = proj.GetPixelUPitch(index);
    kernel.SetConstantValue("c_uPitch", &temp);
    temp = proj.GetPixelVPitch(index);
    kernel.SetConstantValue("c_vPitch", &temp);
    temp = vol.GetSubVolumeBBoxRcp(subVol);
    kernel.SetConstantValue("c_volumeBBoxRcp", &temp);
    temp = vol.GetVolumeBBoxMin();
    kernel.SetConstantValue("c_bBoxMinComplete", &temp);
    temp = vol.GetVolumeBBoxMax();
    kernel.SetConstantValue("c_bBoxMaxComplete", &temp);
    temp = vol.GetSubVolumeBBoxMin(subVol);
	//printf("c_bBoxMin: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_bBoxMin", &temp);
    temp = vol.GetSubVolumeBBoxMax(subVol);
	//printf("c_bBoxMax: %f, %f, %f\n", temp.x, temp.y, temp.z);
    kernel.SetConstantValue("c_bBoxMax", &temp);
    temp = vol.GetDimension();
    kernel.SetConstantValue("c_volumeDimComplete", &temp);
    temp = vol.GetSubVolumeDimension(subVol);
    kernel.SetConstantValue("c_volumeDim", &temp);
    temp = vol.GetVoxelSize();
    kernel.SetConstantValue("c_voxelSize", &temp);
    float t = 0;//vol.GetSubVolumeZShift(subVol);
	//printf("zShift: %f\n", t);
    kernel.SetConstantValue("c_zShiftForPartialVolume", &t);

	kernel.SetConstantValue("c_magAniso", m.GetData());
	kernel.SetConstantValue("c_magAnisoInv", mInv.GetData());
}

void SetConstantValues(BPKernel& kernel, Volume<unsigned short>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv)
{
    //Set constant values
    float3 temp = proj.GetNormalVector(index);
//    temp.x = -temp.x;
//    temp.y = -temp.y;
//    temp.z = -temp.z;
    kernel.SetConstantValue("c_projNorm", &temp);
    temp = proj.GetPosition(index);
    kernel.SetConstantValue("c_detektor", &temp);
    temp = proj.GetPixelUPitch(index);
    kernel.SetConstantValue("c_uPitch", &temp);
    temp = proj.GetPixelVPitch(index);
    kernel.SetConstantValue("c_vPitch", &temp);
    temp = vol.GetSubVolumeBBoxRcp(subVol);
    kernel.SetConstantValue("c_volumeBBoxRcp", &temp);
    temp = vol.GetVolumeBBoxMin();
    kernel.SetConstantValue("c_bBoxMinComplete", &temp);
    temp = vol.GetVolumeBBoxMax();
    kernel.SetConstantValue("c_bBoxMaxComplete", &temp);
    temp = vol.GetDimension();
    kernel.SetConstantValue("c_volumeDimComplete", &temp);
    temp = vol.GetSubVolumeDimension(subVol);
    kernel.SetConstantValue("c_volumeDim", &temp);
    temp = vol.GetVoxelSize();
    kernel.SetConstantValue("c_voxelSize", &temp);
    float t = vol.GetSubVolumeZShift(subVol);
    kernel.SetConstantValue("c_zShiftForPartialVolume", &t);
    int volquat = (int)vol.GetDimension().x / 4;
    kernel.SetConstantValue("c_volumeDim_x_quarter", &volquat);
    temp = vol.GetSubVolumeBBoxMin(subVol);
    kernel.SetConstantValue("c_bBoxMin", &temp);

    float matrix[16];
    proj.GetDetectorMatrix(index, matrix, 1);
    kernel.SetConstantValue("c_DetectorMatrix", matrix);

	kernel.SetConstantValue("c_magAniso", m.GetData());
	kernel.SetConstantValue("c_magAnisoInv", mInv.GetData());
}

void SetConstantValues(BPKernel& kernel, Volume<float>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv)
{
    //Set constant values
    float3 temp = proj.GetNormalVector(index);
//    temp.x = -temp.x;
//    temp.y = -temp.y;
//    temp.z = -temp.z;
    kernel.SetConstantValue("c_projNorm", &temp);
    temp = proj.GetPosition(index);
    kernel.SetConstantValue("c_detektor", &temp);
    temp = proj.GetPixelUPitch(index);
    kernel.SetConstantValue("c_uPitch", &temp);
    temp = proj.GetPixelVPitch(index);
    kernel.SetConstantValue("c_vPitch", &temp);
    temp = vol.GetSubVolumeBBoxRcp(subVol);
    kernel.SetConstantValue("c_volumeBBoxRcp", &temp);
    temp = vol.GetVolumeBBoxMin();
    kernel.SetConstantValue("c_bBoxMinComplete", &temp);
    temp = vol.GetVolumeBBoxMax();
    kernel.SetConstantValue("c_bBoxMaxComplete", &temp);
    temp = vol.GetDimension();
    kernel.SetConstantValue("c_volumeDimComplete", &temp);
    temp = vol.GetSubVolumeDimension(subVol);
    kernel.SetConstantValue("c_volumeDim", &temp);
    temp = vol.GetVoxelSize();
    kernel.SetConstantValue("c_voxelSize", &temp);
    float t = vol.GetSubVolumeZShift(subVol);
    kernel.SetConstantValue("c_zShiftForPartialVolume", &t);
    int volquat = (int)vol.GetDimension().x / 4;
    kernel.SetConstantValue("c_volumeDim_x_quarter", &volquat);
    temp = vol.GetSubVolumeBBoxMin(subVol);
    kernel.SetConstantValue("c_bBoxMin", &temp);

    float matrix[16];
    proj.GetDetectorMatrix(index, matrix, 1);
    kernel.SetConstantValue("c_DetectorMatrix", matrix);

	kernel.SetConstantValue("c_magAniso", m.GetData());
	kernel.SetConstantValue("c_magAnisoInv", mInv.GetData());
}

void SetConstantValues(CTFKernel& kernel, Projection& proj, int index, float cs, float voltage)
{
    //Set constant values
    kernel.SetConstantValue("c_cs", &cs);
    kernel.SetConstantValue("c_voltage", &voltage);
    float _openingAngle = 0.01f;
    kernel.SetConstantValue("c_openingAngle", &_openingAngle);
    float _ampContrast = 0.00f;
    kernel.SetConstantValue("c_ampContrast", &_ampContrast);
    float _phaseContrast = sqrtf(1 - _ampContrast * _ampContrast);
    kernel.SetConstantValue("c_phaseContrast", &_phaseContrast);
    float _pixelsize = proj.GetPixelSize();// * 100.0f;
    //_pixelsize = round(_pixelsize) / 100.0f;

    _pixelsize = _pixelsize * powf(10, -9);
    kernel.SetConstantValue("c_pixelsize", &_pixelsize);
    float _pixelcount = proj.GetMaxDimension();
    kernel.SetConstantValue("c_pixelcount", &_pixelcount);
    float _maxFreq = 1.0 / (_pixelsize * 2.0f);
    kernel.SetConstantValue("c_maxFreq", &_maxFreq);
    float _freqStepSize = _maxFreq / (_pixelcount / 2.0f);
	//printf("_freqStepSize: %f; _pixelsize: %f\n", _freqStepSize, _pixelsize);
    kernel.SetConstantValue("c_freqStepSize", &_freqStepSize);
	
    float _applyScatteringProfile = 0;
    kernel.SetConstantValue("c_applyScatteringProfile", &_applyScatteringProfile);
    float _applyEnvelopeFunction = 0;
    kernel.SetConstantValue("c_applyEnvelopeFunction", &_applyEnvelopeFunction);
}

ConvVolKernel::ConvVolKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("convertVolumeFP16ToFP32", aModule, aGridDim, aBlockDim, 0)
{

}

ConvVolKernel::ConvVolKernel(CUmodule aModule)
	: CudaKernel("convertVolumeFP16ToFP32", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float ConvVolKernel::operator()(Cuda::CudaPitchedDeviceVariable& img, unsigned int z)
{
	CUdeviceptr img_ptr = img.GetDevicePtr();
	int stride = (int)img.GetPitch() / img.GetElementSize();

	void** arglist = (void**)new void*[3];

	arglist[0] = &img_ptr;
	arglist[1] = &stride;
	arglist[2] = &z;

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

ConvVol3DKernel::ConvVol3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
	: CudaKernel("convertVolume3DFP16ToFP32", aModule, aGridDim, aBlockDim, 0)
{

}

ConvVol3DKernel::ConvVol3DKernel(CUmodule aModule)
	: CudaKernel("convertVolume3DFP16ToFP32", aModule, make_dim3(1, 1, 1), make_dim3(8, 8, 8), 0)
{

}

float ConvVol3DKernel::operator()(Cuda::CudaPitchedDeviceVariable& img)
{
	CUdeviceptr img_ptr = img.GetDevicePtr();
	int stride = (int)img.GetPitch() / img.GetElementSize();

	void** arglist = (void**)new void*[2];

	arglist[0] = &img_ptr;
	arglist[1] = &stride;

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

void RotKernel::computeRotMat(float phi, float psi, float theta, float rotMat[3][3])
{
	int i, j;
	float sinphi, sinpsi, sintheta;	/* sin of rotation angles */
	float cosphi, cospsi, costheta;	/* cos of rotation angles */


	float angles[] = { 0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330 };
	float angle_cos[16];
	float angle_sin[16];

	angle_cos[0] = 1.0f;
	angle_cos[1] = sqrt(3.0f) / 2.0f;
	angle_cos[2] = sqrt(2.0f) / 2.0f;
	angle_cos[3] = 0.5f;
	angle_cos[4] = 0.0f;
	angle_cos[5] = -0.5f;
	angle_cos[6] = -sqrt(2.0f) / 2.0f;
	angle_cos[7] = -sqrt(3.0f) / 2.0f;
	angle_cos[8] = -1.0f;
	angle_cos[9] = -sqrt(3.0f) / 2.0f;
	angle_cos[10] = -sqrt(2.0f) / 2.0f;
	angle_cos[11] = -0.5f;
	angle_cos[12] = 0.0f;
	angle_cos[13] = 0.5f;
	angle_cos[14] = sqrt(2.0f) / 2.0f;
	angle_cos[15] = sqrt(3.0f) / 2.0f;
	angle_sin[0] = 0.0f;
	angle_sin[1] = 0.5f;
	angle_sin[2] = sqrt(2.0f) / 2.0f;
	angle_sin[3] = sqrt(3.0f) / 2.0f;
	angle_sin[4] = 1.0f;
	angle_sin[5] = sqrt(3.0f) / 2.0f;
	angle_sin[6] = sqrt(2.0f) / 2.0f;
	angle_sin[7] = 0.5f;
	angle_sin[8] = 0.0f;
	angle_sin[9] = -0.5f;
	angle_sin[10] = -sqrt(2.0f) / 2.0f;
	angle_sin[11] = -sqrt(3.0f) / 2.0f;
	angle_sin[12] = -1.0f;
	angle_sin[13] = -sqrt(3.0f) / 2.0f;
	angle_sin[14] = -sqrt(2.0f) / 2.0f;
	angle_sin[15] = -0.5f;

	for (i = 0, j = 0; i<16; i++)
		if (angles[i] == phi)
		{
			cosphi = angle_cos[i];
			sinphi = angle_sin[i];
			j = 1;
		}

	if (j < 1)
	{
		phi = phi * (float)M_PI / 180.0f;
		cosphi = cos(phi);
		sinphi = sin(phi);
	}

	for (i = 0, j = 0; i<16; i++)
		if (angles[i] == psi)
		{
			cospsi = angle_cos[i];
			sinpsi = angle_sin[i];
			j = 1;
		}

	if (j < 1)
	{
		psi = psi * (float)M_PI / 180.0f;
		cospsi = cos(psi);
		sinpsi = sin(psi);
	}

	for (i = 0, j = 0; i<16; i++)
		if (angles[i] == theta)
		{
			costheta = angle_cos[i];
			sintheta = angle_sin[i];
			j = 1;
		}

	if (j < 1)
	{
		theta = theta * (float)M_PI / 180.0f;
		costheta = cos(theta);
		sintheta = sin(theta);
	}

	/* calculation of rotation matrix */

	rotMat[0][0] = cospsi*cosphi - costheta*sinpsi*sinphi;
	rotMat[1][0] = sinpsi*cosphi + costheta*cospsi*sinphi;
	rotMat[2][0] = sintheta*sinphi;
	rotMat[0][1] = -cospsi*sinphi - costheta*sinpsi*cosphi;
	rotMat[1][1] = -sinpsi*sinphi + costheta*cospsi*cosphi;
	rotMat[2][1] = sintheta*cosphi;
	rotMat[0][2] = sintheta*sinpsi;
	rotMat[1][2] = -sintheta*cospsi;
	rotMat[2][2] = costheta;
}

RotKernel::RotKernel(CUmodule aModule, int aSize)
	: CudaKernel("rot3d", aModule, make_dim3((aSize + 7) / 8, (aSize + 7) / 8, (aSize + 7) / 8), make_dim3(8, 8, 8), 0),
	size(aSize),
	volTexArray(this, "texVol", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, 
		CU_AD_FORMAT_FLOAT, aSize, aSize, aSize, 1)
{
	
}

float RotKernel::operator()(Cuda::CudaDeviceVariable & aVolOut, float phi, float psi, float theta)
{
	//make sure that the texture is properly bound to textref as other rotTools might have changed it!
	volTexArray.BindToTexRef();

	float rotMat[3][3];
	computeRotMat(phi, psi, theta, rotMat);
	CUdeviceptr out_dptr = aVolOut.GetDevicePtr();

	float3 rotMat0 = make_float3(rotMat[0][0], rotMat[0][1], rotMat[0][2]);
	float3 rotMat1 = make_float3(rotMat[1][0], rotMat[1][1], rotMat[1][2]);
	float3 rotMat2 = make_float3(rotMat[2][0], rotMat[2][1], rotMat[2][2]);

	void** arglist = (void**)new void*[5];

	arglist[0] = &size;
	arglist[1] = &rotMat0;
	arglist[2] = &rotMat1;
	arglist[3] = &rotMat2;
	arglist[4] = &out_dptr;

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

void RotKernel::SetData(float* data)
{
	volTexArray.GetArray()->CopyFromHostToArray(data);
}
