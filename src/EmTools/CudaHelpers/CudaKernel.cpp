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


#include "CudaKernel.h"

#ifdef USE_CUDA
namespace Cuda
{
	CudaKernel::CudaKernel(string aKernelName, CUmodule aModule/*, CudaContext* aCtx*/)
		: mKernelName(aKernelName), mModule(aModule), mParamOffset(0), /*mCtx(aCtx), */mSharedMemSize(0),
		_maxThreadsPerBlock(0), _sharedSizeBytes(0), _constSizeBytes(0), _localSizeBytes(0), _numRegs(0),
		_ptxVersion(0), _binaryVersion(0)
	{
		cudaSafeCall(cuModuleGetFunction(&mFunction, mModule, mKernelName.c_str()));
		mBlockDim.x = mBlockDim.y = 32;
		mBlockDim.z = 1;
		mGridDim.x = mGridDim.y = mGridDim.z = 1;
		
		_loadAdditionalInfo();
	}

	CudaKernel::CudaKernel(string aKernelName, CUmodule aModule, /*CudaContext* aCtx, */dim3 aGridDim, dim3 aBlockDim, uint aDynamicSharedMemory)
		: mKernelName(aKernelName), mModule(aModule), mParamOffset(0), /*mCtx(aCtx), */mSharedMemSize(aDynamicSharedMemory),
		_maxThreadsPerBlock(0), _sharedSizeBytes(0), _constSizeBytes(0), _localSizeBytes(0), _numRegs(0),
		_ptxVersion(0), _binaryVersion(0)
	{
		cudaSafeCall(cuModuleGetFunction(&mFunction, mModule, mKernelName.c_str()));
		mBlockDim.x = aBlockDim.x;
		mBlockDim.y = aBlockDim.y;
		mBlockDim.z = aBlockDim.z;
		mGridDim.x = aGridDim.x;
		mGridDim.y = aGridDim.y;
		mGridDim.z = aGridDim.z;

		_loadAdditionalInfo();
	}

	void CudaKernel::_loadAdditionalInfo()
	{
			//Load additional info from kernel image
			int temp = 0;
			
			cudaSafeCall(cuFuncGetAttribute(&_maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, mFunction));
			cudaSafeCall(cuFuncGetAttribute(&_sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, mFunction));
			cudaSafeCall(cuFuncGetAttribute(&_constSizeBytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, mFunction));
			cudaSafeCall(cuFuncGetAttribute(&_localSizeBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, mFunction));
			cudaSafeCall(cuFuncGetAttribute(&_numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, mFunction));
			cudaSafeCall(cuFuncGetAttribute(&temp, CU_FUNC_ATTRIBUTE_PTX_VERSION, mFunction));
			_ptxVersion = temp / 10 + (temp % 10) / 10.0f;
			cudaSafeCall(cuFuncGetAttribute(&temp, CU_FUNC_ATTRIBUTE_BINARY_VERSION, mFunction));
			_binaryVersion = temp / 10 + (temp % 10) / 10.0f;
	}

	CudaKernel::~CudaKernel()
	{
		//mCtx->UnloadModule(mModule);
	}


	void CudaKernel::SetConstantValue(string aName, void* aValue)
	{
		CUdeviceptr dVarPtr;
		size_t varSize;
		cudaSafeCall(cuModuleGetGlobal(&dVarPtr, &varSize, mModule, aName.c_str()));
		cudaSafeCall(cuMemcpyHtoD(dVarPtr, aValue, varSize));
	}
		
	void CudaKernel::SetIntegerParameter(const int aValue)
	{
		mParamOffset = (mParamOffset + __alignof(aValue) - 1) & ~(__alignof(aValue) - 1);
		cudaSafeCall(cuParamSeti(mFunction, mParamOffset, aValue));
		mParamOffset += sizeof(aValue);
	}

	void CudaKernel::SetFloatParameter(const float aValue)
	{
		mParamOffset = (mParamOffset + __alignof(aValue) - 1) & ~(__alignof(aValue) - 1);
		cudaSafeCall(cuParamSetf(mFunction, mParamOffset, aValue));
		mParamOffset += sizeof(aValue);
	}

	void CudaKernel::SetDevicePtrParameter(CUdeviceptr aDevicePtr)
	{
		void* aPtr = (void*)(size_t)(aDevicePtr);
		mParamOffset = (mParamOffset + __alignof(aPtr) - 1) & ~(__alignof(aPtr) - 1);
		cudaSafeCall(cuParamSetv(mFunction, mParamOffset, &aPtr, sizeof(aPtr)));
		mParamOffset += sizeof(aPtr);
	}

	void CudaKernel::ResetParameterOffset()
	{
		mParamOffset = 0;
	}


	void CudaKernel::SetBlockDimensions(uint aX, uint aY, uint aZ)
	{
		mBlockDim.x = aX;
		mBlockDim.y = aY;
		mBlockDim.z = aZ;
	}

	void CudaKernel::SetBlockDimensions(dim3 aBlockDim)
	{
		mBlockDim.x = aBlockDim.x;
		mBlockDim.y = aBlockDim.y;
		mBlockDim.z = aBlockDim.z;
	}

	void CudaKernel::SetGridDimensions(uint aX, uint aY, uint aZ)
	{
		mGridDim.x = aX;
		mGridDim.y = aY;
		mGridDim.z = aZ;
	}

	void CudaKernel::SetGridDimensions(dim3 aGridDim)
	{
		mGridDim.x = aGridDim.x;
		mGridDim.y = aGridDim.y;
		mGridDim.z = aGridDim.z;
	}

	void CudaKernel::SetDynamicSharedMemory(uint aSizeInBytes)
	{
		mSharedMemSize = aSizeInBytes;
	}
	
	CUmodule& CudaKernel::GetCUmodule()
	{ 
		return mModule;
	}

	CUfunction& CudaKernel::GetCUfunction()
	{
		return mFunction;
	}
		
	int CudaKernel::GetMaxThreadsPerBlock()
	{
		return _maxThreadsPerBlock;
	}

	int CudaKernel::GetSharedSizeBytes()
	{
		return _sharedSizeBytes;
	}

	int CudaKernel::GetConstSizeBytes()
	{
		return _constSizeBytes;
	}

	int CudaKernel::GetLocalSizeBytes()
	{
		return _localSizeBytes;
	}

	int CudaKernel::GetNumRegs()
	{
		return _numRegs;
	}

	float CudaKernel::GetPtxVersion()
	{
		return _ptxVersion;
	}

	float CudaKernel::GetBinaryVersion()
	{
		return _binaryVersion;
	}


	//float CudaKernel::operator()()
	//{
	//	float ms = 0;
	//	cudaSafeCall(cuParamSetSize(mFunction, mParamOffset));

	//	cudaSafeCall(cuFuncSetBlockShape(mFunction, mBlockDim[0], mBlockDim[1], mBlockDim[2]));
	//	cudaSafeCall(cuFuncSetSharedSize(mFunction, mSharedMemSize));

	//	cudaSafeCall(cuCtxSynchronize());

	//	CUevent eventStart;
	//	CUevent eventEnd;
	//	CUstream stream = 0;
	//	cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_DEFAULT));
	//	cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_DEFAULT));
	//	
	//	cudaSafeCall(cuStreamQuery(stream));
	//	cudaSafeCall(cuEventRecord(eventStart, stream));

	//	cudaSafeCall(cuLaunchGrid(mFunction, mGridDim[0], mGridDim[1]));

	//	cudaSafeCall(cuCtxSynchronize());

	//	cudaSafeCall(cuStreamQuery(stream));
	//	cudaSafeCall(cuEventRecord(eventEnd, stream));
	//	cudaSafeCall(cuEventSynchronize(eventEnd));
	//	cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

	//	//reset the parameter stack
	//	mParamOffset = 0;

	//	return ms;
	//}	

	float CudaKernel::operator()(int dummy, ...)
	{
		float ms;
		va_list vl;
		va_start(vl,dummy);

		void* argList = (void*) vl;
		
		CUevent eventStart;
		CUevent eventEnd;
		CUstream stream = 0;
		cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
		cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));
		
		//cudaSafeCall(cuStreamQuery(stream));
		cudaSafeCall(cuEventRecord(eventStart, stream));
		cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, (void**)argList, NULL));
		
		cudaSafeCall(cuCtxSynchronize());

		//cudaSafeCall(cuStreamQuery(stream));
		cudaSafeCall(cuEventRecord(eventEnd, stream));
		cudaSafeCall(cuEventSynchronize(eventEnd));
		cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

		va_end(vl);

		return ms;
	}

	void CudaKernel::SetComputeSize(dim3 computeDimensions)
	{
		dim3 gridSize = { (computeDimensions.x + mBlockDim.x - 1) / mBlockDim.x,
						  (computeDimensions.y + mBlockDim.y - 1) / mBlockDim.y, 
						  (computeDimensions.z + mBlockDim.z - 1) / mBlockDim.z };
		SetGridDimensions(gridSize);
	}

	void CudaKernel::SetComputeSize(uint x, uint y, uint z)
	{
		SetComputeSize(dim3{ x, y, z });
	}	
}
#endif //USE_CUDA