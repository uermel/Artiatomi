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


#include "CudaDeviceProperties.h"

#ifdef USE_CUDA
namespace Cuda
{

	CudaDeviceProperties::CudaDeviceProperties(CUdevice aDevice, int aDeviceID)
		:
		mDeviceID(aDeviceID),
		mClockRate(-1),
		mMaxThreadsPerBlock(-1),
		mMemPitch(-1),
		mRegsPerBlock(-1),
		mSharedMemPerBlock(-1),
		mTextureAlign(-1),
		mTotalConstantMemory(-1),
		mMultiProcessorCount(-1),
		mWarpSize(-1),
		mGpuOverlap(-1),
		mKernelExecTimeoutEnabled(-1),
		mIntegrated(-1),
		mCanMapHostMemory(-1),
		mMaximumTexture1DWidth(-1),
		mMaximumTexture2DWidth(-1),
		mMaximumTexture2DHeight(-1),
		mMaximumTexture3DWidth(-1),
		mMaximumTexture3DHeight(-1),
		mMaximumTexture3DDepth(-1),
		mMaximumTexture2DArrayWidth(-1),
		mMaximumTexture2DArrayHeight(-1),
		mMaximumTexture2DArrayNumSlices(-1),
		mSurfaceAllignment(-1),
		mConcurrentKernels(-1),
		mECCEnabled(-1),
		mPCIBusID(-1),
		mPCIDeviceID(-1),
		mTCCDriver(-1),
        mMemoryClockRate(-1),
        mGlobalMemoryBusWidth(-1),
        mL2CacheSize(-1),
        mMaxThreadsPerMultiProcessor(-1),
        mAsyncEngineCount(-1),
        mUnifiedAddressing(-1),
        mMaximumTexture1DLayeredWidth(-1),
        mMaximumTexture1DLayeredLayers(-1),
        mPCIDomainID(-1)
	{
		char devName[256];
		int major = 0, minor = 0;


		cudaSafeCall(cuDeviceGetName(devName, 256, aDevice));
		mDeviceName = std::string(devName);


		cudaSafeCall(cuDeviceComputeCapability(&major, &minor, aDevice));
		mComputeCapability = major + minor / 10.0f;


		cudaSafeCall(cuDriverGetVersion(&major));
		minor = major % 100;
		major = major / 1000;
		mDriverVersion = float(major) + (minor / 100.0f);



		cudaSafeCall(cuDeviceTotalMem_v2(&mTotalGlobalMemory, aDevice));

		if (mDriverVersion >= 2.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mMultiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mTotalConstantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mSharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mRegsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mWarpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mMaxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mMaxBlockDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, aDevice));
		cudaSafeCall(cuDeviceGetAttribute(&mMaxBlockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, aDevice));
		cudaSafeCall(cuDeviceGetAttribute(&mMaxBlockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mMaxGridDim[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, aDevice));
		cudaSafeCall(cuDeviceGetAttribute(&mMaxGridDim[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, aDevice));
		cudaSafeCall(cuDeviceGetAttribute(&mMaxGridDim[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mMemPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mTextureAlign, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mClockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, aDevice));



		if (mDriverVersion >= 2.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mGpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, aDevice));

		if (mDriverVersion >= 2.2f)
			cudaSafeCall(cuDeviceGetAttribute(&mKernelExecTimeoutEnabled, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, aDevice));

		if (mDriverVersion >= 2.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mIntegrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, aDevice));

		if (mDriverVersion >= 2.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mCanMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, aDevice));

		int compMode = 0;
		cudaSafeCall(cuDeviceGetAttribute(&compMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, aDevice));
		mComputeMode = (CUcomputemode)compMode;

		cudaSafeCall(cuDeviceGetAttribute(&mMaximumTexture1DWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mMaximumTexture2DWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mMaximumTexture2DHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mMaximumTexture3DWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mMaximumTexture3DHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mMaximumTexture3DDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mMaximumTexture2DArrayWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mMaximumTexture2DArrayHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT, aDevice));

		cudaSafeCall(cuDeviceGetAttribute(&mMaximumTexture2DArrayNumSlices, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES, aDevice));

		if (mDriverVersion >= 3.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mSurfaceAllignment, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, aDevice));

		if (mDriverVersion >= 3.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mConcurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, aDevice));

		if (mDriverVersion >= 3.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mECCEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, aDevice));

		if (mDriverVersion >= 3.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mPCIBusID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, aDevice));

		if (mDriverVersion >= 3.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mPCIDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, aDevice));

		if (mDriverVersion >= 3.2f)
			cudaSafeCall(cuDeviceGetAttribute(&mTCCDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, aDevice));

		if (mDriverVersion >= 4.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mMemoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, aDevice));

		if (mDriverVersion >= 4.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mGlobalMemoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, aDevice));

		if (mDriverVersion >= 4.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mL2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, aDevice));

		if (mDriverVersion >= 4.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mMaxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, aDevice));

		if (mDriverVersion >= 4.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mAsyncEngineCount, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, aDevice));

		if (mDriverVersion >= 4.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mUnifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, aDevice));

		if (mDriverVersion >= 4.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mMaximumTexture1DLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, aDevice));

		if (mDriverVersion >= 4.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mMaximumTexture1DLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, aDevice));

		if (mDriverVersion >= 4.0f)
			cudaSafeCall(cuDeviceGetAttribute(&mPCIDomainID, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, aDevice));
	}

	CudaDeviceProperties::~CudaDeviceProperties()
	{

	}

	int CudaDeviceProperties::GetClockRate()
	{
		return mClockRate;
	}
	int* CudaDeviceProperties::GetMaxBlockDim()
	{
		return mMaxBlockDim;
	}
	int* CudaDeviceProperties::GetMaxGridDim()
	{
		return mMaxGridDim;
	}
	int CudaDeviceProperties::GetMaxThreadsPerBlock()
	{
		return mMaxThreadsPerBlock;
	}
	int CudaDeviceProperties::GetMemPitch()
	{
		return mMemPitch;
	}
	int CudaDeviceProperties::GetRegsPerBlock()
	{
		return mRegsPerBlock;
	}
	int CudaDeviceProperties::GetSharedMemPerBlock()
	{
		return mSharedMemPerBlock;
	}
	int CudaDeviceProperties::GetTextureAlign()
	{
		return mTextureAlign;
	}
	int CudaDeviceProperties::GetTotalConstantMemory()
	{
		return mTotalConstantMemory;
	}
	string CudaDeviceProperties::GetDeviceName()
	{
		return mDeviceName;
	}
	float CudaDeviceProperties::GetComputeCapability()
	{
		return mComputeCapability;
	}
	float CudaDeviceProperties::GetDriverVersion()
	{
		return mDriverVersion;
	}
	size_t CudaDeviceProperties::GetTotalGlobalMemory()
	{
		return mTotalGlobalMemory;
	}
	int CudaDeviceProperties::GetMultiProcessorCount()
	{
		return mMultiProcessorCount;
	}
	int CudaDeviceProperties::GetWarpSize()
	{
		return mWarpSize;
	}
	bool CudaDeviceProperties::GetGpuOverlap()
	{
		return mGpuOverlap > 0;
	}
	bool CudaDeviceProperties::GetGernelExecTimeoutEnabled()
	{
		return mKernelExecTimeoutEnabled > 0;
	}
	bool CudaDeviceProperties::GetIntegrated()
	{
		return mIntegrated > 0;
	}
	bool CudaDeviceProperties::GetCanMapHostMemory()
	{
		return mCanMapHostMemory > 0;
	}
	CUcomputemode CudaDeviceProperties::GetComputeMode()
	{
		return mComputeMode;
	}
	int CudaDeviceProperties::GetMaximumTexture1DWidth()
	{
		return mMaximumTexture1DWidth;
	}
	int CudaDeviceProperties::GetMaximumTexture2DWidth()
	{
		return mMaximumTexture2DWidth;
	}
	int CudaDeviceProperties::GetMaximumTexture2DHeight()
	{
		return mMaximumTexture2DHeight;
	}
	int CudaDeviceProperties::GetMaximumTexture3DWidth()
	{
		return mMaximumTexture3DWidth;
	}
	int CudaDeviceProperties::GetMaximumTexture3DHeight()
	{
		return mMaximumTexture3DHeight;
	}
	int CudaDeviceProperties::GetMaximumTexture3DDepth()
	{
		return mMaximumTexture3DDepth;
	}
	int CudaDeviceProperties::GetMaximumTexture2DArrayWidth()
	{
		return mMaximumTexture2DArrayWidth;
	}
	int CudaDeviceProperties::GetMaximumTexture2DArrayHeight()
	{
		return mMaximumTexture2DArrayHeight;
	}
	int CudaDeviceProperties::GetMaximumTexture2DArrayNumSlices()
	{
		return mMaximumTexture2DArrayNumSlices;
	}
	int CudaDeviceProperties::GetSurfaceAllignment()
	{
		return mSurfaceAllignment;
	}
	bool CudaDeviceProperties::GetConcurrentKernels()
	{
		return mConcurrentKernels > 0;
	}
	bool CudaDeviceProperties::GetECCEnabled()
	{
		return mECCEnabled > 0;
	}
	int CudaDeviceProperties::GetPCIBusID()
	{
		return mPCIBusID;
	}
	int CudaDeviceProperties::GetPCIDeviceID()
	{
		return mPCIDeviceID;
	}
	bool CudaDeviceProperties::GetTCCDriver()
	{
		return mTCCDriver > 0;
	}
    int CudaDeviceProperties::GetMemoryClockRate()
	{
		return mMemoryClockRate;
	}
    int CudaDeviceProperties::GetGlobalMemoryBusWidth()
	{
		return mGlobalMemoryBusWidth;
	}
    int CudaDeviceProperties::GetL2CacheSize()
	{
		return mL2CacheSize;
	}
    int CudaDeviceProperties::GetMaxThreadsPerMultiProcessor()
	{
		return mMaxThreadsPerMultiProcessor;
	}
    int CudaDeviceProperties::GetAsyncEngineCount()
	{
		return mAsyncEngineCount;
	}
    bool CudaDeviceProperties::GetUnifiedAddressing()
	{
		return mUnifiedAddressing > 0;
	}
    int CudaDeviceProperties::GetMaximumTexture1DLayeredWidth()
	{
		return mMaximumTexture1DLayeredWidth;
	}
    int CudaDeviceProperties::GetMaximumTexture1DLayeredLayers()
	{
		return mMaximumTexture1DLayeredLayers;
	}
    int CudaDeviceProperties::GetPCIDomainID()
	{
		return mPCIDomainID;
	}


	void CudaDeviceProperties::PrintProperties()
	{
		cout << "\nDetailed device info:\n";
		cout << "--------------------------------------------------------------------------------\n\n" ;
		cout << "  Device name:                                      " << mDeviceName << endl;
		cout << "  Device ID:                                        " << mDeviceID << endl;

		cout << "  CUDA compute capability revision                  " << mComputeCapability << endl;
		cout << "  CUDA driver version:                              " << mDriverVersion << endl;
		cout << "  Total amount of global memory:                    " << mTotalGlobalMemory / 1024 / 1024 << " MByte" << endl;
		cout << "  ClockRate:                                        " << mClockRate / 1000 << " MHz"<< endl;
		cout << "  Maximum block dimensions:                         " << mMaxBlockDim[0] << "x" << mMaxBlockDim[1] << "x" << mMaxBlockDim[2] << endl;
		cout << "  Maximum grid dimensions:                          " << mMaxGridDim[0] << "x" << mMaxGridDim[1] << "x" << mMaxGridDim[2] << endl;
		cout << endl;
		if (mMaxThreadsPerBlock > -1)
			cout << "  Maximum number of threads per block:              " << mMaxThreadsPerBlock << endl;
		if (mRegsPerBlock > -1)
			cout << "  Total number of registers available per block:    " << mRegsPerBlock << endl;
		if (mSharedMemPerBlock > -1)
			cout << "  Total amount of shared memory per block:          " << mSharedMemPerBlock << " Bytes" << endl;
		if (mTotalConstantMemory > -1)
			cout << "  Total amount of constant memory:                  " << mTotalConstantMemory << " Bytes"  << endl;
		if (mMultiProcessorCount > -1)
			cout << "  Number of multiprocessors:                        " << mMultiProcessorCount << endl;
		if (mMultiProcessorCount > -1 && mComputeCapability < 2)
			cout << "  Number of cores:                                  " << mMultiProcessorCount * 8 << endl;
		if (mMultiProcessorCount > -1 && mComputeCapability >= 2)
			cout << "  Number of cores:                                  " << mMultiProcessorCount * 32 << endl;
		if (mWarpSize > -1)
			cout << "  Warp size:                                        " << mWarpSize << endl;
		if (mMemPitch > -1)
			cout << "  Maximum memory pitch:                             " << mMemPitch << endl;
		cout << endl;
		if (mGpuOverlap > -1)
		{
			string val;
			if (mGpuOverlap > 0) val = "True"; else val = "False";
			cout << "  Can copy memory and execute kernel concurrently:  " << val << endl;
		}
		if (mKernelExecTimeoutEnabled > -1)
		{
			string val;
			if (mKernelExecTimeoutEnabled > 0) val = "True"; else val = "False";
			cout << "  Run time limit on Kernels is enabled:             " << val << endl;
		}
		if (mIntegrated > -1)
		{
			string val;
			if (mIntegrated > 0) val = "True"; else val = "False";
			cout << "  Is integrated with host memory:                   " << val << endl;
		}
		if (mCanMapHostMemory > -1)
		{
			string val;
			if (mCanMapHostMemory > 0) val = "True"; else val = "False";
			cout << "  Can map host memory:                              " << val << endl;
		}
		if (mConcurrentKernels > -1)
		{
			string val;
			if (mConcurrentKernels > 0) val = "True"; else val = "False";
			cout << "  Can execute multiple kernels concurrently:        " << val << endl;
		}
		cout << endl;
		if (mTextureAlign > -1)
			cout << "  Alignment requirement for textures:               " << mTextureAlign << endl;
		if (mMaximumTexture1DWidth > -1)
			cout << "  Maximum 1D texture dimensions:                    " << mMaximumTexture1DWidth << endl;
		if (mMaximumTexture2DWidth > -1)
			cout << "  Maximum 2D texture dimensions:                    " << mMaximumTexture2DWidth << "x" << mMaximumTexture2DHeight << endl;
		if (mMaximumTexture3DWidth > -1)
			cout << "  Maximum 3D texture dimensions:                    " << mMaximumTexture3DWidth << "x" << mMaximumTexture3DHeight << "x" << mMaximumTexture3DDepth << endl;
		if (mMaximumTexture2DArrayWidth > -1)
			cout << "  Maximum texture array dimensions:                 " << mMaximumTexture2DArrayWidth << "x" << mMaximumTexture2DArrayHeight<< endl;
		if (mMaximumTexture2DArrayNumSlices > -1)
			cout << "  Maximum slices in a texture array:                " << mMaximumTexture2DArrayNumSlices << endl;
		if (mSurfaceAllignment > -1)
			cout << "  Surface alignment requirement:                    " << mSurfaceAllignment << endl;
		cout << endl;
		if (mECCEnabled > -1)
		{
			string val;
			if (mECCEnabled > 0) val = "True"; else val = "False";
			cout << "  ECC support is enabled:                           " << val << endl;
		}
		if (mPCIBusID > -1)
			cout << "  PCI bus ID:                                       " << mPCIBusID << endl;
		if (mPCIDeviceID > -1)
			cout << "  PCI device ID:                                    " << mPCIDeviceID << endl;
		if (mTCCDriver > -1)
		{
			string val;
			if (mTCCDriver > 0) val = "True"; else val = "False";
			cout << "  Device is using TCC driver model:                 " << val << endl;
		}

		if (mComputeMode == CU_COMPUTEMODE_DEFAULT)
			cout << "  CUDA compute mode:                                " << "Default compute mode" << endl;
		/*	if (mComputeMode == CU_COMPUTEMODE_EXCLUSIVE)
			cout << "  CUDA compute mode:                                " << "Compute-exclusive mode" << endl;*/
		if (mComputeMode == CU_COMPUTEMODE_PROHIBITED)
			cout << "  CUDA compute mode:                                " << "Compute-prohibited mode" << endl;

		if (mMemoryClockRate > -1)
			cout << "  Peak memory clock frequency:                      " << mMemoryClockRate / 1000 << " MHz" << endl;
		if (mGlobalMemoryBusWidth > -1)
			cout << "  Global memory bus width in bits:                  " << mGlobalMemoryBusWidth << endl;
		if (mL2CacheSize > -1)
			cout << "  Size of L2 cache in bytes:                        " << mL2CacheSize << endl;
		if (mMaxThreadsPerMultiProcessor > -1)
			cout << "  Maximum resident threads per multiprocessor:      " << mMaxThreadsPerMultiProcessor << endl;
		if (mAsyncEngineCount > -1)
			cout << "  Number of asynchronous engines:                   " << mAsyncEngineCount << endl;
		if (mUnifiedAddressing > -1)
		{
			string val;
			if (mUnifiedAddressing > 0) val = "True"; else val = "False";
			cout << "  Device shares a unified address space with host:  " << val << endl;
		}
		if (mMaximumTexture1DLayeredWidth > -1)
			cout << "  Maximum 1D layered texture width:                 " << mMaximumTexture1DLayeredWidth << endl;
		if (mMaximumTexture1DLayeredLayers > -1)
			cout << "  Maximum layers in a 1D layered texture:           " << mMaximumTexture1DLayeredLayers << endl;
		if (mPCIDomainID > -1)
			cout << "  PCI domain ID of the device:                      " << mPCIDomainID << endl;
		cout << endl;
		cout << endl;
	}
}
#endif //USE_CUDA
