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


#include "OpenCLDeviceVariable.h"
#include "OpenCLHelpers.h"

#ifdef USE_OPENCL
namespace OpenCL
{
	cl_mem allocBuffer(size_t aSizeInBytes, cl_int* errCode)
	{
		return clCreateBuffer(OpenCLThreadBoundContext::GetCtx(), CL_MEM_READ_WRITE, aSizeInBytes, NULL, errCode);
	}

	cl_int copyBuffer(cl_mem aSource, cl_mem aDevPtr, size_t aSizeInBytes)
	{
		return clEnqueueCopyBuffer(OpenCLThreadBoundContext::GetQueue(), aSource, aDevPtr, 0, 0, aSizeInBytes, 0, NULL, NULL);
	}

	cl_int writeBuffer(void* aSource, cl_mem aDest, size_t aSizeInBytes)
	{
		return clEnqueueWriteBuffer(OpenCLThreadBoundContext::GetQueue(), aDest, true, 0, aSizeInBytes, aSource, 0, NULL, NULL);
	}

	cl_int readBuffer(cl_mem aSource, void* aDest, size_t aSizeInBytes)
	{
		return clEnqueueReadBuffer(OpenCLThreadBoundContext::GetQueue(), aSource, true, 0, aSizeInBytes, aDest, 0, NULL, NULL);
	}

	cl_int memset(cl_mem aDest, size_t aSizeInBytes, uchar aValue)
	{
		return clEnqueueFillBuffer, OpenCLThreadBoundContext::GetQueue(), aDest, &aValue, 1, 0, aSizeInBytes, 0, NULL, NULL;
	}

	OpenCLDeviceVariable::OpenCLDeviceVariable(size_t aSizeInBytes)
		:mDevPtr(0), mSizeInBytes(aSizeInBytes), mIsOwner(true)
	{
		cl_int errCode;
		auto erg = RunInOpenCLThread(allocBuffer, aSizeInBytes, &errCode);
		mDevPtr = erg.get();
		openCLSafeCall(errCode);
	}

	OpenCLDeviceVariable::OpenCLDeviceVariable(size_t aSizeInBytes, bool aIsOwner)
		:mDevPtr(0), mSizeInBytes(aSizeInBytes), mIsOwner(aIsOwner)
	{
		cl_int errCode;
		auto erg = RunInOpenCLThread(allocBuffer, aSizeInBytes, &errCode);
		mDevPtr = erg.get();
		openCLSafeCall(errCode);
	}

	OpenCLDeviceVariable::OpenCLDeviceVariable(const cl_mem & aDevPtr, size_t aSizeInBytes)
		:mDevPtr(aDevPtr), mSizeInBytes(aSizeInBytes), mIsOwner(false)
	{
		if (mSizeInBytes == 0)
		{
			size_t sizeRet = 0;
			auto erg = RunInOpenCLThread(clGetMemObjectInfo, mDevPtr, CL_MEM_SIZE, 1, &mSizeInBytes, &sizeRet);
			openCLSafeCall(erg.get());
		}
	}

	OpenCLDeviceVariable::~OpenCLDeviceVariable()
	{
		if (mIsOwner)
		{
			auto erg = RunInOpenCLThread(clReleaseMemObject, mDevPtr);
			openCLSafeCall(erg.get());
		}
	}

	void OpenCLDeviceVariable::CopyDeviceToDevice(cl_mem aSource)
	{
		auto erg = RunInOpenCLThread(copyBuffer, aSource, mDevPtr, mSizeInBytes);
		openCLSafeCall(erg.get());
	}

	void OpenCLDeviceVariable::CopyDeviceToDevice(OpenCLDeviceVariable& aSource)
	{
		auto erg = RunInOpenCLThread(copyBuffer, aSource.mDevPtr, mDevPtr, mSizeInBytes);
		openCLSafeCall(erg.get());
	}


	void OpenCLDeviceVariable::CopyHostToDevice(void* aSource, size_t aSizeInBytes)
	{
		size_t size = aSizeInBytes;
		if (size == 0)
			size = mSizeInBytes;

		auto erg = RunInOpenCLThread(writeBuffer, aSource, mDevPtr, size);
		openCLSafeCall(erg.get());
	}

	void OpenCLDeviceVariable::CopyDeviceToHost(void* aDest, size_t aSizeInBytes)
	{
		size_t size = aSizeInBytes;
		if (size == 0)
			size = mSizeInBytes;

		auto erg = RunInOpenCLThread(readBuffer, mDevPtr, aDest, size);
		openCLSafeCall(erg.get());
	}

	size_t OpenCLDeviceVariable::GetSize()
	{
		return mSizeInBytes;
	}

	cl_mem OpenCLDeviceVariable::GetDevicePtr()
	{
		return mDevPtr;
	}

	void OpenCLDeviceVariable::Memset(uchar aValue)
	{
		auto erg = RunInOpenCLThread(memset, mDevPtr, mSizeInBytes, aValue);
		openCLSafeCall(erg.get());
	}
}
#endif