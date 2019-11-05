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


#include "MemoryPool.h"
#include <stdio.h>
#include "../MKLog/MKLog.h"

using namespace std;

inline void* aligned_malloc(size_t size, size_t align) 
{
	void *result;
#ifdef _USE_WINDOWS_COMPILER_SETTINGS
	result = _aligned_malloc(size, align);
#else 
#ifdef _USE_LINUX_COMPILER_SETTINGS
	if (posix_memalign(&result, align, size)) result = 0;
#else
#ifdef _USE_APPLE_COMPILER_SETTINGS
	if (posix_memalign(&result, align, size)) result = 0;
#else
#error Either Linux or Windows compiler settings must be set.
#endif
#endif
#endif
	return result;
}

inline void aligned_free(void *ptr)
{
#ifdef _USE_WINDOWS_COMPILER_SETTINGS 
	_aligned_free(ptr);
#else
#ifdef _USE_LINUX_COMPILER_SETTINGS 
	free(ptr);
#else
#ifdef _USE_APPLE_COMPILER_SETTINGS 
	free(ptr);
#endif
#endif
#endif

}

#ifdef USE_CUDA
#include "../CudaHelpers/CudaVariables.h"
using namespace Cuda;
#endif

#ifdef USE_OPENCL
#include "../OpenCLHelpers/OpenCLDeviceVariable.h"
#include "../OpenCLHelpers/OpenCLHelpers.h"
using namespace OpenCL;
#endif

MemoryPool* MemoryPool::_instance = NULL;
std::mutex MemoryPool::_mutex;

BufferAllocation::BufferAllocation() :
	BufferType(), Width(), Height(), Depth(), Pitch(), TypeSize(1), Ptr(NULL), AllocatedPtr(NULL), AllocCounter(0)
{
}

BufferAllocation::BufferAllocation(size_t aWidth, size_t aHeight, size_t aDepth, size_t aPitch, size_t aTypeSize, void * aPtr, void * aAllocatedPtr, BufferType_enum aBufferType) :
	BufferType(aBufferType), Width(aWidth), Height(aHeight), Depth(aDepth), Pitch(aPitch), TypeSize(aTypeSize), Ptr(aPtr), AllocatedPtr(aAllocatedPtr), AllocCounter(1)
{
}

MemoryPool * MemoryPool::Get()
{
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_instance)
	{
		_instance = new MemoryPool();
		MKLOG("MemoryPool created.");
	}

	return _instance;
}

void MemoryPool::Allocate(BufferRequestSet & aRequestSet)
{
	std::lock_guard<std::mutex> lock(_mutex);

	vector<shared_ptr<BufferRequest> > requests = aRequestSet.GetRequests();
	vector<size_t> ignoreList;

	for (size_t i = 0; i < requests.size(); i++)
	{
		BufferAllocation* buffer = AllocationNeeded(requests[i], ignoreList);

		if (!buffer) //Need to allocate the memory
		{
			switch (requests[i]->mBufferType)
			{
			case BT_DefaultHost:
				ignoreList.push_back(AllocateDefault(requests[i]));
				break;
#ifdef USE_CUDA
			case BT_CudaHost:
				ignoreList.push_back(AllocateCudaHost(requests[i]));
				break;
			case BT_CudaDevice:
				ignoreList.push_back(AllocateCudaDevice(requests[i]));
				break;
#endif
#ifdef USE_OPENCL
			case BT_OpenCL:
				ignoreList.push_back(AllocateOpenCL(requests[i]));
				break;
#endif
			default:
				break;
			}			
		}
		else
		{

			requests[i]->mAllocatedPitch = buffer->Pitch;
			requests[i]->mAllocatedSizeInBytes = buffer->Pitch * buffer->Height * buffer->Depth;
			requests[i]->mAllocatedPtr = buffer->AllocatedPtr;
			requests[i]->mPtr = buffer->Ptr;
			buffer->AllocCounter++;

			MKLOG("Found reusable Buffer: ");
			MKLOG("Allocated Size: %ld", buffer->Pitch * buffer->Height * buffer->Depth);
			MKLOG("Allocated Pitch: %ld", buffer->Pitch);
			MKLOG("Alloc counter: %ld", buffer->AllocCounter);
		}
		
	}
}

void MemoryPool::Allocate(std::shared_ptr<BufferRequest> request)
{
	BufferRequestSet set;
	set.AddBufferRequest(request);
	Allocate(set);
}

void MemoryPool::FreeAllocations(BufferRequestSet & requestSet)
{
	std::lock_guard<std::mutex> lock(_mutex);

	vector<shared_ptr<BufferRequest> > requests = requestSet.GetRequests();

	for (size_t request = 0; request < requests.size(); request++)
	{
		for (size_t i = 0; i < mAllocationPool.size(); i++)
		{
			if (mAllocationPool[i].AllocatedPtr == requests[request]->mAllocatedPtr && mAllocationPool[i].BufferType == requests[request]->mBufferType)
			{
				mAllocationPool[i].AllocCounter--;

				MKLOG("Freed: %ld bytes", mAllocationPool[i].Pitch * mAllocationPool[i].Height * mAllocationPool[i].Depth);
				MKLOG("New counter: %ld", mAllocationPool[i].AllocCounter);

				if (mAllocationPool[i].AllocCounter <= 0)
				{
					MKLOG("counter dropped to 0: FREE entirely!");
					if (mAllocationPool[i].BufferType == BT_DefaultHost)
					{
						aligned_free(mAllocationPool[i].AllocatedPtr);
					}
					else
					{
#ifdef USE_CUDA
						if (mAllocationPool[i].BufferType == BT_CudaDevice)
						{
							auto erg = Cuda::CudaThreadBoundContext::Run(cuMemFree, (CUdeviceptr)(mAllocationPool[i].AllocatedPtr));
							cudaSafeCall(erg.get());
						}
						else if (mAllocationPool[i].BufferType == BT_CudaHost)
						{
							//TODO!!
						}
#endif // USE_CUDA
#ifdef USE_OPENCL
						if (mAllocationPool[i].BufferType == BT_OpenCL)
						{
							auto erg = RunInOpenCLThread(clReleaseMemObject, (cl_mem)(mAllocationPool[i].AllocatedPtr));
							openCLSafeCall(erg.get());
						}
#endif // USE_OPENCL
					}
					mAllocationPool.erase(mAllocationPool.begin() + i);
				}
			}
		}
	}
}

void MemoryPool::FreeAllocation(std::shared_ptr<BufferRequest> request)
{
	BufferRequestSet set;
	set.AddBufferRequest(request);
	FreeAllocations(set);
}

BufferAllocation * MemoryPool::AllocationNeeded(std::shared_ptr<BufferRequest> aRequest, std::vector<size_t>& aIgnoreList)
{
	bool foundEntry = false;
	size_t BestWastedBytes = SIZE_MAX;
	long64 bestIndex = -1;

	for (size_t i = 0; i < mAllocationPool.size(); i++)
	{
		bool ignoreThisIndex = false;
		for (size_t ignore = 0; ignore < aIgnoreList.size(); ignore++)
		{
			if (i == aIgnoreList[ignore])
			{
				MKLOG("Ignoring index %ld (%ld)", i, ignore);
				ignoreThisIndex = true;
				break;
			}
		}

		if (mAllocationPool[i].BufferType == aRequest->mBufferType && !ignoreThisIndex)
		{
			//default 1D case:
			if (aRequest->mHeight == 1 && aRequest->mDepth == 1)
			{
				if (aRequest->mRequestedSizeInBytes <= mAllocationPool[i].Pitch * mAllocationPool[i].Height * mAllocationPool[i].Depth)
				{
					size_t wastedBytes = (mAllocationPool[i].Pitch * mAllocationPool[i].Height * mAllocationPool[i].Depth) - aRequest->mRequestedSizeInBytes;
					if (wastedBytes < BestWastedBytes)
					{
						BestWastedBytes = wastedBytes;
						bestIndex = i;
						foundEntry = true;
						MKLOG("Found candidate at index %ld with %ld wasted bytes.", i, wastedBytes);
					}
				}
			}
			//2D case:
			else if (aRequest->mDepth == 1)
			{
				if (aRequest->mWidth * aRequest->mTypeSize <= mAllocationPool[i].Pitch && aRequest->mHeight <= mAllocationPool[i].Height * mAllocationPool[i].Depth)
				{
					size_t wastedBytes = (mAllocationPool[i].Pitch * mAllocationPool[i].Height * mAllocationPool[i].Depth) - aRequest->mRequestedSizeInBytes;
					if (wastedBytes < BestWastedBytes)
					{
						BestWastedBytes = wastedBytes;
						bestIndex = i;
						foundEntry = true;
						MKLOG("Found candidate at index %ld with %ld wasted bytes.", i, wastedBytes);
					}
				}
			}
			//3D case:
			else
			{
				if (aRequest->mWidth * aRequest->mTypeSize <= mAllocationPool[i].Pitch && aRequest->mHeight * aRequest->mDepth <= mAllocationPool[i].Height * mAllocationPool[i].Depth)
				{
					size_t wastedBytes = (mAllocationPool[i].Pitch * mAllocationPool[i].Height * mAllocationPool[i].Depth) - aRequest->mRequestedSizeInBytes;
					if (wastedBytes < BestWastedBytes)
					{
						BestWastedBytes = wastedBytes;
						bestIndex = i;
						foundEntry = true;
						MKLOG("Found candidate at index %ld with %ld wasted bytes.", i, wastedBytes);
					}
				}
			}
		}
	}

	if (foundEntry && bestIndex >= 0)
	{
		MKLOG("Choose candidate at index %ld with %ld wasted bytes.", bestIndex, BestWastedBytes);
		aIgnoreList.push_back(bestIndex);
		return &mAllocationPool[bestIndex];
	}

	return NULL;
}

size_t MemoryPool::AllocateDefault(std::shared_ptr<BufferRequest> request)
{
	if (request->mDepth == 1 && request->mHeight == 1)
	{
		size_t pitch = request->mTypeSize * request->mWidth;
		pitch = DivUp(pitch, HOST_PITCH_ALIGNMENT);
		void* ptr = aligned_malloc(pitch, HOST_BYTE_ALIGNMENT);

		if (!ptr)
		{
			throw MemoryException(pitch);
		}
		memset(ptr, 0, pitch);
		BufferAllocation alloc(request->mWidth, 1, 1, pitch, request->mTypeSize, ptr, ptr, request->mBufferType);
		mAllocationPool.push_back(alloc);
		request->mAllocatedPitch = pitch;
		request->mAllocatedSizeInBytes = pitch;
		request->mPtr = ptr;
		request->mAllocatedPtr = ptr;

		MKLOG("Allocated %ld bytes.", pitch);

		return mAllocationPool.size() - 1;
	}
	else if (request->mDepth == 1)
	{
		size_t pitch = request->mTypeSize * request->mWidth;
		pitch = DivUp(pitch, HOST_PITCH_ALIGNMENT);

		void* ptr = aligned_malloc(pitch * request->mHeight, HOST_BYTE_ALIGNMENT);
		if (!ptr)
		{
			throw MemoryException(pitch * request->mHeight);
		}
		memset(ptr, 0, pitch * request->mHeight);
		BufferAllocation alloc(request->mWidth, request->mHeight, 1, pitch, request->mTypeSize, ptr, ptr, request->mBufferType);
		mAllocationPool.push_back(alloc);
		request->mAllocatedPitch = pitch;
		request->mAllocatedSizeInBytes = pitch * request->mHeight;
		request->mPtr = ptr;
		request->mAllocatedPtr = ptr;

		MKLOG("Allocated %ld bytes.", pitch * request->mHeight);
		MKLOG("Pitch %ld bytes.", pitch);

		return mAllocationPool.size() - 1;
	}
	else
	{
		size_t pitch = request->mTypeSize * request->mWidth;
		pitch = DivUp(pitch, HOST_PITCH_ALIGNMENT);

		void* ptr = aligned_malloc(pitch * request->mHeight * request->mDepth, HOST_BYTE_ALIGNMENT);
		if (!ptr)
		{
			throw MemoryException(pitch * request->mHeight * request->mDepth);
		}
		memset(ptr, 0, pitch * request->mHeight * request->mDepth);
		BufferAllocation alloc(request->mWidth, request->mHeight, request->mDepth, pitch, request->mTypeSize, ptr, ptr, request->mBufferType);
		mAllocationPool.push_back(alloc);
		request->mAllocatedPitch = pitch;
		request->mAllocatedSizeInBytes = pitch * request->mHeight * request->mDepth;
		request->mPtr = ptr;
		request->mAllocatedPtr = ptr;

		MKLOG("Allocated %ld bytes.", pitch * request->mHeight * request->mDepth);
		MKLOG("Pitch %ld bytes.", pitch);

		return mAllocationPool.size() - 1;
	}	
}

#ifdef USE_CUDA
size_t MemoryPool::AllocateCudaDevice(std::shared_ptr<BufferRequest> request)
{
	if (request->mDepth == 1 && request->mHeight == 1)
	{
		size_t pitch = request->mTypeSize * request->mWidth;

		CUdeviceptr ptr = (CUdeviceptr)NULL;
		auto erg = Cuda::CudaThreadBoundContext::Run(cuMemAlloc, &ptr, pitch);
		cudaSafeCall(erg.get());

		if (!ptr)
		{
			throw MemoryException(pitch);
		}

		erg = Cuda::CudaThreadBoundContext::Run(cuMemsetD8, ptr, 0, pitch);
		cudaSafeCall(erg.get());

		BufferAllocation alloc(request->mWidth, 1, 1, pitch, request->mTypeSize, (void*)ptr, (void*)ptr, request->mBufferType);
		mAllocationPool.push_back(alloc);
		request->mAllocatedPitch = pitch;
		request->mAllocatedSizeInBytes = pitch;
		request->mPtr = (void*)ptr;
		request->mAllocatedPtr = (void*)ptr;

		MKLOG("Allocated %ld bytes.", pitch);

		return mAllocationPool.size() - 1;
	}
	else if (request->mDepth == 1)
	{
		size_t pitch = 0;
		
		CUdeviceptr ptr = (CUdeviceptr)NULL;

		if (Cuda::CudaPitchedDeviceVariable::IsReallyPitched((uint)request->mTypeSize))
		{
			auto erg = Cuda::CudaThreadBoundContext::Run(cuMemAllocPitch, &ptr, &pitch, request->mTypeSize * request->mWidth, request->mHeight, (uint)request->mTypeSize);
			cudaSafeCall(erg.get());
		}
		else
		{
			pitch = request->mTypeSize * request->mWidth;
			auto erg = Cuda::CudaThreadBoundContext::Run(cuMemAlloc, &ptr, request->mTypeSize * request->mWidth * request->mHeight);
			cudaSafeCall(erg.get());
		}		
		
		if (!ptr)
		{
			throw MemoryException(pitch * request->mHeight);
		}

		auto erg = Cuda::CudaThreadBoundContext::Run(cuMemsetD2D8_v2, ptr, pitch, 0, request->mTypeSize * request->mWidth, request->mHeight);
		cudaSafeCall(erg.get());

		BufferAllocation alloc(request->mWidth, request->mHeight, 1, pitch, request->mTypeSize, (void*)ptr, (void*)ptr, request->mBufferType);
		mAllocationPool.push_back(alloc);
		request->mAllocatedPitch = pitch;
		request->mAllocatedSizeInBytes = pitch * request->mHeight;
		request->mPtr = (void*)ptr;
		request->mAllocatedPtr = (void*)ptr;

		MKLOG("Allocated %ld bytes.", pitch * request->mHeight);
		MKLOG("Pitch %ld bytes.", pitch);

		return mAllocationPool.size() - 1;
	}
	else
	{
		size_t pitch = 0;

		CUdeviceptr ptr = (CUdeviceptr)NULL;
		if (Cuda::CudaPitchedDeviceVariable::IsReallyPitched((uint)request->mTypeSize))
		{
			auto erg = Cuda::CudaThreadBoundContext::Run(cuMemAllocPitch, &ptr, &pitch, request->mTypeSize * request->mWidth, request->mHeight * request->mDepth, (uint)request->mTypeSize);
			cudaSafeCall(erg.get());
		}
		else
		{
			pitch = request->mTypeSize * request->mWidth;
			auto erg = Cuda::CudaThreadBoundContext::Run(cuMemAlloc, &ptr, request->mTypeSize * request->mWidth * request->mHeight * request->mDepth);
			cudaSafeCall(erg.get());
		}

		if (!ptr)
		{
			throw MemoryException(pitch * request->mHeight * request->mDepth);
		}

		auto erg = Cuda::CudaThreadBoundContext::Run(cuMemsetD2D8_v2, ptr, pitch, 0, request->mTypeSize * request->mWidth, request->mHeight * request->mDepth);
		cudaSafeCall(erg.get());

		BufferAllocation alloc(request->mWidth, request->mHeight, request->mDepth, pitch, request->mTypeSize, (void*)ptr, (void*)ptr, request->mBufferType);
		mAllocationPool.push_back(alloc);
		request->mAllocatedPitch = pitch;
		request->mAllocatedSizeInBytes = pitch * request->mHeight * request->mDepth;
		request->mPtr = (void*)ptr;
		request->mAllocatedPtr = (void*)ptr;

		MKLOG("Allocated %ld bytes.", pitch * request->mHeight * request->mDepth);
		MKLOG("Pitch %ld bytes.", pitch);

		return mAllocationPool.size() - 1;
	}
}
#endif

#ifdef USE_CUDA
size_t MemoryPool::AllocateCudaHost(std::shared_ptr<BufferRequest> request)
{
	throw runtime_error("Not Implemented!");
}
#endif

#ifdef USE_OPENCL
cl_mem AllocMemOpenCL(size_t aSize, cl_int* errCode)
{
	cl_mem ret = clCreateBuffer(OpenCL::OpenCLThreadBoundContext::GetCtx(), CL_MEM_READ_WRITE, aSize, NULL, errCode);
	uchar value = 0;
	if (*errCode == 0)
	{
		*errCode = clEnqueueFillBuffer(OpenCLThreadBoundContext::GetQueue(), ret, &value, 1, 0, aSize, 0, NULL, NULL);
	}
	return ret;
}

size_t MemoryPool::AllocateOpenCL(std::shared_ptr<BufferRequest> request)
{
	size_t pitch = 0;

	cl_mem ptr = NULL;
	
	pitch = request->mTypeSize * request->mWidth;

	cl_int errCode;
	auto erg = RunInOpenCLThread(AllocMemOpenCL, request->mTypeSize * request->mWidth * request->mHeight * request->mDepth, &errCode);
	ptr = erg.get();
	openCLSafeCall(errCode);	

	if (!ptr)
	{
		throw MemoryException(pitch * request->mHeight * request->mDepth);
	}

	BufferAllocation alloc(request->mWidth, request->mHeight, request->mDepth, pitch, request->mTypeSize, (void*)ptr, (void*)ptr, request->mBufferType);
	mAllocationPool.push_back(alloc);
	request->mAllocatedPitch = pitch;
	request->mAllocatedSizeInBytes = pitch * request->mHeight * request->mDepth;
	request->mPtr = (void*)ptr;
	request->mAllocatedPtr = (void*)ptr;

	MKLOG("Allocated %ld bytes.", pitch * request->mHeight * request->mDepth);
	MKLOG("Pitch %ld bytes.", pitch);

	return mAllocationPool.size() - 1;
}
#endif

size_t MemoryPool::DivUp(size_t x, size_t d)
{
	return d * ((x + d - 1) / d);
}