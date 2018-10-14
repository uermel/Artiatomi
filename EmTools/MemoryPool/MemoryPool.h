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


#ifndef MEMORYPOOL_H
#define MEMORYPOOL_H

#include "../Basics/Default.h"
#include "BufferRequest.h"
#include "BufferRequestSet.h"
#include "MemoryException.h"
#include <mutex>

#define HOST_PITCH_ALIGNMENT 4
#define HOST_BYTE_ALIGNMENT 16 //for SSE

#ifdef USE_CUDA
#include "../CudaHelpers/CudaHelpers.h"
#endif


class BufferAllocation
{
public:
	BufferAllocation();
	BufferAllocation(size_t aWidth, size_t aHeight, size_t aDepth, size_t aPitch, size_t aTypeSize, void* aPtr, void* aAllocatedPtr, BufferType_enum aBufferType);

	BufferType_enum BufferType;
	size_t Width;
	size_t Height;
	size_t Depth;
	size_t Pitch;
	size_t TypeSize;
	void*  Ptr;
	void*  AllocatedPtr; //Currently they are the same, but this might change in future...
	long64 AllocCounter;
};

class MemoryPool
{
public:
	static MemoryPool* Get();

	void Allocate(BufferRequestSet& aRequestSet);
	void Allocate(std::shared_ptr<BufferRequest> request);
	void FreeAllocations(BufferRequestSet& requests);
	void FreeAllocation(std::shared_ptr<BufferRequest> request);

private:
	static std::mutex _mutex;
	static MemoryPool* _instance;

	MemoryPool() = default;
	MemoryPool(const MemoryPool&) = delete;
	MemoryPool(MemoryPool&&) = delete;

	std::vector<BufferAllocation> mAllocationPool;

	BufferAllocation* AllocationNeeded(std::shared_ptr<BufferRequest> aRequest, std::vector<size_t>& aIgnoreList);

	size_t AllocateDefault(std::shared_ptr<BufferRequest> request);
#ifdef USE_CUDA
	size_t AllocateCudaDevice(std::shared_ptr<BufferRequest> request);
	size_t AllocateCudaHost(std::shared_ptr<BufferRequest> request);
#endif
#ifdef USE_OPENCL
	size_t AllocateOpenCL(std::shared_ptr<BufferRequest> request);
#endif
	size_t DivUp(size_t x, size_t d);
};

#endif
