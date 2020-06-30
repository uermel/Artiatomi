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


#ifndef BUFFERREQUEST_H
#define BUFFERREQUEST_H

#include "../Basics/Default.h"

enum BufferType_enum
{
	BT_DefaultHost,
#ifdef USE_CUDA
	BT_CudaDevice,
	BT_CudaHost,
#endif
#ifdef USE_OPENCL
	BT_OpenCL,
#endif
};

//forward declarations
class MemoryPool;
class IFilterElementBase;
#ifdef USE_CUDA
namespace Cuda
{
	class CudaDeviceVariable;
	class CudaPitchedDeviceVariable;
}
#endif
#ifdef USE_OPENCL
namespace OpenCL
{
	class OpenCLDeviceVariable;
}
#endif


//! BufferRequest  
/*!
BufferRequest
\author Michael Kunz
\date   October 2016
\version 1.0
*/
class BufferRequest
{
public:
	BufferRequest(BufferType_enum aBufferType, DataType_enum aDataType, size_t aWidth, size_t aHeight = 1, size_t aDepth = 1);

	BufferRequest(const BufferRequest& copy);
	BufferRequest(BufferRequest&& move);
	void* mPtr;
	size_t mAllocatedPitch;

protected:
	BufferType_enum mBufferType;
	size_t mRequestedSizeInBytes;
	size_t mTypeSize;
	DataType_enum mDataType;
	void* mAllocatedPtr;
	size_t mAllocatedSizeInBytes;
	size_t mWidth;
	size_t mHeight;
	size_t mDepth;

	friend class MemoryPool;
	friend class IFilterElementBase;
#ifdef USE_CUDA
	friend class Cuda::CudaDeviceVariable;
	friend class Cuda::CudaPitchedDeviceVariable;
#endif
#ifdef USE_OPENCL
	friend class OpenCL::OpenCLDeviceVariable;
#endif
};

#endif