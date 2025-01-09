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


#include "NPPImageBase.h"
#include "CudaException.h"

using namespace Cuda;

NPPImageBase::~NPPImageBase()
{
	if (_isOwner)
	{
		nppiFree((void*)_devPtr);
		_devPtr = 0;
		_devPtrRoi = 0;
	}
}

NPPImageBase::NPPImageBase(CUdeviceptr devPtr, int aWidth, int aHeight, int aTypeSize, int aPitch, int aChannels, bool aIsOwner):
	_devPtr(devPtr),
	_devPtrRoi(devPtr),
	_sizeOriginal{ aWidth, aHeight },
	_sizeRoi{aWidth, aHeight},
	_pointRoi{0, 0},
	_pitch(aPitch),
	_channels(aChannels),
	_typeSize(aTypeSize),
	_isOwner(aIsOwner)
{
}

void NPPImageBase::CopyToDevice(void * hostSrc, size_t stride)
{
	CUDA_MEMCPY2D copyParams;
	memset(&copyParams, 0, sizeof(copyParams));
	copyParams.srcHost = hostSrc;
	copyParams.srcPitch = stride;
	copyParams.srcMemoryType = CU_MEMORYTYPE_HOST;
	copyParams.dstDevice = _devPtr;
	copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstPitch = _pitch;
	copyParams.Height = _sizeOriginal.height;
	copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

	cudaSafeCall(cuMemcpy2D(&copyParams));
}

void NPPImageBase::CopyToDevice(Cuda::CudaPitchedDeviceVariable & deviceSrc)
{
	CUDA_MEMCPY2D copyParams;
	memset(&copyParams, 0, sizeof(copyParams));
	copyParams.srcDevice = deviceSrc.GetDevicePtr();
	copyParams.srcPitch = deviceSrc.GetPitch();
	copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstDevice = _devPtr;
	copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstPitch = _pitch;
	copyParams.Height = _sizeOriginal.height;
	copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

	cudaSafeCall(cuMemcpy2D(&copyParams));
}

void NPPImageBase::CopyToDevice(NPPImageBase & deviceSrc)
{
	CUDA_MEMCPY2D copyParams;
	memset(&copyParams, 0, sizeof(copyParams));
	copyParams.srcDevice = deviceSrc.GetDevicePointer();
	copyParams.srcPitch = deviceSrc.GetPitch();
	copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstDevice = _devPtr;
	copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstPitch = _pitch;
	copyParams.Height = _sizeOriginal.height;
	copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

	cudaSafeCall(cuMemcpy2D(&copyParams));
}

void NPPImageBase::CopyToDevice(Cuda::CudaDeviceVariable & deviceSrc)
{
	CUDA_MEMCPY2D copyParams;
	memset(&copyParams, 0, sizeof(copyParams));
	copyParams.srcDevice = deviceSrc.GetDevicePtr();
	copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstDevice = _devPtr;
	copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstPitch = _pitch;
	copyParams.Height = _sizeOriginal.height;
	copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

	cudaSafeCall(cuMemcpy2D(&copyParams));
}

void NPPImageBase::CopyToDevice(CUdeviceptr deviceSrc)
{
	CUDA_MEMCPY2D copyParams;
	memset(&copyParams, 0, sizeof(copyParams));
	copyParams.srcDevice = deviceSrc;
	copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstDevice = _devPtr;
	copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstPitch = _pitch;
	copyParams.Height = _sizeOriginal.height;
	copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

	cudaSafeCall(cuMemcpy2D(&copyParams));
}

void NPPImageBase::CopyToDevice(CUdeviceptr deviceSrc, size_t pitch)
{
	CUDA_MEMCPY2D copyParams;
	memset(&copyParams, 0, sizeof(copyParams));
	copyParams.srcDevice = deviceSrc;
	copyParams.srcPitch = pitch;
	copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstDevice = _devPtr;
	copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstPitch = _pitch;
	copyParams.Height = _sizeOriginal.height;
	copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

	cudaSafeCall(cuMemcpy2D(&copyParams));
}

void NPPImageBase::CopyToDevice(void * hostSrc)
{
	CUDA_MEMCPY2D copyParams;
	memset(&copyParams, 0, sizeof(copyParams));
	copyParams.srcHost = hostSrc;
	copyParams.srcMemoryType = CU_MEMORYTYPE_HOST;
	copyParams.dstDevice = _devPtr;
	copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstPitch = _pitch;
	copyParams.Height = _sizeOriginal.height;
	copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

	cudaSafeCall(cuMemcpy2D(&copyParams));
}

void NPPImageBase::CopyToHost(void * hostDest)
{
	CUDA_MEMCPY2D copyParams;
	memset(&copyParams, 0, sizeof(copyParams));
	copyParams.srcDevice = _devPtr;
	copyParams.srcPitch = _pitch;
	copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstHost = hostDest;
	copyParams.dstMemoryType = CU_MEMORYTYPE_HOST;
	copyParams.Height = _sizeOriginal.height;
	copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

	cudaSafeCall(cuMemcpy2D(&copyParams));
}

void NPPImageBase::CopyToHost(void * hostDest, size_t stride)
{
	CUDA_MEMCPY2D copyParams;
	memset(&copyParams, 0, sizeof(copyParams));
	copyParams.srcDevice = _devPtr;
	copyParams.srcPitch = _pitch;
	copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParams.dstHost = hostDest;
	copyParams.dstMemoryType = CU_MEMORYTYPE_HOST;
	copyParams.dstPitch = stride;
	copyParams.Height = _sizeOriginal.height;
	copyParams.WidthInBytes = _sizeOriginal.width * _typeSize * _channels;

	cudaSafeCall(cuMemcpy2D(&copyParams));
}

NppiSize NPPImageBase::GetSize()
{
	return _sizeOriginal;
}

NppiSize NPPImageBase::GetSizeRoi()
{
	return _sizeRoi;
}

NppiPoint NPPImageBase::GetPointRoi()
{
	return _pointRoi;
}

CUdeviceptr NPPImageBase::GetDevicePointer()
{
	return _devPtr;
}

CUdeviceptr NPPImageBase::GetDevicePointerRoi()
{
	return _devPtrRoi;
}

int NPPImageBase::GetWidth()
{
	return _sizeOriginal.width;
}

int NPPImageBase::GetWidthInBytes()
{
	return _sizeOriginal.width * _typeSize * _channels;
}

int NPPImageBase::GetHeight()
{
	return _sizeOriginal.height;
}

int NPPImageBase::GetWidthRoi()
{
	return _sizeRoi.width;
}

int NPPImageBase::GetWidthRoiInBytes()
{
	return _sizeRoi.width * _typeSize * _channels;
}

int NPPImageBase::GetHeightRoi()
{
	return _sizeRoi.height;
}

int NPPImageBase::GetPitch()
{
	return _pitch;
}

int NPPImageBase::GetTotalSizeInBytes()
{
	return _pitch * _sizeOriginal.height;
}

int NPPImageBase::GetChannels()
{
	return _channels;
}

void NPPImageBase::SetRoi(NppiRect roi)
{
	_devPtrRoi = _devPtr + _typeSize * _channels * roi.x + _pitch * roi.y;
	_pointRoi = NppiPoint{ roi.x, roi.y };
	_sizeRoi = NppiSize{ roi.width, roi.height };
}

void NPPImageBase::SetRoi(int x, int y, int width, int height)
{
	SetRoi(NppiRect{ x, y, width, height });
}

void NPPImageBase::ResetRoi()
{
	_devPtrRoi = _devPtr;
	_pointRoi = NppiPoint{ 0, 0 };
	_sizeRoi = _sizeOriginal;
}
