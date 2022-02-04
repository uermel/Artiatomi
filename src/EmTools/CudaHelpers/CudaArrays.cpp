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


#include "CudaArrays.h"

#ifdef USE_CUDA
namespace Cuda
{
	CudaArray1D::CudaArray1D(CUarray_format aFormat, size_t aSizeInElements, uint aNumChannels)
		: mCUarray(0)
	{
		if (!(aNumChannels == 1 || aNumChannels == 2 || aNumChannels == 4))
		{
			CudaException ex("Number of channels for a CUDA array must be 1, 2 or 4.");
		}

		memset(&mDescriptor, 0, sizeof(mDescriptor));
		mDescriptor.Format = aFormat;
		mDescriptor.Height = 0;
		mDescriptor.Width = aSizeInElements;
		mDescriptor.NumChannels = aNumChannels;

		cudaSafeCall(cuArrayCreate(&mCUarray, &mDescriptor));
	}

	CudaArray1D::~CudaArray1D()
	{
		cuArrayDestroy(mCUarray);
	}

	void CudaArray1D::CopyFromDeviceToArray(CudaDeviceVariable& aSource, size_t aOffsetInBytes)
	{
		cudaSafeCall(cuMemcpyDtoA(mCUarray, aOffsetInBytes, aSource.GetDevicePtr(), aSource.GetSize()));
	}
	void CudaArray1D::CopyFromArrayToDevice(CudaDeviceVariable& aDest, size_t aOffsetInBytes)
	{
		cudaSafeCall(cuMemcpyAtoD(aDest.GetDevicePtr(), mCUarray, aOffsetInBytes, aDest.GetSize()));
	}

	void CudaArray1D::CopyFromHostToArray(void* aSource, size_t aOffsetInBytes)
	{
		cudaSafeCall(cuMemcpyHtoA(mCUarray, aOffsetInBytes, aSource, mDescriptor.Width * GetChannelSize(mDescriptor.Format) * mDescriptor.NumChannels));
	}
	void CudaArray1D::CopyFromArrayToHost(void* aDest, size_t aOffsetInBytes)
	{
		cudaSafeCall(cuMemcpyAtoH(aDest, mCUarray, aOffsetInBytes, mDescriptor.Width * GetChannelSize(mDescriptor.Format) * mDescriptor.NumChannels));
	}

	CUDA_ARRAY_DESCRIPTOR CudaArray1D::GetArrayDescriptor()
	{
		return mDescriptor;
	}

	CUarray CudaArray1D::GetCUarray()
	{
		return mCUarray;
	}



	CudaArray2D::CudaArray2D(CUarray_format aFormat, size_t aWidthInElements, size_t aHeightInElements, uint aNumChannels)
		: mCUarray(0)
	{
		if (!(aNumChannels == 1 || aNumChannels == 2 || aNumChannels == 4))
		{
			CudaException ex("Number of channels for a CUDA array must be 1, 2 or 4.");
		}

		memset(&mDescriptor, 0, sizeof(mDescriptor));
		mDescriptor.Format = aFormat;
		mDescriptor.Height = aHeightInElements;
		mDescriptor.Width = aWidthInElements;
		mDescriptor.NumChannels = aNumChannels;

		cudaSafeCall(cuArrayCreate(&mCUarray, &mDescriptor));
	}
	CudaArray2D::~CudaArray2D()
	{
		cuArrayDestroy(mCUarray);
	}

	void CudaArray2D::CopyFromDeviceToArray(CudaPitchedDeviceVariable& aSource)
	{
		CUDA_MEMCPY2D params;
		memset(&params, 0, sizeof(params));
		params.srcDevice = aSource.GetDevicePtr();
		params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		params.srcPitch = aSource.GetPitch();
		params.dstArray = mCUarray;
		params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Height = aSource.GetHeight();
		params.WidthInBytes = aSource.GetWidthInBytes();

		cudaSafeCall(cuMemcpy2D(&params));
	}
	void CudaArray2D::CopyFromArrayToDevice(CudaPitchedDeviceVariable& aDest)
	{
		CUDA_MEMCPY2D params;
		memset(&params, 0, sizeof(params));
		params.dstDevice = aDest.GetDevicePtr();
		params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		params.dstPitch = aDest.GetPitch();
		params.srcArray = mCUarray;
		params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Height = aDest.GetHeight();
		params.WidthInBytes = aDest.GetWidthInBytes();

		cudaSafeCall(cuMemcpy2D(&params));
	}

	void CudaArray2D::CopyFromHostToArray(void* aSource)
	{
		CUDA_MEMCPY2D params;
		memset(&params, 0, sizeof(params));
		params.srcHost = aSource;
		params.srcMemoryType = CU_MEMORYTYPE_HOST;
		params.srcPitch = mDescriptor.Width * mDescriptor.NumChannels * GetChannelSize(mDescriptor.Format);
		params.dstArray = mCUarray;
		params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Height = mDescriptor.Height;
		params.WidthInBytes = params.srcPitch;

		cudaSafeCall(cuMemcpy2D(&params));
	}

	void CudaArray2D::CopyFromArrayToHost(void* aDest)
	{
		CUDA_MEMCPY2D params;
		memset(&params, 0, sizeof(params));
		params.dstHost = aDest;
		params.dstMemoryType = CU_MEMORYTYPE_HOST;
		params.dstPitch = mDescriptor.Width * mDescriptor.NumChannels * GetChannelSize(mDescriptor.Format);
		params.srcArray = mCUarray;
		params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Height = mDescriptor.Height;
		params.WidthInBytes = params.dstPitch;

		cudaSafeCall(cuMemcpy2D(&params));
	}

	CUDA_ARRAY_DESCRIPTOR CudaArray2D::GetArrayDescriptor()
	{
		return mDescriptor;
	}

	CUarray CudaArray2D::GetCUarray()
	{
		return mCUarray;
	}



	CudaArray3D::CudaArray3D(CUarray_format aFormat, size_t aWidthInElements, size_t aHeightInElements, size_t aDepthInElements, uint aNumChannels, uint aFlags)
		: mCUarray(0)
	{
		if (!(aNumChannels == 1 || aNumChannels == 2 || aNumChannels == 4))
		{
			CudaException ex("Number of channels for a CUDA array must be 1, 2 or 4.");
		}

		memset(&mDescriptor, 0, sizeof(mDescriptor));
		mDescriptor.Format = aFormat;
		mDescriptor.Height = aHeightInElements;
		mDescriptor.Width = aWidthInElements;
		mDescriptor.Depth = aDepthInElements;
		mDescriptor.NumChannels = aNumChannels;
		mDescriptor.Flags = aFlags;

		cudaSafeCall(cuArray3DCreate(&mCUarray, &mDescriptor));
	}
	CudaArray3D::~CudaArray3D()
	{
		cuArrayDestroy(mCUarray);
	}

	void CudaArray3D::CopyFromDeviceToArray(CudaDeviceVariable& aSource)
	{
		CUDA_MEMCPY3D params;
		memset(&params, 0, sizeof(params));
		params.srcDevice = aSource.GetDevicePtr();
		params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        //params.srcHeight = mDescriptor.Height;
		params.srcPitch = mDescriptor.Width * mDescriptor.NumChannels * GetChannelSize(mDescriptor.Format);
		params.dstArray = mCUarray;
		params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Depth = mDescriptor.Depth;
		params.Height = mDescriptor.Height;
		params.WidthInBytes = params.srcPitch;

		cudaSafeCall(cuMemcpy3D_v2(&params));
	}
	void CudaArray3D::CopyFromArrayToDevice(CudaDeviceVariable& aDest)
	{
		CUDA_MEMCPY3D params;
		memset(&params, 0, sizeof(params));
		params.dstDevice = aDest.GetDevicePtr();
		params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		params.dstPitch = mDescriptor.Width * mDescriptor.NumChannels * GetChannelSize(mDescriptor.Format);
		params.srcArray = mCUarray;
		params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Depth = mDescriptor.Depth;
		params.Height = mDescriptor.Height;
		params.WidthInBytes = params.srcPitch;

		cudaSafeCall(cuMemcpy3D(&params));
	}

	void CudaArray3D::CopyFromDeviceToArray(CudaPitchedDeviceVariable& aSource)
	{
		CUDA_MEMCPY3D params;
		memset(&params, 0, sizeof(params));
		params.srcDevice = aSource.GetDevicePtr();
		params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		params.srcPitch = aSource.GetPitch();
		params.dstArray = mCUarray;
		params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Depth = mDescriptor.Depth;
		params.Height = mDescriptor.Height;
		params.WidthInBytes = params.srcPitch;

		cudaSafeCall(cuMemcpy3D(&params));
	}
	void CudaArray3D::CopyFromArrayToDevice(CudaPitchedDeviceVariable& aDest)
	{
		CUDA_MEMCPY3D params;
		memset(&params, 0, sizeof(params));
		params.dstDevice = aDest.GetDevicePtr();
		params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		params.dstPitch = aDest.GetPitch();
		params.srcArray = mCUarray;
		params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Depth = mDescriptor.Depth;
		params.Height = mDescriptor.Height;
		params.WidthInBytes = params.srcPitch;

		cudaSafeCall(cuMemcpy3D(&params));
	}

	void CudaArray3D::CopyFromHostToArray(void* aSource)
	{
		CUDA_MEMCPY3D params;
		memset(&params, 0, sizeof(params));
		params.srcHost = aSource;
		params.srcMemoryType = CU_MEMORYTYPE_HOST;
		params.srcPitch = mDescriptor.Width * mDescriptor.NumChannels * GetChannelSize(mDescriptor.Format);
		params.dstArray = mCUarray;
		params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Depth = mDescriptor.Depth;
		params.Height = mDescriptor.Height;
		params.WidthInBytes = params.srcPitch;

		cudaSafeCall(cuMemcpy3D(&params));
	}
	void CudaArray3D::CopyFromArrayToHost(void* aDest)
	{
		CUDA_MEMCPY3D params;
		memset(&params, 0, sizeof(params));
		params.dstHost = aDest;
		params.dstMemoryType = CU_MEMORYTYPE_HOST;
		params.dstPitch = mDescriptor.Width * mDescriptor.NumChannels * GetChannelSize(mDescriptor.Format);
		params.srcArray = mCUarray;
		params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Depth = mDescriptor.Depth;
		params.Height = mDescriptor.Height;
		params.WidthInBytes = params.dstPitch;

		cudaSafeCall(cuMemcpy3D(&params));
	}

	CUDA_ARRAY3D_DESCRIPTOR CudaArray3D::GetArrayDescriptor()
	{
		return mDescriptor;
	}

	CUarray CudaArray3D::GetCUarray()
	{
		return mCUarray;
	}


	unsigned int GetChannelSize(const CUarray_format aFormat)
	{
		unsigned int result = 0;
		switch(aFormat)
		{
			case CU_AD_FORMAT_FLOAT:
				result = sizeof(float);
				break;
			case CU_AD_FORMAT_HALF:
				result = sizeof(short);
				break;
			case CU_AD_FORMAT_UNSIGNED_INT8:
				result = sizeof(unsigned char);
				break;
			case CU_AD_FORMAT_UNSIGNED_INT16:
				result = sizeof(unsigned short);
				break;
			case CU_AD_FORMAT_UNSIGNED_INT32:
				result = sizeof(unsigned int);
				break;
			case CU_AD_FORMAT_SIGNED_INT8:
				result = sizeof(char);
				break;
			case CU_AD_FORMAT_SIGNED_INT16:
				result = sizeof(short);
				break;
			case CU_AD_FORMAT_SIGNED_INT32:
				result = sizeof(int);
				break;
			default:
				CudaException ex("Unknown texture format");
				throw ex;
				break;
		}

		return result;
	}
}
#endif //USE_CUDA
