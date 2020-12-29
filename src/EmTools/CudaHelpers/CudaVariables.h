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


#ifndef CUDAVARIABLES_H
#define CUDAVARIABLES_H

#include "CudaDefault.h"
#ifdef USE_CUDA
#include "CudaException.h"
//#include "../MemoryPool/BufferRequest.h"

namespace Cuda
{
	//!  A wrapper class for a CUDA Device variable.
	/*!
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	//A wrapper class for a CUDA Device variable.
	class CudaDeviceVariable
	{
	private:
		CUdeviceptr mDevPtr; //Wrapped device pointer
		size_t mSizeInBytes; //Size in bytes of the wrapped device variable
		bool mIsOwner;

	public:
		//! CudaDeviceVariable constructor
		/*!
			Allocates \p aSizeInBytes bytes in device memory
			\param aSizeInBytes Amount of memory to allocate
		*/
		//CudaDeviceVariable constructor
		CudaDeviceVariable(size_t aSizeInBytes);
		CudaDeviceVariable(const CUdeviceptr& aDevPtr, bool aIsOwner);
		CudaDeviceVariable(const CudaDeviceVariable& aDevVar, bool aIsOwner);
		CudaDeviceVariable(const CudaDeviceVariable& aDevVar) = delete;
		CudaDeviceVariable(CudaDeviceVariable&& aDevVar);

		//! Initializes the object but doesn't allocate GPU memory. Inner ptr is 0;
		CudaDeviceVariable();

		//! Reallocates the inner CUdeviceptr. If inner ptr isn't 0 it is freed before. If not isOwner it takes over ownership
		void Alloc(size_t aSizeInBytes);

		//! CudaDeviceVariable destructor
		//CudaDeviceVariable destructor
		~CudaDeviceVariable();

		//! Copy data from device memory to this CudaDeviceVariable
		/*!
			\param aSource Data source in device memory
		*/
		//Copy data from device memory to this CudaDeviceVariable
		void CopyDeviceToDevice(CUdeviceptr aSource);

		//! Copy data from device memory to this CudaDeviceVariable
		/*!
			\param aSource Data source in device memory
		*/
		//Copy data from device memory to this CudaDeviceVariable
		void CopyDeviceToDevice(CudaDeviceVariable& aSource);


		//! Copy data from host memory to this CudaDeviceVariable
		/*!
			\param aSource Data source in host memory
			\param aSizeInBytes Number of bytes to copy
		*/
		//Copy data from host memory to this CudaDeviceVariable
		void CopyHostToDevice(void* aSource, size_t aSizeInBytes = 0);

		//! Copy data from this CudaDeviceVariable to host memory
		/*!
			\param aDest Data destination in host memory
			\param aSizeInBytes Number of bytes to copy
		*/
		//Copy data from this CudaDeviceVariable to host memory
		void CopyDeviceToHost(void* aDest, size_t aSizeInBytes = 0);

		//! Copy data from device memory to this CudaDeviceVariable
		/*!
			\param aSource Data source in device memory
		*/
		//Copy data from device memory to this CudaDeviceVariable
		void CopyDeviceToDeviceAsync(CUstream stream, CUdeviceptr aSource);

		//! Copy data from device memory to this CudaDeviceVariable
		/*!
			\param aSource Data source in device memory
		*/
		//Copy data from device memory to this CudaDeviceVariable
		void CopyDeviceToDeviceAsync(CUstream stream, CudaDeviceVariable& aSource);


		//! Copy data from host memory to this CudaDeviceVariable
		/*!
			\param aSource Data source in host memory
			\param aSizeInBytes Number of bytes to copy
		*/
		//Copy data from host memory to this CudaDeviceVariable
		void CopyHostToDeviceAsync(CUstream stream, void* aSource, size_t aSizeInBytes = 0);

		//! Copy data from this CudaDeviceVariable to host memory
		/*!
			\param aDest Data destination in host memory
			\param aSizeInBytes Number of bytes to copy
		*/
		//Copy data from this CudaDeviceVariable to host memory
		void CopyDeviceToHostAsync(CUstream stream, void* aDest, size_t aSizeInBytes = 0);

		//! Returns the size in bytes of the allocated device memory
		size_t GetSize();

		//! Returns the wrapped CUdeviceptr
		CUdeviceptr GetDevicePtr();

		//! Sets the allocated memory to \p aValue
		void Memset(uchar aValue);

		//! Sets the allocated memory to \p aValue
		void MemsetAsync(CUstream stream, uchar aValue);
	};

	//!  A wrapper class for a pitched CUDA Device variable.
	/*!
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	//A wrapper class for a pitched CUDA Device variable.
	class CudaPitchedDeviceVariable
	{
	private:
		CUdeviceptr mDevPtr;  //Wrapped CUdeviceptr
		size_t mSizeInBytes;  //Total size in bytes allocated in device memory
		size_t mPitch;		  //Memory pitch as returned by cuMemAllocPitch
		size_t mWidthInBytes; //Width in bytes of the allocated memory area (<= mPitch)
		size_t mHeight;		  //Height in elements of the allocated memory area
		uint   mElementSize;  //Size of one data element
		bool mIsOwner;
		bool IsReallyPitched();

	public:
		//! CudaPitchedDeviceVariable constructor
		/*!
			Allocates at least \p aHeight x \p aWidthInBytes bytes in device memory
			\param aHeight Height in elements
			\param aWidthInBytes Width in bytes (<= mPitch)
			\param aElementSize Size of one data element
		*/
		//CudaPitchedDeviceVariable constructor
		CudaPitchedDeviceVariable(size_t aWidthInBytes, size_t aHeight, uint aElementSize);
		CudaPitchedDeviceVariable(const CudaPitchedDeviceVariable& aDevVar, bool aIsOwner);
		CudaPitchedDeviceVariable(CUdeviceptr aPtr, size_t aWidthInBytes, size_t aHeight, size_t aPitch, uint aElementSize, bool aIsOwner = false);
		CudaPitchedDeviceVariable(const CudaPitchedDeviceVariable& aDevVar) = delete;
		CudaPitchedDeviceVariable(CudaPitchedDeviceVariable&& aDevVar);

		//! Initializes the object but doesn't allocate GPU memory. Inner ptr is 0;
		CudaPitchedDeviceVariable();

		//! Reallocates the inner CUdeviceptr. If inner ptr isn't 0 it is freed before. If not isOwner it takes over ownership
		void Alloc(size_t aWidthInBytes, size_t aHeight, uint aElementSize);

		//! CudaPitchedDeviceVariable destructor
		//CudaPitchedDeviceVariable destructor
		~CudaPitchedDeviceVariable();

		//! Copy data from device memory to this CudaPitchedDeviceVariable
		/*!
			\param aSource Data source in device memory
		*/
		//Copy data from device memory to this CudaPitchedDeviceVariable
		//void CopyDeviceToDevice(CUdeviceptr aSource);

		//! Copy data from device memory to this CudaPitchedDeviceVariable
		/*!
			\param aSource Data source in device memory
		*/
		//Copy data from device memory to this CudaPitchedDeviceVariable
		void CopyDeviceToDevice(CudaPitchedDeviceVariable& aSource);


		//! Copy data from host memory to this CudaPitchedDeviceVariable
		/*!
			\param aSource Data source in host memory
		*/
		//Copy data from host memory to this CudaPitchedDeviceVariable
		void CopyHostToDevice(void* aSource);

		//! Copy data from this CudaPitchedDeviceVariable to host memory
		/*!
			\param aDest Data destination in host memory
		*/
		//Copy data from this CudaPitchedDeviceVariable to host memory
		void CopyDeviceToHost(void* aDest);

		//! Copy data from device memory to this CudaPitchedDeviceVariable
		/*!
			\param aSource Data source in device memory
		*/
		//Copy data from device memory to this CudaPitchedDeviceVariable
		void CopyDeviceToDeviceAsync(CUstream stream, CudaPitchedDeviceVariable& aSource);


		//! Copy data from host memory to this CudaPitchedDeviceVariable
		/*!
			\param aSource Data source in host memory
		*/
		//Copy data from host memory to this CudaPitchedDeviceVariable
		void CopyHostToDeviceAsync(CUstream stream, void* aSource);

		//! Copy data from this CudaPitchedDeviceVariable to host memory
		/*!
			\param aDest Data destination in host memory
		*/
		//Copy data from this CudaPitchedDeviceVariable to host memory
		void CopyDeviceToHostAsync(CUstream stream, void* aDest);

		//! Returns the data size in bytes. NOT the pitched allocated size in device memory
		//Returns the data size in bytes. NOT the pitched allocated size in device memory
		size_t GetSize();

		//! Return the allocation pitch
		//Return the allocation pitch
		size_t GetPitch();

		//! Returns the data element size
		//Returns the data element size
		uint GetElementSize();

		//! Return the width in elements
		//Return the width in elements
		size_t GetWidth();

		//! Return the width in bytes
		//Return the width in bytes
		size_t GetWidthInBytes();

		//! Return the height in elements
		//Return the height in elements
		size_t GetHeight();

		//! Return the wrapped CUdeviceptr
		//Return the wrapped CUdeviceptr
		CUdeviceptr GetDevicePtr();

		//! Sets the allocated memory to \p aValue
		void Memset(uchar aValue);

		//! Sets the allocated memory to \p aValue
		void MemsetAsync(CUstream stream, uchar aValue);


		static bool IsReallyPitched(uint aElementSize);
	};

	//!  A wrapper class for a Page Locked Host variable.
	/*!
		CudaPageLockedHostVariable are allocated using cuMemAllocHost
		\author Michael Kunz
		\date   January 2010
		\version 1.0
	*/
	class CudaPageLockedHostVariable
	{
	private:
		void* mHostPtr; //Host pointer allocated by cuMemAllocHost
		size_t mSizeInBytes; //Size in bytes

	public:
		//! CudaPageLockedHostVariable constructor
		/*!
			Allocates \p aSizeInBytes bytes in page locked host memory using cuMemAllocHost
			\param aSizeInBytes Number of bytes to allocate
			\param aFlags Allocation flags.
		*/
		//CudaPageLockedHostVariable constructor
		CudaPageLockedHostVariable(size_t aSizeInBytes, uint aFlags);
		CudaPageLockedHostVariable(const CudaPageLockedHostVariable&) = delete;
		CudaPageLockedHostVariable(CudaPageLockedHostVariable&&);

		//!CudaPageLockedHostVariable destructor
		//CudaPageLockedHostVariable destructor
		~CudaPageLockedHostVariable();

		//! Returns the size of the allocated memory area in bytes
		//Returns the size of the allocated memory area in bytes
		size_t GetSize();

		//! Returns the wrapped page locked host memory pointer
		//Returns the wrapped page locked host memory pointer
		void* GetHostPtr();
	};
}
#endif //USE_CUDA
#endif //CUDAVARIABLES_H
