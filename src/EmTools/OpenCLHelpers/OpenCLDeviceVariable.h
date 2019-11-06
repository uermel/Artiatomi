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


#ifndef OPENCLDEVICEVARIABLE_H
#define OPENCLDEVICEVARIABLE_H


#include "../Basics/Default.h"

#ifdef USE_OPENCL
#include "OpenCLException.h"
#include <CL/cl.h>

namespace OpenCL
{
	//!  A wrapper class for an OpenCL Device variable.
	/*!
		The wrapped mDevPtr is only released in destructor, if mIsOwner is true.
		In contrast to the CUDA counterpart, all OpenCL API calls are performed on the unique OpenCL host thread, i.e. all methods can safely be called from any host thread without locking.
	\author Michael Kunz
	\date   November 2016
	\version 1.0
	*/
	//A wrapper class for a OpenCL Device variable.
	class OpenCLDeviceVariable
	{
	private:
		cl_mem mDevPtr; //Wrapped device pointer
		size_t mSizeInBytes; //Size in bytes of the wrapped device variable
		bool mIsOwner; //Destructor only releases mDevPtr if OpenCLDeviVariable is owner.

	public:
		//! OpenCLDeviceVariable constructor
		/*!
		Allocates \p aSizeInBytes bytes in device memory
		\param aSizeInBytes Amount of memory to allocate
		*/
		//OpenCLDeviceVariable constructor
		OpenCLDeviceVariable(size_t aSizeInBytes);
		OpenCLDeviceVariable(size_t aSizeInBytes, bool aIsOwner);
		OpenCLDeviceVariable(const cl_mem& aDevPtr, size_t aSizeInBytes = 0);

		//! OpenCLDeviceVariable destructor
		//OpenCLDeviceVariable destructor
		~OpenCLDeviceVariable();

		//! Copy data from device memory to this OpenCLDeviceVariable
		/*!
		\param aSource Data source in device memory
		*/
		//Copy data from device memory to this OpenCLDeviceVariable
		void CopyDeviceToDevice(cl_mem aSource);

		//! Copy data from device memory to this OpenCLDeviceVariable
		/*!
		\param aSource Data source in device memory
		*/
		//Copy data from device memory to this OpenCLDeviceVariable
		void CopyDeviceToDevice(OpenCLDeviceVariable& aSource);


		//! Copy data from host memory to this OpenCLDeviceVariable
		/*!
		\param aSource Data source in host memory
		\param aSizeInBytes Number of bytes to copy
		*/
		//Copy data from host memory to this OpenCLDeviceVariable
		void CopyHostToDevice(void* aSource, size_t aSizeInBytes = 0);

		//! Copy data from this OpenCLDeviceVariable to host memory
		/*!
		\param aDest Data destination in host memory
		\param aSizeInBytes Number of bytes to copy
		*/
		//Copy data from this OpenCLDeviceVariable to host memory
		void CopyDeviceToHost(void* aDest, size_t aSizeInBytes = 0);

		//! Returns the size in bytes of the allocated device memory
		size_t GetSize();

		//! Returns the wrapped CUdeviceptr
		cl_mem GetDevicePtr();

		//! Sets the allocated memory to \p aValue
		void Memset(uchar aValue);
	};
}
#endif //USE_OPENCL
#endif
