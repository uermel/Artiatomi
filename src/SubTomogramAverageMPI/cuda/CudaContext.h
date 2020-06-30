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


#ifndef CUDACONTEXT_H
#define CUDACONTEXT_H

#include "CudaDefault.h"
#include "CudaKernel.h"
#include "CudaException.h"
#include "CudaDeviceProperties.h"

namespace Cuda
{
	//!  A wrapper class for a CUDA Context. 
	/*!
	  CudaContext manages all the functionality of a CUDA Context.
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	//A wrapper class for a CUDA Context. 
	class CudaContext
	{
	private:
		CUcontext	mHcuContext; //The wrapped CUDA Context
		CUdevice	mHcuDevice;  //The CUDA Device linked to this CUDA Context
		int			mDeviceID;   //The ID of the CUDA Device
		int			mCtxFlags;   //Context creation flags
				
		//! CudaContext constructor
		/*!
			Creates a new CUDA Context bound to the CUDA Device with the ID \p deviceID
			using the \p ctxFlags context creation flags.
			\param deviceID The ID of the CUDA Device to use
			\param ctxFlags Context creation flags
		*/
		//CudaContext constructor
		CudaContext(int deviceID, CUctx_flags ctxFlags);
	
		//! CudaContext destructor
		/*!
			The Wrapped CUDA Context will be detached.
		*/
		//CudaContext destructor
		~CudaContext();
	public:
		
		//! Create a new instance of a CudaContext.
		/*!
			Creates a new CUDA Context bound to the CUDA Device with the ID \p aDeviceID
			using the \p ctxFlags context creation flags.
			\param aDeviceID The ID of the CUDA Device to use
			\param ctxFlags Context creation flags
		*/
		//Create a new instance of a CudaContext.
		static CudaContext* CreateInstance(int aDeviceID, CUctx_flags ctxFlags = CU_CTX_SCHED_AUTO);

		//! Destroys an instance of a CudaContext
		/*!
			The Wrapped CUDA Context will be detached.
			/param aCtx The CudaContext to destroy
		*/
		//Destroys an instance of a CudaContext
		static void DestroyInstance(CudaContext* aCtx);

		//! Destroys an CudaContext
		/*!
			The Wrapped CUDA Context will be destroyed.
			/param aCtx The CudaContext to destroy
		*/
		//Destroys an CudaContext
		static void DestroyContext(CudaContext* aCtx);

		//! Pushes the wrapped CUDA Context on the current CPU thread
		/*!
			Pushes the wrapped CUDA Context onto the CPU thread's stack of current
			contexts. The specified context becomes the CPU thread's current context, so
			all CUDA functions that operate on the current context are affected.
			The wrapped CUDA Context must be "floating" before calling CUDA::PushContext().
			I.e. not attached to any thread. Contexts are
			made to float by calling CUDA::PopContext().
		*/
		//Pushes the wrapped CUDA Context on the current CPU thread
		void PushContext();		

		//! Pops the wrapped CUDA Context from the current CPU thread
		/*!
			Pops the wrapped CUDA context from the CPU thread. The CUDA context must
			have a usage count of 1. CUDA contexts have a usage count of 1 upon
			creation.
		*/
		//Pops the wrapped CUDA Context from the current CPU thread
		void PopContext();	

		//! Binds the wrapped CUDA context to the calling CPU thread
		/*!
			Binds the wrapped CUDA context to the calling CPU thread.
			If there exists a CUDA context stack on the calling CPU thread, this
			will replace the top of that stack with this context.  
		*/
		//Binds the wrapped CUDA context to the calling CPU thread
		void SetCurrent();		

		//! Block for a context's tasks to complete
		/*!
			Blocks until the device has completed all preceding requested tasks.
			Synchronize() throws an exception if one of the preceding tasks failed.
			If the context was created with the ::CU_CTX_SCHED_BLOCKING_SYNC flag, the 
			CPU thread will block until the GPU context has finished its work.
		*/
		//Block for a context's tasks to complete
		void Synchronize();	

		//! Load a *.cubin/*.ptx Cuda Module
		/*!
			Loads the *.cubin file given by \p modulePath and creates a CUmodule bound to this CUDA context.
			\param modulePath Path and filename of the *.cubin file to load.
			\return The created CUmodule
		*/
		//Load a *.cubin Cuda Module
		CudaKernel* LoadKernel(std::string aModulePath, std::string aKernelName, dim3 aGridDim, dim3 aBlockDim, uint aDynamicSharedMemory = 0);
		
		//! Load a *.cubin/*.ptx Cuda Module
		/*!
			Loads the *.cubin file given by \p modulePath and creates a CUmodule bound to this CUDA context.
			\param modulePath Path and filename of the *.cubin file to load.
			\return The created CUmodule
		*/
		//Load a *.cubin Cuda Module
		CudaKernel* LoadKernel(std::string aModulePath, std::string aKernelName, uint aGridDimX, uint aGridDimY, uint aGridDimZ, uint aBlockDimX, uint aBlockDimY, uint aDynamicSharedMemory = 0);

		//! Load a *.cubin Cuda Module
		/*!
			Loads the *.cubin file given by \p modulePath and creates a CUmodule bound to this CUDA context.
			\param modulePath Path and filename of the *.cubin file to load.
			\return The created CUmodule
		*/
		//Load a *.cubin Cuda Module
		CUmodule LoadModule(const char* modulePath);

		//! Load a *.ptx Cuda Module
		/*!
			Loads the *.ptx file given by \p aModulePath and creates a CUmodule bound to this CUDA context.
			The PTX file will be compiled using \p aOptionCount compiling options determind in \p aOptions and the option
			values given in \p aOptionValues.
			\param aModulePath Path and filename of the *.ptx file to load.
			\param aOptionCount Number of options
			\param aOptions Options for JIT
			\param aOptionValues Option values for JIT
			\return The created CUmodule
		*/
		//Load a *.ptx Cuda Module
		CUmodule LoadModulePTX(const char* aModulePath, uint aOptionCount, CUjit_option* aOptions, void** aOptionValues);
		
		//! Load a PTX Cuda Module from byte array
		/*!
			Loads the *.ptx module given by \p aModuleImage and creates a CUmodule bound to this CUDA context.
			The PTX file will be compiled using \p aOptionCount compiling options determind in \p aOptions and the option
			values given in \p aOptionValues.
			\param aOptionCount Number of options
			\param aOptions Options for JIT
			\param aOptionValues Option values for JIT
			\param aModuleImage Binary image of the *.ptx file to load.
			\return The created CUmodule
		*/
		//Load a PTX Cuda Module from byte array
		CUmodule LoadModulePTX(uint aOptionCount, CUjit_option* aOptions, void** aOptionValues, const void* aModuleImage);
		
		//! Load a PTX Cuda Module from byte array
		/*!
			Loads the *.ptx module given by \p aModuleImage and creates a CUmodule bound to this CUDA context.
			\return The created CUmodule
		*/
		//Load a PTX Cuda Module from byte array
		CUmodule LoadModulePTX(const void* aModuleImage, uint aMaxRegCount, bool showInfoBuffer, bool showErrorBuffer);

		//! Unload a Cuda Module
		/*!
			Unloads a module \p aModule from the current context.
			\param aModule to unload
		*/
		//Unload a Cuda Module
		void UnloadModule(CUmodule& aModule);
				
		//! A memset function for device memory
		/*!
			Sets the memory range of \p aSizeInBytes / sizeof(unsigned int) 32-bit values to the specified value
			\p aValue.
			\param aPtr Destination device pointer
			\param aValue Value to set
			\param aSizeInBytes Size of the memory area to set.
		*/
		//A memset function for device memory
		void ClearMemory(CUdeviceptr aPtr, unsigned int aValue, size_t aSizeInBytes);
		
		//! Retrieves informations on the current CUDA Device
		/*!
			Retrieves informations on the current CUDA Device
			\return A CudaDeviceProperties object
		*/
		//Retrieves informations on the current CUDA Device
		CudaDeviceProperties* GetDeviceProperties();
		
		size_t GetFreeMemorySize();

		size_t GetMemorySize();
	};
	
} //Namespace Cuda

#endif //CUDACONTEXT_H