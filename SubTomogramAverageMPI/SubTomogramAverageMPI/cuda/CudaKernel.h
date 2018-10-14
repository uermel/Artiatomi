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


#ifndef CUDAKERNELS_H
#define CUDAKERNELS_H

#include "CudaDefault.h"
#include "CudaException.h"
//#include "CudaContext.h"
#include "CudaVariables.h"
#include "CudaArrays.h"
#include <stdarg.h>

namespace Cuda
{
	//!  A wrapper class for a CUDA Kernel.
	/*!
	  CudaKernel manages all the functionality of a CUDA Kernel.
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	//A wrapper class for a CUDA Kernel.
	class CudaKernel
	{
	private:
		void _loadAdditionalInfo();

	protected:
		//CudaContext* mCtx;
		CUmodule mModule;
		CUfunction mFunction;
		uint mSharedMemSize;
		int mParamOffset;
		dim3 mBlockDim;
		dim3 mGridDim;
		string mKernelName;
		int _maxThreadsPerBlock;
		int _sharedSizeBytes;
		int _constSizeBytes;
		int _localSizeBytes;
		int _numRegs;
		float _ptxVersion;
		float _binaryVersion;

	public:
		//! CudaKernel constructor
		/*!
			Loads a CUDA Kernel with name \p aKernelName bound to the CudaContext \p aCtx
			from the CUmodule \p aModule.
			\param aKernelName Name of the kernel to load
			\param aModule The module to load the kernel from
			\param aCtx The CUDA Context to use with this kernel.
		*/
		//CudaKernel constructor
		CudaKernel(string aKernelName, CUmodule aModule/*, CudaContext* aCtx*/);

		//! CudaKernel constructor
		/*!
			Loads a CUDA Kernel with name \p aKernelName bound to the CudaContext \p aCtx
			from the CUmodule \p aModule.
			\param aKernelName Name of the kernel to load
			\param aModule The module to load the kernel from
			\param aCtx The CUDA Context to use with this kernel.
		*/
		//CudaKernel constructor
		CudaKernel(string aKernelName, CUmodule aModule/*, CudaContext* aCtx*/, dim3 aGridDim, dim3 aBlockDim, uint aDynamicSharedMemory);

		//! CudaKernel destructor
		//CudaKernel destructor
		~CudaKernel();

		//! Set a constant variable value before kernel launch.
		/*!
			\param aName Name of the constant variable as defined in the *.cu kernel source file.
			\param aValue The value to set.
		*/
		//Set a constant variable value before kernel launch.
		void SetConstantValue(string aName, void* aValue);


		//! Set an integer kernel launch paramter.
		/*!
			The kernel launch parameters must be set in the same order as defined in the *.cu kernel source file.
			\param aValue The value to set.
		*/
		//Set an integer kernel launch paramter.
		void SetIntegerParameter(const int aValue);

		//! Set a float kernel launch paramter.
		/*!
			The kernel launch parameters must be set in the same order as defined in the *.cu kernel source file.
			\param aValue The value to set.
		*/
		//Set a float kernel launch paramter.
		void SetFloatParameter(const float aValue);

		//! Set a device pointer kernel launch paramter.
		/*!
			The kernel launch parameters must be set in the same order as defined in the *.cu kernel source file.
			\param aDevicePtr The value to set.
		*/
		//Set a device pointer kernel launch paramter.
		void SetDevicePtrParameter(CUdeviceptr aDevicePtr);

		//! Manually reset the parameter offset counter
		//Manually reset the parameter offset counter
		void ResetParameterOffset();

		//! Set the kernels thread block dimensions before first launch.
		/*!
			\param aX Block X dimension.
			\param aY Block Y dimension.
			\param aZ Block Z dimension.
		*/
		//Set the kernels thread block dimensions before first launch.
		void SetBlockDimensions(uint aX, uint aY, uint aZ);

		//! Set the kernels thread block dimensions before first launch.
		/*!
			\param aBlockDim Block dimensions.
		*/
		//Set the kernels thread block dimensions before first launch.
		void SetBlockDimensions(dim3 aBlockDim);

		//! Set the kernels block grid dimensions before first launch.
		/*!
			\param aX Grid X dimension.
			\param aY Grid Y dimension.
		*/
		//Set the kernels block grid dimensions before first launch.
		void SetGridDimensions(uint aX, uint aY);

		//! Set the kernels block grid dimensions before first launch.
		/*!
			\param aGridDim Grid dimensions.
		*/
		//Set the kernels block grid dimensions before first launch.
		void SetGridDimensions(dim3 aGridDim);

		//! Set the dynamic amount of shared memory before first launch.
		/*!
			\param aSizeInBytes Size of shared memory in bytes.
		*/
		// Set the dynamic amount of shared memory before first launch.
		void SetDynamicSharedMemory(uint aSizeInBytes);

		//! Get the wrapped CUmodule
		//Get the wrapped CUmodule
		CUmodule& GetCUmodule();

		//! Get the wrapped CUfunction
		//Get the wrapped CUfunction
		CUfunction& GetCUfunction();

		//! Get the maximum number of threads per block, beyond which a launch of the function would fail.
		/*!
			This number depends on both the function and the device on which the function is currently loaded.
		*/
		//Get the wrapped CUfunction
		int GetMaxThreadsPerBlock();

		//! Get the size in bytes of statically-allocated shared memory per block required by this function.
		/*!
			 This does not include dynamically-allocated shared memory requested by
			 the user at runtime.
		*/
		//Get the size in bytes of statically-allocated shared memory per block required by this function.
		int GetSharedSizeBytes();

		//! The size in bytes of user-allocated constant memory required by this function.
		//The size in bytes of user-allocated constant memory required by this function.
		int GetConstSizeBytes();

		//! Get the size in bytes of local memory used by each thread of this function.
		//Get the size in bytes of local memory used by each thread of this function.
		int GetLocalSizeBytes();

		//! Get the number of registers used by each thread of this function.
		//Get the number of registers used by each thread of this function.
		int GetNumRegs();

		//! Get he PTX virtual architecture version for which the function was compiled.
		//Get he PTX virtual architecture version for which the function was compiled.
		float GetPtxVersion();

		//! Get the binary architecture version for which the function was compiled.
		//Get the binary architecture version for which the function was compiled.
		float GetBinaryVersion();

		//! Launches the kernel.
		/*!
			Before kernel launch all kernel parameters, constant variable values and block / grid dimensions must be set.
			\return kernel runtime in [ms]
		*/
		//Launches the kernel. Returns kernel runtime in [ms]
		virtual float operator()(int dummy, ...);
	};
}
#endif //CUDAKERNELS_H
