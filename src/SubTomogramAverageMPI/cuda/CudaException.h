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


#ifndef CUDAEXCEPTION_H
#define CUDAEXCEPTION_H

#include "CudaDefault.h"

using namespace std;

#define cudaSafeCall(err) __cudaSafeCall (err, __FILE__, __LINE__)


namespace Cuda
{
	//!  An exception wrapper class for CUDA CUresult. 
	/*!
	  CudaException is thrown, if a CUDA Driver API call via cudaSafeCall does not return CUDA_SUCCESS
	  \author Michael Kunz
	  \date   September 2011
	  \version 1.1
	*/
	//An exception wrapper class for CUDA CUresult. 
	class CudaException: public exception
	{
	protected:
		string mFileName;
		string mMessage;
		int mLine;
		CUresult mErr;

	public:			
		//! Default constructor
		//Default constructor
		CudaException();
			
		~CudaException() throw();
			
		//! CudaException constructor
		/*!
			\param aMessage Ecxeption message
		*/
		//CudaException constructor
		CudaException(string aMessage);

		//! CudaException constructor
		/*!
			\param aFileName Source code file where the exception was thrown
			\param aLine Code line where the exception was thrown
			\param aMessage Ecxeption message
			\param aErr CUresult error code
		*/
		//CudaException constructor
		CudaException(string aFileName, int aLine, string aMessage, CUresult aErr);
		
		//! Returns "CudaException"
		//Returns "CudaException"
		virtual const char* what() const throw();
		
		//! Returns an error message
		//Returns an error message
		virtual string GetMessage() const;
	};
	
	//! Translates a CUresult error code into a human readable error description, if \p err is not CUDA_SUCCESS.
	/*!		
		\param file Source code file where the exception was thrown
		\param line Code line where the exception was thrown
		\param err CUresult error code
	*/
	//Translates a CUresult error code into a human readable error description, if err is not CUDA_SUCCESS.
	inline void __cudaSafeCall(CUresult err, const char *file, const int line)
	{        
		if( CUDA_SUCCESS != err)
		{
			std::string errMsg;
			switch(err)
			{                
			case CUDA_ERROR_INVALID_VALUE:
				errMsg = "Invalid value";
				break;
			case CUDA_ERROR_OUT_OF_MEMORY:
				errMsg = "Out of memory";
				break;
			case CUDA_ERROR_NOT_INITIALIZED:
				errMsg = "Driver not initialized";
				break;
			case CUDA_ERROR_DEINITIALIZED:
				errMsg = "Driver deinitialized";            
				break;
			case CUDA_ERROR_PROFILER_DISABLED:
				errMsg = "Profiling APIs are called while application is running in visual profiler mode.";            
				break;
			case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
				errMsg = "Profiling has not been initialized for this context. Call cuProfilerInitialize() to resolve this.";            
				break;
			case CUDA_ERROR_PROFILER_ALREADY_STARTED:
				errMsg = "Profiler has already been started and probably cuProfilerStart() is incorrectly called.";
				break;
			case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
				errMsg = "Profiler has already been stopped and probably cuProfilerStop() is incorrectly called.";
				break;
			case CUDA_ERROR_NO_DEVICE:
				errMsg = "No CUDA-capable device available";            
				break;
			case CUDA_ERROR_INVALID_DEVICE:
				errMsg = "Invalid device";            
				break;
			case CUDA_ERROR_INVALID_IMAGE:
				errMsg = "Invalid kernel image";            
				break;
			case CUDA_ERROR_INVALID_CONTEXT:
				errMsg = "Invalid context";            
				break;
			case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
				errMsg = "Context already current";
				break;
			case CUDA_ERROR_MAP_FAILED:
				errMsg = "Map failed";
				break;
			case CUDA_ERROR_UNMAP_FAILED:
				errMsg = "Unmap failed";            
				break;
			case CUDA_ERROR_ARRAY_IS_MAPPED:
				errMsg = "Array is mapped";						
				break;
			case CUDA_ERROR_ALREADY_MAPPED:
				errMsg = "Already mapped";
				break;
			case CUDA_ERROR_NO_BINARY_FOR_GPU:
				errMsg = "No binary for GPU";            
				break;
			case CUDA_ERROR_ALREADY_ACQUIRED:
				errMsg = "Already acquired";            
				break;
			case CUDA_ERROR_NOT_MAPPED:
				errMsg = "Not mapped";            
				break;
			case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
				errMsg = "Not mapped as array";            
				break;
			case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
				errMsg = "Not mapped as pointer";            
				break;
			case CUDA_ERROR_ECC_UNCORRECTABLE:
				errMsg = "Uncorrectable ECC error";            
				break;
			case CUDA_ERROR_UNSUPPORTED_LIMIT:
				errMsg = "Unsupported limit";            
				break;
			case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
				errMsg = "The context can only be bound to a single CPU thread at a time but is already bound to a CPU thread.";            
				break;
			case CUDA_ERROR_INVALID_SOURCE:
				errMsg = "Invalid source";            
				break;
			case CUDA_ERROR_FILE_NOT_FOUND:
				errMsg = "File not found";
				break;
			case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
				errMsg = "Link to a shared object failed to resolve";
				break;
			case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
				errMsg = "Initialization of a shared object failed";
				break;
			case CUDA_ERROR_OPERATING_SYSTEM:
				errMsg = "Operating system call failed";
				break;
			case CUDA_ERROR_INVALID_HANDLE:
				errMsg = "Invalid handle";            
				break;
			case CUDA_ERROR_NOT_FOUND:
				errMsg = "Not found";            
				break;
			case CUDA_ERROR_NOT_READY:
				errMsg = "CUDA not ready";            
				break;
			case CUDA_ERROR_LAUNCH_FAILED:
				errMsg = "Launch failed";            
				break;
			case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
				errMsg = "Launch exceeded resources";            
				break;
			case CUDA_ERROR_LAUNCH_TIMEOUT:
				errMsg = "Launch exceeded timeout";            
				break;
			case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
				errMsg = "Launch with incompatible texturing";            
				break;
			case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
				errMsg = "Trying to re-enable peer access to a context which has already had peer access to it enabled.";
				break;
			case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
				errMsg = "Trying to disable peer access which has not been enabled yet  via ::cuCtxEnablePeerAccess(). ";
				break;
			case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
				errMsg = "The primary context for the specified device has already been initialized.";
				break;
			case CUDA_ERROR_CONTEXT_IS_DESTROYED:
				errMsg = "The context current to the calling thread has been destroyed using ::cuCtxDestroy, or is a primary context which has not yet been initialized.";
				break;
			case CUDA_ERROR_UNKNOWN:
				errMsg = "Unknown error";
				break;
			}

			CudaException ex(file, line, errMsg, err);
			throw ex;
		} //if CUDA_SUCCESS
	}
}
#endif //CUDAEXCEPTION_H