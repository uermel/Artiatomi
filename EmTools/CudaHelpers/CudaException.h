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
#ifdef USE_CUDA
#include "cufft.h"
#include "nppdefs.h"

using namespace std;

#define cudaSafeCall(err) __cudaSafeCall (err, __FILE__, __LINE__)
#define cufftSafeCall(err) __cufftSafeCall (err, __FILE__, __LINE__)
#define nppSafeCall(err) __nppSafeCall (err, __FILE__, __LINE__, false)


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

	
	//!  An exception wrapper class for CUDA CUresult. 
	/*!
	  CudaException is thrown, if a CUDA Driver API call via cudaSafeCall does not return CUDA_SUCCESS
	  \author Michael Kunz
	  \date   September 2011
	  \version 1.1
	*/
	//An exception wrapper class for CUDA CUresult. 
	class CufftException: public exception
	{
	protected:
		string mFileName;
		string mMessage;
		int mLine;
		cufftResult mErr;

	public:			
		//! Default constructor
		//Default constructor
		CufftException();
			
		~CufftException() throw();
			
		//! CudaException constructor
		/*!
			\param aMessage Ecxeption message
		*/
		//CudaException constructor
		CufftException(string aMessage);

		//! CudaException constructor
		/*!
			\param aFileName Source code file where the exception was thrown
			\param aLine Code line where the exception was thrown
			\param aMessage Ecxeption message
			\param aErr CUresult error code
		*/
		//CudaException constructor
		CufftException(string aFileName, int aLine, string aMessage, cufftResult aErr);
		
		//! Returns "CudaException"
		//Returns "CudaException"
		virtual const char* what() const throw();
		
		//! Returns an error message
		//Returns an error message
		virtual string GetMessage() const;
	};
	
	//! Translates a cufftResult error code into a human readable error description, if \p err is not CUDA_SUCCESS.
	/*!		
		\param file Source code file where the exception was thrown
		\param line Code line where the exception was thrown
		\param err CUresult error code
	*/
	//Translates a CUresult error code into a human readable error description, if err is not CUDA_SUCCESS.
	inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
	{        
		if( CUFFT_SUCCESS != err)
		{
			std::string errMsg;
			switch(err)
			{         
			case CUFFT_INVALID_PLAN:
				errMsg = "CUFFT_INVALID_PLAN";
				break;       
			case CUFFT_ALLOC_FAILED:
				errMsg = "CUFFT_ALLOC_FAILED";
				break;    
			case CUFFT_INVALID_TYPE:
				errMsg = "CUFFT_INVALID_TYPE";
				break;    
			case CUFFT_INVALID_VALUE:
				errMsg = "CUFFT_INVALID_VALUE";
				break;    
			case CUFFT_INTERNAL_ERROR:
				errMsg = "CUFFT_INTERNAL_ERROR";
				break;    
			case CUFFT_EXEC_FAILED:
				errMsg = "CUFFT_EXEC_FAILED";
				break;    
			case CUFFT_SETUP_FAILED:
				errMsg = "CUFFT_SETUP_FAILED";
				break;    
			case CUFFT_INVALID_SIZE:
				errMsg = "CUFFT_INVALID_SIZE";
				break;    
			case CUFFT_UNALIGNED_DATA:
				errMsg = "CUFFT_UNALIGNED_DATA";
				break;    
			case CUFFT_INCOMPLETE_PARAMETER_LIST:
				errMsg = "CUFFT_INCOMPLETE_PARAMETER_LIST";
				break;   
			case CUFFT_INVALID_DEVICE:
				errMsg = "CUFFT_INVALID_DEVICE";
				break;   
			case CUFFT_PARSE_ERROR:
				errMsg = "CUFFT_PARSE_ERROR";
				break;   
			case CUFFT_NO_WORKSPACE:
				errMsg = "CUFFT_NO_WORKSPACE";
				break;  
			}

			CufftException ex(file, line, errMsg, err);
			throw ex;
		} //if CUDA_SUCCESS
	}

	
	//!  An exception wrapper class for CUDA CUresult. 
	/*!
	  CudaException is thrown, if a CUDA Driver API call via cudaSafeCall does not return CUDA_SUCCESS
	  \author Michael Kunz
	  \date   September 2011
	  \version 1.1
	*/
	//An exception wrapper class for CUDA CUresult. 
	class NppException: public exception
	{
	protected:
		string mFileName;
		string mMessage;
		int mLine;
		NppStatus mErr;

	public:			
		//! Default constructor
		//Default constructor
		NppException();
			
		~NppException() throw();
			
		//! CudaException constructor
		/*!
			\param aMessage Ecxeption message
		*/
		//CudaException constructor
		NppException(string aMessage);

		//! CudaException constructor
		/*!
			\param aFileName Source code file where the exception was thrown
			\param aLine Code line where the exception was thrown
			\param aMessage Ecxeption message
			\param aErr CUresult error code
		*/
		//CudaException constructor
		NppException(string aFileName, int aLine, string aMessage, NppStatus aErr);
		
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
	inline void __nppSafeCall(NppStatus err, const char *file, const int line, bool warningAsError)
	{        
		if( NPP_SUCCESS != err)
		{
			std::string errMsg;
			
			switch(err)
			{       
			/* negative return-codes indicate errors */
			case NPP_NOT_SUPPORTED_MODE_ERROR:
				errMsg = "NPP_NOT_SUPPORTED_MODE_ERROR";
				break;         
			case NPP_INVALID_HOST_POINTER_ERROR:
				errMsg = "NPP_INVALID_HOST_POINTER_ERROR";
				break;              
			case NPP_INVALID_DEVICE_POINTER_ERROR:
				errMsg = "NPP_INVALID_DEVICE_POINTER_ERROR";
				break;               
			case NPP_LUT_PALETTE_BITSIZE_ERROR:
				errMsg = "NPP_LUT_PALETTE_BITSIZE_ERROR";
				break;                  
			case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
				errMsg = "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";
				break;                   
			case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
				errMsg = "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";
				break;                   
			case NPP_TEXTURE_BIND_ERROR:
				errMsg = "NPP_TEXTURE_BIND_ERROR";
				break;                   
			case NPP_WRONG_INTERSECTION_ROI_ERROR:
				errMsg = "NPP_WRONG_INTERSECTION_ROI_ERROR";
				break;                   
			case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
				errMsg = "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";
				break;                   
			case NPP_MEMFREE_ERROR:
				errMsg = "NPP_MEMFREE_ERROR";
				break;                   
			case NPP_MEMSET_ERROR:
				errMsg = "NPP_MEMSET_ERROR";
				break;                   
			case NPP_MEMCPY_ERROR:
				errMsg = "NPP_MEMCPY_ERROR";
				break;                    
			case NPP_ALIGNMENT_ERROR:
				errMsg = "NPP_ALIGNMENT_ERROR";
				break;                    
			case NPP_CUDA_KERNEL_EXECUTION_ERROR:
				errMsg = "NPP_CUDA_KERNEL_EXECUTION_ERROR";
				break;                    
			case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
				errMsg = "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";
				break;                    
			case NPP_QUALITY_INDEX_ERROR:
				errMsg = "Image pixels are constant for quality index";
				break;                    
			case NPP_RESIZE_NO_OPERATION_ERROR:
				errMsg = "One of the output image dimensions is less than 1 pixel";
				break;                    
			case NPP_NOT_EVEN_STEP_ERROR:
				errMsg = "Step value is not pixel multiple";
				break;                    
			case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
				errMsg = "Number of levels for histogram is less than 2";
				break;                      
			case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
				errMsg = "Number of levels for LUT is less than 2";
				break;                      
			case NPP_CHANNEL_ORDER_ERROR:
				errMsg = "Wrong order of the destination channels";
				break;                      
			case NPP_ZERO_MASK_VALUE_ERROR:
				errMsg = "All values of the mask are zero";
				break;                      
			case NPP_QUADRANGLE_ERROR:
				errMsg = "The quadrangle is nonconvex or degenerates into triangle, line or point";
				break;                      
			case NPP_RECTANGLE_ERROR:
				errMsg = "Size of the rectangle region is less than or equal to 1";
				break;                      
			case NPP_COEFFICIENT_ERROR:
				errMsg = "Unallowable values of the transformation coefficients";
				break;                      
			case NPP_NUMBER_OF_CHANNELS_ERROR:
				errMsg = "Bad or unsupported number of channels";
				break;                      
			case NPP_COI_ERROR:
				errMsg = "Channel of interest is not 1, 2, or 3";
				break;                         
			case NPP_DIVISOR_ERROR:
				errMsg = "Divisor is equal to zero";
				break;                         
			case NPP_CHANNEL_ERROR:
				errMsg = "Illegal channel index";
				break;                         
			case NPP_STRIDE_ERROR:
				errMsg = "Stride is less than the row length";
				break;                         
			case NPP_ANCHOR_ERROR:
				errMsg = "Anchor point is outside mask";
				break;                         
			case NPP_MASK_SIZE_ERROR:
				errMsg = "Lower bound is larger than upper bound";
				break;                         
			case NPP_RESIZE_FACTOR_ERROR:
				errMsg = "NPP_RESIZE_FACTOR_ERROR";
				break;                         
			case NPP_INTERPOLATION_ERROR:
				errMsg = "NPP_INTERPOLATION_ERROR";
				break;                         
			case NPP_MIRROR_FLIP_ERROR:
				errMsg = "NPP_MIRROR_FLIP_ERROR";
				break;                         
			case NPP_MOMENT_00_ZERO_ERROR:
				errMsg = "NPP_MOMENT_00_ZERO_ERROR";
				break;                         
			case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
				errMsg = "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";
				break;                            
			case NPP_THRESHOLD_ERROR:
				errMsg = "NPP_THRESHOLD_ERROR";
				break;                          
			case NPP_CONTEXT_MATCH_ERROR:
				errMsg = "NPP_CONTEXT_MATCH_ERROR";
				break;                          
			case NPP_FFT_FLAG_ERROR:
				errMsg = "NPP_FFT_FLAG_ERROR";
				break;                          
			case NPP_FFT_ORDER_ERROR:
				errMsg = "NPP_FFT_ORDER_ERROR";
				break;                          
			case NPP_STEP_ERROR:
				errMsg = "Step is less or equal zero";
				break;                          
			case NPP_SCALE_RANGE_ERROR:
				errMsg = "NPP_SCALE_RANGE_ERROR";
				break;                          
			case NPP_DATA_TYPE_ERROR:
				errMsg = "NPP_DATA_TYPE_ERROR";
				break;                              
			case NPP_OUT_OFF_RANGE_ERROR:
				errMsg = "NPP_OUT_OFF_RANGE_ERROR";
				break;                          
			case NPP_DIVIDE_BY_ZERO_ERROR:
				errMsg = "NPP_DIVIDE_BY_ZERO_ERROR";
				break;                          
			case NPP_MEMORY_ALLOCATION_ERR:
				errMsg = "NPP_MEMORY_ALLOCATION_ERR";
				break;                          
			case NPP_NULL_POINTER_ERROR:
				errMsg = "NPP_NULL_POINTER_ERROR";
				break;                          
			case NPP_RANGE_ERROR:
				errMsg = "NPP_RANGE_ERROR";
				break;                          
			case NPP_SIZE_ERROR:
				errMsg = "NPP_SIZE_ERROR";
				break;                          
			case NPP_BAD_ARGUMENT_ERROR:
				errMsg = "NPP_BAD_ARGUMENT_ERROR";
				break;                          
			case NPP_NO_MEMORY_ERROR:
				errMsg = "NPP_NO_MEMORY_ERROR";
				break;                   
			case NPP_NOT_IMPLEMENTED_ERROR:
				errMsg = "NPP_NOT_IMPLEMENTED_ERROR";
				break;                             
			case NPP_ERROR:
				errMsg = "NPP_ERROR";
				break;                             
			case NPP_ERROR_RESERVED:
				errMsg = "NPP_ERROR_RESERVED";
				break;  

				/* positive return-codes indicate warnings */
				                           
			case NPP_NO_OPERATION_WARNING:
				errMsg = "Indicates that no operation was performed";
				break;                          
			case NPP_DIVIDE_BY_ZERO_WARNING:
				errMsg = "Divisor is zero however does not terminate the execution";
				break;                          
			case NPP_AFFINE_QUAD_INCORRECT_WARNING:
				errMsg = "Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded.";
				break;                          
			case NPP_WRONG_INTERSECTION_ROI_WARNING:
				errMsg = "The given ROI has no interestion with either the source or destination ROI. Thus no operation was performed.";
				break;                          
			case NPP_WRONG_INTERSECTION_QUAD_WARNING:
				errMsg = "The given quadrangle has no intersection with either the source or destination ROI. Thus no operation was performed.";
				break;                          
			case NPP_DOUBLE_SIZE_WARNING:
				errMsg = "Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing.";
				break;                          
			case NPP_MISALIGNED_DST_ROI_WARNING:
				errMsg = "Speed reduction due to uncoalesced memory accesses warning.";
				break;  

			}

			if (err < 0 || (warningAsError && err > 0))
			{
				NppException ex(file, line, errMsg, err);
				throw ex;
			}
			else
			{
				printf("NPP Warning: %s", errMsg.c_str());
			}
		} //if CUDA_SUCCESS
	}
}
#endif //USE_CUDA
#endif //CUDAEXCEPTION_H