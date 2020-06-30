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


#ifndef OPENCLEXCEPTION_H
#define OPENCLEXCEPTION_H

#include <CL/cl.h>
#include <exception>
#include <string>
//#include <Windows.h>

using namespace std;

#define openCLSafeCall(err) __openCLSafeCall (err, __FILE__, __LINE__)
#define openCLSafeCallMsg(err, message) __openCLSafeCall (err, __FILE__, __LINE__, message)

namespace OpenCL
{
	//!  An exception wrapper class for OpenCL error codes. 
	/*!
	OpenCLException is thrown, if an OpenCL API call via openCLSafeCall does not return CL_SUCCESS
	\author Michael Kunz
	\date   November 2016
	\version 1.0
	*/
	//An exception wrapper class for OpenCL error codes. 
	class OpenCLException : public exception
	{
	protected:
		string mFileName;
		string mMessage;
		int mLine;
		cl_int mErr;

	public:
		//! Default constructor
		//Default constructor
		OpenCLException();

		~OpenCLException() throw();

		//! OpenCLException constructor
		/*!
		\param aMessage Ecxeption message
		*/
		//OpenCLException constructor
		OpenCLException(string aMessage);

		//! OpenCLException constructor
		/*!
		\param aFileName Source code file where the exception was thrown
		\param aLine Code line where the exception was thrown
		\param aMessage Ecxeption message
		\param aErr cl_int error code
		*/
		//OpenCLException constructor
		OpenCLException(string aFileName, int aLine, string aMessage, cl_int aErr);

		//! Returns "OpenCLException"
		//Returns "OpenCLException"
		virtual const char* what() const throw();

		//! Returns an error message
		//Returns an error message
		//virtual string GetMessage() const;

		//! Returns an error message
		//Returns an error message
		//virtual string GetMessageW() const;
	};

	//! Translates a cl_int error code into a human readable error description, if \p err is not CL_SUCCESS.
	/*!
	\param file Source code file where the exception was thrown
	\param line Code line where the exception was thrown
	\param err cl_int error code
	*/
	//Translates a cl_int error code into a human readable error description, if err is not CL_SUCCESS.
	inline void __openCLSafeCall(cl_int err, const char *file, const int line, const char* message = NULL)
	{
		if (CL_SUCCESS != err)
		{
			std::string errMsg;
			switch (err)
			{
			case CL_DEVICE_NOT_FOUND:
				errMsg = "CL_DEVICE_NOT_FOUND";
				break;
			case CL_DEVICE_NOT_AVAILABLE:
				errMsg = "CL_DEVICE_NOT_AVAILABLE";
				break;
			case CL_COMPILER_NOT_AVAILABLE:
				errMsg = "CL_COMPILER_NOT_AVAILABLE";
				break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				errMsg = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
				break;
			case CL_OUT_OF_RESOURCES:
				errMsg = "CL_OUT_OF_RESOURCES";
				break;
			case CL_OUT_OF_HOST_MEMORY:
				errMsg = "CL_OUT_OF_HOST_MEMORY";
				break;
			case CL_PROFILING_INFO_NOT_AVAILABLE:
				errMsg = "CL_PROFILING_INFO_NOT_AVAILABLE";
				break;
			case CL_MEM_COPY_OVERLAP:
				errMsg = "CL_MEM_COPY_OVERLAP";
				break;
			case CL_IMAGE_FORMAT_MISMATCH:
				errMsg = "CL_IMAGE_FORMAT_MISMATCH";
				break;
			case CL_IMAGE_FORMAT_NOT_SUPPORTED:
				errMsg = "CL_IMAGE_FORMAT_NOT_SUPPORTED";
				break;
			case CL_BUILD_PROGRAM_FAILURE:
				errMsg = "CL_BUILD_PROGRAM_FAILURE";
				break;
			case CL_MAP_FAILURE:
				errMsg = "CL_MAP_FAILURE";
				break;
			case CL_MISALIGNED_SUB_BUFFER_OFFSET:
				errMsg = "CL_MISALIGNED_SUB_BUFFER_OFFSET";
				break;
			case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
				errMsg = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
				break;
			case CL_COMPILE_PROGRAM_FAILURE:
				errMsg = "CL_COMPILE_PROGRAM_FAILURE";
				break;
			case CL_LINKER_NOT_AVAILABLE:
				errMsg = "CL_LINKER_NOT_AVAILABLE";
				break;
			case CL_LINK_PROGRAM_FAILURE:
				errMsg = "CL_LINK_PROGRAM_FAILURE";
				break;
			case CL_DEVICE_PARTITION_FAILED:
				errMsg = "CL_DEVICE_PARTITION_FAILED";
				break;
			case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
				errMsg = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
				break;

			case CL_INVALID_VALUE:
				errMsg = "CL_INVALID_VALUE";
				break;
			case CL_INVALID_DEVICE_TYPE:
				errMsg = "CL_INVALID_DEVICE_TYPE";
				break;
			case CL_INVALID_PLATFORM:
				errMsg = "CL_INVALID_PLATFORM";
				break;
			case CL_INVALID_DEVICE:
				errMsg = "CL_INVALID_DEVICE";
				break;
			case CL_INVALID_CONTEXT:
				errMsg = "CL_INVALID_CONTEXT";
				break;
			case CL_INVALID_QUEUE_PROPERTIES:
				errMsg = "CL_INVALID_QUEUE_PROPERTIES";
				break;
			case CL_INVALID_COMMAND_QUEUE:
				errMsg = "CL_INVALID_COMMAND_QUEUE";
				break;
			case CL_INVALID_HOST_PTR:
				errMsg = "CL_INVALID_HOST_PTR";
				break;
			case CL_INVALID_MEM_OBJECT:
				errMsg = "CL_INVALID_MEM_OBJECT";
				break;
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
				errMsg = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
				break;
			case CL_INVALID_IMAGE_SIZE:
				errMsg = "CL_INVALID_IMAGE_SIZE";
				break;
			case CL_INVALID_SAMPLER:
				errMsg = "CL_INVALID_SAMPLER";
				break;
			case CL_INVALID_BINARY:
				errMsg = "CL_INVALID_BINARY";
				break;
			case CL_INVALID_BUILD_OPTIONS:
				errMsg = "CL_INVALID_BUILD_OPTIONS";
				break;
			case CL_INVALID_PROGRAM:
				errMsg = "CL_INVALID_PROGRAM";
				break;
			case CL_INVALID_PROGRAM_EXECUTABLE:
				errMsg = "CL_INVALID_PROGRAM_EXECUTABLE";
				break;
			case CL_INVALID_KERNEL_NAME:
				errMsg = "CL_INVALID_KERNEL_NAME";
				break;
			case CL_INVALID_KERNEL_DEFINITION:
				errMsg = "CL_INVALID_KERNEL_DEFINITION";
				break;
			case CL_INVALID_KERNEL:
				errMsg = "CL_INVALID_KERNEL";
				break;
			case CL_INVALID_ARG_INDEX:
				errMsg = "CL_INVALID_ARG_INDEX";
				break;
			case CL_INVALID_ARG_VALUE:
				errMsg = "CL_INVALID_ARG_VALUE";
				break;
			case CL_INVALID_ARG_SIZE:
				errMsg = "CL_INVALID_ARG_SIZE";
				break;
			case CL_INVALID_KERNEL_ARGS:
				errMsg = "CL_INVALID_KERNEL_ARGS";
				break;
			case CL_INVALID_WORK_DIMENSION:
				errMsg = "CL_INVALID_WORK_DIMENSION";
				break;
			case CL_INVALID_WORK_GROUP_SIZE:
				errMsg = "CL_INVALID_WORK_GROUP_SIZE";
				break;
			case CL_INVALID_WORK_ITEM_SIZE:
				errMsg = "CL_INVALID_WORK_ITEM_SIZE";
				break;
			case CL_INVALID_GLOBAL_OFFSET:
				errMsg = "CL_INVALID_GLOBAL_OFFSET";
				break;
			case CL_INVALID_EVENT_WAIT_LIST:
				errMsg = "CL_INVALID_EVENT_WAIT_LIST";
				break;
			case CL_INVALID_EVENT:
				errMsg = "CL_INVALID_EVENT";
				break;
			case CL_INVALID_OPERATION:
				errMsg = "CL_INVALID_OPERATION";
				break;
			case CL_INVALID_GL_OBJECT:
				errMsg = "CL_INVALID_GL_OBJECT";
				break;
			case CL_INVALID_BUFFER_SIZE:
				errMsg = "CL_INVALID_BUFFER_SIZE";
				break;
			case CL_INVALID_MIP_LEVEL:
				errMsg = "CL_INVALID_MIP_LEVEL";
				break;
			case CL_INVALID_GLOBAL_WORK_SIZE:
				errMsg = "CL_INVALID_GLOBAL_WORK_SIZE";
				break;
			case CL_INVALID_PROPERTY:
				errMsg = "CL_INVALID_PROPERTY";
				break;
			case CL_INVALID_IMAGE_DESCRIPTOR:
				errMsg = "CL_INVALID_IMAGE_DESCRIPTOR";
				break;
			case CL_INVALID_COMPILER_OPTIONS:
				errMsg = "CL_INVALID_COMPILER_OPTIONS";
				break;
			case CL_INVALID_LINKER_OPTIONS:
				errMsg = "CL_INVALID_LINKER_OPTIONS";
				break;
			case CL_INVALID_DEVICE_PARTITION_COUNT:
				errMsg = "CL_INVALID_DEVICE_PARTITION_COUNT";
				break;
			}

			if (message)
			{
				errMsg += "\n";
				errMsg += message;
			}

			throw OpenCLException(file, line, errMsg, err);
		} //if CUDA_SUCCESS
	}

}
#endif //OPENCLEXCEPTION_H