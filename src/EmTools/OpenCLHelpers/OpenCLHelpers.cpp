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


#include "OpenCLHelpers.h"
#include "../MKLog/MKLog.h"
#include <CL/cl_gl.h>

#if defined(_USE_WINDOWS_COMPILER_SETTINGS)
#include <Windows.h>
#else
#if defined (_USE_LINUX_COMPILER_SETTINGS)
#include <QVariant>
#include <QtPlatformHeaders/QGLXNativeContext>
#include <QOpenGLContext>
//#include nothing
#else
#if defined(_USE_APPLE_COMPILER_SETTINGS)
#include apple nothing
#endif
#endif
#endif

namespace OpenCL
{

	// Helper function to get OpenCL image format string (channel order and type) from constant
	// *********************************************************************
	const char* oclImageFormatString(cl_uint uiImageFormat)
	{
		// cl_channel_order 
		if (uiImageFormat == CL_R)return "CL_R";
		if (uiImageFormat == CL_A)return "CL_A";
		if (uiImageFormat == CL_RG)return "CL_RG";
		if (uiImageFormat == CL_RA)return "CL_RA";
		if (uiImageFormat == CL_RGB)return "CL_RGB";
		if (uiImageFormat == CL_RGBA)return "CL_RGBA";
		if (uiImageFormat == CL_BGRA)return "CL_BGRA";
		if (uiImageFormat == CL_ARGB)return "CL_ARGB";
		if (uiImageFormat == CL_INTENSITY)return "CL_INTENSITY";
		if (uiImageFormat == CL_LUMINANCE)return "CL_LUMINANCE";

		// cl_channel_type 
		if (uiImageFormat == CL_SNORM_INT8)return "CL_SNORM_INT8";
		if (uiImageFormat == CL_SNORM_INT16)return "CL_SNORM_INT16";
		if (uiImageFormat == CL_UNORM_INT8)return "CL_UNORM_INT8";
		if (uiImageFormat == CL_UNORM_INT16)return "CL_UNORM_INT16";
		if (uiImageFormat == CL_UNORM_SHORT_565)return "CL_UNORM_SHORT_565";
		if (uiImageFormat == CL_UNORM_SHORT_555)return "CL_UNORM_SHORT_555";
		if (uiImageFormat == CL_UNORM_INT_101010)return "CL_UNORM_INT_101010";
		if (uiImageFormat == CL_SIGNED_INT8)return "CL_SIGNED_INT8";
		if (uiImageFormat == CL_SIGNED_INT16)return "CL_SIGNED_INT16";
		if (uiImageFormat == CL_SIGNED_INT32)return "CL_SIGNED_INT32";
		if (uiImageFormat == CL_UNSIGNED_INT8)return "CL_UNSIGNED_INT8";
		if (uiImageFormat == CL_UNSIGNED_INT16)return "CL_UNSIGNED_INT16";
		if (uiImageFormat == CL_UNSIGNED_INT32)return "CL_UNSIGNED_INT32";
		if (uiImageFormat == CL_HALF_FLOAT)return "CL_HALF_FLOAT";
		if (uiImageFormat == CL_FLOAT)return "CL_FLOAT";

		// unknown constant
		return "Unknown";
	}

	void oclPrintDevInfo(cl_device_id device)
	{
		char device_string[1024];
		bool nv_device_attibute_query = false;

		// CL_DEVICE_NAME
		clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
		MKLOG("  CL_DEVICE_NAME: \t\t\t%s", &device_string[0]);

		// CL_DEVICE_VENDOR
		clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
		MKLOG("  CL_DEVICE_VENDOR: \t\t\t%s", &device_string[0]);

		// CL_DRIVER_VERSION
		clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_string), &device_string, NULL);
		MKLOG("  CL_DRIVER_VERSION: \t\t\t%s", &device_string[0]);

		// CL_DEVICE_VERSION
		clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_string), &device_string, NULL);
		MKLOG("  CL_DEVICE_VERSION: \t\t\t%s", &device_string[0]);

#if !defined(__APPLE__) && !defined(__MACOSX)
		// CL_DEVICE_OPENCL_C_VERSION (if CL_DEVICE_VERSION version > 1.0)
		if (strncmp("OpenCL 1.0", device_string, 10) != 0)
		{
			// This code is unused for devices reporting OpenCL 1.0, but a def is needed anyway to allow compilation using v 1.0 headers 
			// This constant isn't #defined in 1.0
#ifndef CL_DEVICE_OPENCL_C_VERSION
#define CL_DEVICE_OPENCL_C_VERSION 0x103D   
#endif

			clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(device_string), &device_string, NULL);
			MKLOG("  CL_DEVICE_OPENCL_C_VERSION: \t\t%s", &device_string[0]);
		}
#endif

		// CL_DEVICE_TYPE
		cl_device_type type;
		clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
		if (type & CL_DEVICE_TYPE_CPU)
			MKLOG("  CL_DEVICE_TYPE:\t\t\t%s", "CL_DEVICE_TYPE_CPU");
		if (type & CL_DEVICE_TYPE_GPU)
			MKLOG("  CL_DEVICE_TYPE:\t\t\t%s", "CL_DEVICE_TYPE_GPU");
		if (type & CL_DEVICE_TYPE_ACCELERATOR)
			MKLOG("  CL_DEVICE_TYPE:\t\t\t%s", "CL_DEVICE_TYPE_ACCELERATOR");
		if (type & CL_DEVICE_TYPE_DEFAULT)
			MKLOG("  CL_DEVICE_TYPE:\t\t\t%s", "CL_DEVICE_TYPE_DEFAULT");

		// CL_DEVICE_MAX_COMPUTE_UNITS
		cl_uint compute_units;
		clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
		MKLOG("  CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u", compute_units);

		// CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
		size_t workitem_dims;
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
		MKLOG("  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%u", workitem_dims);

		// CL_DEVICE_MAX_WORK_ITEM_SIZES
		size_t workitem_size[3];
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
		MKLOG("  CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%u / %u / %u ", workitem_size[0], workitem_size[1], workitem_size[2]);

		// CL_DEVICE_MAX_WORK_GROUP_SIZE
		size_t workgroup_size;
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
		MKLOG("  CL_DEVICE_MAX_WORK_GROUP_SIZE:\t%u", workgroup_size);

		// CL_DEVICE_MAX_CLOCK_FREQUENCY
		cl_uint clock_frequency;
		clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
		MKLOG("  CL_DEVICE_MAX_CLOCK_FREQUENCY:\t%u MHz", clock_frequency);

		// CL_DEVICE_ADDRESS_BITS
		cl_uint addr_bits;
		clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
		MKLOG("  CL_DEVICE_ADDRESS_BITS:\t\t%u", addr_bits);

		// CL_DEVICE_MAX_MEM_ALLOC_SIZE
		cl_ulong max_mem_alloc_size;
		clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
		MKLOG("  CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%u MByte", (unsigned int)(max_mem_alloc_size / (1024 * 1024)));

		// CL_DEVICE_GLOBAL_MEM_SIZE
		cl_ulong mem_size;
		clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
		MKLOG("  CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte", (unsigned int)(mem_size / (1024 * 1024)));

		// CL_DEVICE_ERROR_CORRECTION_SUPPORT
		cl_bool error_correction_support;
		clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
		MKLOG("  CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s", error_correction_support == CL_TRUE ? "yes" : "no");

		// CL_DEVICE_LOCAL_MEM_TYPE
		cl_device_local_mem_type local_mem_type;
		clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
		MKLOG("  CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s", local_mem_type == 1 ? "local" : "global");

		// CL_DEVICE_LOCAL_MEM_SIZE
		clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
		MKLOG("  CL_DEVICE_LOCAL_MEM_SIZE:\t\t%u KByte", (unsigned int)(mem_size / 1024));

		// CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
		clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
		MKLOG("  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%u KByte", (unsigned int)(mem_size / 1024));

		// CL_DEVICE_QUEUE_PROPERTIES
		cl_command_queue_properties queue_properties;
		clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
		if (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
			MKLOG("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s", "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
		if (queue_properties & CL_QUEUE_PROFILING_ENABLE)
			MKLOG("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s", "CL_QUEUE_PROFILING_ENABLE");

		// CL_DEVICE_IMAGE_SUPPORT
		cl_bool image_support;
		clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
		MKLOG("  CL_DEVICE_IMAGE_SUPPORT:\t\t%u", image_support);

		// CL_DEVICE_MAX_READ_IMAGE_ARGS
		cl_uint max_read_image_args;
		clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
		MKLOG("  CL_DEVICE_MAX_READ_IMAGE_ARGS:\t%u", max_read_image_args);

		// CL_DEVICE_MAX_WRITE_IMAGE_ARGS
		cl_uint max_write_image_args;
		clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
		MKLOG("  CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t%u", max_write_image_args);

		// CL_DEVICE_SINGLE_FP_CONFIG
		cl_device_fp_config fp_config;
		clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &fp_config, NULL);
		MKLOG("  CL_DEVICE_SINGLE_FP_CONFIG:\t\t%s%s%s%s%s%s",
			fp_config & CL_FP_DENORM ? "denorms " : "",
			fp_config & CL_FP_INF_NAN ? "INF-quietNaNs " : "",
			fp_config & CL_FP_ROUND_TO_NEAREST ? "round-to-nearest " : "",
			fp_config & CL_FP_ROUND_TO_ZERO ? "round-to-zero " : "",
			fp_config & CL_FP_ROUND_TO_INF ? "round-to-inf " : "",
			fp_config & CL_FP_FMA ? "fma " : "");

		// CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH
		size_t szMaxDims[5];
		MKLOG("  CL_DEVICE_IMAGE <dim>");
		clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
		MKLOG("\t\t\t\t\t2D_MAX_WIDTH\t %u", szMaxDims[0]);
		clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
		MKLOG("\t\t\t\t\t2D_MAX_HEIGHT\t %u", szMaxDims[1]);
		clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
		MKLOG("\t\t\t\t\t3D_MAX_WIDTH\t %u", szMaxDims[2]);
		clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
		MKLOG("\t\t\t\t\t3D_MAX_HEIGHT\t %u", szMaxDims[3]);
		clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
		MKLOG("\t\t\t\t\t3D_MAX_DEPTH\t %u", szMaxDims[4]);
		MKLOG("");
		// CL_DEVICE_EXTENSIONS: get device extensions, and if any then parse & log the string onto separate lines
		clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(device_string), &device_string, NULL);
		if (device_string != 0)
		{
			MKLOG("  CL_DEVICE_EXTENSIONS:");
			std::string stdDevString;
			stdDevString = std::string(device_string);
			size_t szOldPos = 0;
			size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
			while (szSpacePos != stdDevString.npos)
			{
				if (strcmp("cl_nv_device_attribute_query", stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0)
					nv_device_attibute_query = true;

				if (szOldPos > 0)
				{
					//MKLOG("\t\t");
				}
				MKLOG("\t\t\t%s", stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str());

				do {
					szOldPos = szSpacePos + 1;
					szSpacePos = stdDevString.find(' ', szOldPos);
				} while (szSpacePos == szOldPos);
			}
			//MKLOG("\n");
		}
		else
		{
			MKLOG("  CL_DEVICE_EXTENSIONS: None");
		}

		MKLOG("");
		

		// CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
		MKLOG("  CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\t");
		cl_uint vec_width[6];
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &vec_width[0], NULL);
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &vec_width[1], NULL);
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &vec_width[2], NULL);
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &vec_width[3], NULL);
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &vec_width[4], NULL);
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &vec_width[5], NULL);
		MKLOG("  CHAR %u, SHORT %u, INT %u, LONG %u, FLOAT %u, DOUBLE %u",
			vec_width[0], vec_width[1], vec_width[2], vec_width[3], vec_width[4], vec_width[5]);
	}

	void QueryDeviceProps(cl_device_id device, cl_context cxGPUContext)
	{
		oclPrintDevInfo(device);

		// Determine and show image format support 
		cl_uint uiNumSupportedFormats = 0;

		// 2D
		clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY,
			CL_MEM_OBJECT_IMAGE2D,
			0, NULL, &uiNumSupportedFormats);
		cl_image_format* ImageFormats = new cl_image_format[uiNumSupportedFormats];
		clGetSupportedImageFormats(cxGPUContext, CL_MEM_READ_ONLY,
			CL_MEM_OBJECT_IMAGE2D,
			uiNumSupportedFormats, ImageFormats, NULL);

		MKLOG("");
		MKLOG("  ---------------------------------");
		MKLOG("  2D Image Formats Supported (%u)", uiNumSupportedFormats);
		MKLOG("  ---------------------------------");
		MKLOG("  %-6s%-16s%-22s", "#", "Channel Order", "Channel Type");
		for (unsigned int i = 0; i < uiNumSupportedFormats; i++)
		{
			MKLOG("  %-6u%-16s%-22s", (i + 1),
				oclImageFormatString(ImageFormats[i].image_channel_order),
				oclImageFormatString(ImageFormats[i].image_channel_data_type));
		}
		MKLOG("");
		delete[] ImageFormats;
	}





	OpenCL::LocalMemory::LocalMemory(size_t aSize) : mSize(aSize)
	{
	}

	size_t OpenCL::LocalMemory::GetSize()
	{
		return mSize;
	}

	int createContext(int platformID, int deviceID, cl_context* context, cl_command_queue* queue, cl_device_id* device)
	{
		// Get OpenCL platform count
		cl_uint NumPlatforms;
		cl_int errCode = CL_SUCCESS;

		openCLSafeCall(clGetPlatformIDs(0, NULL, &NumPlatforms));


		if (platformID >= (int)NumPlatforms || platformID < 0)
		{
			return -1;
		}
		
		// Get all OpenCL platform IDs
		cl_platform_id* PlatformIDs;
		PlatformIDs = new cl_platform_id[NumPlatforms];
		openCLSafeCall(clGetPlatformIDs(NumPlatforms, PlatformIDs, NULL));

		//get all devices supported by platform
		cl_uint num_devices;
		openCLSafeCall(clGetDeviceIDs(PlatformIDs[platformID], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));

		if (deviceID >= (int)num_devices || deviceID < 0)
		{
			delete[] PlatformIDs;
			return -2;
		}

		cl_device_id* devices;
		devices = new cl_device_id[num_devices];
		openCLSafeCall(clGetDeviceIDs(PlatformIDs[platformID], CL_DEVICE_TYPE_ALL, num_devices * sizeof(cl_device_id), devices, NULL));
			
		// Create a context
		*context = clCreateContext(0, 1, &devices[deviceID], NULL, NULL, &errCode);
		openCLSafeCall(errCode);

		*queue = clCreateCommandQueue(*context, devices[deviceID], 0, &errCode);
		openCLSafeCall(errCode);

		char cBuffer[1024];
		clGetDeviceInfo(devices[deviceID], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
		MKLOG("Context created on device '%s'", &cBuffer[0]);
		*device = devices[deviceID];

		delete[] PlatformIDs;
		delete[] devices;

		return 0;
	}

	int createContextOpenGL(int& platformID, cl_context* context, cl_command_queue* queue, cl_device_id* device)
	{
		// Set up OpenCL.
		cl_uint n;
		cl_int err = clGetPlatformIDs(0, 0, &n);
		openCLSafeCall(err);

		if (n == 0) 
		{
			throw OpenCLException("No openCL devices found");
		}
		
		cl_platform_id* platformIds = new cl_platform_id[n];
		err = clGetPlatformIDs(n, platformIds, 0);
		openCLSafeCall(err);

		for (platformID	= 0; platformID < n; platformID++)
		{
			cl_platform_id platform = platformIds[platformID];

		// Set up the context with OpenCL/OpenGL interop.
#if defined (_USE_APPLE_COMPILER_SETTINGS)
		cl_context_properties contextProps[] = { CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
			(cl_context_properties)CGLGetShareGroup(CGLGetCurrentContext()),
			0 };
#elif defined(_USE_WINDOWS_COMPILER_SETTINGS)
			cl_context_properties contextProps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
				CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
				CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
				0 };
#elif defined(_USE_LINUX_COMPILER_SETTINGS)
		QVariant nativeGLXHandle = QOpenGLContext::currentContext()->nativeHandle();
		QGLXNativeContext nativeGLXContext;
		if (!nativeGLXHandle.isNull() && nativeGLXHandle.canConvert<QGLXNativeContext>())
			nativeGLXContext = nativeGLXHandle.value<QGLXNativeContext>();
		else
			qWarning("Failed to get the underlying GLX context from the current QOpenGLContext");
		cl_context_properties contextProps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
			CL_GL_CONTEXT_KHR, (cl_context_properties)nativeGLXContext.context(),
			CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
			0 };
#endif

		// Get the GPU device id
#if defined(_USE_APPLE_COMPILER_SETTINGS)
		// On OS X, get the "online" device/GPU. This is required for OpenCL/OpenGL context sharing.
		err = clGetGLContextInfoAPPLE(m_clContext, CGLGetCurrentContext(),
			CL_CGL_DEVICE_FOR_CURRENT_VIRTUAL_SCREEN_APPLE,
			sizeof(cl_device_id), &m_clDeviceId, 0);
		if (err != CL_SUCCESS) {
			qWarning("Failed to get OpenCL device for current screen: %d", err);
			return;
		}
#else
			clGetGLContextInfoKHR_fn getGLContextInfo = NULL;
			getGLContextInfo = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddressForPlatform(platform, "clGetGLContextInfoKHR");

			if (getGLContextInfo)
			{
				size_t retSize = 0;
				cl_int error = getGLContextInfo(contextProps, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, 0, 0, &retSize);
				//openCLSafeCall(error);
				*context = 0;
				if (retSize > 0)
				{
					error = getGLContextInfo(contextProps, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, sizeof(cl_device_id), device, 0);
					openCLSafeCall(error);
					cl_uint devCount = 0;
					err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, 0, &devCount);
					openCLSafeCall(err);
					if (devCount > 0)
					{
						err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, device, 0);
						openCLSafeCall(err);
						*context = clCreateContext(contextProps, 1, device, 0, 0, &err);
						if (err == CL_SUCCESS)
						{
							QueryDeviceProps(*device, *context);
							break;
						}
					}
				}
			}
#endif
		}
		if (!(*context) || err != CL_SUCCESS)
		{
			throw OpenCLException("Failed to create OpenCL context shared with OpenGL");
		}

		*queue = clCreateCommandQueue(*context, *device, 0, &err);
		openCLSafeCall(err);

		char cBuffer[1024];
		clGetDeviceInfo(*device, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
		MKLOG("Context created on device '%s'", &cBuffer[0]);

		delete[] platformIds;
		return 0;
	}

	cl_context OpenCLThreadBoundContext::GetCtx(int platformID, int deviceID)
	{
		std::lock_guard<std::recursive_mutex> lock(_mutex);

		if (!_instance)
		{
			_instance = new OpenCLThreadBoundContext();
			MKLOG("OpenCLThreadBoundContext created.");

			cl_context ctx;
			cl_command_queue queue;
			cl_device_id device;

			auto ret = RunInOpenCLThread(createContext, platformID, deviceID, &ctx, &queue, &device);
			if (ret.get() >= 0)
			{
				_instance->_ctx = ctx;
				_instance->_queue = queue;
				_instance->_device_id = device;
				MKLOG("OpenCL context created on platform/device %d/%d.", platformID, deviceID);
			}
			else
			{
				throw OpenCLException("Could not create OpenCL context");
			}
		}
		return _instance->_ctx;
	}

	cl_context OpenCLThreadBoundContext::GetCtxOpenGL()
	{
		std::lock_guard<std::recursive_mutex> lock(_mutex);
		int platformID = 0;

		if (!_instance)
		{
			_instance = new OpenCLThreadBoundContext();
			MKLOG("OpenCLThreadBoundContext created.");

			cl_context ctx;
			cl_command_queue queue;
			cl_device_id device;

			auto ret = RunInOpenCLThread(createContextOpenGL, platformID, &ctx, &queue, &device);
			if (ret.get() >= 0)
			{
				_instance->_ctx = ctx;
				_instance->_queue = queue;
				_instance->_device_id = device;
				MKLOG("OpenCL context created on platform %d.", platformID);
			}
			else
			{
				throw OpenCLException("Could not create OpenCL context");
			}
		}
		return _instance->_ctx;
	}

	cl_command_queue OpenCLThreadBoundContext::GetQueue()
	{
		if (!_instance)
		{
			throw OpenCLException("Error getting OpenCL command queue: No OpenCL context created");
		}
		return _instance->_queue;
	}

	cl_device_id OpenCLThreadBoundContext::GetDeviceID()
	{
		if (!_instance)
		{
			throw OpenCLException("Error getting OpenCL command queue: No OpenCL context created");
		}
		return _instance->_device_id;
	}

	int OpenCLThreadBoundContext::GetProgram(cl_program * program, const char * code, const size_t codeLength)
	{
		cl_int errCode = CL_SUCCESS;
		auto ret = RunInOpenCLThread(clCreateProgramWithSource, GetCtx(), 1, &code, &codeLength, &errCode);
		*program = ret.get();
		openCLSafeCall(errCode);
		return 0;
	}

	int build(cl_program program, const char* options)
	{
		return clBuildProgram(program, 0, NULL, options, NULL, NULL);
	}

	int buildInfo(cl_program program, char** logBuffer)
	{
		size_t lenBuffer;
		clGetProgramBuildInfo(program, OpenCLThreadBoundContext::GetDeviceID(), CL_PROGRAM_BUILD_LOG, 0, NULL, &lenBuffer);
		*logBuffer = new char[lenBuffer];
		return clGetProgramBuildInfo(program, OpenCLThreadBoundContext::GetDeviceID(), CL_PROGRAM_BUILD_LOG, lenBuffer, *logBuffer, NULL);
	}

	int OpenCLThreadBoundContext::Build(cl_program program, const char* options)
	{
		cl_int errCode = CL_SUCCESS;
		auto ret = RunInOpenCLThread(build, program, options);
		errCode = ret.get();

		char* logBuffer = NULL;
		if (errCode != CL_SUCCESS)
		{
			auto erg = RunInOpenCLThread(buildInfo, program, &logBuffer);
			erg.get();
			MKLOG("OpenCL Build error log:\n%s", logBuffer);
		}
		openCLSafeCallMsg(errCode, logBuffer);
		delete[] logBuffer;
		MKLOG("Building OpenCL-program with options '%s'.", options);
		return 0;
	}

	int OpenCLThreadBoundContext::GetProgramAndBuild(cl_program * program, const char * code, const size_t codeLength, const char * options)
	{
		int ret = GetProgram(program, code, codeLength);
		if (ret != 0) return ret;

		ret = Build(*program, options);
		return ret;
	}

	int OpenCLThreadBoundContext::GetKernel(cl_program program, cl_kernel* kernel, const char * kernelName)
	{
		cl_int errCode = CL_SUCCESS;
		auto ret = RunInOpenCLThread(clCreateKernel, program, kernelName, &errCode);
		*kernel = ret.get();
		openCLSafeCall(errCode);
		MKLOG("Get OpenCL-kernel '%s'.", kernelName);
		return 0;
	}

	int OpenCLThreadBoundContext::GetKernel(cl_program * program, cl_kernel * kernel, const char * code, const size_t codeLength, const char * kernelName, const char * options)
	{
		int ret = GetProgramAndBuild(program, code, codeLength, options);
		if (ret != 0)
			return ret;
		return GetKernel(*program, kernel, kernelName);
	}

	void OpenCLThreadBoundContext::Sync()
	{
		auto ret = RunInOpenCLThread(clFlush, GetQueue());
		openCLSafeCall(ret.get());
		ret = RunInOpenCLThread(clFinish, GetQueue());
		openCLSafeCall(ret.get());
	}

	int acquire(cl_mem buffer, cl_command_queue queue)
	{
		return clEnqueueAcquireGLObjects(queue, 1, &buffer, 0, 0, 0);
	}

	int release(cl_mem buffer, cl_command_queue queue)
	{
		return clEnqueueReleaseGLObjects(queue, 1, &buffer, 0, 0, 0);
	}

	void OpenCLThreadBoundContext::AcquireOpenGL(cl_mem buffer)
	{
		cl_command_queue queue = GetQueue();
		auto ret = RunInOpenCLThread(acquire, buffer, queue);
		openCLSafeCall(ret.get());
	}

	void OpenCLThreadBoundContext::ReleaseOpenGL(cl_mem buffer)
	{
		cl_command_queue queue = GetQueue();
		auto ret = RunInOpenCLThread(release, buffer, queue);
		openCLSafeCall(ret.get());
	}

	void OpenCLThreadBoundContext::Cleanup()
	{
		std::lock_guard<std::recursive_mutex> lock(_mutex);
		if (_instance)
		{
			if (_instance->_queue)
			{
				auto ret = RunInOpenCLThread(clReleaseCommandQueue, _instance->_queue);
				openCLSafeCall(ret.get());
				_instance->_queue = 0;
				MKLOG("OpenCL command queue released.");
			}
			if (_instance->_ctx)
			{
				auto ret = RunInOpenCLThread(clReleaseContext, _instance->_ctx);
				openCLSafeCall(ret.get());
				_instance->_ctx = 0;
				MKLOG("OpenCL context released.");
			}
			delete _instance;
			_instance = NULL;
			MKLOG("OpenCLThreadBoundContext deleted.");
		}
	}

	void OpenCLThreadBoundContext::Release(cl_program program)
	{
		auto ret = RunInOpenCLThread(clReleaseProgram, program);
		openCLSafeCall(ret.get());
	}

	void OpenCLThreadBoundContext::Release(cl_kernel kernel)
	{
		auto ret = RunInOpenCLThread(clReleaseKernel, kernel);
		openCLSafeCall(ret.get());
	}

	cl_int OpenCLThreadBoundContext::EnqueueNDRange1(cl_kernel aKernel, size_t global_work_size, size_t local_work_size)
        {
            return clEnqueueNDRangeKernel(GetQueue(), aKernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
            //return clEnqueueNDRangeKernel(GetQueue(), aKernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
	}

	cl_int OpenCLThreadBoundContext::EnqueueNDRange(cl_kernel aKernel, cl_uint work_dim, size_t* global_work_size, size_t* local_work_size)
        {
            return clEnqueueNDRangeKernel(GetQueue(), aKernel, work_dim, NULL, global_work_size, NULL, 0, NULL, NULL);
            //return clEnqueueNDRangeKernel(GetQueue(), aKernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	}

	OpenCLThreadBoundContext* OpenCLThreadBoundContext::_instance = NULL;
	std::recursive_mutex OpenCLThreadBoundContext::_mutex;
}
