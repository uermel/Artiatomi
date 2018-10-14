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


#ifndef OPENCLHELPERS_H
#define OPENCLHELPERS_H

#include "../Basics/Default.h"

#ifdef USE_OPENCL
#include "../Threading/ThreadPool.h"
#include "../Threading/SpecificBackgroundThread.h"
#include "OpenCLException.h"
#include <CL/cl.h>
#include <mutex>

namespace OpenCL
{
	class LocalMemory
	{
		size_t mSize;
	public:
		LocalMemory(size_t aSize);
		size_t GetSize();
	};

	class OpenCLThreadBoundContext
	{
	public:
		static cl_context GetCtx(int platformID = 0, int deviceID = 0);
		static cl_context GetCtxOpenGL();
		static cl_command_queue GetQueue();
		static cl_device_id GetDeviceID();
		static int GetProgram(cl_program* program, const char* code, const size_t codeLength);
		static int Build(cl_program program, const char* options = NULL);
		static int GetProgramAndBuild(cl_program* program, const char* code, const size_t codeLength, const char* options = NULL);
		static int GetKernel(cl_program program, cl_kernel* kernel, const char* kernelName);
		static int GetKernel(cl_program* program, cl_kernel* kernel, const char* code, const size_t codeLength, const char* kernelName, const char* options = NULL);
		static void Sync();
		static void Cleanup();
		static void Release(cl_program program);
		static void Release(cl_kernel kernel);
		static void AcquireOpenGL(cl_mem buffer);
		static void ReleaseOpenGL(cl_mem buffer);

		template<class F1, class... Args1> static inline auto Run(F1&& f, Args1&&... args)
		{
			return SingletonThread::Get(OPENCL_THREAD_ID)->enqueue(f, args...);
		}

		template<class... Args1> static void Execute(cl_kernel aKernel, cl_uint work_dim, size_t* global_work_size, size_t* local_work_size, Args1&&... args)
		{
			//Set arguments
			cl_uint argCounter = 0;
			AppendArgument(aKernel, &argCounter, args...);

			// Launch kernel
			auto erg = RunInOpenCLThread(EnqueueNDRange, aKernel, work_dim, global_work_size, local_work_size);
			openCLSafeCall(erg.get());
		}
		template<class... Args1> static void Execute(cl_kernel aKernel, size_t global_work_size, size_t local_work_size, Args1&&... args)
		{
			//Set arguments
			cl_uint argCounter = 0;
			AppendArgument(aKernel, &argCounter, args...);

			// Launch kernel
			auto erg = RunInOpenCLThread(EnqueueNDRange1, aKernel, global_work_size, local_work_size);
			openCLSafeCall(erg.get());
		}
	private:
		static std::recursive_mutex _mutex;
		static OpenCLThreadBoundContext* _instance;
		cl_context _ctx;
		cl_command_queue _queue;
		cl_device_id _device_id;

		template<typename Arg>
		static void AppendArgument(cl_kernel aKernel, cl_uint* argCounter, Arg arg)
		{
			auto erg = RunInOpenCLThread(clSetKernelArg, aKernel, *argCounter, sizeof(Arg), (void*)&arg);
			openCLSafeCall(erg.get());
			(*argCounter)++;
		}

		template<typename Arg, typename ...Args>
		static void AppendArgument(cl_kernel aKernel, cl_uint* argCounter, Arg arg, Args&&... args)
		{
			auto erg = RunInOpenCLThread(clSetKernelArg, aKernel, *argCounter, sizeof(Arg), (void*)&arg);
			openCLSafeCall(erg.get());
			(*argCounter)++;
			AppendArgument(aKernel, argCounter, args...);
		}
		//template<>
		static void AppendArgument(cl_kernel aKernel, cl_uint* argCounter, LocalMemory arg)
		{
			auto erg = RunInOpenCLThread(clSetKernelArg, aKernel, *argCounter, arg.GetSize(), (void*)NULL);
			openCLSafeCall(erg.get());
			(*argCounter)++;
		}

		template<typename ...Args>
		static void AppendArgument(cl_kernel aKernel, cl_uint* argCounter, LocalMemory arg, Args&&... args)
		{
			auto erg = RunInOpenCLThread(clSetKernelArg, aKernel, *argCounter, arg.GetSize(), (void*)NULL);
			openCLSafeCall(erg.get());
			(*argCounter)++;
			AppendArgument(aKernel, argCounter, args...);
		}
		static cl_int EnqueueNDRange1(cl_kernel aKernel, size_t global_work_size, size_t local_work_size);
		static cl_int EnqueueNDRange(cl_kernel aKernel, cl_uint work_dim, size_t* global_work_size, size_t* local_work_size);

		OpenCLThreadBoundContext() = default;
		OpenCLThreadBoundContext(const OpenCLThreadBoundContext&) = delete;
		OpenCLThreadBoundContext(OpenCLThreadBoundContext&&) = delete;
	};
}


#endif //USE_OPENCL
#endif // !OPENCLHELPERS_H
