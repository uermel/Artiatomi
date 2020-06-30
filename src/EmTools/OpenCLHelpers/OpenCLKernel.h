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


#ifndef OPENCLKERNEL_H
#define OPENCLKERNEL_H
#include "../Basics/Default.h"
#ifdef USE_OPENCL
#include "OpenCLHelpers.h"
#include "OpenCLException.h"
#include "OpenCLDeviceVariable.h"

namespace OpenCL
{
	
	//!  A wrapper class for an OpenCL kernel.
	/*!
	The wrapped cl_kernel is released in destructor.
	The wrapped cl_program is released if kernel was created from source code.
	In contrast to the CUDA counterpart, all OpenCL API calls are performed on the unique OpenCL host thread, i.e. all methods can safely be called from any host thread without locking.
	\author Michael Kunz
	\date   November 2016
	\version 1.0
	*/
	//A wrapper class for an OpenCL kernel.
	class OpenCLKernel
	{
	private:
		cl_kernel _kernel;
		cl_program _program;
		uint _workDim;
		size_t* _globalWorkSize;
		size_t* _localWorkSize;
		size_t _maxWorkGroupSize;

	public:
		//! Instantiates a wrapper around aKernel
		OpenCLKernel(cl_kernel aKernel);
		//! Instantiates a wrapper around a kernel from already built program \p aProgram
		OpenCLKernel(std::string aKernelName, cl_program aProgram);
		//! Instantiates a wrapper around a kernel from source code
		OpenCLKernel(std::string aKernelName, const unsigned char* code, const char* options = NULL);
		//! Instantiates a wrapper around a kernel from source code
		OpenCLKernel(std::string aKernelName, const char* code, const char* options = NULL);
		//! Releases all internally created OpenCL handlers.
		~OpenCLKernel();

		//! Executes the kernel with previously set work sizes.
		template<class... Args1> void Run(Args1&&... args)
		{
			OpenCLThreadBoundContext::Sync();
			if (_workDim == 1)
				OpenCLThreadBoundContext::Execute(_kernel, *_globalWorkSize, *_localWorkSize, args...);
			else
				OpenCLThreadBoundContext::Execute(_kernel, _workDim, _globalWorkSize, _localWorkSize, args...);
			OpenCLThreadBoundContext::Sync();
		}

		//! Set the global and local work sizes (multi-dimensional). Has to be called before first kernel launch.
		void SetProblemSize(uint aWorkDim, size_t aLocalWorkSize[3], size_t aProblemSize[3]);
		//! Set the global and local work sizes (one dimensional). Has to be called before first kernel launch.
		void SetProblemSize(size_t aLocalWorkSize, size_t aProblemSize);
		//! Set the global and local work sizes (one dimensional). Has to be called before first kernel launch.
		void SetProblemSize(size_t aLocalWorkSizeX, size_t aLocalWorkSizeY, size_t aProblemSizeX, size_t aProblemSizeY);
		//! Set the global and local work sizes (one dimensional). Has to be called before first kernel launch.
		void SetProblemSize(size_t aLocalWorkSizeX, size_t aLocalWorkSizeY, size_t aLocalWorkSizeZ, size_t aProblemSizeX, size_t aProblemSizeY, size_t aProblemSizeZ);
	};
}

#endif //USE_OPENCL
#endif //OPENCLKERNEL_H
