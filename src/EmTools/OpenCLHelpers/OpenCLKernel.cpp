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


#include "OpenCLKernel.h"

OpenCL::OpenCLKernel::OpenCLKernel(cl_kernel aKernel) :
	_kernel(aKernel), _program(0), _workDim(1), _globalWorkSize(new size_t[3]), _localWorkSize(new size_t[3])
{
	_globalWorkSize[0] = 1;
	_globalWorkSize[1] = 0;
	_globalWorkSize[2] = 0;
	_localWorkSize[0] = 1;
	_localWorkSize[1] = 0;
	_localWorkSize[2] = 0;
	size_t kernel_work_group_size;
	clGetKernelWorkGroupInfo(aKernel, OpenCLThreadBoundContext::GetDeviceID(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_maxWorkGroupSize, NULL);
}

OpenCL::OpenCLKernel::OpenCLKernel(std::string aKernelName, cl_program aProgram) :
_kernel(0), _program(0), _workDim(1), _globalWorkSize(new size_t[3]), _localWorkSize(new size_t[3])
{
	//we are not the owner of the cl_program, why we won't clean it up in the destructor: init it to 0.
	OpenCLThreadBoundContext::GetKernel(aProgram, &_kernel, aKernelName.c_str());
	_globalWorkSize[0] = 1;
	_globalWorkSize[1] = 0;
	_globalWorkSize[2] = 0;
	_localWorkSize[0] = 1;
	_localWorkSize[1] = 0;
	_localWorkSize[2] = 0;
	clGetKernelWorkGroupInfo(_kernel, OpenCLThreadBoundContext::GetDeviceID(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_maxWorkGroupSize, NULL);
}

OpenCL::OpenCLKernel::OpenCLKernel(std::string aKernelName, const unsigned char * code, const char * options) :
_kernel(0), _program(0), _workDim(1), _globalWorkSize(new size_t[3]), _localWorkSize(new size_t[3])
{
	size_t codeLength = strlen((char*)code);
	OpenCLThreadBoundContext::GetKernel(&_program, &_kernel, (char*)code, codeLength, aKernelName.c_str(), options);
	_globalWorkSize[0] = 1;
	_globalWorkSize[1] = 0;
	_globalWorkSize[2] = 0;
	_localWorkSize[0] = 1;
	_localWorkSize[1] = 0;
	_localWorkSize[2] = 0;
	clGetKernelWorkGroupInfo(_kernel, OpenCLThreadBoundContext::GetDeviceID(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_maxWorkGroupSize, NULL);
}

OpenCL::OpenCLKernel::OpenCLKernel(std::string aKernelName, const char * code, const char * options) :
_kernel(0), _program(0), _workDim(1), _globalWorkSize(new size_t[3]), _localWorkSize(new size_t[3])
{
	size_t codeLength = strlen(code);
	OpenCLThreadBoundContext::GetKernel(&_program, &_kernel, code, codeLength, aKernelName.c_str(), options);
	_globalWorkSize[0] = 1;
	_globalWorkSize[1] = 0;
	_globalWorkSize[2] = 0;
	_localWorkSize[0] = 1;
	_localWorkSize[1] = 0;
	_localWorkSize[2] = 0;
	clGetKernelWorkGroupInfo(_kernel, OpenCLThreadBoundContext::GetDeviceID(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_maxWorkGroupSize, NULL);
}

OpenCL::OpenCLKernel::~OpenCLKernel()
{
	if (_globalWorkSize)
		delete[] _globalWorkSize;
	_globalWorkSize = NULL;
	if (_localWorkSize)
		delete[] _localWorkSize;
	_localWorkSize = NULL;

	if (_kernel)
		OpenCLThreadBoundContext::Release(_kernel);
	_kernel = 0;

	if (_program)
		OpenCLThreadBoundContext::Release(_program);
	_program = 0;
}

void OpenCL::OpenCLKernel::SetProblemSize(uint aWorkDim, size_t aLocalWorkSize[3], size_t aProblemSize[3])
{
	if (aWorkDim > 3 || aWorkDim == 0)
	{
		throw std::invalid_argument("aWorkDim must be in range 1..3");
	}

	_workDim = aWorkDim;
	for (size_t i = 0; i < aWorkDim; i++)
	{
		_localWorkSize[i] = aLocalWorkSize[i];
	}
	for (size_t i = 0; i < aWorkDim; i++)
	{
		_globalWorkSize[i] = ((aProblemSize[i] + _localWorkSize[i] - 1) / _localWorkSize[i]) * _localWorkSize[i];
	}
}

void OpenCL::OpenCLKernel::SetProblemSize(size_t aLocalWorkSize, size_t aProblemSize)
{
	SetProblemSize(1, &aLocalWorkSize, &aProblemSize);
}

void OpenCL::OpenCLKernel::SetProblemSize(size_t aLocalWorkSizeX, size_t aLocalWorkSizeY, size_t aProblemSizeX, size_t aProblemSizeY)
{
	size_t tempLocal[3] = {aLocalWorkSizeX, aLocalWorkSizeY, 0};
	size_t tempWork[3] = { aProblemSizeX, aProblemSizeY, 0 };
	SetProblemSize(2, tempLocal, tempWork);
}

void OpenCL::OpenCLKernel::SetProblemSize(size_t aLocalWorkSizeX, size_t aLocalWorkSizeY, size_t aLocalWorkSizeZ, size_t aProblemSizeX, size_t aProblemSizeY, size_t aProblemSizeZ)
{
	size_t tempLocal[3] = { aLocalWorkSizeX, aLocalWorkSizeY, aLocalWorkSizeZ };
	size_t tempWork[3] = { aProblemSizeX, aProblemSizeY, aProblemSizeZ };
	SetProblemSize(3, tempLocal, tempWork);
}
