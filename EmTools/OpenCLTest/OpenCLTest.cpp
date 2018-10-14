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


// OpenCLTest.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include "../Basics/Default.h"
#include "../OpenCLHelpers/OpenCLHelpers.h"
#include "../OpenCLHelpers/OpenCLDeviceVariable.h"
#include "../OpenCLHelpers/OpenCLKernel.h"
#include "../MKLog/MKLog.h"


//const char* kernel =  "__kernel void VectorAdd(__global const float* a, __global const float* b, __global float* c, int iNumElements)\n"
//							"{\n"
//							"	// get index into global data array\n"
//							"	int iGID = get_global_id(0);\n"
//
//							"	// bound check (equivalent to the limit on a 'for' loop for standard/serial C code\n"
//							"	if (iGID >= iNumElements)\n"
//							"	{\n"
//							"		return;\n"
//							"	}\n"
//
//							"	// add the vector elements\n"
//							"	c[iGID] = a[iGID] + b[iGID];\n"
//							"}";
#include "bin_kernel.h"

int dimX = 10;
int dimY = 10;
int iNumElements = dimX * dimY;


int main()
{
	MKLog::Init("openCL.log");
	try
	{
		OpenCL::OpenCLThreadBoundContext::GetCtx(0, 0);
		OpenCL::OpenCLKernel kernel("Test2D", kernel);

		OpenCL::OpenCLDeviceVariable devSrcA(iNumElements * sizeof(float));
		OpenCL::OpenCLDeviceVariable devSrcB(iNumElements * sizeof(float));
		OpenCL::OpenCLDeviceVariable devDest(iNumElements * sizeof(float));
		float* srcA = new float[iNumElements];
		float* srcB = new float[iNumElements];
		float* dest = new float[iNumElements];

		for (size_t i = 0; i < iNumElements; i++)
		{
			srcA[i] = (float)i;
			srcB[i] = (float)i;
			dest[i] = 0;
		}

		devSrcA.CopyHostToDevice(srcA);
		devSrcB.CopyHostToDevice(srcB);

		kernel.SetProblemSize(16, 8, dimX, dimY);
		kernel.Run(OpenCL::LocalMemory(1024), devSrcA.GetDevicePtr(), devSrcB.GetDevicePtr(), devDest.GetDevicePtr(), dimX, dimY);

		devDest.CopyDeviceToHost(dest);
		try
		{
		}
		catch (const std::exception& ex)
		{
			printf(ex.what());
		}
		printf("");
	}
	catch (const std::exception& ex)
	{
		printf(ex.what());
	}
	OpenCL::OpenCLThreadBoundContext::Cleanup();

    return 0;
}

