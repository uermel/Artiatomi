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


#ifndef COPYTOSQUARE_CU
#define COPYTOSQUARE_CU


#include <cuda.h>
#include "cutil.h"
#include "cutil_math.h"

#include "DeviceVariables.cuh"
#include "float.h"

  
extern "C"
__global__ 
void makeSquare(int proj_x, int proj_y, int maxsize, int stride, float* aIn, float* aOut, int borderSizeX, int borderSizeY, bool mirrorY, bool fillZero)
{
	// integer pixel coordinates	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;	

	if (x >= maxsize || y >= maxsize)
		return;

	if (fillZero)
	{
		float val = 0;

		int xIn = x - borderSizeX;
		int yIn = y - borderSizeY;

		if (xIn >= 0 && xIn < proj_x && yIn >= 0 && yIn < proj_y)
		{
			if (mirrorY)
			{
				yIn = proj_y - yIn - 1;
			}			
			val = *(((float*)((char*)aIn + stride * yIn)) + xIn);
		}
		aOut[y * maxsize + x] = val;
	}
	else //wrap
	{
		int xIn = x - borderSizeX;
		if (xIn < 0) xIn = -xIn - 1;
		if (xIn >= proj_x)
		{
			xIn = xIn - proj_x;
			xIn = proj_x - xIn - 1;
		}

		int yIn = y - borderSizeY;
		if (yIn < 0) yIn = -yIn - 1;
		if (yIn >= proj_y)
		{
			yIn = yIn - proj_y;
			yIn = proj_y - yIn - 1;
		}
		if (mirrorY)
		{
			yIn = proj_y - yIn - 1;
		}
	
		aOut[y * maxsize + x] = *(((float*)((char*)aIn + stride * yIn)) + xIn);
	}
}

#endif
