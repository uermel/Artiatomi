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

extern "C"
__global__
void rect2squareSlices(int proj_x,
                      int proj_y,
                      int sliceNumber,
                      int maxsize,
                      int stride,
                      float* aIn,
                      float* aOut,
                      int borderSizeX,
                      int borderSizeY,
                      bool mirrorY,
                      bool fillZero)
{
    // integer pixel coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= maxsize || y >= maxsize || z >= sliceNumber)
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
        aOut[z * maxsize * maxsize + y * maxsize + x] = val;
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

        aOut[z * maxsize * maxsize + y * maxsize + x] = *(((float*)((char*)aIn + stride * yIn)) + xIn);
    }
}

extern "C"
__global__
void squareSlices2rectSlices(int proj_x, int proj_y, int sliceNumber, int maxsize, float* aIn, float* aOut, int borderSizeX, int borderSizeY, bool mirrorY, bool fillZero)
{
    // integer pixel coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= maxsize || y >= maxsize || z >= sliceNumber)
        return;

    if (fillZero)
    {
        float val = 0;

        int xOut = x - borderSizeX;
        int yOut = y - borderSizeY;

        if (xOut >= 0 && xOut < proj_x && yOut >= 0 && yOut < proj_y)
        {
            if (mirrorY)
            {
                yOut = proj_y - yOut - 1;
            }
            //val = *(((float*)((char*)aIn + stride * yIn)) + xIn);

            //aOut[z * proj_y * proj_x + yOut * proj_x + xOut]
            aOut[z * proj_y * proj_x + yOut * proj_x + xOut] = aIn[z * maxsize * maxsize + y * maxsize + x];
        }
        //aOut[z * slicenum * maxsize + y * maxsize + x] = val;
    }
    else //wrap
    {
        int xOut = x - borderSizeX;
        if (xOut < 0) xOut = -xOut - 1;
        if (xOut >= proj_x)
        {
            xOut = xOut - proj_x;
            xOut = proj_x - xOut - 1;
        }

        int yOut = y - borderSizeY;
        if (yOut < 0) yOut = -yOut - 1;
        if (yOut >= proj_y)
        {
            yOut = yOut - proj_y;
            yOut = proj_y - yOut - 1;
        }
        if (mirrorY)
        {
            yOut = proj_y - yOut - 1;
        }

        //aOut[z * slicenum * maxsize + y * maxsize + x] = *(((float*)((char*)aIn + stride * yIn)) + xIn);
        //*(((float*)((char*)aIn + stride * yIn)) + xIn) = aOut[z * slicenum * maxsize + y * maxsize + x]
        aOut[z * proj_y * proj_x + yOut * proj_x + xOut] = aIn[z * maxsize * maxsize + y * maxsize + x];
    }
}

extern "C"
__global__
void rectSlices2squareSlices(int proj_x, int proj_y, int sliceNumber, int maxsize, float* aIn, float* aOut, int borderSizeX, int borderSizeY, bool mirrorY, bool fillZero)
{
    // integer pixel coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= maxsize || y >= maxsize || z >= sliceNumber)
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
            val = aIn[z * proj_x * proj_y + yIn * proj_x + xIn];//*(((float*)((char*)aIn + stride * yIn)) + xIn);
        }
        aOut[z * maxsize * maxsize + y * maxsize + x] = val;
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

        aOut[z * maxsize * maxsize + y * maxsize + x] = aIn[z * proj_x * proj_y + yIn * proj_x + xIn];//*(((float*)((char*)aIn + stride * yIn)) + xIn);
    }
}

extern "C"
__global__
void squareSlices2rect(int proj_x,
                       int proj_y,
                       int sliceNumber,
                       int maxsize,
                       int stride,
                       float* aIn,
                       float* aOut,
                       int borderSizeX,
                       int borderSizeY,
                       bool mirrorY,
                       bool fillZero)
{
    // integer pixel coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= maxsize || y >= maxsize || z >= sliceNumber)
        return;

    if (fillZero)
    {
        float val = 0;

        int xOut = x - borderSizeX;
        int yOut = y - borderSizeY;

        if (xOut >= 0 && xOut < proj_x && yOut >= 0 && yOut < proj_y)
        {
            if (mirrorY)
            {
                yOut = proj_y - yOut - 1;
            }
            //val = *(((float*)((char*)aIn + stride * yIn)) + xIn);

            //aOut[z * proj_y * proj_x + yOut * proj_x + xOut]
            //atomicAdd(aOut + (yOut * proj_x + xOut), aIn[z * maxsize * maxsize + y * maxsize + x]);
            atomicAdd((((float*)((char*)aOut + stride * yOut)) + xOut), aIn[z * maxsize * maxsize + y * maxsize + x]);
        }
        //aOut[z * slicenum * maxsize + y * maxsize + x] = val;
    }
    else //wrap
    {
        int xOut = x - borderSizeX;
        if (xOut < 0) xOut = -xOut - 1;
        if (xOut >= proj_x)
        {
            xOut = xOut - proj_x;
            xOut = proj_x - xOut - 1;
        }

        int yOut = y - borderSizeY;
        if (yOut < 0) yOut = -yOut - 1;
        if (yOut >= proj_y)
        {
            yOut = yOut - proj_y;
            yOut = proj_y - yOut - 1;
        }
        if (mirrorY)
        {
            yOut = proj_y - yOut - 1;
        }

        //aOut[z * slicenum * maxsize + y * maxsize + x] = *(((float*)((char*)aIn + stride * yIn)) + xIn);
        //*(((float*)((char*)aIn + stride * yIn)) + xIn) = aOut[z * slicenum * maxsize + y * maxsize + x]
        //aOut[z * proj_y * proj_x + yOut * proj_x + xOut] = aIn[z * maxsize * maxsize + y * maxsize + x];
        //atomicAdd(aOut + (yOut * proj_x + xOut), aIn[z * maxsize * maxsize + y * maxsize + x]);
        atomicAdd((((float*)((char*)aOut + stride * yOut)) + xOut), aIn[z * maxsize * maxsize + y * maxsize + x]);
    }
}


#endif
