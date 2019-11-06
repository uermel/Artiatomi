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


#include "baseframecontroller.h"
#include <QMessageBox>
#include <QDebug>


const char * BaseFrameController::openCLSource =

"__kernel void WriteToSurfaceKernel8uC1(__global unsigned char* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX || y >=dimY) return;\n"
" float pixel = imageIn[y * dimX + x] * scale;\n"
" pixel -= minVal;\n"
" pixel /= (maxVal - minVal);\n"
" pixel -= minContrast;\n"
" pixel /= (maxContrast - minContrast);\n"
" pixel = max(min(pixel, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel, 0.0, 0.0, 1.0);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"
"__kernel void WriteToSurfaceKernel8uC1Filter(__global unsigned char* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX-2 || y >=dimY-2) return;\n"
" if (x<2 || y <2) return;\n"
" float pixel = imageIn[y * dimX  + x] * 41 * scale;\n"
"       pixel += imageIn[(y+1) * dimX  + x+1] * 16 * scale;\n"
"       pixel += imageIn[(y+1) * dimX  + x] * 26 * scale;\n"
"       pixel += imageIn[(y+1) * dimX  + x-1] * 16 * scale;\n"
"       pixel += imageIn[y * dimX  + x+1] * 26 * scale;\n"
"       pixel += imageIn[y * dimX  + x-1] * 26 * scale;\n"
"       pixel += imageIn[(y-1) * dimX  + x+1] * 16 * scale;\n"
"       pixel += imageIn[(y-1) * dimX  + x] * 26 * scale;\n"
"       pixel += imageIn[(y-1) * dimX  + x-1] * 16 * scale;\n"

"       pixel += imageIn[(y+2) * dimX  + x+2] * 1 * scale;\n"
"       pixel += imageIn[(y+2) * dimX  + x+1] * 4 * scale;\n"
"       pixel += imageIn[(y+2) * dimX  + x] * 7 * scale;\n"
"       pixel += imageIn[(y+2) * dimX  + x-1] * 4 * scale;\n"
"       pixel += imageIn[(y+2) * dimX  + x-2] * 1 * scale;\n"

"       pixel += imageIn[(y-2) * dimX  + x+2] * 1 * scale;\n"
"       pixel += imageIn[(y-2) * dimX  + x+1] * 4 * scale;\n"
"       pixel += imageIn[(y-2) * dimX  + x] * 7 * scale;\n"
"       pixel += imageIn[(y-2) * dimX  + x-1] * 4 * scale;\n"
"       pixel += imageIn[(y-2) * dimX  + x-2] * 1 * scale;\n"

"       pixel += imageIn[(y+1) * dimX  + x-2] * 4 * scale;\n"
"       pixel += imageIn[(y+1) * dimX  + x+2] * 4 * scale;\n"
"       pixel += imageIn[(y-1) * dimX  + x+2] * 4 * scale;\n"
"       pixel += imageIn[(y-1) * dimX  + x-2] * 4 * scale;\n"
"       pixel += imageIn[(y) * dimX  + x+2] * 7 * scale;\n"
"       pixel += imageIn[(y) * dimX  + x-2] * 7 * scale;\n"
" pixel /= 273.0f;\n"
" pixel -= minVal;\n"
" pixel /= (maxVal - minVal);\n"
" pixel -= minContrast;\n"
" pixel /= (maxContrast - minContrast);\n"
" pixel = max(min(pixel, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel, 0.0, 0.0, 1.0);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"
"__kernel void WriteToSurfaceKernel8uC2(__global unsigned char* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX || y >=dimY) return;\n"
" float pixel1 = imageIn[y * dimX + x] * scale;\n"
" float pixel2 = imageIn[y * dimX + x + dimX*dimY] * scale; //this is a planar image\n"
" pixel1 -= minVal;\n"
" pixel1 /= (maxVal - minVal);\n"
" pixel2 -= minVal;\n"
" pixel2 /= (maxVal - minVal);\n"
" pixel1 -= minContrast;\n"
" pixel1 /= (maxContrast - minContrast);\n"
" pixel2 -= minContrast;\n"
" pixel2 /= (maxContrast - minContrast);\n"
" pixel1 = max(min(pixel1, 1.0f),0.0f);\n"
" pixel2 = max(min(pixel2, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel1, pixel2, 0, 1.0f);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"
"__kernel void WriteToSurfaceKernel8uC3(__global unsigned char* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX || y >=dimY) return;\n"
" float pixel1 = imageIn[y * dimX * 3 + 3*x] * scale;\n"
" float pixel2 = imageIn[y * dimX * 3 + 3*x + 1] * scale;\n"
" float pixel3 = imageIn[y * dimX * 3 + 3*x + 2] * scale;\n"
" pixel1 -= minVal;\n"
" pixel1 /= (maxVal - minVal);\n"
" pixel2 -= minVal;\n"
" pixel2 /= (maxVal - minVal);\n"
" pixel3 -= minVal;\n"
" pixel3 /= (maxVal - minVal);\n"
" pixel1 -= minContrast;\n"
" pixel1 /= (maxContrast - minContrast);\n"
" pixel2 -= minContrast;\n"
" pixel2 /= (maxContrast - minContrast);\n"
" pixel3 -= minContrast;\n"
" pixel3 /= (maxContrast - minContrast);\n"
" pixel1 = max(min(pixel1, 1.0f),0.0f);\n"
" pixel2 = max(min(pixel2, 1.0f),0.0f);\n"
" pixel3 = max(min(pixel3, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel1, pixel2, pixel3, 1.0f);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"
"__kernel void WriteToSurfaceKernel8uC4(__global unsigned char* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX || y >=dimY) return;\n"
" float pixel1 = imageIn[y * dimX * 4 + 4*x] * scale;\n"
" float pixel2 = imageIn[y * dimX * 4 + 4*x + 1] * scale;\n"
" float pixel3 = imageIn[y * dimX * 4 + 4*x + 2] * scale;\n"
" float pixel4 = imageIn[y * dimX * 4 + 4*x + 3] * scale;\n"
" pixel1 -= minVal;\n"
" pixel1 /= (maxVal - minVal);\n"
" pixel2 -= minVal;\n"
" pixel2 /= (maxVal - minVal);\n"
" pixel3 -= minVal;\n"
" pixel3 /= (maxVal - minVal);\n"
" pixel4 -= minVal;\n"
" pixel4 /= (maxVal - minVal);\n"
" pixel1 -= minContrast;\n"
" pixel1 /= (maxContrast - minContrast);\n"
" pixel2 -= minContrast;\n"
" pixel2 /= (maxContrast - minContrast);\n"
" pixel3 -= minContrast;\n"
" pixel3 /= (maxContrast - minContrast);\n"
" pixel4 -= minContrast;\n"
" pixel4 /= (maxContrast - minContrast);\n"
" pixel1 = max(min(pixel1, 1.0f),0.0f);\n"
" pixel2 = max(min(pixel2, 1.0f),0.0f);\n"
" pixel3 = max(min(pixel3, 1.0f),0.0f);\n"
" pixel4 = max(min(pixel4, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel1, pixel2, pixel3, pixel4);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"
"__kernel void WriteToSurfaceKernel16uC1(__global unsigned short* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX || y >=dimY) return;\n"
" float pixel1 = imageIn[y * dimX  + x] * scale;\n"
" pixel1 -= minVal;\n"
" pixel1 /= (maxVal - minVal);\n"
" pixel1 -= minContrast;\n"
" pixel1 /= (maxContrast - minContrast);\n"
" pixel1 = max(min(pixel1, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel1, 0.0, 0.0, 1.0);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"
"__kernel void WriteToSurfaceKernel16uC1Filter(__global unsigned short* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX-2 || y >=dimY-2) return;\n"
" if (x<2 || y <2) return;\n"
" float pixel1 = imageIn[y * dimX  + x] * 41 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x+1] * 16 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x] * 26 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x-1] * 16 * scale;\n"
"       pixel1 += imageIn[y * dimX  + x+1] * 26 * scale;\n"
"       pixel1 += imageIn[y * dimX  + x-1] * 26 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x+1] * 16 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x] * 26 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x-1] * 16 * scale;\n"

"       pixel1 += imageIn[(y+2) * dimX  + x+2] * 1 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x+1] * 4 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x] * 7 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x-1] * 4 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x-2] * 1 * scale;\n"

"       pixel1 += imageIn[(y-2) * dimX  + x+2] * 1 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x+1] * 4 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x] * 7 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x-1] * 4 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x-2] * 1 * scale;\n"

"       pixel1 += imageIn[(y+1) * dimX  + x-2] * 4 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x+2] * 4 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x+2] * 4 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x-2] * 4 * scale;\n"
"       pixel1 += imageIn[(y) * dimX  + x+2] * 7 * scale;\n"
"       pixel1 += imageIn[(y) * dimX  + x-2] * 7 * scale;\n"
" pixel1 /= 273.0f;\n"
" pixel1 -= minVal;\n"
" pixel1 /= (maxVal - minVal);\n"
" pixel1 -= minContrast;\n"
" pixel1 /= (maxContrast - minContrast);\n"
" pixel1 = max(min(pixel1, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel1, 0.0, 0.0, 1.0);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"
"__kernel void WriteToSurfaceKernel16sC1(__global short* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX || y >=dimY) return;\n"
" float pixel1 = imageIn[y * dimX  + x] * scale;\n"
" pixel1 -= minVal;\n"
" pixel1 /= (maxVal - minVal);\n"
" pixel1 -= minContrast;\n"
" pixel1 /= (maxContrast - minContrast);\n"
" pixel1 = max(min(pixel1, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel1, 0.0, 0.0, 1.0);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"
"__kernel void WriteToSurfaceKernel16sC1Filter(__global short* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX-2 || y >=dimY-2) return;\n"
" if (x<2 || y <2) return;\n"
" float pixel1 = imageIn[y * dimX  + x] * 41 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x+1] * 16 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x] * 26 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x-1] * 16 * scale;\n"
"       pixel1 += imageIn[y * dimX  + x+1] * 26 * scale;\n"
"       pixel1 += imageIn[y * dimX  + x-1] * 26 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x+1] * 16 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x] * 26 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x-1] * 16 * scale;\n"

"       pixel1 += imageIn[(y+2) * dimX  + x+2] * 1 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x+1] * 4 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x] * 7 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x-1] * 4 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x-2] * 1 * scale;\n"

"       pixel1 += imageIn[(y-2) * dimX  + x+2] * 1 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x+1] * 4 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x] * 7 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x-1] * 4 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x-2] * 1 * scale;\n"

"       pixel1 += imageIn[(y+1) * dimX  + x-2] * 4 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x+2] * 4 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x+2] * 4 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x-2] * 4 * scale;\n"
"       pixel1 += imageIn[(y) * dimX  + x+2] * 7 * scale;\n"
"       pixel1 += imageIn[(y) * dimX  + x-2] * 7 * scale;\n"
" pixel1 /= 273.0f;\n"
" pixel1 -= minVal;\n"
" pixel1 /= (maxVal - minVal);\n"
" pixel1 -= minContrast;\n"
" pixel1 /= (maxContrast - minContrast);\n"
" pixel1 = max(min(pixel1, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel1, 0.0, 0.0, 1.0);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"
"__kernel void WriteToSurfaceKernel32sC1(__global int* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX || y >=dimY) return;\n"
" float pixel1 = imageIn[y * dimX  + x] * scale;\n"
" pixel1 -= minVal;\n"
" pixel1 /= (maxVal - minVal);\n"
" pixel1 -= minContrast;\n"
" pixel1 /= (maxContrast - minContrast);\n"
" pixel1 = max(min(pixel1, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel1, 0.0, 0.0, 1.0);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"
"__kernel void WriteToSurfaceKernel32sC1Filter(__global int* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX-2 || y >=dimY-2) return;\n"
" if (x<2 || y <2) return;\n"
" float pixel1 = imageIn[y * dimX  + x] * 41 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x+1] * 16 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x] * 26 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x-1] * 16 * scale;\n"
"       pixel1 += imageIn[y * dimX  + x+1] * 26 * scale;\n"
"       pixel1 += imageIn[y * dimX  + x-1] * 26 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x+1] * 16 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x] * 26 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x-1] * 16 * scale;\n"

"       pixel1 += imageIn[(y+2) * dimX  + x+2] * 1 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x+1] * 4 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x] * 7 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x-1] * 4 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x-2] * 1 * scale;\n"

"       pixel1 += imageIn[(y-2) * dimX  + x+2] * 1 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x+1] * 4 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x] * 7 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x-1] * 4 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x-2] * 1 * scale;\n"

"       pixel1 += imageIn[(y+1) * dimX  + x-2] * 4 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x+2] * 4 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x+2] * 4 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x-2] * 4 * scale;\n"
"       pixel1 += imageIn[(y) * dimX  + x+2] * 7 * scale;\n"
"       pixel1 += imageIn[(y) * dimX  + x-2] * 7 * scale;\n"
" pixel1 /= 273.0f;\n"
" pixel1 -= minVal;\n"
" pixel1 /= (maxVal - minVal);\n"
" pixel1 -= minContrast;\n"
" pixel1 /= (maxContrast - minContrast);\n"
" pixel1 = max(min(pixel1, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel1, 0.0, 0.0, 1.0);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"
"__kernel void WriteToSurfaceKernel32fC1(__global float* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX || y >=dimY) return;\n"
" float pixel1 = imageIn[y * dimX  + x] * scale;\n"
" pixel1 -= minVal;\n"
" pixel1 /= (maxVal - minVal);\n"
" pixel1 -= minContrast;\n"
" pixel1 /= (maxContrast - minContrast);\n"
" pixel1 = max(min(pixel1, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel1, 0.0, 0.0, 1.0);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"

"__kernel void WriteToSurfaceKernel32fC1Filter(__global float* imageIn, __write_only image2d_t image, int dimX, int dimY, float minVal, float maxVal, float minContrast, float maxContrast, float scale)\n"
"{\n"
" int x = get_global_id(0);\n"
" int y = get_global_id(1);\n"
" if (x>=dimX-2 || y >=dimY-2) return;\n"
" if (x<2 || y <2) return;\n"
" float pixel1 = imageIn[y * dimX  + x] * 41 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x+1] * 16 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x] * 26 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x-1] * 16 * scale;\n"
"       pixel1 += imageIn[y * dimX  + x+1] * 26 * scale;\n"
"       pixel1 += imageIn[y * dimX  + x-1] * 26 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x+1] * 16 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x] * 26 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x-1] * 16 * scale;\n"

"       pixel1 += imageIn[(y+2) * dimX  + x+2] * 1 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x+1] * 4 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x] * 7 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x-1] * 4 * scale;\n"
"       pixel1 += imageIn[(y+2) * dimX  + x-2] * 1 * scale;\n"

"       pixel1 += imageIn[(y-2) * dimX  + x+2] * 1 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x+1] * 4 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x] * 7 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x-1] * 4 * scale;\n"
"       pixel1 += imageIn[(y-2) * dimX  + x-2] * 1 * scale;\n"

"       pixel1 += imageIn[(y+1) * dimX  + x-2] * 4 * scale;\n"
"       pixel1 += imageIn[(y+1) * dimX  + x+2] * 4 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x+2] * 4 * scale;\n"
"       pixel1 += imageIn[(y-1) * dimX  + x-2] * 4 * scale;\n"
"       pixel1 += imageIn[(y) * dimX  + x+2] * 7 * scale;\n"
"       pixel1 += imageIn[(y) * dimX  + x-2] * 7 * scale;\n"
" pixel1 /= 273.0f;\n"
" pixel1 -= minVal;\n"
" pixel1 /= (maxVal - minVal);\n"
" pixel1 -= minContrast;\n"
" pixel1 /= (maxContrast - minContrast);\n"
" pixel1 = max(min(pixel1, 1.0f),0.0f);\n"
" float4 color = (float4)(pixel1, 0.0, 0.0, 1.0);\n"
" write_imagef(image, (int2)(x, y), color); \n"
"}\n"




"__kernel void SumDiffKernel8uC1(__global unsigned char* imageIn, __global float* output, __local float* temp, float subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = imageIn[x] - subVal;\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffKernel8uC2(__global unsigned char* imageIn, __global float2* output, __local float2* temp, float2 subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = (float2)(imageIn[x],imageIn[x+dimX*dimY]) - subVal;\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffKernel8uC3(__global unsigned char* imageIn, __global float3* output, __local float3* temp, float3 subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = (float3)(imageIn[3*x + 0],imageIn[3*x + 1], imageIn[3*x + 2]) - subVal;\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffKernel8uC4(__global unsigned char* imageIn, __global float3* output, __local float3* temp, float3 subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = (float3)(imageIn[4*x + 0],imageIn[4*x + 1], imageIn[4*x + 2]) - subVal;\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffKernel16uC1(__global unsigned short* imageIn, __global float* output, __local float* temp, float subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = imageIn[x] - subVal;\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffKernel16sC1(__global short* imageIn, __global float* output, __local float* temp, float subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = imageIn[x] - subVal;\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffKernel32sC1(__global int* imageIn, __global float* output, __local float* temp, float subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = imageIn[x] - subVal;\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffKernel32fC1(__global float* imageIn, __global float* output, __local float* temp, float subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = imageIn[x] - subVal;\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffKernel32fC1Final(__global float* inputOutput, __local float* temp, int dimX)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX) temp[xl] = inputOutput[x];\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) inputOutput[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffKernel32fC2Final(__global float2* inputOutput, __local float2* temp, int dimX)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX) temp[xl] = inputOutput[x];\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) inputOutput[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffKernel32fC3Final(__global float3* inputOutput, __local float3* temp, int dimX)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX) temp[xl] = inputOutput[x];\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) inputOutput[get_group_id(0)] = temp[0];\n"
"}\n"

"__kernel void SumDiffSqrKernel8uC1(__global unsigned char* imageIn, __global float* output, __local float* temp, float subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = (imageIn[x] - subVal) * (imageIn[x] - subVal);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffSqrKernel8uC2(__global unsigned char* imageIn, __global float2* output, __local float2* temp, float2 subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = ((float2)(imageIn[x],imageIn[x+dimX*dimY]) - subVal) * ((float2)(imageIn[x],imageIn[x+dimX*dimY]) - subVal);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffSqrKernel8uC3(__global unsigned char* imageIn, __global float3* output, __local float3* temp, float3 subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = ((float3)(imageIn[3*x + 0],imageIn[3*x + 1], imageIn[3*x + 2]) - subVal) * ((float3)(imageIn[3*x + 0],imageIn[3*x + 1], imageIn[3*x + 2]) - subVal);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffSqrKernel8uC4(__global unsigned char* imageIn, __global float3* output, __local float3* temp, float3 subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = ((float3)(imageIn[4*x + 0],imageIn[4*x + 1], imageIn[4*x + 2]) - subVal) * ((float3)(imageIn[4*x + 0],imageIn[4*x + 1], imageIn[4*x + 2]) - subVal);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffSqrKernel16uC1(__global unsigned short* imageIn, __global float* output, __local float* temp, float subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = (imageIn[x] - subVal) * (imageIn[x] - subVal);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffSqrKernel16sC1(__global short* imageIn, __global float* output, __local float* temp, float subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = (imageIn[x] - subVal) * (imageIn[x] - subVal);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffSqrKernel32sC1(__global int* imageIn, __global float* output, __local float* temp, float subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = (imageIn[x] - subVal) * (imageIn[x] - subVal);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"
"__kernel void SumDiffSqrKernel32fC1(__global float* imageIn, __global float* output, __local float* temp, float subVal, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" temp[xl] = 0;"
" if (x < dimX * dimY) temp[xl] = (imageIn[x] - subVal) * (imageIn[x] - subVal);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     temp[xl] += temp[xl + s];\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) output[get_group_id(0)] = temp[0];\n"
"}\n"

"__kernel void MinMaxKernel8uC1(__global unsigned char* imageIn, __global float* outputMin, __local float* tempMin, __global float* outputMax, __local float* tempMax, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" tempMin[xl] = 0;"
" if (x < dimX * dimY) tempMin[xl] = imageIn[x];\n"
" tempMax[xl] = tempMin[xl];"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     tempMin[xl] = min(tempMin[xl],tempMin[xl + s]);\n"
"     tempMax[xl] = max(tempMax[xl],tempMax[xl + s]);\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) outputMin[get_group_id(0)] = tempMin[0];\n"
" if (xl == 0) outputMax[get_group_id(0)] = tempMax[0];\n"
"}\n"
"__kernel void MinMaxKernel8uC2(__global unsigned char* imageIn, __global float2* outputMin, __local float2* tempMin, __global float2* outputMax, __local float2* tempMax, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" tempMin[xl] = 0;"
" if (x < dimX * dimY) tempMin[xl] = (float2)(imageIn[x],imageIn[x+dimX*dimY]);\n"
" tempMax[xl] = tempMin[xl];"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     tempMin[xl] = min(tempMin[xl],tempMin[xl + s]);\n"
"     tempMax[xl] = max(tempMax[xl],tempMax[xl + s]);\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) outputMin[get_group_id(0)] = tempMin[0];\n"
" if (xl == 0) outputMax[get_group_id(0)] = tempMax[0];\n"
"}\n"
"__kernel void MinMaxKernel8uC3(__global unsigned char* imageIn, __global float3* outputMin, __local float3* tempMin, __global float3* outputMax, __local float3* tempMax, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" tempMin[xl] = 0;"
" if (x < dimX * dimY) tempMin[xl] = (float3)(imageIn[3*x + 0],imageIn[3*x + 1], imageIn[3*x + 2]);\n"
" tempMax[xl] = tempMin[xl];"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     tempMin[xl] = min(tempMin[xl],tempMin[xl + s]);\n"
"     tempMax[xl] = max(tempMax[xl],tempMax[xl + s]);\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) outputMin[get_group_id(0)] = tempMin[0];\n"
" if (xl == 0) outputMax[get_group_id(0)] = tempMax[0];\n"
"}\n"
"__kernel void MinMaxKernel8uC4(__global unsigned char* imageIn, __global float3* outputMin, __local float3* tempMin, __global float3* outputMax, __local float3* tempMax, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" tempMin[xl] = 0;"
" if (x < dimX * dimY) tempMin[xl] = (float3)(imageIn[4*x + 0],imageIn[4*x + 1], imageIn[4*x + 2]);\n"
" tempMax[xl] = tempMin[xl];"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     tempMin[xl] = min(tempMin[xl],tempMin[xl + s]);\n"
"     tempMax[xl] = max(tempMax[xl],tempMax[xl + s]);\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) outputMin[get_group_id(0)] = tempMin[0];\n"
" if (xl == 0) outputMax[get_group_id(0)] = tempMax[0];\n"
"}\n"
"__kernel void MinMaxKernel16uC1(__global unsigned short* imageIn, __global float* outputMin, __local float* tempMin, __global float* outputMax, __local float* tempMax, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" tempMin[xl] = 0;"
" if (x < dimX * dimY) tempMin[xl] = imageIn[x];\n"
" tempMax[xl] = tempMin[xl];"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     tempMin[xl] = min(tempMin[xl],tempMin[xl + s]);\n"
"     tempMax[xl] = max(tempMax[xl],tempMax[xl + s]);\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) outputMin[get_group_id(0)] = tempMin[0];\n"
" if (xl == 0) outputMax[get_group_id(0)] = tempMax[0];\n"
"}\n"
"__kernel void MinMaxKernel16sC1(__global short* imageIn, __global float* outputMin, __local float* tempMin, __global float* outputMax, __local float* tempMax, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" tempMin[xl] = 0;"
" if (x < dimX * dimY) tempMin[xl] = imageIn[x];\n"
" tempMax[xl] = tempMin[xl];"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     tempMin[xl] = min(tempMin[xl],tempMin[xl + s]);\n"
"     tempMax[xl] = max(tempMax[xl],tempMax[xl + s]);\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) outputMin[get_group_id(0)] = tempMin[0];\n"
" if (xl == 0) outputMax[get_group_id(0)] = tempMax[0];\n"
"}\n"
"__kernel void MinMaxKernel32sC1(__global int* imageIn, __global float* outputMin, __local float* tempMin, __global float* outputMax, __local float* tempMax, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" tempMin[xl] = 0;"
" if (x < dimX * dimY) tempMin[xl] = imageIn[x];\n"
" tempMax[xl] = tempMin[xl];"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     tempMin[xl] = min(tempMin[xl],tempMin[xl + s]);\n"
"     tempMax[xl] = max(tempMax[xl],tempMax[xl + s]);\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) outputMin[get_group_id(0)] = tempMin[0];\n"
" if (xl == 0) outputMax[get_group_id(0)] = tempMax[0];\n"
"}\n"
"__kernel void MinMaxKernel32fC1(__global float* imageIn, __global float* outputMin, __local float* tempMin, __global float* outputMax, __local float* tempMax, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" tempMin[xl] = 0;"
" if (x < dimX * dimY) tempMin[xl] = imageIn[x];\n"
" tempMax[xl] = tempMin[xl];"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     tempMin[xl] = min(tempMin[xl],tempMin[xl + s]);\n"
"     tempMax[xl] = max(tempMax[xl],tempMax[xl + s]);\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) outputMin[get_group_id(0)] = tempMin[0];\n"
" if (xl == 0) outputMax[get_group_id(0)] = tempMax[0];\n"
"}\n"
"__kernel void MinMaxKernel32fC1Final(__global float* inputOutputMin, __local float* tempMin, __global float* inputOutputMax, __local float* tempMax, int dimX)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" tempMin[xl] = 0;"
" if (x < dimX) tempMin[xl] = inputOutputMin[x];\n"
" if (x < dimX) tempMax[xl] = inputOutputMax[x];\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     tempMin[xl] = min(tempMin[xl],tempMin[xl + s]);\n"
"     tempMax[xl] = max(tempMax[xl],tempMax[xl + s]);\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) inputOutputMin[get_group_id(0)] = tempMin[0];\n"
" if (xl == 0) inputOutputMax[get_group_id(0)] = tempMax[0];\n"
"}\n"
"__kernel void MinMaxKernel32fC2Final(__global float2* inputOutputMin, __local float2* tempMin, __global float2* inputOutputMax, __local float2* tempMax, int dimX)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" tempMin[xl] = 0;"
" if (x < dimX) tempMin[xl] = inputOutputMin[x];\n"
" if (x < dimX) tempMax[xl] = inputOutputMax[x];\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     tempMin[xl] = min(tempMin[xl],tempMin[xl + s]);\n"
"     tempMax[xl] = max(tempMax[xl],tempMax[xl + s]);\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) inputOutputMin[get_group_id(0)] = tempMin[0];\n"
" if (xl == 0) inputOutputMax[get_group_id(0)] = tempMax[0];\n"
"}\n"
"__kernel void MinMaxKernel32fC3Final(__global float3* inputOutputMin, __local float3* tempMin, __global float3* inputOutputMax, __local float3* tempMax, int dimX)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" int sizexl = get_local_size(0);"
" tempMin[xl] = 0;"
" if (x < dimX) tempMin[xl] = inputOutputMin[x];\n"
" if (x < dimX) tempMax[xl] = inputOutputMax[x];\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" for (int s = sizexl/2; s > 0; s>>=1)\n"
" {\n"
"   if (xl < s)\n"
"   {\n"
"     tempMin[xl] = min(tempMin[xl],tempMin[xl + s]);\n"
"     tempMax[xl] = max(tempMax[xl],tempMax[xl + s]);\n"
"   }\n"
"   barrier(CLK_LOCAL_MEM_FENCE);\n"
" }\n"
" if (xl == 0) inputOutputMin[get_group_id(0)] = tempMin[0];\n"
" if (xl == 0) inputOutputMax[get_group_id(0)] = tempMax[0];\n"
"}\n"


"__kernel void HistogramKernel8uC1(__global unsigned char* imageIn, __global int* output, __local int* temp, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" temp[xl] = 0;"
" if (get_group_id(0) == 0) output[xl] = 0;\n"
" barrier(CLK_GLOBAL_MEM_FENCE);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" if (x < dimX * dimY)\n"
" {\n"
"    int index = imageIn[x];\n"
"    atomic_inc(&temp[index]);\n"
" }\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" atomic_add(&output[xl], temp[xl]);\n"
"}\n"
"__kernel void HistogramKernel8uC2(__global unsigned char* imageIn, __global int* outputA, __global int* outputB, __local int2* temp, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" temp[xl] = 0;"
" if (get_group_id(0) == 0) outputA[xl] = 0;\n"
" if (get_group_id(0) == 0) outputB[xl] = 0;\n"
" barrier(CLK_GLOBAL_MEM_FENCE);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" if (x < dimX * dimY)\n"
" {\n"
"    int indexR = imageIn[x];\n"
"    int indexG = imageIn[x + dimX * dimY];\n"
"    __local int* t = (__local int*)temp;\n"
"    atomic_inc(&t[2 * indexR]);\n"
"    atomic_inc(&t[2 * indexG + 1]);\n"
" }\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" atomic_add(&outputA[xl], temp[xl].x);\n"
" atomic_add(&outputB[xl], temp[xl].y);\n"
"}\n"
"__kernel void HistogramKernel8uC3(__global unsigned char* imageIn, __global int* outputA, __global int* outputB, __global int* outputC, __local int4* temp, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" temp[xl] = 0;"
" if (get_group_id(0) == 0) outputA[xl] = 0;\n"
" if (get_group_id(0) == 0) outputB[xl] = 0;\n"
" if (get_group_id(0) == 0) outputC[xl] = 0;\n"
" barrier(CLK_GLOBAL_MEM_FENCE);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" if (x < dimX * dimY)\n"
" {\n"
"    int indexR = imageIn[3*x];\n"
"    int indexG = imageIn[3*x + 1];\n"
"    int indexB = imageIn[3*x + 2];\n"
"    __local int* t = (__local int*)temp;\n"
"    atomic_inc(&t[4 * indexR]);\n"
"    atomic_inc(&t[4 * indexG + 1]);\n"
"    atomic_inc(&t[4 * indexB + 2]);\n"
" }\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" atomic_add(&outputA[xl], temp[xl].x);\n"
" atomic_add(&outputB[xl], temp[xl].y);\n"
" atomic_add(&outputC[xl], temp[xl].z);\n"
"}\n"
"__kernel void HistogramKernel8uC4(__global unsigned char* imageIn, __global int* outputA, __global int* outputB, __global int* outputC, __local int4* temp, int dimX, int dimY)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" temp[xl] = 0;"
" if (get_group_id(0) == 0) outputA[xl] = 0;\n"
" if (get_group_id(0) == 0) outputB[xl] = 0;\n"
" if (get_group_id(0) == 0) outputC[xl] = 0;\n"
" barrier(CLK_GLOBAL_MEM_FENCE);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" if (x < dimX * dimY)\n"
" {\n"
"    int indexR = imageIn[3*x];\n"
"    int indexG = imageIn[3*x + 1];\n"
"    int indexB = imageIn[3*x + 2];\n"
"    __local int* t = (__local int*)temp;\n"
"    atomic_inc(&t[4 * indexR]);\n"
"    atomic_inc(&t[4 * indexG + 1]);\n"
"    atomic_inc(&t[4 * indexB + 2]);\n"
" }\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" atomic_add(&outputA[xl], temp[xl].x);\n"
" atomic_add(&outputB[xl], temp[xl].y);\n"
" atomic_add(&outputC[xl], temp[xl].z);\n"
"}\n"
"__kernel void HistogramKernel16uC1(__global unsigned short* imageIn, __global int* output, __local int* temp, int dimX, int dimY, float minVal, float maxVal)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" temp[xl] = 0;"
" if (get_group_id(0) == 0) output[xl] = 0;\n"
" barrier(CLK_GLOBAL_MEM_FENCE);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" if (x < dimX * dimY)\n"
" {\n"
"    int index = floor((imageIn[x] - minVal) / (maxVal - minVal) * (get_local_size(0) - 1));\n"
"    index = min((int)get_local_size(0)-1, max(index, 0));\n"
"    atomic_inc(&temp[index]);\n"
" }\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" atomic_add(&output[xl], temp[xl]);\n"
"}\n"
"__kernel void HistogramKernel16sC1(__global short* imageIn, __global int* output, __local int* temp, int dimX, int dimY, float minVal, float maxVal)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" temp[xl] = 0;"
" if (get_group_id(0) == 0) output[xl] = 0;\n"
" barrier(CLK_GLOBAL_MEM_FENCE);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" if (x < dimX * dimY)\n"
" {\n"
"    int index = floor((imageIn[x] - minVal) / (maxVal - minVal) * (get_local_size(0) - 1));\n"
"    index = min((int)get_local_size(0)-1, max(index, 0));\n"
"    atomic_inc(&temp[index]);\n"
" }\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" atomic_add(&output[xl], temp[xl]);\n"
"}\n"
"__kernel void HistogramKernel32sC1(__global int* imageIn, __global int* output, __local int* temp, int dimX, int dimY, float minVal, float maxVal)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" temp[xl] = 0;"
" if (get_group_id(0) == 0) output[xl] = 0;\n"
" barrier(CLK_GLOBAL_MEM_FENCE);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" if (x < dimX * dimY)\n"
" {\n"
"    int index = floor((imageIn[x] - minVal) / (maxVal - minVal) * (get_local_size(0) - 1));\n"
"    index = min((int)get_local_size(0)-1, max(index, 0));\n"
"    atomic_inc(&temp[index]);\n"
" }\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" atomic_add(&output[xl], temp[xl]);\n"
"}\n"
"__kernel void HistogramKernel32fC1(__global float* imageIn, __global int* output, __local int* temp, int dimX, int dimY, float minVal, float maxVal)\n"
"{\n"
" int x = get_global_id(0);\n"
" int xl = get_local_id(0);\n"
" temp[xl] = 0;\n"
" if (get_group_id(0) == 0) output[xl] = 0;\n"
" barrier(CLK_GLOBAL_MEM_FENCE);\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" if (x < dimX * dimY)\n"
" {\n"
"    int index = floor((imageIn[x] - minVal) / (maxVal - minVal) * (get_local_size(0) - 1));\n"
"    index = min((int)get_local_size(0)-1, max(index, 0));\n"
"    atomic_inc(&temp[index]);\n"
" }\n"
" barrier(CLK_LOCAL_MEM_FENCE);\n"
" atomic_add(&output[xl], temp[xl]);\n"
"}\n";


OpenCL::OpenCLKernel* BaseFrameController::mKernel8uC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernel8uC1Filter = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernel8uC2 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernel8uC3 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernel8uC4 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernel16uC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernel16sC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernel32sC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernel32fC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernel16uC1Filter = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernel16sC1Filter = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernel32sC1Filter = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernel32fC1Filter = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiff8uC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiff8uC2 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiff8uC3 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiff8uC4 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiff16uC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiff16sC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiff32sC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiff32fC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiff32fC1Final = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiff32fC2Final = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiff32fC3Final = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiffSqr8uC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiffSqr8uC2 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiffSqr8uC3 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiffSqr8uC4 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiffSqr16uC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiffSqr16sC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiffSqr32sC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelSumDiffSqr32fC1 = NULL;

OpenCL::OpenCLKernel* BaseFrameController::mKernelMinMax8uC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelMinMax8uC2 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelMinMax8uC3 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelMinMax8uC4 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelMinMax16uC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelMinMax16sC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelMinMax32sC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelMinMax32fC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelMinMax32fC1Final = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelMinMax32fC2Final = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelMinMax32fC3Final = NULL;

OpenCL::OpenCLKernel* BaseFrameController::mKernelHistogram8uC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelHistogram8uC2 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelHistogram8uC3 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelHistogram8uC4 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelHistogram16uC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelHistogram16sC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelHistogram32sC1 = NULL;
OpenCL::OpenCLKernel* BaseFrameController::mKernelHistogram32fC1 = NULL;

cl_context BaseFrameController::mCtx = 0;
cl_program BaseFrameController::mProgram = 0;


BaseFrameController::BaseFrameController(QObject *parent) :
    QObject(parent),
    mDevVar(NULL),

    mDimX(0),
    mDimY(0),
    mPixelSize(0),
    mIsRGB(false),
    mStdValues{0, 0, 0},
    mMeanValues{0, 0, 0},
    mMaxValues{0, 0, 0},
    mMinValues{0, 0, 0},
    mHistogramBinCount(256),
    mIsFileLoaded(false),
    mIsLockedForLoading(false)
{

}

BaseFrameController::~BaseFrameController()
{
    if (mDevVar)
        delete mDevVar;
    mDevVar = NULL;
}

void BaseFrameController::computeImageStatisticsOpenCL()
{
    int workSize = 256;
    int size = (mDimX * mDimY + workSize - 1) / workSize;
    int sizeCompute = size;
    OpenCL::OpenCLDeviceVariable bufferA(size * 4 * sizeof(float));
    OpenCL::OpenCLDeviceVariable bufferB(size * 4 * sizeof(float));
    cl_float2 val2 = {0, 0};
    cl_float3 val3 = {0,0,0};



    switch (mDatatype)
    {
    case DT_UCHAR:
        mKernelSumDiff8uC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelSumDiff8uC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), 0.0f, mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMeanValues[0],4);
        mMeanValues[0] /= mDimX * mDimY;
        mMeanValues[1] = mMeanValues[0];
        mMeanValues[2] = mMeanValues[0];

    break;
    case DT_UCHAR2:
        mKernelSumDiff8uC2->SetProblemSize(workSize, mDimX* mDimY);
        val2 = {0, 0};
        mKernelSumDiff8uC2->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*2), val2, mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC2Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC2Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*2), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMeanValues[0],4*2);
        mMeanValues[0] /= mDimX * mDimY;
        mMeanValues[1] /= mDimX * mDimY;
        mMeanValues[2] = 0;
    break;
    case DT_UCHAR3:
        mKernelSumDiff8uC3->SetProblemSize(workSize, mDimX* mDimY);
        val3 = {0, 0, 0};
        mKernelSumDiff8uC3->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), val3, mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC3Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC3Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMeanValues[0],4*3);
        mMeanValues[0] /= mDimX * mDimY;
        mMeanValues[1] /= mDimX * mDimY;
        mMeanValues[2] /= mDimX * mDimY;
    break;
    case DT_UCHAR4:
        mKernelSumDiff8uC4->SetProblemSize(workSize, mDimX* mDimY);
        val3 = {0, 0, 0};
        mKernelSumDiff8uC4->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*3), val3, mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC3Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC3Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMeanValues[0],4*3);
        mMeanValues[0] /= mDimX * mDimY;
        mMeanValues[1] /= mDimX * mDimY;
        mMeanValues[2] /= mDimX * mDimY;
    break;
    case DT_USHORT:
        mKernelSumDiff16uC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelSumDiff16uC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), 0.0f, mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMeanValues[0],4);
        mMeanValues[0] /= mDimX * mDimY;
        mMeanValues[1] = mMeanValues[0];
        mMeanValues[2] = mMeanValues[0];
    break;
    case DT_SHORT:
        mKernelSumDiff16sC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelSumDiff16sC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), 0.0f, mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMeanValues[0],4);
        mMeanValues[0] /= mDimX * mDimY;
        mMeanValues[1] = mMeanValues[0];
        mMeanValues[2] = mMeanValues[0];
    break;
    case DT_INT:
        mKernelSumDiff32sC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelSumDiff32sC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), 0.0f, mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMeanValues[0],4);
        mMeanValues[0] /= mDimX * mDimY;
        mMeanValues[1] = mMeanValues[0];
        mMeanValues[2] = mMeanValues[0];
    break;
    case DT_FLOAT:
        mKernelSumDiff32fC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelSumDiff32fC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), 0.0f, mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMeanValues[0],4);
        mMeanValues[0] /= mDimX * mDimY;
        mMeanValues[1] = mMeanValues[0];
        mMeanValues[2] = mMeanValues[0];
    break;
    }

    sizeCompute = size;
    switch (mDatatype)
    {
    case DT_UCHAR:
        mKernelSumDiffSqr8uC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelSumDiffSqr8uC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), mMeanValues[0], mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mStdValues[0],4);
        mStdValues[0] /= mDimX * mDimY;
        mStdValues[1] = mStdValues[0];
        mStdValues[2] = mStdValues[0];

    break;
    case DT_UCHAR2:
        mKernelSumDiffSqr8uC2->SetProblemSize(workSize, mDimX* mDimY);
        val2 = {mMeanValues[0], mMeanValues[1]};
        mKernelSumDiffSqr8uC2->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*2), val2, mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC2Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC2Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*2), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mStdValues[0],4*2);
        mStdValues[0] /= mDimX * mDimY;
        mStdValues[1] /= mDimX * mDimY;
        mStdValues[2] = 0;
    break;
    case DT_UCHAR3:
        mKernelSumDiffSqr8uC3->SetProblemSize(workSize, mDimX* mDimY);
        val3 = {mMeanValues[0], mMeanValues[1], mMeanValues[2]};
        mKernelSumDiffSqr8uC3->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), val3, mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC3Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC3Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mStdValues[0],4*3);
        mStdValues[0] /= mDimX * mDimY;
        mStdValues[1] /= mDimX * mDimY;
        mStdValues[2] /= mDimX * mDimY;
    break;
    case DT_UCHAR4:
        mKernelSumDiffSqr8uC4->SetProblemSize(workSize, mDimX* mDimY);
        val3 = {mMeanValues[0], mMeanValues[1], mMeanValues[2]};
        mKernelSumDiffSqr8uC4->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), val3, mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC3Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC3Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mStdValues[0],4*3);
        mStdValues[0] /= mDimX * mDimY;
        mStdValues[1] /= mDimX * mDimY;
        mStdValues[2] /= mDimX * mDimY;
    break;
    case DT_USHORT:
        mKernelSumDiffSqr16uC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelSumDiffSqr16uC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), mMeanValues[0], mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mStdValues[0],4);
        mStdValues[0] /= mDimX * mDimY;
        mStdValues[1] = mStdValues[0];
        mStdValues[2] = mStdValues[0];
    break;
    case DT_SHORT:
        mKernelSumDiffSqr16sC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelSumDiffSqr16sC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), mMeanValues[0], mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mStdValues[0],4);
        mStdValues[0] /= mDimX * mDimY;
        mStdValues[1] = mStdValues[0];
        mStdValues[2] = mStdValues[0];
    break;
    case DT_INT:
        mKernelSumDiffSqr32sC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelSumDiffSqr32sC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), mMeanValues[0], mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mStdValues[0],4);
        mStdValues[0] /= mDimX * mDimY;
        mStdValues[1] = mStdValues[0];
        mStdValues[2] = mStdValues[0];
    break;
    case DT_FLOAT:
        mKernelSumDiffSqr32fC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelSumDiffSqr32fC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), mMeanValues[0], mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelSumDiff32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelSumDiff32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mStdValues[0],4);
        mStdValues[0] /= mDimX * mDimY;
        mStdValues[1] = mStdValues[0];
        mStdValues[2] = mStdValues[0];
    break;
    }

    sizeCompute = size;
    switch (mDatatype)
    {
    case DT_UCHAR:
        mKernelMinMax8uC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelMinMax8uC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4),  mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelMinMax32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelMinMax32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMinValues[0],4);
        mMinValues[1] = mMinValues[0];
        mMinValues[2] = mMinValues[0];
        bufferB.CopyDeviceToHost(&mMaxValues[0],4);
        mMaxValues[1] = mMaxValues[0];
        mMaxValues[2] = mMaxValues[0];

    break;
    case DT_UCHAR2:
        mKernelMinMax8uC2->SetProblemSize(workSize, mDimX* mDimY);
        mKernelMinMax8uC2->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*2), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*2),  mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelMinMax32fC2Final->SetProblemSize(workSize, sizeCompute);
            mKernelMinMax32fC2Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*2), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*2), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMinValues[0],4*2);
        mMinValues[2] = 0;
        bufferB.CopyDeviceToHost(&mMaxValues[0],4*2);
        mMaxValues[2] = 0;
    break;
    case DT_UCHAR3:
        mKernelMinMax8uC3->SetProblemSize(workSize, mDimX* mDimY);
        mKernelMinMax8uC3->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4),  mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelMinMax32fC3Final->SetProblemSize(workSize, sizeCompute);
            mKernelMinMax32fC3Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMinValues[0],4*3);
        bufferB.CopyDeviceToHost(&mMaxValues[0],4*3);
    break;
    case DT_UCHAR4:
        mKernelMinMax8uC4->SetProblemSize(workSize, mDimX* mDimY);
        mKernelMinMax8uC4->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelMinMax32fC3Final->SetProblemSize(workSize, sizeCompute);
            mKernelMinMax32fC3Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMinValues[0],4*3);
        bufferB.CopyDeviceToHost(&mMaxValues[0],4*3);
    break;
    case DT_USHORT:
        mKernelMinMax16uC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelMinMax16uC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelMinMax32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelMinMax32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMinValues[0],4);
        mMinValues[1] = mMinValues[0];
        mMinValues[2] = mMinValues[0];
        bufferB.CopyDeviceToHost(&mMaxValues[0],4);
        mMaxValues[1] = mMaxValues[0];
        mMaxValues[2] = mMaxValues[0];
    break;
    case DT_SHORT:
        mKernelMinMax16sC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelMinMax16sC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelMinMax32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelMinMax32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMinValues[0],4);
        mMinValues[1] = mMinValues[0];
        mMinValues[2] = mMinValues[0];
        bufferB.CopyDeviceToHost(&mMaxValues[0],4);
        mMaxValues[1] = mMaxValues[0];
        mMaxValues[2] = mMaxValues[0];
    break;
    case DT_INT:
        mKernelMinMax32sC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelMinMax32sC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelMinMax32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelMinMax32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMinValues[0],4);
        mMinValues[1] = mMinValues[0];
        mMinValues[2] = mMinValues[0];
        bufferB.CopyDeviceToHost(&mMaxValues[0],4);
        mMaxValues[1] = mMaxValues[0];
        mMaxValues[2] = mMaxValues[0];
    break;
    case DT_FLOAT:
        mKernelMinMax32fC1->SetProblemSize(workSize, mDimX* mDimY);
        mKernelMinMax32fC1->Run(mDevVar->GetDevicePtr(), bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), mDimX, mDimY);

        while (sizeCompute > 1)
        {
            mKernelMinMax32fC1Final->SetProblemSize(workSize, sizeCompute);
            mKernelMinMax32fC1Final->Run(bufferA.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), bufferB.GetDevicePtr(), OpenCL::LocalMemory(workSize*4), sizeCompute);
            sizeCompute = (sizeCompute + workSize - 1) / workSize;
        }

        bufferA.CopyDeviceToHost(&mMinValues[0],4);
        mMinValues[1] = mMinValues[0];
        mMinValues[2] = mMinValues[0];
        bufferB.CopyDeviceToHost(&mMaxValues[0],4);
        mMaxValues[1] = mMaxValues[0];
        mMaxValues[2] = mMaxValues[0];
    break;
    }

    //qDebug() << mMeanValues[0];

}

void BaseFrameController::computeHistogramOpenCL(OpenCL::OpenCLDeviceVariable* histDeviceA, OpenCL::OpenCLDeviceVariable* histDeviceB, OpenCL::OpenCLDeviceVariable* histDeviceC)
{
    mHistogramRorGray.clear();
    mHistogramG.clear();
    mHistogramB.clear();



    for (int i = 0; i < mHistogramBinCount; i++)
    {
        mHistogramRorGray.push_back(0);
        mHistogramG.push_back(0);
        mHistogramB.push_back(0);
    }

    switch (mDatatype)
    {
    case DT_UCHAR:
        mKernelHistogram8uC1->SetProblemSize(mHistogramBinCount, mDimX * mDimY);
        mKernelHistogram8uC1->Run(mDevVar->GetDevicePtr(), histDeviceA->GetDevicePtr(), OpenCL::LocalMemory(256*4), mDimX, mDimY);
        histDeviceA->CopyDeviceToHost(&mHistogramRorGray[0]);
    break;
    case DT_UCHAR2:
        mKernelHistogram8uC2->SetProblemSize(mHistogramBinCount, mDimX * mDimY);
        mKernelHistogram8uC2->Run(mDevVar->GetDevicePtr(), histDeviceA->GetDevicePtr(), histDeviceB->GetDevicePtr(), OpenCL::LocalMemory(256*4*2), mDimX, mDimY);
        histDeviceA->CopyDeviceToHost(&mHistogramRorGray[0]);
        histDeviceB->CopyDeviceToHost(&mHistogramG[0]);
    break;
    case DT_UCHAR3:
        mKernelHistogram8uC3->SetProblemSize(mHistogramBinCount, mDimX * mDimY);
        mKernelHistogram8uC3->Run(mDevVar->GetDevicePtr(), histDeviceA->GetDevicePtr(), histDeviceB->GetDevicePtr(), histDeviceC->GetDevicePtr(), OpenCL::LocalMemory(256*4*4), mDimX, mDimY);
        histDeviceA->CopyDeviceToHost(&mHistogramRorGray[0]);
        histDeviceB->CopyDeviceToHost(&mHistogramG[0]);
        histDeviceC->CopyDeviceToHost(&mHistogramB[0]);
    break;
    case DT_UCHAR4:
        mKernelHistogram8uC4->SetProblemSize(mHistogramBinCount, mDimX * mDimY);
        mKernelHistogram8uC4->Run(mDevVar->GetDevicePtr(), histDeviceA->GetDevicePtr(), histDeviceB->GetDevicePtr(), histDeviceC->GetDevicePtr(), OpenCL::LocalMemory(256*4*4), mDimX, mDimY);
        histDeviceA->CopyDeviceToHost(&mHistogramRorGray[0]);
        histDeviceB->CopyDeviceToHost(&mHistogramG[0]);
        histDeviceC->CopyDeviceToHost(&mHistogramB[0]);
    break;
    case DT_USHORT:
        mKernelHistogram16uC1->SetProblemSize(mHistogramBinCount, mDimX * mDimY);
        mKernelHistogram16uC1->Run(mDevVar->GetDevicePtr(), histDeviceA->GetDevicePtr(), OpenCL::LocalMemory(256*4), mDimX, mDimY, mMinValues[0], mMaxValues[0]);
        histDeviceA->CopyDeviceToHost(&mHistogramRorGray[0]);
    break;
    case DT_SHORT:
        mKernelHistogram16sC1->SetProblemSize(mHistogramBinCount, mDimX * mDimY);
        mKernelHistogram16sC1->Run(mDevVar->GetDevicePtr(), histDeviceA->GetDevicePtr(), OpenCL::LocalMemory(256*4), mDimX, mDimY, mMinValues[0], mMaxValues[0]);
        histDeviceA->CopyDeviceToHost(&mHistogramRorGray[0]);
    break;
    case DT_INT:
        mKernelHistogram32sC1->SetProblemSize(mHistogramBinCount, mDimX * mDimY);
        mKernelHistogram32sC1->Run(mDevVar->GetDevicePtr(), histDeviceA->GetDevicePtr(), OpenCL::LocalMemory(256*4), mDimX, mDimY, mMinValues[0], mMaxValues[0]);
        histDeviceA->CopyDeviceToHost(&mHistogramRorGray[0]);
    break;
    case DT_FLOAT:
        mKernelHistogram32fC1->SetProblemSize(mHistogramBinCount, mDimX * mDimY);
        mKernelHistogram32fC1->Run(mDevVar->GetDevicePtr(), histDeviceA->GetDevicePtr(), OpenCL::LocalMemory(256*4), mDimX, mDimY, mMinValues[0], mMaxValues[0]);
        histDeviceA->CopyDeviceToHost(&mHistogramRorGray[0]);
    break;
    }
}

void BaseFrameController::processWithOpenCL(int mDimX, int mDimY, cl_mem output, float minVal, float maxVal, void *userData)
{
    FrameControllerData* data = (FrameControllerData*)userData;
    if (data->useFilter && data->kernelFilter)
    {
        data->kernelFilter->SetProblemSize(16,16,mDimX, mDimY);
        data->kernelFilter->Run(data->devPtr, output, mDimX, mDimY, data->minVal, data->maxVal, minVal, maxVal, data->scale);
    }
    else
    {
        data->kernel->SetProblemSize(16,16,mDimX, mDimY);
        data->kernel->Run(data->devPtr, output, mDimX, mDimY, data->minVal, data->maxVal, minVal, maxVal, data->scale);
    }
}

void BaseFrameController::LoadKernel()
{
    switch (mDatatype)
    {
    case DT_UCHAR:
        if (!mKernel8uC1) mKernel8uC1 = new OpenCL::OpenCLKernel("WriteToSurfaceKernel8uC1", mProgram);
        if (!mKernel8uC1Filter) mKernel8uC1Filter = new OpenCL::OpenCLKernel("WriteToSurfaceKernel8uC1Filter", mProgram);
        if (!mKernelSumDiff8uC1) mKernelSumDiff8uC1 = new OpenCL::OpenCLKernel("SumDiffKernel8uC1", mProgram);
        if (!mKernelSumDiffSqr8uC1) mKernelSumDiffSqr8uC1 = new OpenCL::OpenCLKernel("SumDiffSqrKernel8uC1", mProgram);
        if (!mKernelSumDiff32fC1Final) mKernelSumDiff32fC1Final = new OpenCL::OpenCLKernel("SumDiffKernel32fC1Final", mProgram);
        if (!mKernelMinMax8uC1) mKernelMinMax8uC1 = new OpenCL::OpenCLKernel("MinMaxKernel8uC1", mProgram);
        if (!mKernelMinMax32fC1Final) mKernelMinMax32fC1Final = new OpenCL::OpenCLKernel("MinMaxKernel32fC1Final", mProgram);
        if (!mKernelHistogram8uC1) mKernelHistogram8uC1 = new OpenCL::OpenCLKernel("HistogramKernel8uC1", mProgram);
        data.kernel = mKernel8uC1;
        data.kernelFilter = mKernel8uC1Filter;
    break;
    case DT_UCHAR2:
        if (!mKernel8uC2) mKernel8uC2 = new OpenCL::OpenCLKernel("WriteToSurfaceKernel8uC2", mProgram);
        if (!mKernelSumDiff8uC2) mKernelSumDiff8uC2 = new OpenCL::OpenCLKernel("SumDiffKernel8uC2", mProgram);
        if (!mKernelSumDiffSqr8uC2) mKernelSumDiffSqr8uC2 = new OpenCL::OpenCLKernel("SumDiffSqrKernel8uC2", mProgram);
        if (!mKernelSumDiff32fC2Final) mKernelSumDiff32fC2Final = new OpenCL::OpenCLKernel("SumDiffKernel32fC2Final", mProgram);
        if (!mKernelMinMax8uC2) mKernelMinMax8uC2 = new OpenCL::OpenCLKernel("MinMaxKernel8uC2", mProgram);
        if (!mKernelMinMax32fC2Final) mKernelMinMax32fC2Final = new OpenCL::OpenCLKernel("MinMaxKernel32fC2Final", mProgram);
        if (!mKernelHistogram8uC2) mKernelHistogram8uC2 = new OpenCL::OpenCLKernel("HistogramKernel8uC2", mProgram);
        data.kernel = mKernel8uC2;
        mIsRGB = true;
    break;
    case DT_UCHAR3:
        if (!mKernel8uC3) mKernel8uC3 = new OpenCL::OpenCLKernel("WriteToSurfaceKernel8uC3", mProgram);
        if (!mKernelSumDiff8uC3) mKernelSumDiff8uC3 = new OpenCL::OpenCLKernel("SumDiffKernel8uC3", mProgram);
        if (!mKernelSumDiffSqr8uC3) mKernelSumDiffSqr8uC3 = new OpenCL::OpenCLKernel("SumDiffSqrKernel8uC3", mProgram);
        if (!mKernelSumDiff32fC3Final) mKernelSumDiff32fC3Final = new OpenCL::OpenCLKernel("SumDiffKernel32fC3Final", mProgram);
        if (!mKernelMinMax8uC3) mKernelMinMax8uC3 = new OpenCL::OpenCLKernel("MinMaxKernel8uC3", mProgram);
        if (!mKernelMinMax32fC3Final) mKernelMinMax32fC3Final = new OpenCL::OpenCLKernel("MinMaxKernel32fC3Final", mProgram);
        if (!mKernelHistogram8uC3) mKernelHistogram8uC3 = new OpenCL::OpenCLKernel("HistogramKernel8uC3", mProgram);
        data.kernel = mKernel8uC3;
        mIsRGB = true;
    break;
    case DT_UCHAR4:
        if (!mKernel8uC4) mKernel8uC4 = new OpenCL::OpenCLKernel("WriteToSurfaceKernel8uC4", mProgram);
        if (!mKernelSumDiff8uC4) mKernelSumDiff8uC4 = new OpenCL::OpenCLKernel("SumDiffKernel8uC4", mProgram);
        if (!mKernelSumDiffSqr8uC4) mKernelSumDiffSqr8uC4 = new OpenCL::OpenCLKernel("SumDiffSqrKernel8uC4", mProgram);
        if (!mKernelSumDiff32fC3Final) mKernelSumDiff32fC3Final = new OpenCL::OpenCLKernel("SumDiffKernel32fC3Final", mProgram);
        if (!mKernelMinMax8uC4) mKernelMinMax8uC4 = new OpenCL::OpenCLKernel("MinMaxKernel8uC4", mProgram);
        if (!mKernelMinMax32fC3Final) mKernelMinMax32fC3Final = new OpenCL::OpenCLKernel("MinMaxKernel32fC3Final", mProgram);
        if (!mKernelHistogram8uC4) mKernelHistogram8uC4 = new OpenCL::OpenCLKernel("HistogramKernel8uC4", mProgram);
        data.kernel = mKernel8uC4;
        mIsRGB = true;
    break;
    case DT_USHORT:
        if (!mKernel16uC1) mKernel16uC1 = new OpenCL::OpenCLKernel("WriteToSurfaceKernel16uC1", mProgram);
        if (!mKernel16uC1Filter) mKernel16uC1Filter = new OpenCL::OpenCLKernel("WriteToSurfaceKernel16uC1Filter", mProgram);
        if (!mKernelSumDiff16uC1) mKernelSumDiff16uC1 = new OpenCL::OpenCLKernel("SumDiffKernel16uC1", mProgram);
        if (!mKernelSumDiffSqr16uC1) mKernelSumDiffSqr16uC1 = new OpenCL::OpenCLKernel("SumDiffSqrKernel16uC1", mProgram);
        if (!mKernelSumDiff32fC1Final) mKernelSumDiff32fC1Final = new OpenCL::OpenCLKernel("SumDiffKernel32fC1Final", mProgram);
        if (!mKernelMinMax16uC1) mKernelMinMax16uC1 = new OpenCL::OpenCLKernel("MinMaxKernel16uC1", mProgram);
        if (!mKernelMinMax32fC1Final) mKernelMinMax32fC1Final = new OpenCL::OpenCLKernel("MinMaxKernel32fC1Final", mProgram);
        if (!mKernelHistogram16uC1) mKernelHistogram16uC1 = new OpenCL::OpenCLKernel("HistogramKernel16uC1", mProgram);
        data.kernel = mKernel16uC1;
        data.kernelFilter = mKernel16uC1Filter;
    break;
    case DT_SHORT:
        if (!mKernel16sC1) mKernel16sC1 = new OpenCL::OpenCLKernel("WriteToSurfaceKernel16sC1", mProgram);
        if (!mKernel16sC1Filter) mKernel16sC1Filter = new OpenCL::OpenCLKernel("WriteToSurfaceKernel16sC1Filter", mProgram);
        if (!mKernelSumDiff16sC1) mKernelSumDiff16sC1 = new OpenCL::OpenCLKernel("SumDiffKernel16sC1", mProgram);
        if (!mKernelSumDiffSqr16sC1) mKernelSumDiffSqr16sC1 = new OpenCL::OpenCLKernel("SumDiffSqrKernel16sC1", mProgram);
        if (!mKernelSumDiff32fC1Final) mKernelSumDiff32fC1Final = new OpenCL::OpenCLKernel("SumDiffKernel32fC1Final", mProgram);
        if (!mKernelMinMax16sC1) mKernelMinMax16sC1 = new OpenCL::OpenCLKernel("MinMaxKernel16sC1", mProgram);
        if (!mKernelMinMax32fC1Final) mKernelMinMax32fC1Final = new OpenCL::OpenCLKernel("MinMaxKernel32fC1Final", mProgram);
        if (!mKernelHistogram16sC1) mKernelHistogram16sC1 = new OpenCL::OpenCLKernel("HistogramKernel16sC1", mProgram);
        data.kernel = mKernel16sC1;
        data.kernelFilter = mKernel16sC1Filter;
    break;
    case DT_INT:
        if (!mKernel32sC1) mKernel32sC1 = new OpenCL::OpenCLKernel("WriteToSurfaceKernel32sC1", mProgram);
        if (!mKernel32sC1Filter) mKernel32sC1Filter = new OpenCL::OpenCLKernel("WriteToSurfaceKernel32sC1Filter", mProgram);
        if (!mKernelSumDiff32sC1) mKernelSumDiff32sC1 = new OpenCL::OpenCLKernel("SumDiffKernel32sC1", mProgram);
        if (!mKernelSumDiffSqr32sC1) mKernelSumDiffSqr32sC1 = new OpenCL::OpenCLKernel("SumDiffSqrKernel32sC1", mProgram);
        if (!mKernelSumDiff32fC1Final) mKernelSumDiff32fC1Final = new OpenCL::OpenCLKernel("SumDiffKernel32fC1Final", mProgram);
        if (!mKernelMinMax32sC1) mKernelMinMax32sC1 = new OpenCL::OpenCLKernel("MinMaxKernel32sC1", mProgram);
        if (!mKernelMinMax32fC1Final) mKernelMinMax32fC1Final = new OpenCL::OpenCLKernel("MinMaxKernel32fC1Final", mProgram);
        if (!mKernelHistogram32sC1) mKernelHistogram32sC1 = new OpenCL::OpenCLKernel("HistogramKernel32sC1", mProgram);
        data.kernel = mKernel32sC1;
        data.kernelFilter = mKernel32sC1Filter;
    break;
    case DT_FLOAT:
        if (!mKernel32fC1) mKernel32fC1 = new OpenCL::OpenCLKernel("WriteToSurfaceKernel32fC1", mProgram);
        if (!mKernel32fC1Filter) mKernel32fC1Filter = new OpenCL::OpenCLKernel("WriteToSurfaceKernel32fC1Filter", mProgram);
        if (!mKernelSumDiff32fC1) mKernelSumDiff32fC1 = new OpenCL::OpenCLKernel("SumDiffKernel32fC1", mProgram);
        if (!mKernelSumDiffSqr32fC1) mKernelSumDiffSqr32fC1 = new OpenCL::OpenCLKernel("SumDiffSqrKernel32fC1", mProgram);
        if (!mKernelSumDiff32fC1Final) mKernelSumDiff32fC1Final = new OpenCL::OpenCLKernel("SumDiffKernel32fC1Final", mProgram);
        if (!mKernelMinMax32fC1) mKernelMinMax32fC1 = new OpenCL::OpenCLKernel("MinMaxKernel32fC1", mProgram);
        if (!mKernelMinMax32fC1Final) mKernelMinMax32fC1Final = new OpenCL::OpenCLKernel("MinMaxKernel32fC1Final", mProgram);
        if (!mKernelHistogram32fC1) mKernelHistogram32fC1 = new OpenCL::OpenCLKernel("HistogramKernel32fC1", mProgram);
        data.kernel = mKernel32fC1;
        data.kernelFilter = mKernel32fC1Filter;
    break;
    default:
        return;
    }
}

BaseFrameController::FrameControllerData *BaseFrameController::GetUserData()
{
    return &data;
}

bool BaseFrameController::GetIsLoaded()
{
    return mIsFileLoaded;
}

QString BaseFrameController::GetFilename()
{
    return mFilename;
}

void BaseFrameController::customEvent(QEvent* e)
{
    if(e->type() == QEvent::Type(ProgressUpdateEventID))
    {
        QProgressUpdateEvent* flu = static_cast<QProgressUpdateEvent *>(e);
        emit ProgressStatusChanged(flu->GetValue());
    }
}

QMatrix4x4 BaseFrameController::GetAlignmentMatrix()
{
    QMatrix4x4 m;
    m.setToIdentity();
    return m;
}



QProgressUpdateEvent::QProgressUpdateEvent(int aValue)
    : QEvent(QEvent::Type(ProgressUpdateEventID)), value(aValue)
{

}

int QProgressUpdateEvent::GetValue()
{
    return value;
}
