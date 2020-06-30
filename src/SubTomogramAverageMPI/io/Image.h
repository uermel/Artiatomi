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


#ifndef IMAGE_H
#define IMAGE_H

#include "IODefault.h"

//union value
//{
//	float f;
//	uint ui;
//	int i;
//};

class Image
{
protected:
	void SetArraysNULL();
	void AllocArrays(int aSize);
	void DeAllocArrays();
	char* _data;
	
#pragma region Converters
	//uchar* ConvertPixelTypeUCHAR(uchar* in, int aAdd);
	uchar* ConvertPixelTypeUCHAR(ushort* in, int aAdd);
	uchar* ConvertPixelTypeUCHAR(uint* in, int aAdd);
	uchar* ConvertPixelTypeUCHAR(ulong64* in, int aAdd);
	uchar* ConvertPixelTypeUCHAR(float* in, int aAdd);
	uchar* ConvertPixelTypeUCHAR(double* in, int aAdd);
	uchar* ConvertPixelTypeUCHAR(char* in, int aAdd);
	uchar* ConvertPixelTypeUCHAR(short* in, int aAdd);
	uchar* ConvertPixelTypeUCHAR(int* in, int aAdd);
	uchar* ConvertPixelTypeUCHAR(long* in, int aAdd);
	
	ushort* ConvertPixelTypeUSHORT(uchar* in, int aAdd);
	//ushort* ConvertPixelTypeUSHORT(ushort* in, int aAdd);
	ushort* ConvertPixelTypeUSHORT(uint* in, int aAdd);
	ushort* ConvertPixelTypeUSHORT(ulong64* in, int aAdd);
	ushort* ConvertPixelTypeUSHORT(float* in, int aAdd);
	ushort* ConvertPixelTypeUSHORT(double* in, int aAdd);
	ushort* ConvertPixelTypeUSHORT(char* in, int aAdd);
	ushort* ConvertPixelTypeUSHORT(short* in, int aAdd);
	ushort* ConvertPixelTypeUSHORT(int* in, int aAdd);
	ushort* ConvertPixelTypeUSHORT(long* in, int aAdd);
	
	uint* ConvertPixelTypeUINT(uchar* in, int aAdd);
	uint* ConvertPixelTypeUINT(ushort* in, int aAdd);
	//uint* ConvertPixelTypeUINT(uint* in, int aAdd);
	uint* ConvertPixelTypeUINT(ulong64* in, int aAdd);
	uint* ConvertPixelTypeUINT(float* in, int aAdd);
	uint* ConvertPixelTypeUINT(double* in, int aAdd);
	uint* ConvertPixelTypeUINT(char* in, int aAdd);
	uint* ConvertPixelTypeUINT(short* in, int aAdd);
	uint* ConvertPixelTypeUINT(int* in, int aAdd);
	uint* ConvertPixelTypeUINT(long* in, int aAdd);
	
	ulong64* ConvertPixelTypeULONG(uchar* in, int aAdd);
	ulong64* ConvertPixelTypeULONG(ushort* in, int aAdd);
	ulong64* ConvertPixelTypeULONG(uint* in, int aAdd);
	//ulong64* ConvertPixelTypeULONG(ulong64* in, int aAdd);
	ulong64* ConvertPixelTypeULONG(float* in, int aAdd);
	ulong64* ConvertPixelTypeULONG(double* in, int aAdd);
	ulong64* ConvertPixelTypeULONG(char* in, int aAdd);
	ulong64* ConvertPixelTypeULONG(short* in, int aAdd);
	ulong64* ConvertPixelTypeULONG(int* in, int aAdd);
	ulong64* ConvertPixelTypeULONG(long* in, int aAdd);
	
	float* ConvertPixelTypeFLOAT(uchar* in, int aAdd);
	float* ConvertPixelTypeFLOAT(ushort* in, int aAdd);
	float* ConvertPixelTypeFLOAT(uint* in, int aAdd);
	float* ConvertPixelTypeFLOAT(ulong64* in, int aAdd);
	//float* ConvertPixelTypeFLOAT(float* in, int aAdd);
	float* ConvertPixelTypeFLOAT(double* in, int aAdd);
	float* ConvertPixelTypeFLOAT(char* in, int aAdd);
	float* ConvertPixelTypeFLOAT(short* in, int aAdd);
	float* ConvertPixelTypeFLOAT(int* in, int aAdd);
	float* ConvertPixelTypeFLOAT(long* in, int aAdd);
	
	double* ConvertPixelTypeDOUBLE(uchar* in, int aAdd);
	double* ConvertPixelTypeDOUBLE(ushort* in, int aAdd);
	double* ConvertPixelTypeDOUBLE(uint* in, int aAdd);
	double* ConvertPixelTypeDOUBLE(ulong64* in, int aAdd);
	double* ConvertPixelTypeDOUBLE(float* in, int aAdd);
	//double* ConvertPixelTypeDOUBLE(double* in, int aAdd);
	double* ConvertPixelTypeDOUBLE(char* in, int aAdd);
	double* ConvertPixelTypeDOUBLE(short* in, int aAdd);
	double* ConvertPixelTypeDOUBLE(int* in, int aAdd);
	double* ConvertPixelTypeDOUBLE(long* in, int aAdd);
	
	char* ConvertPixelTypeCHAR(uchar* in, int aAdd);
	char* ConvertPixelTypeCHAR(ushort* in, int aAdd);
	char* ConvertPixelTypeCHAR(uint* in, int aAdd);
	char* ConvertPixelTypeCHAR(ulong64* in, int aAdd);
	char* ConvertPixelTypeCHAR(float* in, int aAdd);
	char* ConvertPixelTypeCHAR(double* in, int aAdd);
	//char* ConvertPixelTypeCHAR(char* in, int aAdd);
	char* ConvertPixelTypeCHAR(short* in, int aAdd);
	char* ConvertPixelTypeCHAR(int* in, int aAdd);
	char* ConvertPixelTypeCHAR(long* in, int aAdd);
	
	short* ConvertPixelTypeSHORT(uchar* in, int aAdd);
	short* ConvertPixelTypeSHORT(ushort* in, int aAdd);
	short* ConvertPixelTypeSHORT(uint* in, int aAdd);
	short* ConvertPixelTypeSHORT(ulong64* in, int aAdd);
	short* ConvertPixelTypeSHORT(float* in, int aAdd);
	short* ConvertPixelTypeSHORT(double* in, int aAdd);
	short* ConvertPixelTypeSHORT(char* in, int aAdd);
	//short* ConvertPixelTypeSHORT(short* in, int aAdd);
	short* ConvertPixelTypeSHORT(int* in, int aAdd);
	short* ConvertPixelTypeSHORT(long* in, int aAdd);
	
	int* ConvertPixelTypeINT(uchar* in, int aAdd);
	int* ConvertPixelTypeINT(ushort* in, int aAdd);
	int* ConvertPixelTypeINT(uint* in, int aAdd);
	int* ConvertPixelTypeINT(ulong64* in, int aAdd);
	int* ConvertPixelTypeINT(float* in, int aAdd);
	int* ConvertPixelTypeINT(double* in, int aAdd);
	int* ConvertPixelTypeINT(char* in, int aAdd);
	int* ConvertPixelTypeINT(short* in, int aAdd);
	//int* ConvertPixelTypeINT(int* in, int aAdd);
	int* ConvertPixelTypeINT(long* in, int aAdd);
	
	long* ConvertPixelTypeLONG(uchar* in, int aAdd);
	long* ConvertPixelTypeLONG(ushort* in, int aAdd);
	long* ConvertPixelTypeLONG(uint* in, int aAdd);
	long* ConvertPixelTypeLONG(ulong64* in, int aAdd);
	long* ConvertPixelTypeLONG(float* in, int aAdd);
	long* ConvertPixelTypeLONG(double* in, int aAdd);
	long* ConvertPixelTypeLONG(char* in, int aAdd);
	long* ConvertPixelTypeLONG(short* in, int aAdd);
	long* ConvertPixelTypeLONG(int* in, int aAdd);
	//long* ConvertPixelTypeLONG(long* in, int aAdd);
#pragma endregion

public:
	Image();
	Image(const Image& aImage);
	virtual ~Image();

	virtual void ReadHeaderInfo() = 0;
	virtual void WriteInfoToHeader() = 0;

	void ClearImageMetadata();
	
#pragma region Members
	//! Image size in pixels X
	int DimX;
	//! Image size in pixels Y
	int DimY;
	//! Image size in pixels Z. Also defines the size of array members.
	int DimZ;
	
	//! Pixel data type
	FileDataType_enum DataType;

	//! Pixel/Voxel size in Angstroms X
	float CellSizeX; 
	//! Pixel/Voxel size in Angstroms X
	float CellSizeY;
	//! Pixel/Voxel size in Angstroms X
	float CellSizeZ;
	
	//! Cell Angles (Degrees)
	float CellAngleAlpha;
	//! Cell Angles (Degrees)
	float CellAngleBeta;
	//! Cell Angles (Degrees)
	float CellAngleGamma;
	
	//! Minimum density value
	float* MinValue;
	//! Maximum density value
	float* MaxValue;
	//! Mean density value (Average)
	float* MeanValue;
	
	//! X origin
	float OriginX;
	//! Y origin
	float OriginY;
	//! Z origin
	float OriginZ;

	//! RMS
	float RMS;
	
	//! Alpha tilt (deg)
	float* TiltAlpha;
	//! Beta tilt (deg)
	float* TiltBeta;
	
	//! Stage x position (Unit=m. But if value>1, unit=???m)
	float* StageXPosition;
	//! Stage y position (Unit=m. But if value>1, unit=???m)
	float* StageYPosition;
	//! Stage z position (Unit=m. But if value>1, unit=???m)
	float* StageZPosition;
	
	//! x Image shift (Unit=m. But if value>1, unit=???m)
	float* ShiftX;
	//! y Image shift (Unit=m. But if value>1, unit=???m)
	float* ShiftY;
	//! z Image shift (Unit=m. But if value>1, unit=???m)
	float* ShiftZ;

	//! Defocus
	float* Defocus;
	//! ExposureTime
	float* ExposureTime;
	//! Tilt axis (deg)
	float* TiltAxis;
	//! Pixel size of image (m)
	float* PixelSize;
	//! Magnification used
	float* Magnification;
	//! Binning
	float* Binning;
	//! AppliedDefocus
	float* AppliedDefocus;

	//! accelerating Voltage
	int Voltage;
	//! Cs
	int Cs;
#pragma endregion

	uint GetPixelSizeInBytes() const;
	uint GetImageSizeInBytes() const;
	void ConvertPixelType(FileDataType_enum aNewType, int aAdd);
	char* GetXYSlice(uint aZ);
	char* GetYZSlice(uint aX);
	char* GetXZSlice(uint aY);
};


#endif