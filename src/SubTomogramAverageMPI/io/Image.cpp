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


#include "Image.h"

Image::Image() : 
	DimX(0), DimY(0), DimZ(0), DataType(FDT_UNKNOWN), CellSizeX(0), CellSizeY(0), CellSizeZ(0),
	CellAngleAlpha(0), CellAngleBeta(0), CellAngleGamma(0), OriginX(0), OriginY(0), OriginZ(0),
	RMS(0)
{	
	SetArraysNULL();
	_data = NULL;
}

Image::Image(const Image& aImage) : 
	DimX(aImage.DimX), DimY(aImage.DimY), DimZ(aImage.DimZ), DataType(aImage.DataType), CellSizeX(aImage.CellSizeX), CellSizeY(aImage.CellSizeY), CellSizeZ(aImage.CellSizeZ),
	CellAngleAlpha(aImage.CellAngleAlpha), CellAngleBeta(aImage.CellAngleBeta), CellAngleGamma(aImage.CellAngleGamma), OriginX(aImage.OriginX), OriginY(aImage.OriginY), OriginZ(aImage.OriginZ),
	RMS(aImage.RMS)
{	
	SetArraysNULL();
	AllocArrays(DimZ);
	
	memcpy(MinValue, aImage.MinValue, sizeof(float) * DimZ);
	memcpy(MaxValue, aImage.MaxValue, sizeof(float) * DimZ);
	memcpy(MeanValue, aImage.MeanValue, sizeof(float) * DimZ);

	memcpy(TiltAlpha, aImage.TiltAlpha, sizeof(float) * DimZ);
	memcpy(TiltBeta, aImage.TiltBeta, sizeof(float) * DimZ);
	memcpy(StageXPosition, aImage.StageXPosition, sizeof(float) * DimZ);
	memcpy(StageYPosition, aImage.StageYPosition, sizeof(float) * DimZ);
	memcpy(StageZPosition, aImage.StageZPosition, sizeof(float) * DimZ);
	memcpy(ShiftX, aImage.ShiftX, sizeof(float) * DimZ);
	memcpy(ShiftY, aImage.ShiftY, sizeof(float) * DimZ);
	memcpy(ShiftZ, aImage.ShiftZ, sizeof(float) * DimZ);
	memcpy(Defocus, aImage.Defocus, sizeof(float) * DimZ);
	memcpy(ExposureTime, aImage.ExposureTime, sizeof(float) * DimZ);
	memcpy(TiltAxis, aImage.TiltAxis, sizeof(float) * DimZ);
	memcpy(PixelSize, aImage.PixelSize, sizeof(float) * DimZ);
	memcpy(Magnification, aImage.Magnification, sizeof(float) * DimZ);
	memcpy(Binning, aImage.Binning, sizeof(float) * DimZ);
	memcpy(AppliedDefocus, aImage.AppliedDefocus, sizeof(float) * DimZ);

	uint size = aImage.GetImageSizeInBytes();
	_data = new char[size];
	memcpy(_data, aImage._data, size);
}

Image::~Image()
{	
	DeAllocArrays();
	if (_data)
		delete[] _data;
	_data = NULL;
}

void Image::SetArraysNULL()
{
	MinValue = NULL;
	MaxValue = NULL;
	MeanValue = NULL;
	
	TiltAlpha = NULL;
	TiltBeta = NULL;
	
	StageXPosition = NULL;
	StageYPosition = NULL;
	StageZPosition = NULL;
	
	ShiftX = NULL;
	ShiftY = NULL;
	ShiftZ = NULL;

	Defocus = NULL;
	ExposureTime = NULL;
	TiltAxis = NULL;
	PixelSize = NULL;
	Magnification = NULL;
	Binning = NULL;
	AppliedDefocus = NULL;
}
	
void Image::AllocArrays(int aSize)
{
	DeAllocArrays();
	MinValue = new float[aSize];
	MaxValue = new float[aSize];
	MeanValue = new float[aSize];
	
	TiltAlpha = new float[aSize];
	TiltBeta = new float[aSize];
	
	StageXPosition = new float[aSize];
	StageYPosition = new float[aSize];
	StageZPosition = new float[aSize];
	
	ShiftX = new float[aSize];
	ShiftY = new float[aSize];
	ShiftZ = new float[aSize];

	Defocus = new float[aSize];
	ExposureTime = new float[aSize];
	TiltAxis = new float[aSize];
	PixelSize = new float[aSize];
	Magnification = new float[aSize];
	AppliedDefocus = new float[aSize];
	Binning = new float[aSize];
	
	memset(MinValue, 0, aSize * sizeof(float));
	memset(MaxValue, 0, aSize * sizeof(float));
	memset(MeanValue, 0, aSize * sizeof(float));
	
	memset(TiltAlpha, 0, aSize * sizeof(float));
	memset(TiltBeta, 0, aSize * sizeof(float));
	
	memset(StageXPosition, 0, aSize * sizeof(float));
	memset(StageYPosition, 0, aSize * sizeof(float));
	memset(StageZPosition, 0, aSize * sizeof(float));
	
	memset(ShiftX, 0, aSize * sizeof(float));
	memset(ShiftY, 0, aSize * sizeof(float));
	memset(ShiftZ, 0, aSize * sizeof(float));

	memset(Defocus, 0, aSize * sizeof(float));
	memset(ExposureTime, 0, aSize * sizeof(float));
	memset(TiltAxis, 0, aSize * sizeof(float));
	memset(PixelSize, 0, aSize * sizeof(float));
	memset(Magnification, 0, aSize * sizeof(float));
	memset(AppliedDefocus, 0, aSize * sizeof(float));
	memset(Binning, 0, aSize * sizeof(float));
}

void Image::DeAllocArrays()
{
	if (MinValue)
		delete[] MinValue;
	
	if (MaxValue)
		delete[] MaxValue;
	if (MeanValue)
		delete[] MeanValue;
	
	if (TiltAlpha)
		delete[] TiltAlpha;
	if (TiltBeta)
		delete[] TiltBeta;
	
	if (StageXPosition)
		delete[] StageXPosition;
	if (StageYPosition)
		delete[] StageYPosition;
	if (StageZPosition)
		delete[] StageZPosition;
	
	if (ShiftX)
		delete[] ShiftX;
	if (ShiftY)
		delete[] ShiftY;
	if (ShiftZ)
		delete[] ShiftZ;
	
	if (Defocus)
		delete[] Defocus;
	if (ExposureTime)
		delete[] ExposureTime;
	if (TiltAxis)
		delete[] TiltAxis;
	if (PixelSize)
		delete[] PixelSize;
	if (Magnification)
		delete[] Magnification;
	if (AppliedDefocus)
		delete[] AppliedDefocus;
	if (Binning)
		delete[] Binning;
	

	SetArraysNULL();
}

void Image::ClearImageMetadata()
{
	DimX = 0;
	DimY = 0;
	DimZ = 0;
	
	DataType = FDT_UNKNOWN;

	CellSizeX = 0; 
	CellSizeY = 0;
	CellSizeZ = 0;
	
	CellAngleAlpha = 0;
	CellAngleBeta = 0;
	CellAngleGamma = 0;

	OriginX = 0;
	OriginY = 0;
	OriginZ = 0;

	RMS = 0;

	Voltage = 0;
	Cs = 0;

	DeAllocArrays();
}

uint Image::GetPixelSizeInBytes() const
{
	switch (DataType)
	{	
	case FDT_UCHAR:
		return 1;
	case FDT_USHORT:
		return 2;
	case FDT_UINT:
		return 4;
	case FDT_ULONG:
		return 8;
	case FDT_FLOAT:
		return 4;
	case FDT_DOUBLE:
		return 8;
	case FDT_CHAR:
		return 1;
	case FDT_SHORT:
		return 2;
	case FDT_INT:
		return 4;
	case FDT_LONG:
		return 8;
	case FDT_FLOAT2:
		return 8;
	case FDT_SHORT2:
		return 4;
	default:
		return 0;
	}
}

uint Image::GetImageSizeInBytes() const
{
	return GetPixelSizeInBytes() * DimX * DimY * DimZ;
}

void Image::ConvertPixelType(FileDataType_enum aNewType, int aAdd)
{	
	if (DataType == FDT_UNKNOWN ||
		DataType == FDT_FLOAT2 ||
		DataType == FDT_SHORT2)
		return;


	char* tmp;

	switch (DataType)
	{	
	case FDT_UCHAR:
		switch (aNewType)
		{			
		case FDT_UCHAR:
			break;
		case FDT_USHORT:
			tmp = (char*)ConvertPixelTypeUSHORT((uchar*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_UINT:
			tmp = (char*)ConvertPixelTypeUINT((uchar*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_ULONG:
			tmp = (char*)ConvertPixelTypeULONG((uchar*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_FLOAT:
			tmp = (char*)ConvertPixelTypeFLOAT((uchar*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_DOUBLE:
			tmp = (char*)ConvertPixelTypeDOUBLE((uchar*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_CHAR:
			tmp = (char*)ConvertPixelTypeCHAR((uchar*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_SHORT:
			tmp = (char*)ConvertPixelTypeSHORT((uchar*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_INT:
			tmp = (char*)ConvertPixelTypeINT((uchar*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_LONG:
			tmp = (char*)ConvertPixelTypeLONG((uchar*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		}
		return;
	case FDT_USHORT:
		switch (aNewType)
		{			
		case FDT_UCHAR:
			tmp = (char*)ConvertPixelTypeUCHAR((ushort*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_USHORT:
			break;
		case FDT_UINT:
			tmp = (char*)ConvertPixelTypeUINT((ushort*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_ULONG:
			tmp = (char*)ConvertPixelTypeULONG((ushort*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_FLOAT:
			tmp = (char*)ConvertPixelTypeFLOAT((ushort*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_DOUBLE:
			tmp = (char*)ConvertPixelTypeDOUBLE((ushort*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_CHAR:
			tmp = (char*)ConvertPixelTypeCHAR((ushort*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_SHORT:
			tmp = (char*)ConvertPixelTypeSHORT((ushort*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_INT:
			tmp = (char*)ConvertPixelTypeINT((ushort*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_LONG:
			tmp = (char*)ConvertPixelTypeLONG((ushort*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		}
		return;
	case FDT_UINT:
		switch (aNewType)
		{			
		case FDT_UCHAR:
			tmp = (char*)ConvertPixelTypeUCHAR((uint*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_USHORT:
			tmp = (char*)ConvertPixelTypeUSHORT((uint*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_UINT:
			break;
		case FDT_ULONG:
			tmp = (char*)ConvertPixelTypeULONG((uint*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_FLOAT:
			tmp = (char*)ConvertPixelTypeFLOAT((uint*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_DOUBLE:
			tmp = (char*)ConvertPixelTypeDOUBLE((uint*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_CHAR:
			tmp = (char*)ConvertPixelTypeCHAR((uint*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_SHORT:
			tmp = (char*)ConvertPixelTypeSHORT((uint*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_INT:
			tmp = (char*)ConvertPixelTypeINT((uint*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_LONG:
			tmp = (char*)ConvertPixelTypeLONG((uint*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		}
		return;
	case FDT_ULONG:
		switch (aNewType)
		{			
		case FDT_UCHAR:
			tmp = (char*)ConvertPixelTypeUCHAR((ulong64*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_USHORT:
			tmp = (char*)ConvertPixelTypeUSHORT((ulong64*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_UINT:
			tmp = (char*)ConvertPixelTypeUINT((ulong64*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_ULONG:
			break;
		case FDT_FLOAT:
			tmp = (char*)ConvertPixelTypeFLOAT((ulong64*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_DOUBLE:
			tmp = (char*)ConvertPixelTypeDOUBLE((ulong64*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_CHAR:
			tmp = (char*)ConvertPixelTypeCHAR((ulong64*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_SHORT:
			tmp = (char*)ConvertPixelTypeSHORT((ulong64*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_INT:
			tmp = (char*)ConvertPixelTypeINT((ulong64*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_LONG:
			tmp = (char*)ConvertPixelTypeLONG((ulong64*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		}
		return;
	case FDT_FLOAT:
		switch (aNewType)
		{			
		case FDT_UCHAR:
			tmp = (char*)ConvertPixelTypeUCHAR((float*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_USHORT:
			tmp = (char*)ConvertPixelTypeUSHORT((float*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_UINT:
			tmp = (char*)ConvertPixelTypeUINT((float*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_ULONG:
			tmp = (char*)ConvertPixelTypeULONG((float*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_FLOAT:
			break;
		case FDT_DOUBLE:
			tmp = (char*)ConvertPixelTypeDOUBLE((float*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_CHAR:
			tmp = (char*)ConvertPixelTypeCHAR((float*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_SHORT:
			tmp = (char*)ConvertPixelTypeSHORT((float*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_INT:
			tmp = (char*)ConvertPixelTypeINT((float*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_LONG:
			tmp = (char*)ConvertPixelTypeLONG((float*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		}
		return;
	case FDT_DOUBLE:
		switch (aNewType)
		{			
		case FDT_UCHAR:
			tmp = (char*)ConvertPixelTypeUCHAR((double*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_USHORT:
			tmp = (char*)ConvertPixelTypeUSHORT((double*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_UINT:
			tmp = (char*)ConvertPixelTypeUINT((double*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_ULONG:
			tmp = (char*)ConvertPixelTypeULONG((double*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_FLOAT:
			tmp = (char*)ConvertPixelTypeFLOAT((double*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_DOUBLE:
			break;
		case FDT_CHAR:
			tmp = (char*)ConvertPixelTypeCHAR((double*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_SHORT:
			tmp = (char*)ConvertPixelTypeSHORT((double*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_INT:
			tmp = (char*)ConvertPixelTypeINT((double*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_LONG:
			tmp = (char*)ConvertPixelTypeLONG((double*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		}
		return;
	case FDT_CHAR:
		switch (aNewType)
		{			
		case FDT_UCHAR:
			tmp = (char*)ConvertPixelTypeUCHAR((char*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_USHORT:
			tmp = (char*)ConvertPixelTypeUSHORT((char*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_UINT:
			tmp = (char*)ConvertPixelTypeUINT((char*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_ULONG:
			tmp = (char*)ConvertPixelTypeULONG((char*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_FLOAT:
			tmp = (char*)ConvertPixelTypeFLOAT((char*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_DOUBLE:
			tmp = (char*)ConvertPixelTypeDOUBLE((char*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_CHAR:
			break;
		case FDT_SHORT:
			tmp = (char*)ConvertPixelTypeSHORT((char*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_INT:
			tmp = (char*)ConvertPixelTypeINT((char*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_LONG:
			tmp = (char*)ConvertPixelTypeLONG((char*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		}
		return;
	case FDT_SHORT:
		switch (aNewType)
		{			
		case FDT_UCHAR:
			tmp = (char*)ConvertPixelTypeUCHAR((short*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_USHORT:
			tmp = (char*)ConvertPixelTypeUSHORT((short*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_UINT:
			tmp = (char*)ConvertPixelTypeUINT((short*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_ULONG:
			tmp = (char*)ConvertPixelTypeULONG((short*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			break;
		case FDT_FLOAT:
			tmp = (char*)ConvertPixelTypeFLOAT((short*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_DOUBLE:
			tmp = (char*)ConvertPixelTypeDOUBLE((short*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_CHAR:
			tmp = (char*)ConvertPixelTypeCHAR((short*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_SHORT:
			DataType = aNewType;
			break;
		case FDT_INT:
			tmp = (char*)ConvertPixelTypeINT((short*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_LONG:
			tmp = (char*)ConvertPixelTypeLONG((short*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		}
		return;
	case FDT_INT:
		switch (aNewType)
		{			
		case FDT_UCHAR:
			tmp = (char*)ConvertPixelTypeUCHAR((int*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_USHORT:
			tmp = (char*)ConvertPixelTypeUSHORT((int*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_UINT:
			tmp = (char*)ConvertPixelTypeUINT((int*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_ULONG:
			tmp = (char*)ConvertPixelTypeULONG((int*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_FLOAT:
			tmp = (char*)ConvertPixelTypeFLOAT((int*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_DOUBLE:
			tmp = (char*)ConvertPixelTypeDOUBLE((int*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_CHAR:
			tmp = (char*)ConvertPixelTypeCHAR((int*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_SHORT:
			tmp = (char*)ConvertPixelTypeSHORT((int*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_INT:
			break;
		case FDT_LONG:
			tmp = (char*)ConvertPixelTypeLONG((int*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		}
		return;
	case FDT_LONG:
		switch (aNewType)
		{			
		case FDT_UCHAR:
			tmp = (char*)ConvertPixelTypeUCHAR((long*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_USHORT:
			tmp = (char*)ConvertPixelTypeUSHORT((long*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_UINT:
			tmp = (char*)ConvertPixelTypeUINT((long*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_ULONG:
			tmp = (char*)ConvertPixelTypeULONG((long*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_FLOAT:
			tmp = (char*)ConvertPixelTypeFLOAT((long*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_DOUBLE:
			tmp = (char*)ConvertPixelTypeDOUBLE((long*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_CHAR:
			tmp = (char*)ConvertPixelTypeCHAR((long*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_SHORT:
			tmp = (char*)ConvertPixelTypeSHORT((long*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_INT:
			tmp = (char*)ConvertPixelTypeINT((long*) _data, aAdd);
			delete[] _data;
			_data = tmp;
			DataType = aNewType;
			break;
		case FDT_LONG:
			break;
		}
		return;
	}
}

char* Image::GetXYSlice(uint aZ)
{
	uint pixel = GetPixelSizeInBytes();
	if (!pixel) return NULL;

	return _data + (pixel * aZ * DimX * DimY);
}

char* Image::GetYZSlice(uint aX)
{
	uint pixel = GetPixelSizeInBytes();
	uint img = pixel * DimY * DimZ;

	if (!pixel) return NULL;
	
	char* slice = new char[img];
	uint counter = 0;
	
	for (int z = 0; z < DimZ; z++)
	{
		for (int y = 0; y < DimY; y++)
		{
			for (uint i = 0; i < pixel; i++)
			{
				slice[counter * pixel + i] = _data[(z * DimX * DimY + y * DimX + aX) * pixel + i];
			}
			counter++;
		}
	}

	return slice;
}

char* Image::GetXZSlice(uint aY)
{
	uint pixel = GetPixelSizeInBytes();
	uint img = pixel * DimY * DimZ;

	if (!pixel) return NULL;
	
	char* slice = new char[img];
	uint counter = 0;
	
	for (int z = 0; z < DimZ; z++)
	{
		for (int x = 0; x < DimX; x++)
		{
			for (uint i = 0; i < pixel; i++)
			{
				slice[counter * pixel + i] = _data[(z * DimX * DimY + aY * DimX + x) * pixel + i];
			}
			counter++;
		}
	}

	return slice;
}





	/*int DimX;
	int DimY;
	int DimZ;
	
	FileDataType_enum DataType;

	float CellSizeX; 
	float CellSizeY;
	float CellSizeZ;
	
	float CellAngleAlpha;
	float CellAngleBeta;
	float CellAngleGamma;
	
	float* MinValue;
	float* MaxValue;
	float* MeanValue;
	
	float OriginX;
	float OriginY;
	float OriginZ;

	float RMS;
	
	float* TiltAlpha;
	float* TiltBeta;
	
	float* StageXPosition;
	float* StageYPosition;
	float* StageZPosition;
	
	float* ShiftX;
	float* ShiftY;
	float* ShiftZ;

	float* Defocus;
	float* ExposureTime;
	float* TiltAxis;
	float* PixelSize;
	float* Magnification;*/