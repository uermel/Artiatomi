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


#include "ShiftFile.h"

#ifndef NO_CUDA

ShiftFile::ShiftFile(string aFileName)
	: EmFile(aFileName)
{
	OpenAndRead();
	//ReadHeaderInfo();
}

ShiftFile::ShiftFile(string aFileName, int aProjectionCount, int aMotiveCount)
	: EmFile(aFileName)
{
	//DimX = aMotiveCount;
	//DimY = aProjectionCount;
	//DimZ = 2;
	_fileHeader.DimX = aMotiveCount;
	_fileHeader.DimY = aProjectionCount;
	_fileHeader.DimZ = 2;
	
	//SetDataType(FDT_FLOAT);
	_fileHeader.DataType = EMDATATYPE_FLOAT;
	
	_data = new char[aProjectionCount * aMotiveCount * 2 * sizeof(float)]; //x and y coordinate
	memset(_data, 0, aProjectionCount * aMotiveCount * 2 * sizeof(float));
}

float* ShiftFile::GetData()
{
	return (float*)EmFile::GetData();
}

int ShiftFile::GetMotiveCount()
{
	return _fileHeader.DimX;
}

int ShiftFile::GetProjectionCount()
{
	return _fileHeader.DimY;
}

float2 ShiftFile::operator() (const int aProjection, const int aMotive)
{
	float2 erg;

	float* fdata = (float*)_data;
	erg.x = -fdata[0 * _fileHeader.DimX * _fileHeader.DimY + aProjection * _fileHeader.DimX + aMotive];
	erg.y = -fdata[1 * _fileHeader.DimX * _fileHeader.DimY + aProjection * _fileHeader.DimX + aMotive];
	return erg;
}

void ShiftFile::SetValue(const int aProjection, const int aMotive, float2 aVal)
{
	float* fdata = (float*)_data;
	fdata[0 * _fileHeader.DimX * _fileHeader.DimY + aProjection * _fileHeader.DimX + aMotive] = aVal.x;
	fdata[1 * _fileHeader.DimX * _fileHeader.DimY + aProjection * _fileHeader.DimX + aMotive] = aVal.y;
}
#endif
