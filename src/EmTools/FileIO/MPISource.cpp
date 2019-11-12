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


#include "MPISource.h"
//#include "../utils/Config.h"

MPISource::MPISource(int aDimX, int aDimY, int aDimZ, float aPixelSize)
	: ProjectionSource(), _dimX(aDimX), _dimY(aDimY), _dimZ(aDimZ), _pixelSize(aPixelSize)
{
}

MPISource::~MPISource()
{
	/*if (_projectionCache)
        for (int i = 0; i < DimZ; i++)
            if (_projectionCache[i])
                delete[] _projectionCache[i];*/
}

//bool MPISource::OpenAndRead()
//{
//	return true;
//}

//bool MPISource::OpenAndWrite()
//{
//	return true;
//}

DataType_enum MPISource::GetDataType()
{
	return DT_FLOAT;
}

//void MPISource::SetDataType(FileDataType_enum aType)
//{
//	
//}

//size_t MPISource::GetDataSize()
//{
//	return 0;
//}

//char* MPISource::GetData()
//{
//	return _data;
//}

char* MPISource::GetProjection(uint aIndex)
{
	return NULL;
}

//float* MPISource::GetProjectionFloat(uint aIndex)
//{
//	return NULL;
//}
//
//float* MPISource::GetProjectionInvertFloat(uint aIndex)
//{
//	return NULL;
//}

//void MPISource::ReadHeaderInfo()
//{}
//
//void MPISource::WriteInfoToHeader()
//{}
float MPISource::GetPixelSize()
{
	return _pixelSize;
}

int MPISource::GetWidth()
{
	return _dimX;
}
int MPISource::GetHeight()
{
	return _dimY;
}
int MPISource::GetProjectionCount()
{
	return _dimZ;
}
