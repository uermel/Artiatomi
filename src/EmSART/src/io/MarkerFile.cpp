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


#include "MarkerFile.h"

MarkerFile::MarkerFile(string aFileName, int aRefMarker)
	: EMFile(aFileName), mRefMarker(aRefMarker)
{
	OpenAndRead();
	ReadHeaderInfo();
}

float* MarkerFile::GetData()
{
	return (float*)EMFile::GetData();
}

int MarkerFile::GetMarkerCount()
{
	return _fileHeader.DimZ;
}

int MarkerFile::GetProjectionCount()
{
	int count = 0;
	for (int i = 0; i < _fileHeader.DimY; i++)
		if((*this)(MFI_X_Coordinate, i, mRefMarker) > 0 && (*this)(MFI_Y_Coordinate, i, mRefMarker) > 0) count++;
	return count;
}

bool MarkerFile::CheckIfProjIndexIsGood(const int index)
{
	return ((*this)(MFI_X_Coordinate, index, mRefMarker) > 0 && (*this)(MFI_Y_Coordinate, index, mRefMarker) > 0);
}

float& MarkerFile::operator() (const MarkerFileItem_enum aItem, const int aProjection, const int aMarker)
{
	float* fdata = (float*) _data;
//    if (aItem == MFI_RotationPsi) {
//        fdata[aMarker * DimX * DimY + aProjection * DimX + aItem] = 0;}
//    if (aItem == MFI_X_Shift) {
//        fdata[aMarker * DimX * DimY + aProjection * DimX + aItem] = 0;}
//    if (aItem == MFI_Y_Shift) {
//        fdata[aMarker * DimX * DimY + aProjection * DimX + aItem] = 0;}

	return fdata[aMarker * DimX * DimY + aProjection * DimX + aItem];
}
