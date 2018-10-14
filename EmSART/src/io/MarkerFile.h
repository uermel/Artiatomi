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


#ifndef MARKERFILE_H
#define MARKERFILE_H

#include "IODefault.h"
#include "EMFile.h"


using namespace std;

//! Definition of marker file items
enum MarkerFileItem_enum
{
	MFI_TiltAngle = 0,
	MFI_X_Coordinate,
	MFI_Y_Coordinate,
	MFI_DevOfMark,
	MFI_X_Shift,
	MFI_Y_Shift,
	MFI_X_MeanShift,
	MFI_MagnifiactionX=8,
	MFI_MagnifiactionY=8,
	MFI_RotationPsi=9
};

//! Represents a marker file stored in EM-file format.
/*!
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class MarkerFile : private EMFile
{
protected:
	int mRefMarker;
public:
	//! Creates a new MarkerFile instance. The file is directly read from file.
	MarkerFile(string aFileName, int aRefMarker);

	//! Returns the number of markers in the marker file.
	int GetMarkerCount();
	
	//! Returns the number of projections in the marker file.
	int GetProjectionCount();

	//! Returns a pointer to the inner data array.
	float* GetData();

	//! Returns a reference to value with index (\p aItem, \p aProjection, \p aMarker).
	float& operator() (const MarkerFileItem_enum aItem, const int aProjection, const int aMarker);

	bool CheckIfProjIndexIsGood(const int index);
};

#endif