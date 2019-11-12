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


#ifndef SHIFTFILE_H
#define SHIFTFILE_H

#include "EmFile.h"
#include <vector_types.h>


using namespace std;

//! Represents a shift file stored in EM-file format.
/*!
\author Michael Kunz
\date   September 2011
\version 1.0
*/

struct my_float2 {
	float x;
	float y;
};


class ShiftFile : public EmFile
{
protected:
public:
	//! Creates a new ShiftFile instance. The file is directly read from file.
	ShiftFile(string aFileName);

	//! Creates a new ShiftFile instance. The file ist not yet created.
	ShiftFile(string aFileName, int aProjectionCount, int aMotiveCount);

	//! Returns the number of markers in the marker file.
	int GetMotiveCount();

	//! Returns the number of projections in the marker file.
	int GetProjectionCount();

	//! Returns a pointer to the inner data array.
	float* GetData();

	//! Returns value with index (\p aProjection, \p aMotive).
	my_float2 operator() (const int aProjection, const int aMotive);

	//! Returns value with index (\p aProjection, \p aMotive).
	void SetValue(const int aProjection, const int aMotive, float2 aVal);
};

#endif