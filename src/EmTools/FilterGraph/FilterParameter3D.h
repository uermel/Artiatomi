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


#ifndef FILTERPARAMETER3D_H
#define FILTERPARAMETER3D_H

#include "../Basics/Default.h"
#include "FilterROI3D.h"
#include "FilterSize3D.h"

class FilterParameter3D
{
public:
	//! Size of the memory allocation the filter operates with
	FilterSize3D Size;
	//! The ROI the filter is computing output values
	FilterROI3D Roi;
	//! Data type
	DataType_enum DataType;
};

#endif