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


#ifndef FILTERSIZE3D_H
#define FILTERSIZE3D_H

#include "FilterROI3D.h"

class FilterSize3D
{
public:
	int width;
	int height;
	int depth;

	FilterSize3D();
	FilterSize3D(int aWidth, int aHeight, int aDepth);
	FilterSize3D(const FilterSize3D& aSize);


	FilterSize3D& operator=(const FilterSize3D& arg);
	bool operator==(const FilterSize3D& arg);

	FilterSize3D operator+(int value) const;
	FilterSize3D operator+(const FilterSize3D& value) const;

	FilterSize3D& operator+=(int value);
	FilterSize3D& operator+=(const FilterSize3D& value);

	FilterSize3D operator-(int value) const;
	FilterSize3D operator-(const FilterSize3D& value) const;

	FilterSize3D& operator-=(int value);
	FilterSize3D& operator-=(const FilterSize3D& value);

	FilterSize3D operator*(int value) const;
	FilterSize3D operator*(const FilterSize3D& value) const;

	FilterSize3D& operator*=(int value);
	FilterSize3D& operator*=(const FilterSize3D& value);

	FilterSize3D operator/(int value) const;
	FilterSize3D operator/(const FilterSize3D& value) const;

	FilterSize3D& operator/=(int value);
	FilterSize3D& operator/=(const FilterSize3D& value);

	FilterROI3D operator+(const FilterROI3D& value) const;
	FilterROI3D operator-(const FilterROI3D& value) const;
	FilterROI3D operator*(const FilterROI3D& value) const;
	FilterROI3D operator/(const FilterROI3D& value) const;
};


FilterSize3D operator+(int lhs, const FilterSize3D& rhs);
FilterSize3D operator-(int lhs, const FilterSize3D& rhs);
FilterSize3D operator*(int lhs, const FilterSize3D& rhs);
FilterSize3D operator/(int lhs, const FilterSize3D& rhs);

#endif