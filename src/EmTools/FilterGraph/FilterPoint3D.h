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


#ifndef FILTERPOINT3D_H
#define FILTERPOINT3D_H

#include "../Basics/Default.h"

//Forward declatation
class FilterSize3D;
class FilterROI3D;

class FilterPoint3D
{
public:
	int x;
	int y;
	int z;

	FilterPoint3D();
	FilterPoint3D(int aX, int aY, int z);
	FilterPoint3D(const FilterPoint3D& aPoint);

	FilterPoint3D& operator=(const FilterPoint3D& arg);
	bool operator==(const FilterPoint3D& arg);

	FilterPoint3D operator+(const FilterPoint3D& value) const;
	FilterPoint3D operator+(const FilterSize3D& value) const;
	FilterPoint3D operator+(int value) const;

	FilterPoint3D operator+=(const FilterPoint3D& value);
	FilterPoint3D operator+=(const FilterSize3D& value);
	FilterPoint3D operator+=(int value);

	FilterPoint3D operator-(const FilterPoint3D& value) const;
	FilterPoint3D operator-(const FilterSize3D& value) const;
	FilterPoint3D operator-(int value) const;

	FilterPoint3D operator-=(const FilterPoint3D& value);
	FilterPoint3D operator-=(const FilterSize3D& value);
	FilterPoint3D operator-=(int value);

	FilterPoint3D operator*(const FilterPoint3D& value) const;
	FilterPoint3D operator*(const FilterSize3D& value) const;
	FilterPoint3D operator*(int value) const;

	FilterPoint3D operator*=(const FilterPoint3D& value);
	FilterPoint3D operator*=(const FilterSize3D& value);
	FilterPoint3D operator*=(int value);

	FilterPoint3D operator/(const FilterPoint3D& value) const;
	FilterPoint3D operator/(const FilterSize3D& value) const;
	FilterPoint3D operator/(int value) const;

	FilterPoint3D operator/=(const FilterPoint3D& value);
	FilterPoint3D operator/=(const FilterSize3D& value);
	FilterPoint3D operator/=(int value);

	FilterROI3D operator+(const FilterROI3D& value) const;
	FilterROI3D operator-(const FilterROI3D& value) const;
	FilterROI3D operator*(const FilterROI3D& value) const;
	FilterROI3D operator/(const FilterROI3D& value) const;
};

FilterPoint3D operator+(int lhs, const FilterPoint3D& rhs);
FilterPoint3D operator-(int lhs, const FilterPoint3D& rhs);
FilterPoint3D operator*(int lhs, const FilterPoint3D& rhs);
FilterPoint3D operator/(int lhs, const FilterPoint3D& rhs);

#endif