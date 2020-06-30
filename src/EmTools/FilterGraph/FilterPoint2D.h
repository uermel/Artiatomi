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


#ifndef FILTERPOINT2D_H
#define FILTERPOINT2D_H

#include "../Basics/Default.h"

//Forward declatation
class FilterSize;
class FilterROI;

class FilterPoint2D
{
public:
	int x;
	int y;

	FilterPoint2D();
	FilterPoint2D(int aX, int aY);
	FilterPoint2D(const FilterPoint2D& aPoint);

	FilterPoint2D& operator=(const FilterPoint2D& arg);
	bool operator==(const FilterPoint2D& arg);

	FilterPoint2D operator+(const FilterPoint2D& value) const;
	FilterPoint2D operator+(const FilterSize& value) const;
	FilterPoint2D operator+(int value) const;

	FilterPoint2D operator+=(const FilterPoint2D& value);
	FilterPoint2D operator+=(const FilterSize& value);
	FilterPoint2D operator+=(int value);

	FilterPoint2D operator-(const FilterPoint2D& value) const;
	FilterPoint2D operator-(const FilterSize& value) const;
	FilterPoint2D operator-(int value) const;

	FilterPoint2D operator-=(const FilterPoint2D& value);
	FilterPoint2D operator-=(const FilterSize& value);
	FilterPoint2D operator-=(int value);

	FilterPoint2D operator*(const FilterPoint2D& value) const;
	FilterPoint2D operator*(const FilterSize& value) const;
	FilterPoint2D operator*(int value) const;

	FilterPoint2D operator*=(const FilterPoint2D& value);
	FilterPoint2D operator*=(const FilterSize& value);
	FilterPoint2D operator*=(int value);

	FilterPoint2D operator/(const FilterPoint2D& value) const;
	FilterPoint2D operator/(const FilterSize& value) const;
	FilterPoint2D operator/(int value) const;

	FilterPoint2D operator/=(const FilterPoint2D& value);
	FilterPoint2D operator/=(const FilterSize& value);
	FilterPoint2D operator/=(int value);

	FilterROI operator+(const FilterROI& value) const;
	FilterROI operator-(const FilterROI& value) const;
	FilterROI operator*(const FilterROI& value) const;
	FilterROI operator/(const FilterROI& value) const;
};

FilterPoint2D operator+(int lhs, const FilterPoint2D& rhs);
FilterPoint2D operator-(int lhs, const FilterPoint2D& rhs);
FilterPoint2D operator*(int lhs, const FilterPoint2D& rhs);
FilterPoint2D operator/(int lhs, const FilterPoint2D& rhs);

#endif