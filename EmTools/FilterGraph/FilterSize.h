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


#ifndef FILTERSIZE_H
#define FILTERSIZE_H

#include "FilterROI.h"

class FilterSize
{
public:
	int width;
	int height;

	FilterSize();
	FilterSize(int aWidth, int aHeight);
	FilterSize(const FilterSize& aSize);


	FilterSize& operator=(const FilterSize& arg);
	bool operator==(const FilterSize& arg);

	FilterSize operator+(int value) const;
	FilterSize operator+(const FilterSize& value) const;

	FilterSize& operator+=(int value);
	FilterSize& operator+=(const FilterSize& value);

	FilterSize operator-(int value) const;
	FilterSize operator-(const FilterSize& value) const;

	FilterSize& operator-=(int value);
	FilterSize& operator-=(const FilterSize& value);

	FilterSize operator*(int value) const;
	FilterSize operator*(const FilterSize& value) const;

	FilterSize& operator*=(int value);
	FilterSize& operator*=(const FilterSize& value);

	FilterSize operator/(int value) const;
	FilterSize operator/(const FilterSize& value) const;

	FilterSize& operator/=(int value);
	FilterSize& operator/=(const FilterSize& value);

	FilterROI operator+(const FilterROI& value) const;
	FilterROI operator-(const FilterROI& value) const;
	FilterROI operator*(const FilterROI& value) const;
	FilterROI operator/(const FilterROI& value) const;
};


FilterSize operator+(int lhs, const FilterSize& rhs);
FilterSize operator-(int lhs, const FilterSize& rhs);
FilterSize operator*(int lhs, const FilterSize& rhs);
FilterSize operator/(int lhs, const FilterSize& rhs);

#endif