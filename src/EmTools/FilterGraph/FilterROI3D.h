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


#ifndef FILTERROI3D_H
#define FILTERROI3D_H


//Forward declaration;
class FilterPoint3D;
class FilterSize3D;

class FilterROI3D
{
public:
	int x;
	int y;
	int z;
	int width;
	int height;
	int depth;

	FilterROI3D();
	FilterROI3D(int aX, int aY, int aZ, int aWidth, int aHeight, int aDepth);
	FilterROI3D(int aX, int aY, int aZ, const FilterSize3D& aFilterSize);
	FilterROI3D(const FilterPoint3D& aPoint, int aWidth, int aHeight, int aDepth);
	FilterROI3D(const FilterPoint3D& aPoint, const FilterSize3D& aFilterSize);

	FilterROI3D(const FilterROI3D& aROI);

	//! Gets the x-coordinate of the left edge.
	int Left() const;
	//! Gets the x-coordinate that is the sum of x and width values - 1.
	int Right() const;
	//! Gets the y-coordinate of the top edge.
	int Top() const;
	//! Gets the y-coordinate that is the sum of the y and height values - 1.
	int Bottom() const;
	//! Gets the z-coordinate of the top edge.
	int Front() const;
	//! Gets the z-coordinate that is the sum of the z and depth values - 1.
	int Back() const;

	//! Returns the x, y and z component as FilterPoint3D
	FilterPoint3D Location() const;

	//! Returns the width, height, depth component as FilterSize3D
	FilterSize3D Size() const;

	//! Tests whether all numeric properties of this Box have values of zero.
	bool IsEmpty() const;

	//! Determines if the specified point is contained within this Box structure.
	bool Contains(int aX, int aY, int aZ) const;
	//! Determines if the specified point is contained within this Box structure.
	bool Contains(const FilterPoint3D& aPoint) const;
	//! Determines if the rectangular region represented by aFilterRoi is entirely contained within this FilterROI3D.
	bool Contains(const FilterROI3D& aFilterRoi) const;

	//! Enlarges this Box by the specified amount.
	void Inflate(int aVal);
	//! Enlarges this Box by the specified amount.
	void Inflate(int aValX, int aValY, int aValZ);
	//! Enlarges this Box by the specified amount.
	void Inflate(const FilterSize3D& aVal);
	//! Reduces this Box by the specified amount.
	void Deflate(int aVal);
	//! Reduces this Box by the specified amount.
	void Deflate(int aValX, int aValY, int aValZ);
	//! Reduces this Box by the specified amount.
	void Deflate(const FilterSize3D& aVal);

	//! Returns the intersection of this ROI with and the specified Box.
	FilterROI3D Intersect(const FilterROI3D& aROI) const;
	//! Determines if this ROI intersects with aROI.
	bool IntersectsWith(const FilterROI3D& aROI) const;


	FilterROI3D& operator=(const FilterROI3D& arg);
	bool operator==(const FilterROI3D& arg);

	FilterROI3D operator+(const FilterPoint3D& value) const;
	FilterROI3D operator+(const FilterSize3D& value) const;
	FilterROI3D operator+(const FilterROI3D& value) const;
	FilterROI3D operator+(int value) const;

	FilterROI3D& operator+=(const FilterPoint3D& value);
	FilterROI3D& operator+=(const FilterSize3D& value);
	FilterROI3D& operator+=(const FilterROI3D& value);
	FilterROI3D& operator+=(int value);

	FilterROI3D operator-(const FilterPoint3D& value) const;
	FilterROI3D operator-(const FilterSize3D& value) const;
	FilterROI3D operator-(const FilterROI3D& value) const;
	FilterROI3D operator-(int value) const;

	FilterROI3D& operator-=(const FilterPoint3D& value);
	FilterROI3D& operator-=(const FilterSize3D& value);
	FilterROI3D& operator-=(const FilterROI3D& value);
	FilterROI3D& operator-=(int value);

	FilterROI3D operator*(const FilterPoint3D& value) const;
	FilterROI3D operator*(const FilterSize3D& value) const;
	FilterROI3D operator*(const FilterROI3D& value) const;
	FilterROI3D operator*(int value) const;

	FilterROI3D& operator*=(const FilterPoint3D& value);
	FilterROI3D& operator*=(const FilterSize3D& value);
	FilterROI3D& operator*=(const FilterROI3D& value);
	FilterROI3D& operator*=(int value);

	FilterROI3D operator/(const FilterPoint3D& value) const;
	FilterROI3D operator/(const FilterSize3D& value) const;
	FilterROI3D operator/(const FilterROI3D& value) const;
	FilterROI3D operator/(int value) const;

	FilterROI3D& operator/=(const FilterPoint3D& value);
	FilterROI3D& operator/=(const FilterSize3D& value);
	FilterROI3D& operator/=(const FilterROI3D& value);
	FilterROI3D& operator/=(const int value);
};

FilterROI3D operator+(int lhs, const FilterROI3D& rhs);
FilterROI3D operator-(int lhs, const FilterROI3D& rhs);
FilterROI3D operator*(int lhs, const FilterROI3D& rhs);
FilterROI3D operator/(int lhs, const FilterROI3D& rhs);

#endif