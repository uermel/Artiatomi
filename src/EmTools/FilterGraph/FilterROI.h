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


#ifndef FILTERROI_H
#define FILTERROI_H


//Forward declaration;
class FilterPoint2D;
class FilterSize;

class FilterROI
{
public:
	int x;
	int y;
	int width;
	int height;

	FilterROI();
	FilterROI(int aX, int aY, int aWidth, int aHeight);
	FilterROI(int aX, int aY, const FilterSize& aFilterSize);
	FilterROI(const FilterPoint2D& aPoint, int aWidth, int aHeight);
	FilterROI(const FilterPoint2D& aPoint, const FilterSize& aFilterSize);

	FilterROI(const FilterROI& aROI);

	//! Gets the x-coordinate of the left edge.
	int Left() const;
	//! Gets the x-coordinate that is the sum of x and width values - 1.
	int Right() const;
	//! Gets the y-coordinate of the top edge.
	int Top() const;
	//! Gets the y-coordinate that is the sum of the y and height values - 1.
	int Bottom() const;

	//! Returns the x and y component as FilterPoint2D
	FilterPoint2D Location() const;

	//! Returns the width and height component as FilterSize
	FilterSize Size() const;

	//! Tests whether all numeric properties of this Rectangle have values of zero.
	bool IsEmpty() const;

	//! Determines if the specified point is contained within this Rectangle structure.
	bool Contains(int aX, int aY) const;
	//! Determines if the specified point is contained within this Rectangle structure.
	bool Contains(const FilterPoint2D& aPoint) const;
	//! Determines if the rectangular region represented by aFilterRoi is entirely contained within this FilterROI.
	bool Contains(const FilterROI& aFilterRoi) const;

	//! Enlarges this Rectangle by the specified amount.
	void Inflate(int aVal);
	//! Enlarges this Rectangle by the specified amount.
	void Inflate(int aValX, int aValY);
	//! Enlarges this Rectangle by the specified amount.
	void Inflate(const FilterSize& aVal);
	//! Reduces this Rectangle by the specified amount.
	void Deflate(int aVal);
	//! Reduces this Rectangle by the specified amount.
	void Deflate(int aValX, int aValY);
	//! Reduces this Rectangle by the specified amount.
	void Deflate(const FilterSize& aVal);

	//! Returns the intersection of this ROI with and the specified Rectangle.
	FilterROI Intersect(const FilterROI& aROI) const;
	//! Determines if this ROI intersects with aROI.
	bool IntersectsWith(const FilterROI& aROI) const;


	FilterROI& operator=(const FilterROI& arg);
	bool operator==(const FilterROI& arg);

	FilterROI operator+(const FilterPoint2D& value) const;
	FilterROI operator+(const FilterSize& value) const;
	FilterROI operator+(const FilterROI& value) const;
	FilterROI operator+(int value) const;

	FilterROI& operator+=(const FilterPoint2D& value);
	FilterROI& operator+=(const FilterSize& value);
	FilterROI& operator+=(const FilterROI& value);
	FilterROI& operator+=(int value);

	FilterROI operator-(const FilterPoint2D& value) const;
	FilterROI operator-(const FilterSize& value) const;
	FilterROI operator-(const FilterROI& value) const;
	FilterROI operator-(int value) const;

	FilterROI& operator-=(const FilterPoint2D& value);
	FilterROI& operator-=(const FilterSize& value);
	FilterROI& operator-=(const FilterROI& value);
	FilterROI& operator-=(int value);

	FilterROI operator*(const FilterPoint2D& value) const;
	FilterROI operator*(const FilterSize& value) const;
	FilterROI operator*(const FilterROI& value) const;
	FilterROI operator*(int value) const;

	FilterROI& operator*=(const FilterPoint2D& value);
	FilterROI& operator*=(const FilterSize& value);
	FilterROI& operator*=(const FilterROI& value);
	FilterROI& operator*=(int value);

	FilterROI operator/(const FilterPoint2D& value) const;
	FilterROI operator/(const FilterSize& value) const;
	FilterROI operator/(const FilterROI& value) const;
	FilterROI operator/(int value) const;

	FilterROI& operator/=(const FilterPoint2D& value);
	FilterROI& operator/=(const FilterSize& value);
	FilterROI& operator/=(const FilterROI& value);
	FilterROI& operator/=(const int value);
};

FilterROI operator+(int lhs, const FilterROI& rhs);
FilterROI operator-(int lhs, const FilterROI& rhs);
FilterROI operator*(int lhs, const FilterROI& rhs);
FilterROI operator/(int lhs, const FilterROI& rhs);

#endif