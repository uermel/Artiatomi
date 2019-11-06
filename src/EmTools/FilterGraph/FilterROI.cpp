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


#include "FilterROI.h"
#include "FilterSize.h"
#include "FilterPoint2D.h"

FilterROI::FilterROI() :
	x(0), y(0), width(0), height(0)
{
}

FilterROI::FilterROI(int aX, int aY, int aWidth, int aHeight):
	x(aX), y(aY), width(aWidth), height(aHeight)
{
}

FilterROI::FilterROI(int aX, int aY, const FilterSize & aFilterSize):
	x(aX), y(aY), width(aFilterSize.width), height(aFilterSize.height)
{
}

FilterROI::FilterROI(const FilterPoint2D & aPoint, int aWidth, int aHeight):
	x(aPoint.x), y(aPoint.y), width(aWidth), height(aHeight)
{
}

FilterROI::FilterROI(const FilterPoint2D & aPoint, const FilterSize & aFilterSize) :
	x(aPoint.x), y(aPoint.y), width(aFilterSize.width), height(aFilterSize.height)
{
}

FilterROI::FilterROI(const FilterROI & aROI) :
	x(aROI.x), y(aROI.y), width(aROI.width), height(aROI.height)
{
}

int FilterROI::Left() const
{
	return x;
}

int FilterROI::Right() const
{
	return x + width - 1;
}

int FilterROI::Top() const
{
	return y;
}

int FilterROI::Bottom() const
{
	return y + height - 1;
}

FilterPoint2D FilterROI::Location() const
{
	return FilterPoint2D(x, y);
}

FilterSize FilterROI::Size() const
{
	return FilterSize(width, height);
}

bool FilterROI::IsEmpty() const
{
	return x == 0 && y == 0 && width == 0 && height == 0;
}

bool FilterROI::Contains(int aX, int aY) const
{
	return (aX >= Left()) && (aX <= Right()) && (aY >= Top()) && (aY <= Bottom());
}

bool FilterROI::Contains(const FilterPoint2D & aPoint) const
{
	return Contains(aPoint.x, aPoint.y);
}

bool FilterROI::Contains(const FilterROI & aFilterRoi) const
{
	return Contains(aFilterRoi.Location()) && Contains(aFilterRoi.Location() + aFilterRoi.Size() - 1);
}

void FilterROI::Inflate(int aVal)
{
	Inflate(aVal, aVal);
}

void FilterROI::Inflate(int aValX, int aValY)
{
	x -= aValX;
	y -= aValY;
	width += 2 * aValX;
	height += 2 * aValY;
}

void FilterROI::Inflate(const FilterSize & aVal)
{
	Inflate(aVal.width, aVal.height);
}

void FilterROI::Deflate(int aVal)
{
	Deflate(aVal, aVal);
}

void FilterROI::Deflate(int aValX, int aValY)
{
	x += aValX;
	y += aValY;
	width -= 2 * aValX;
	height -= 2 * aValY;
}

void FilterROI::Deflate(const FilterSize & aVal)
{
	Deflate(aVal.width, aVal.height);
}

FilterROI FilterROI::Intersect(const FilterROI & aROI) const
{
	int iX = Left();
	if (iX < aROI.Left())
	{
		iX = aROI.Left();
	}

	int iY = Top();
	if (iY < aROI.Top())
	{
		iY = aROI.Top();
	}

	int iX2 = Right();
	if (iX2 > aROI.Right())
	{
		iX2 = aROI.Right();
	}

	int iY2 = Bottom();
	if (iY2 > aROI.Bottom())
	{
		iY2 = aROI.Bottom();
	}

	int iWidth = iX2 - iX + 1;
	int iHeight = iY2 - iY + 1;
	if (iWidth <= 0 || iHeight <= 0)
	{
		iX = 0;
		iY = 0;
		iWidth = 0;
		iHeight = 0;
	}

	return FilterROI(iX, iY, iWidth, iHeight);
}

bool FilterROI::IntersectsWith(const FilterROI & aROI) const
{
	int iX = Left();
	if (iX < aROI.Left())
	{
		iX = aROI.Left();
	}

	int iY = Top();
	if (iY < aROI.Top())
	{
		iY = aROI.Top();
	}

	int iX2 = Right();
	if (iX2 > aROI.Right())
	{
		iX2 = aROI.Right();
	}

	int iY2 = Bottom();
	if (iY2 > aROI.Bottom())
	{
		iY2 = aROI.Bottom();
	}

	int iWidth = iX2 - iX + 1;
	int iHeight = iY2 - iY + 1;
	if (iWidth <= 0 || iHeight <= 0)
	{
		return false;
	}
	return true;
}


FilterROI & FilterROI::operator=(const FilterROI& arg)
{
	this->x = arg.x;
	this->y = arg.y;
	this->width = arg.width;
	this->height = arg.height;
	return *this;
}

bool FilterROI::operator==(const FilterROI & arg)
{
	return x == arg.x && y == arg.y && width == arg.width && height == arg.height;
}

FilterROI FilterROI::operator+(const FilterPoint2D & value) const
{
	return FilterROI(x + value.x, y + value.y, width, height);
}

FilterROI FilterROI::operator+(const FilterSize & value) const
{
	return FilterROI(x, y, width + value.width, height + value.height);
}

FilterROI FilterROI::operator+(const FilterROI & value) const
{
	return FilterROI(x + value.x, y + value.y, width + value.width, height + value.height);
}

FilterROI FilterROI::operator+(int value) const
{
	return FilterROI(x + value, y + value, width + value, height + value);
}

FilterROI & FilterROI::operator+=(const FilterPoint2D & value)
{
	x += value.x;
	y += value.y;
	return *this;
}

FilterROI & FilterROI::operator+=(const FilterSize & value)
{
	width += value.width;
	height += value.height;
	return *this;
}

FilterROI & FilterROI::operator+=(const FilterROI & value)
{
	x += value.x;
	y += value.y;
	width += value.width;
	height += value.height;
	return *this;
}

FilterROI & FilterROI::operator+=(int value)
{
	x += value;
	y += value;
	width += value;
	height += value;
	return *this;
}

FilterROI FilterROI::operator-(const FilterPoint2D & value) const
{
	return FilterROI(x - value.x, y - value.y, width, height);
}

FilterROI FilterROI::operator-(const FilterSize & value) const
{
	return FilterROI(x, y, width - value.width, height - value.height);
}

FilterROI FilterROI::operator-(const FilterROI & value) const
{
	return FilterROI(x - value.x, y - value.y, width - value.width, height - value.height);
}

FilterROI FilterROI::operator-(int value) const
{
	return FilterROI(x - value, y - value, width - value, height - value);
}

FilterROI & FilterROI::operator-=(const FilterPoint2D & value)
{
	x -= value.x;
	y -= value.y;
	return *this;
}

FilterROI & FilterROI::operator-=(const FilterSize & value)
{
	width -= value.width;
	height -= value.height;
	return *this;
}

FilterROI & FilterROI::operator-=(const FilterROI & value)
{
	x -= value.x;
	y -= value.y;
	width -= value.width;
	height -= value.height;
	return *this;
}

FilterROI & FilterROI::operator-=(int value)
{
	x -= value;
	y -= value;
	width -= value;
	height -= value;
	return *this;
}

FilterROI FilterROI::operator*(const FilterPoint2D & value) const
{
	return FilterROI(x * value.x, y * value.y, width, height);
}

FilterROI FilterROI::operator*(const FilterSize & value) const
{
	return FilterROI(x, y, width * value.width, height * value.height);
}

FilterROI FilterROI::operator*(const FilterROI & value) const
{
	return FilterROI(x * value.x, y * value.y, width * value.width, height * value.height);
}

FilterROI FilterROI::operator*(int value) const
{
	return FilterROI(x * value, y * value, width * value, height * value);
}

FilterROI & FilterROI::operator*=(const FilterPoint2D & value)
{
	x *= value.x;
	y *= value.y;
	return *this;
}

FilterROI & FilterROI::operator*=(const FilterSize & value)
{
	width *= value.width;
	height *= value.height;
	return *this;
}

FilterROI & FilterROI::operator*=(const FilterROI & value)
{
	x *= value.x;
	y *= value.y;
	width *= value.width;
	height *= value.height;
	return *this;
}

FilterROI & FilterROI::operator*=(int value)
{
	x *= value;
	y *= value;
	width *= value;
	height *= value;
	return *this;
}

FilterROI FilterROI::operator/(const FilterPoint2D & value) const
{
	return FilterROI(x / value.x, y / value.y, width, height);
}

FilterROI FilterROI::operator/(const FilterSize & value) const
{
	return FilterROI(x, y, width / value.width, height / value.height);
}

FilterROI FilterROI::operator/(const FilterROI & value) const
{
	return FilterROI(x / value.x, y / value.y, width / value.width, height / value.height);
}

FilterROI FilterROI::operator/(int value) const
{
	return FilterROI(x / value, y / value, width / value, height / value);
}

FilterROI & FilterROI::operator/=(const FilterPoint2D & value)
{
	x /= value.x;
	y /= value.y;
	return *this;
}

FilterROI & FilterROI::operator/=(const FilterSize & value)
{
	width /= value.width;
	height /= value.height;
	return *this;
}

FilterROI & FilterROI::operator/=(const FilterROI & value)
{
	x /= value.x;
	y /= value.y;
	width /= value.width;
	height /= value.height;
	return *this;
}

FilterROI & FilterROI::operator/=(int value)
{
	x /= value;
	y /= value;
	width /= value;
	height /= value;
	return *this;
}

FilterROI operator+(int lhs, const FilterROI & rhs)
{
	return FilterROI(lhs + rhs.x, lhs + rhs.y, lhs + rhs.width, lhs + rhs.height);
}

FilterROI operator-(int lhs, const FilterROI & rhs)
{
	return FilterROI(lhs - rhs.x, lhs - rhs.y, lhs - rhs.width, lhs - rhs.height);
}

FilterROI operator*(int lhs, const FilterROI & rhs)
{
	return FilterROI(lhs * rhs.x, lhs * rhs.y, lhs * rhs.width, lhs * rhs.height);
}

FilterROI operator/(int lhs, const FilterROI & rhs)
{
	return FilterROI(lhs / rhs.x, lhs / rhs.y, lhs / rhs.width, lhs / rhs.height);
}