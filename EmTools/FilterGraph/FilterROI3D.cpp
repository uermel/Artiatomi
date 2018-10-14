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


#include "FilterROI3D.h"
#include "FilterSize3D.h"
#include "FilterPoint3D.h"

FilterROI3D::FilterROI3D() :
	x(0), y(0), width(0), height(0)
{
}

FilterROI3D::FilterROI3D(int aX, int aY, int aZ, int aWidth, int aHeight, int aDepth) :
	x(aX), y(aY), z(aZ), width(aWidth), height(aHeight), depth(aDepth)
{
}

FilterROI3D::FilterROI3D(int aX, int aY, int aZ, const FilterSize3D & aFilterSize) :
	x(aX), y(aY), z(aZ), width(aFilterSize.width), height(aFilterSize.height), depth(aFilterSize.depth)
{
}

FilterROI3D::FilterROI3D(const FilterPoint3D & aPoint, int aWidth, int aHeight, int aDepth) :
	x(aPoint.x), y(aPoint.y), z(aPoint.z), width(aWidth), height(aHeight), depth(aDepth)
{
}

FilterROI3D::FilterROI3D(const FilterPoint3D & aPoint, const FilterSize3D & aFilterSize) :
	x(aPoint.x), y(aPoint.y), z(aPoint.z), width(aFilterSize.width), height(aFilterSize.height), depth(aFilterSize.depth)
{
}

FilterROI3D::FilterROI3D(const FilterROI3D & aROI) :
	x(aROI.x), y(aROI.y), z(aROI.z), width(aROI.width), height(aROI.height), depth(aROI.depth)
{
}

int FilterROI3D::Left() const
{
	return x;
}

int FilterROI3D::Right() const
{
	return x + width - 1;
}

int FilterROI3D::Top() const
{
	return y;
}

int FilterROI3D::Bottom() const
{
	return y + height - 1;
}

int FilterROI3D::Front() const
{
	return z;
}

int FilterROI3D::Back() const
{
	return z + depth - 1;
}

FilterPoint3D FilterROI3D::Location() const
{
	return FilterPoint3D(x, y, z);
}

FilterSize3D FilterROI3D::Size() const
{
	return FilterSize3D(width, height, depth);
}

bool FilterROI3D::IsEmpty() const
{
	return x == 0 && y == 0 && z == 0 && width == 0 && height == 0 && depth == 0;
}

bool FilterROI3D::Contains(int aX, int aY, int aZ) const
{
	return (aX >= Left()) && (aX <= Right()) && (aY >= Top()) && (aY <= Bottom()) && (aZ >= Front()) && (aX <= Back());
}

bool FilterROI3D::Contains(const FilterPoint3D & aPoint) const
{
	return Contains(aPoint.x, aPoint.y, aPoint.z);
}

bool FilterROI3D::Contains(const FilterROI3D & aFilterRoi) const
{
	return Contains(aFilterRoi.Location()) && Contains(aFilterRoi.Location() + aFilterRoi.Size());
}

void FilterROI3D::Inflate(int aVal)
{
	Inflate(aVal, aVal, aVal);
}

void FilterROI3D::Inflate(int aValX, int aValY, int aValZ)
{
	x -= aValX;
	y -= aValY;
	z -= aValZ;
	width += 2 * aValX;
	height += 2 * aValY;
	depth += 2 * aValZ;
}

void FilterROI3D::Inflate(const FilterSize3D & aVal)
{
	Inflate(aVal.width, aVal.height, aVal.depth);
}

void FilterROI3D::Deflate(int aVal)
{
	Deflate(aVal, aVal, aVal);
}

void FilterROI3D::Deflate(int aValX, int aValY, int aValZ)
{
	x += aValX;
	y += aValY;
	z += aValZ;
	width -= 2 * aValX;
	height -= 2 * aValY;
	depth -= 2 * aValZ;
}

void FilterROI3D::Deflate(const FilterSize3D & aVal)
{
	Deflate(aVal.width, aVal.height, aVal.depth);
}

FilterROI3D FilterROI3D::Intersect(const FilterROI3D & aROI) const
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

	int iZ = Front();
	if (iZ < aROI.Front())
	{
		iZ = aROI.Front();
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

	int iZ2 = Back();
	if (iZ2 < aROI.Back())
	{
		iZ2 = aROI.Back();
	}

	int iWidth = iX2 - iX + 1;
	int iHeight = iY2 - iY + 1;
	int iDepth = iZ2 - iZ + 1;
	if (iWidth <= 0 || iHeight <= 0 || iDepth <= 0)
	{
		iX = 0;
		iY = 0;
		iZ = 0;
		iWidth = 0;
		iHeight = 0;
		iDepth = 0;
	}

	return FilterROI3D(iX, iY, iZ, iWidth, iHeight, iDepth);
}

bool FilterROI3D::IntersectsWith(const FilterROI3D & aROI) const
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

	int iZ = Front();
	if (iZ < aROI.Front())
	{
		iZ = aROI.Front();
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

	int iZ2 = Back();
	if (iZ2 > aROI.Back())
	{
		iZ2 = aROI.Back();
	}

	int iWidth = iX2 - iX + 1;
	int iHeight = iY2 - iY + 1;
	int iDepth = iZ2 - iZ + 1;
	if (iWidth <= 0 || iHeight <= 0 || iDepth <= 0)
	{
		return false;
	}
	return true;
}


FilterROI3D & FilterROI3D::operator=(const FilterROI3D& arg)
{
	this->x = arg.x;
	this->y = arg.y;
	this->z = arg.z;
	this->width = arg.width;
	this->height = arg.height;
	this->depth = arg.depth;
	return *this;
}

bool FilterROI3D::operator==(const FilterROI3D & arg)
{
	return x == arg.x && y == arg.y && z == arg.z && width == arg.width && height == arg.height && depth == arg.depth;
}

FilterROI3D FilterROI3D::operator+(const FilterPoint3D & value) const
{
	return FilterROI3D(x + value.x, y + value.y, z + value.z, width, height, depth);
}

FilterROI3D FilterROI3D::operator+(const FilterSize3D & value) const
{
	return FilterROI3D(x, y, z, width + value.width, height + value.height, depth + value.depth);
}

FilterROI3D FilterROI3D::operator+(const FilterROI3D & value) const
{
	return FilterROI3D(x + value.x, y + value.y, z + value.z, width + value.width, height + value.height, depth + value.depth);
}

FilterROI3D FilterROI3D::operator+(int value) const
{
	return FilterROI3D(x + value, y + value, z + value, width + value, height + value, depth + value);
}

FilterROI3D & FilterROI3D::operator+=(const FilterPoint3D & value)
{
	x += value.x;
	y += value.y;
	z += value.z;
	return *this;
}

FilterROI3D & FilterROI3D::operator+=(const FilterSize3D & value)
{
	width += value.width;
	height += value.height;
	depth += value.depth;
	return *this;
}

FilterROI3D & FilterROI3D::operator+=(const FilterROI3D & value)
{
	x += value.x;
	y += value.y;
	z += value.z;
	width += value.width;
	height += value.height;
	depth += value.depth;
	return *this;
}

FilterROI3D & FilterROI3D::operator+=(int value)
{
	x += value;
	y += value;
	z += value;
	width += value;
	height += value;
	depth += value;
	return *this;
}

FilterROI3D FilterROI3D::operator-(const FilterPoint3D & value) const
{
	return FilterROI3D(x - value.x, y - value.y, z - value.z, width, height, depth);
}

FilterROI3D FilterROI3D::operator-(const FilterSize3D & value) const
{
	return FilterROI3D(x, y, z, width - value.width, height - value.height, depth - value.depth);
}

FilterROI3D FilterROI3D::operator-(const FilterROI3D & value) const
{
	return FilterROI3D(x - value.x, y - value.y, z - value.z, width - value.width, height - value.height, depth - value.depth);
}

FilterROI3D FilterROI3D::operator-(int value) const
{
	return FilterROI3D(x - value, y - value, z - value, width - value, height - value, depth - value);
}

FilterROI3D & FilterROI3D::operator-=(const FilterPoint3D & value)
{
	x -= value.x;
	y -= value.y;
	z -= value.z;
	return *this;
}

FilterROI3D & FilterROI3D::operator-=(const FilterSize3D & value)
{
	width -= value.width;
	height -= value.height;
	depth -= value.depth;
	return *this;
}

FilterROI3D & FilterROI3D::operator-=(const FilterROI3D & value)
{
	x -= value.x;
	y -= value.y;
	z -= value.z;
	width -= value.width;
	height -= value.height;
	depth -= value.depth;
	return *this;
}

FilterROI3D & FilterROI3D::operator-=(int value)
{
	x -= value;
	y -= value;
	z -= value;
	width -= value;
	height -= value;
	depth -= value;
	return *this;
}

FilterROI3D FilterROI3D::operator*(const FilterPoint3D & value) const
{
	return FilterROI3D(x * value.x, y * value.y, z * value.z, width, height, depth);
}

FilterROI3D FilterROI3D::operator*(const FilterSize3D & value) const
{
	return FilterROI3D(x, y, z, width * value.width, height * value.height, depth * value.depth);
}

FilterROI3D FilterROI3D::operator*(const FilterROI3D & value) const
{
	return FilterROI3D(x * value.x, y * value.y, z * value.z, width * value.width, height * value.height, depth * value.depth);
}

FilterROI3D FilterROI3D::operator*(int value) const
{
	return FilterROI3D(x * value, y * value, z * value, width * value, height * value, depth * value);
}

FilterROI3D & FilterROI3D::operator*=(const FilterPoint3D & value)
{
	x *= value.x;
	y *= value.y;
	z *= value.z;
	return *this;
}

FilterROI3D & FilterROI3D::operator*=(const FilterSize3D & value)
{
	width *= value.width;
	height *= value.height;
	depth *= value.depth;
	return *this;
}

FilterROI3D & FilterROI3D::operator*=(const FilterROI3D & value)
{
	x *= value.x;
	y *= value.y;
	z *= value.z;
	width *= value.width;
	height *= value.height;
	depth *= value.depth;
	return *this;
}

FilterROI3D & FilterROI3D::operator*=(int value)
{
	x *= value;
	y *= value;
	z *= value;
	width *= value;
	height *= value;
	depth *= value;
	return *this;
}

FilterROI3D FilterROI3D::operator/(const FilterPoint3D & value) const
{
	return FilterROI3D(x / value.x, y / value.y, z / value.z, width, height, depth);
}

FilterROI3D FilterROI3D::operator/(const FilterSize3D & value) const
{
	return FilterROI3D(x, y, z, width / value.width, height / value.height, depth / value.depth);
}

FilterROI3D FilterROI3D::operator/(const FilterROI3D & value) const
{
	return FilterROI3D(x / value.x, y / value.y, z / value.z, width / value.width, height / value.height, depth / value.depth);
}

FilterROI3D FilterROI3D::operator/(int value) const
{
	return FilterROI3D(x / value, y / value, z / value, width / value, height / value, depth / value);
}

FilterROI3D & FilterROI3D::operator/=(const FilterPoint3D & value)
{
	x /= value.x;
	y /= value.y;
	z /= value.z;
	return *this;
}

FilterROI3D & FilterROI3D::operator/=(const FilterSize3D & value)
{
	width /= value.width;
	height /= value.height;
	depth /= value.depth;
	return *this;
}

FilterROI3D & FilterROI3D::operator/=(const FilterROI3D & value)
{
	x /= value.x;
	y /= value.y;
	z /= value.z;
	width /= value.width;
	height /= value.height;
	depth /= value.depth;
	return *this;
}

FilterROI3D & FilterROI3D::operator/=(int value)
{
	x /= value;
	y /= value;
	z /= value;
	width /= value;
	height /= value;
	depth /= value;
	return *this;
}

FilterROI3D operator+(int lhs, const FilterROI3D & rhs)
{
	return FilterROI3D(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.width, lhs + rhs.height, lhs + rhs.depth);
}

FilterROI3D operator-(int lhs, const FilterROI3D & rhs)
{
	return FilterROI3D(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.width, lhs - rhs.height, lhs - rhs.depth);
}

FilterROI3D operator*(int lhs, const FilterROI3D & rhs)
{
	return FilterROI3D(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.width, lhs * rhs.height, lhs * rhs.depth);
}

FilterROI3D operator/(int lhs, const FilterROI3D & rhs)
{
	return FilterROI3D(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.width, lhs / rhs.height, lhs / rhs.depth);
}