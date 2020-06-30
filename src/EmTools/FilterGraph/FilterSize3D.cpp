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


#include "FilterSize3D.h"

FilterSize3D::FilterSize3D() :
	width(0), height(0)
{
}

FilterSize3D::FilterSize3D(int aWidth, int aHeight, int aDepth) :
	width(aWidth), height(aHeight), depth(aDepth)
{
}

FilterSize3D::FilterSize3D(const FilterSize3D & aPoint) :
	width(aPoint.width), height(aPoint.height), depth(aPoint.depth)
{
}

FilterSize3D & FilterSize3D::operator=(const FilterSize3D& arg)
{
	this->width = arg.width;
	this->height = arg.height;
	this->depth = arg.depth;
	return *this;
}

bool FilterSize3D::operator==(const FilterSize3D & arg)
{
	return width == arg.width && height == arg.height && depth == arg.depth;
}

FilterSize3D FilterSize3D::operator+(int value) const
{
	return FilterSize3D(width + value, height + value, depth + value);
}

FilterSize3D FilterSize3D::operator+(const FilterSize3D & value) const
{
	return FilterSize3D(width + value.width, height + value.height, depth + value.depth);
}

FilterSize3D & FilterSize3D::operator+=(int value)
{
	width += value;
	height += value;
	depth += value;
	return *this;
}

FilterSize3D & FilterSize3D::operator+=(const FilterSize3D & value)
{
	width += value.width;
	height += value.height;
	depth += value.depth;
	return *this;
}

FilterSize3D FilterSize3D::operator-(int value) const
{
	return FilterSize3D(width - value, height - value, depth - value);
}

FilterSize3D FilterSize3D::operator-(const FilterSize3D & value) const
{
	return FilterSize3D(width - value.width, height - value.height, depth - value.depth);
}

FilterSize3D & FilterSize3D::operator-=(int value)
{
	width -= value;
	height -= value;
	depth -= value;
	return *this;
}

FilterSize3D & FilterSize3D::operator-=(const FilterSize3D & value)
{
	width -= value.width;
	height -= value.height;
	depth -= value.height;
	return *this;
}

FilterSize3D FilterSize3D::operator*(int value) const
{
	return FilterSize3D(width * value, height * value, depth * value);
}

FilterSize3D FilterSize3D::operator*(const FilterSize3D & value) const
{
	return FilterSize3D(width * value.width, height * value.height, depth * value.depth);
}

FilterSize3D & FilterSize3D::operator*=(int value)
{
	width *= value;
	height *= value;
	depth *= value;
	return *this;
}

FilterSize3D & FilterSize3D::operator*=(const FilterSize3D & value)
{
	width *= value.width;
	height *= value.height;
	depth *= value.depth;
	return *this;
}

FilterSize3D FilterSize3D::operator/(int value) const
{
	return FilterSize3D(width / value, height / value, depth / value);
}

FilterSize3D FilterSize3D::operator/(const FilterSize3D & value) const
{
	return FilterSize3D(width / value.width, height / value.height, depth / value.depth);
}

FilterSize3D & FilterSize3D::operator/=(int value)
{
	width /= value;
	height /= value;
	depth /= value;
	return *this;
}

FilterSize3D & FilterSize3D::operator/=(const FilterSize3D & value)
{
	width /= value.width;
	height /= value.height;
	depth /= value.depth;
	return *this;
}

FilterROI3D FilterSize3D::operator+(const FilterROI3D & value) const
{
	return FilterROI3D(value.x, value.y, value.z, width + value.width, height + value.height, depth + value.depth);
}

FilterROI3D FilterSize3D::operator-(const FilterROI3D & value) const
{
	return FilterROI3D(value.x, value.y, value.z, width - value.width, height - value.height, depth - value.depth);
}

FilterROI3D FilterSize3D::operator*(const FilterROI3D & value) const
{
	return FilterROI3D(value.x, value.y, value.z, width * value.width, height * value.height, depth * value.depth);
}

FilterROI3D FilterSize3D::operator/(const FilterROI3D & value) const
{
	return FilterROI3D(value.x, value.y, value.z, width / value.width, height / value.height, depth / value.depth);
}

FilterSize3D operator+(int lhs, const FilterSize3D & rhs)
{
	return FilterSize3D(lhs + rhs.width, lhs + rhs.height, lhs + rhs.depth);
}

FilterSize3D operator-(int lhs, const FilterSize3D & rhs)
{
	return FilterSize3D(lhs - rhs.width, lhs - rhs.height, lhs - rhs.depth);
}

FilterSize3D operator*(int lhs, const FilterSize3D & rhs)
{
	return FilterSize3D(lhs * rhs.width, lhs * rhs.height, lhs * rhs.depth);
}

FilterSize3D operator/(int lhs, const FilterSize3D & rhs)
{
	return FilterSize3D(lhs / rhs.width, lhs / rhs.height, lhs / rhs.depth);
}

