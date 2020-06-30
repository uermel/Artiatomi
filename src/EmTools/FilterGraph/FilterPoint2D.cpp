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


#include "FilterPoint2D.h"
#include "FilterSize.h"
#include "FilterROI.h"

FilterPoint2D::FilterPoint2D():
	x(0), y(0)
{
}

FilterPoint2D::FilterPoint2D(int aX, int aY):
	x(aX), y(aY)
{
}

FilterPoint2D::FilterPoint2D(const FilterPoint2D & aPoint) :
	x(aPoint.x), y(aPoint.y)
{
}

FilterPoint2D & FilterPoint2D::operator=(const FilterPoint2D& arg)
{
	this->x = arg.x;
	this->y = arg.y;
	return *this;
}

bool FilterPoint2D::operator==(const FilterPoint2D & arg)
{
	return x == arg.x && y == arg.y;
}

FilterPoint2D FilterPoint2D::operator+(const FilterPoint2D & value) const
{
	return FilterPoint2D(x + value.x, y + value.y);
}

FilterPoint2D FilterPoint2D::operator+(const FilterSize & value) const
{
	return FilterPoint2D(x + value.width, y + value.height);
}

FilterPoint2D FilterPoint2D::operator+(int value) const
{
	return FilterPoint2D(x + value, y + value);
}

FilterPoint2D FilterPoint2D::operator+=(const FilterPoint2D & value)
{
	x += value.x;
	y += value.y;
	return *this;
}

FilterPoint2D FilterPoint2D::operator+=(const FilterSize & value)
{
	x += value.width;
	y += value.height;
	return *this;
}

FilterPoint2D FilterPoint2D::operator+=(int value)
{
	x += value;
	y += value;
	return *this;
}

FilterPoint2D FilterPoint2D::operator-(const FilterPoint2D & value) const
{
	return FilterPoint2D(x - value.x, y - value.y);
}

FilterPoint2D FilterPoint2D::operator-(const FilterSize & value) const
{
	return FilterPoint2D(x - value.width, y - value.height);
}

FilterPoint2D FilterPoint2D::operator-(int value) const
{
	return FilterPoint2D(x - value, y - value);
}

FilterPoint2D FilterPoint2D::operator-=(const FilterPoint2D & value)
{
	x -= value.x;
	y -= value.y;
	return *this;
}

FilterPoint2D FilterPoint2D::operator-=(const FilterSize & value)
{
	x -= value.width;
	y -= value.height;
	return *this;
}

FilterPoint2D FilterPoint2D::operator-=(int value)
{
	x -= value;
	y -= value;
	return *this;
}

FilterPoint2D FilterPoint2D::operator*(const FilterPoint2D & value) const
{
	return FilterPoint2D(x * value.x, y * value.y);
}

FilterPoint2D FilterPoint2D::operator*(const FilterSize & value) const
{
	return FilterPoint2D(x * value.width, y * value.height);
}

FilterPoint2D FilterPoint2D::operator*(int value) const
{
	return FilterPoint2D(x * value, y * value);
}

FilterPoint2D FilterPoint2D::operator*=(const FilterPoint2D & value)
{
	x *= value.x;
	y *= value.y;
	return *this;
}

FilterPoint2D  FilterPoint2D::operator*=(const FilterSize & value)
{
	x *= value.width;
	y *= value.height;
	return *this;
}

FilterPoint2D  FilterPoint2D::operator*=(int value)
{
	x *= value;
	y *= value;
	return *this;
}

FilterPoint2D FilterPoint2D::operator/(const FilterPoint2D & value) const
{
	return FilterPoint2D(x / value.x, y / value.y);
}

FilterPoint2D FilterPoint2D::operator/(const FilterSize & value) const
{
	return FilterPoint2D(x / value.width, y / value.height);
}

FilterPoint2D FilterPoint2D::operator/(int value) const
{
	return FilterPoint2D(x / value, y / value);
}

FilterPoint2D FilterPoint2D::operator/=(const FilterPoint2D & value)
{
	x /= value.x;
	y /= value.y;
	return *this;
}

FilterPoint2D FilterPoint2D::operator/=(const FilterSize & value)
{
	x /= value.width;
	y /= value.height;
	return *this;
}

FilterPoint2D FilterPoint2D::operator/=(int value)
{
	x /= value;
	y /= value;
	return *this;
}

FilterROI FilterPoint2D::operator+(const FilterROI & value) const
{
	return FilterROI(x + value.x, y + value.y, value.width, value.height);
}

FilterROI FilterPoint2D::operator-(const FilterROI & value) const
{
	return FilterROI(x - value.x, y - value.y, value.width, value.height);
}

FilterROI FilterPoint2D::operator*(const FilterROI & value) const
{
	return FilterROI(x * value.x, y * value.y, value.width, value.height);
}

FilterROI FilterPoint2D::operator/(const FilterROI & value) const
{
	return FilterROI(x / value.x, y / value.y, value.width, value.height);
}

FilterPoint2D operator+(int lhs, const FilterPoint2D & rhs)
{
	return FilterPoint2D(lhs + rhs.x, lhs + rhs.y);
}

FilterPoint2D operator-(int lhs, const FilterPoint2D & rhs)
{
	return FilterPoint2D(lhs - rhs.x, lhs - rhs.y);
}

FilterPoint2D operator*(int lhs, const FilterPoint2D & rhs)
{
	return FilterPoint2D(lhs * rhs.x, lhs * rhs.y);
}

FilterPoint2D operator/(int lhs, const FilterPoint2D & rhs)
{
	return FilterPoint2D(lhs / rhs.x, lhs / rhs.y);
}


