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


#include "PointF.h"


PointF::PointF() :
	x(0), y(0)
{
}

PointF::PointF(float aX, float aY) :
	x(aX), y(aY)
{
}

PointF::PointF(const PointF & aPoint) :
	x(aPoint.x), y(aPoint.y)
{
}

PointF & PointF::operator=(const PointF& arg)
{
	this->x = arg.x;
	this->y = arg.y;
	return *this;
}

bool PointF::operator==(const PointF & arg)
{
	return x == arg.x && y == arg.y;
}

PointF PointF::operator+(const PointF & value) const
{
	return PointF(x + value.x, y + value.y);
}

PointF PointF::operator+(float value) const
{
	return PointF(x + value, y + value);
}

PointF PointF::operator+=(const PointF & value)
{
	x += value.x;
	y += value.y;
	return *this;
}

PointF PointF::operator+=(float value)
{
	x += value;
	y += value;
	return *this;
}

PointF PointF::operator-(const PointF & value) const
{
	return PointF(x - value.x, y - value.y);
}

PointF PointF::operator-(float value) const
{
	return PointF(x - value, y - value);
}

PointF PointF::operator-=(const PointF & value)
{
	x -= value.x;
	y -= value.y;
	return *this;
}

PointF PointF::operator-=(float value)
{
	x -= value;
	y -= value;
	return *this;
}

PointF PointF::operator*(const PointF & value) const
{
	return PointF(x * value.x, y * value.y);
}

PointF PointF::operator*(float value) const
{
	return PointF(x * value, y * value);
}

PointF PointF::operator*=(const PointF & value)
{
	x *= value.x;
	y *= value.y;
	return *this;
}

PointF  PointF::operator*=(float value)
{
	x *= value;
	y *= value;
	return *this;
}

PointF PointF::operator/(const PointF & value) const
{
	return PointF(x / value.x, y / value.y);
}

PointF PointF::operator/(float value) const
{
	return PointF(x / value, y / value);
}

PointF PointF::operator/=(const PointF & value)
{
	x /= value.x;
	y /= value.y;
	return *this;
}

PointF PointF::operator/=(float value)
{
	x /= value;
	y /= value;
	return *this;
}

PointF operator+(float lhs, const PointF & rhs)
{
	return PointF(lhs + rhs.x, lhs + rhs.y);
}

PointF operator-(float lhs, const PointF & rhs)
{
	return PointF(lhs - rhs.x, lhs - rhs.y);
}

PointF operator*(float lhs, const PointF & rhs)
{
	return PointF(lhs * rhs.x, lhs * rhs.y);
}

PointF operator/(float lhs, const PointF & rhs)
{
	return PointF(lhs / rhs.x, lhs / rhs.y);
}


