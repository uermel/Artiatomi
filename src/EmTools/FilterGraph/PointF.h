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


#ifndef POINTF_H
#define POINTF_H

class PointF
{
public:
	float x;
	float y;

	PointF();
	PointF(float aX, float aY);
	PointF(const PointF& aPoint);

	PointF& operator=(const PointF& arg);
	bool operator==(const PointF& arg);

	PointF operator+(const PointF& value) const;
	PointF operator+(float value) const;

	PointF operator+=(const PointF& value);
	PointF operator+=(float value);

	PointF operator-(const PointF& value) const;
	PointF operator-(float value) const;

	PointF operator-=(const PointF& value);
	PointF operator-=(float value);

	PointF operator*(const PointF& value) const;
	PointF operator*(float value) const;

	PointF operator*=(const PointF& value);
	PointF operator*=(float value);

	PointF operator/(const PointF& value) const;
	PointF operator/(float value) const;

	PointF operator/=(const PointF& value);
	PointF operator/=(float value);
};

PointF operator+(float lhs, const PointF& rhs);
PointF operator-(float lhs, const PointF& rhs);
PointF operator*(float lhs, const PointF& rhs);
PointF operator/(float lhs, const PointF& rhs);

#endif