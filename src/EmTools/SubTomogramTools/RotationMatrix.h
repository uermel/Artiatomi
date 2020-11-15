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


#ifndef ROTATIONMATRIX_H
#define ROTATIONMATRIX_H

#include "../Basics/Default.h"

class RotationMatrix
{
private:
	float _data[3][3];

public:
	RotationMatrix();
	RotationMatrix(float aMatrix[3][3]);
	RotationMatrix(float phi, float psi, float theta);
	RotationMatrix(RotationMatrix& rotMat);

	void GetEulerAngles(float& phi, float& psi, float& theta);
	void GetData(float data[3][3]);
	float& operator()(int i, int j);
	float operator()(int i, int j) const;
	RotationMatrix operator*(const RotationMatrix& other);

};

std::ostream& operator<< (std::ostream& stream, const RotationMatrix& matrix);

#endif