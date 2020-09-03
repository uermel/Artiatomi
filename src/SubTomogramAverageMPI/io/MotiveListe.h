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


#ifndef MOTIVELIST_H
#define MOTIVELIST_H

#include "EMFile.h"
#include "../config/Config.h"

struct motive
{
	float ccCoeff;
	float xCoord;
	float yCoord;
	float partNr;
	float tomoNr;
	float partNrInTomo;
	float wedgeIdx;
	float x_Coord;
	float y_Coord;
	float z_Coord;
	float x_Shift;
	float y_Shift;
	float z_Shift;
	float x_ShiftBefore;
	float y_ShiftBefore;
	float z_ShiftBefore;
	float phi;
	float psi;
	float theta;
	float classNo;

	motive();

	string GetIndexCoding(Configuration::NamingConvention nc);
};

class MotiveList : public EMFile
{
public:
	MotiveList(string filename);

	motive GetAt(int index);

	void SetAt(int index, motive& m);

    void getWedgeIndeces(std::vector<int> &unique, int* &correspond, int &count);

    int GetParticleCount();

};
#endif //MOTIVELIST_H