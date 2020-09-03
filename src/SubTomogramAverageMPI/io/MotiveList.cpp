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


#include "MotiveListe.h"
#include <sstream>

motive::motive() :
	ccCoeff(0),
	xCoord(0),
	yCoord(0),
	partNr(0),
	tomoNr(0),
	partNrInTomo(0),
	wedgeIdx(0),
	x_Coord(0),
	y_Coord(0),
	z_Coord(0),
	x_Shift(0),
	y_Shift(0),
	z_Shift(0),
	x_ShiftBefore(0),
	y_ShiftBefore(0),
	z_ShiftBefore(0),
	phi(0),
	psi(0),
	theta(0),
	classNo(0)
{

}

MotiveList::MotiveList(string filename)
	: EMFile(filename)
{
	EMFile::OpenAndRead();
	EMFile::ReadHeaderInfo();
}

motive MotiveList::GetAt(int index)
{
	motive m;
	memcpy(&m, _data + index * sizeof(m), sizeof(m));
	return m;
}

void MotiveList::SetAt(int index, motive& m)
{
	memcpy(_data + index * sizeof(m), &m, sizeof(m));
}

void MotiveList::getWedgeIndeces(std::vector<int> &unique, int* &correspond, int &count)
{
    // The unique IDs
    unique.clear();
    // The number of unique indeces
    count = 0;

    for (int i = 0; i < _fileHeader.DimY; i++)
    {
        // Get the current item
        motive m = GetAt(i);
        bool found = false;

        // Search if this ID is already present
        for (int j = 0; j < count; j++)
        {
            if (unique[j] == (int)m.wedgeIdx)
            {
                found = true;
                correspond[i] = j;
            }
        }

        // If not present append to the end
        if (!found)
        {
            unique.push_back((int)m.wedgeIdx);
            correspond[i] = count;
            count++;
        }
    }
}

int MotiveList::GetParticleCount()
{
    return _fileHeader.DimY;
}


string motive::GetIndexCoding(Configuration::NamingConvention nc)
{
	stringstream ss;
	switch (nc)
	{
	case Configuration::NC_ParticleOnly:
		ss << partNr;
		break;
	case Configuration::NC_TomogramParticle:
		ss << tomoNr << "_" << partNrInTomo;
		break;
	default:
		break;
	}
	return ss.str();
}