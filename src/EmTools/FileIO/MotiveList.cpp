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
#include <algorithm>
#include <math.h>

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

MotiveList::MotiveList(string filename, float aBinningFactorClick, float aBinningShift)
	: EmFile(filename), binningFactorClick(aBinningFactorClick), binningFactorShift(aBinningShift)
{
	EmFile::OpenAndRead();
	//EmFile::ReadHeaderInfo();

	for (int i = 0; i < _fileHeader.DimY; i++)
	{
		motive m = GetAt(i);

		bool exists = false;
		for (int j = 0; j < groupIndices.size(); j++)
		{
			if (groupIndices[j] == m.classNo)
			{
				exists = true;
			}
		}

		if (!exists)
		{
			groupIndices.push_back(m.classNo);
		}
	}
}

motive MotiveList::GetAt(int index)
{
	motive m;
	memcpy(&m, (char*)_data + index * sizeof(m), sizeof(m));
	m.xCoord *= binningFactorClick;
	m.yCoord *= binningFactorClick;
	m.x_Coord *= binningFactorClick;
	m.y_Coord *= binningFactorClick;
	m.z_Coord *= binningFactorClick;
	m.x_Shift *= binningFactorShift;
	m.y_Shift *= binningFactorShift;
	m.z_Shift *= binningFactorShift;

	return m;
}

void MotiveList::SetAt(int index, motive& m)
{
	m.xCoord /= binningFactorClick;
	m.yCoord /= binningFactorClick;
	m.x_Coord /= binningFactorClick;
	m.y_Coord /= binningFactorClick;
	m.z_Coord /= binningFactorClick;
	m.x_Shift /= binningFactorShift;
	m.y_Shift /= binningFactorShift;
	m.z_Shift /= binningFactorShift;
	memcpy((char*)_data + index * sizeof(m), &m, sizeof(m));
}

int MotiveList::GetParticleCount()
{
	return _fileHeader.DimY;
}

float MotiveList::GetDistance(int aIndex1, int aIndex2)
{
	motive mot1 = GetAt(aIndex1);
	motive mot2 = GetAt(aIndex2);

	return GetDistance(mot1, mot2);
}

float MotiveList::GetDistance(motive & mot1, motive & mot2)
{
	float x = mot1.x_Coord - mot2.x_Coord;
	float y = mot1.y_Coord - mot2.y_Coord;
	float z = mot1.z_Coord - mot2.z_Coord;

	float dist = sqrtf(x * x + y * y + z * z);

	return dist;
}

std::vector<motive> MotiveList::GetNeighbours(int index, GroupMode_enum aGroupMode, float aMaxDistance, int aGroupSize)
{
	switch (aGroupMode)
	{
	case GM_BYGROUP:
		return GetNeighbours(index);
	case GM_MAXDIST:
		return GetNeighbours(index, aMaxDistance);
	case GM_MAXCOUNT:
		return GetNeighbours(index, aGroupSize);
	}
	return std::vector<motive>();
}

std::vector<motive> MotiveList::GetNeighbours(int index, int count)
{
	std::vector<motive> ret;
	std::vector<std::pair<float, int> > dists;

	for (int i = 0; i < _fileHeader.DimY; i++)
	{
		float dist = GetDistance(i, index);
		dists.push_back(pair<float, int>(dist, i));
	}

	//actual index is first element as it has distance zero!
	sort(dists.begin(), dists.end());

	if (count > _fileHeader.DimY) //don't run over the end of the motive list if particle count is larger than the motive list.
	{
		count = _fileHeader.DimY;
	}

	for (int i = 0; i < count; i++)
	{
		ret.push_back(GetAt(dists[i].second));
	}

	return ret;
}

std::vector<motive> MotiveList::GetNeighbours(int index, float maxDist)
{
	std::vector<motive> ret;

	ret.push_back(GetAt(index)); //actual index is first element.
	for (int i = 0; i < _fileHeader.DimY; i++)
	{
		if (GetDistance(i, index) <= maxDist && i != index)
		{
			ret.push_back(GetAt(i));
		}
	}

	return ret;
}


std::vector<supportMotive> MotiveList::GetNeighbours(int index, float maxDist, vector<MotiveList>& supporters)
{
	std::vector<supportMotive> ret;

	motive curr = GetAt(index);
	for (int s = 0; s < supporters.size(); s++)
	{
		for (int i = 0; i < supporters[s]._fileHeader.DimY; i++)
		{
			motive m = supporters[s].GetAt(i);
			if (GetDistance(m, curr) <= maxDist)
			{
				supportMotive sm;
				sm.m = m;
				sm.index = s;
				ret.push_back(sm);
			}
		}
	}

	return ret;
}

std::vector<motive> MotiveList::GetNeighbours(int groupNr)
{
	std::vector<motive> ret;

	int idx = GetGroupIdx(groupNr);
	if (idx < 0) return ret;

	for (int i = 0; i < _fileHeader.DimY; i++)
	{
		motive m = GetAt(i);
		if (m.classNo == idx)
		{
			ret.push_back(m);
		}
	}

	return ret;
}

int MotiveList::GetGroupCount(GroupMode_enum aGroupMode)
{
	switch (aGroupMode)
	{
	case GM_BYGROUP:
		return (int)groupIndices.size();
	case GM_MAXDIST:
		return _fileHeader.DimY;
	case GM_MAXCOUNT:
		return _fileHeader.DimY;
	}
	return 0;
}

int MotiveList::GetGroupIdx(int groupNr)
{
	if (groupNr < 0 || groupNr >= groupIndices.size())
		return -1;

	return groupIndices[groupNr];
}
int MotiveList::GetGlobalIdx(motive & m)
{
	for (int i = 0; i < _fileHeader.DimY; i++)
	{
		motive m2 = GetAt(i);
		if (m2.isEqual(m))
			return i;
	}
	return -1;
}
bool motive::isEqual(motive & m)
{
	bool eq = true;
	eq &= ccCoeff == m.ccCoeff;
	eq &= xCoord == m.xCoord;
	eq &= yCoord == m.yCoord;
	eq &= partNr == m.partNr;
	eq &= tomoNr == m.tomoNr;
	eq &= partNrInTomo == m.partNrInTomo;
	eq &= wedgeIdx == m.wedgeIdx;
	eq &= x_Coord == m.x_Coord;
	eq &= y_Coord == m.y_Coord;
	eq &= z_Coord == m.z_Coord;
	eq &= x_Shift == m.x_Shift;
	eq &= y_Shift == m.y_Shift;
	eq &= z_Shift == m.z_Shift;
	eq &= x_ShiftBefore == m.x_ShiftBefore;
	eq &= y_ShiftBefore == m.y_ShiftBefore;
	eq &= z_ShiftBefore == m.z_ShiftBefore;
	eq &= phi == m.phi;
	eq &= psi == m.psi;
	eq &= theta == m.theta;
	eq &= classNo == m.classNo;

	return eq;
}

string motive::GetIndexCoding(MotiveList::NamingConvention_enum nc)
{
	stringstream ss;
	switch (nc)
	{
	case MotiveList::NamingConvention_enum::NC_ParticleOnly:
		ss << partNr;
		break;
	case MotiveList::NamingConvention_enum::NC_TomogramParticle:
		ss << tomoNr << "_" << partNrInTomo;
		break;
	default:
		break;
	}
	return ss.str();
}
