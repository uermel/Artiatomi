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

#include <EmFile.h>
#ifdef SUBVOLREC_MODE
#include "../utils/Config.h"
#endif
#ifdef REFINE_MODE
#endif
#include "../utils/Config.h"

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
	
#ifdef SUBVOLREC_MODE
	string GetIndexCoding(Configuration::NamingConvention nc);
#endif

#ifdef REFINE_MODE
	bool isEqual(motive& m);
#endif
};

struct supportMotive
{
	motive m;
	int index;
};

class MotiveList : public EmFile
{
	float binningFactorClick;
	float binningFactorShift;

	std::vector<int> groupIndices;

public:
	MotiveList(string filename, float aBinningFactorClick, float aBinningShift);

	motive GetAt(int index);

	void SetAt(int index, motive& m);

	int GetParticleCount();

#ifdef REFINE_MODE
	float GetDistance(int aIndex1, int aIndex2);
	float GetDistance(motive& mot1, motive& mot2);
	std::vector<motive> GetNeighbours(int index, Configuration::Config& aConfig);
	std::vector<motive> GetNeighbours(int index, int count);
	std::vector<motive> GetNeighbours(int index, float maxDist);
	std::vector<supportMotive> GetNeighbours(int index, float maxDist, vector<MotiveList>& supporters);
	std::vector<motive> GetNeighbours(int groupNr);
	int GetGroupCount(Configuration::Config& aConfig);
	int GetGroupIdx(int groupNr);
	int GetGlobalIdx(motive& m);
#endif
};
#endif //MOTIVELIST_H