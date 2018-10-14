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


#ifndef ALIGNMENTOPTIONS_H
#define ALIGNMENTOPTIONS_H

#include <string>
#include <vector>
#include <sstream>

enum Algorithm_enum
{
	ALGO_YIFAN_CHENG,
	ALGO_ITERATIVE,
	ALGO_BOTH
};

enum Interpolation_enum
{
	INTER_NEARESTNEIGHBOUR,
	INTER_BILINEAR,
	INTER_CUBIC
};

class AlignmentOptions
{
public:
	AlignmentOptions(int argc, char* argv[]);

	int LP, LPS, HP, HPS;
	std::vector<std::string> FileList;
	std::vector<float> TiltAngles;
	float PixelSize;
	int PatchSize;
	int MaxShift;
	Algorithm_enum Algorithm;
	Interpolation_enum Interpolation;
	bool EntireTiltSeries;
	int DeviceID;
	std::string Path;
	std::string Output;
	int DeadPixelThreshold;
	bool AssumeZeroShift;
	int GroupStack;
};

#endif