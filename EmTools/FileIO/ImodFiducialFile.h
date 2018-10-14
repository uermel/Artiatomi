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


#ifndef IMODFIDUCIALFILE_H
#define IMODFIDUCIALFILE_H

#include "../Basics/Default.h"
#include "FileReader.h"
#include "../FilterGraph/PointF.h"
#include <vector>

#define IMOD (0x494d4f44)
#define V12  (0x56312e32)
#define IEOF (0x49454f46)
#define OBJT (0x4f424a54)
#define CONT (0x434f4e54)

struct ImodModelDataStructure
{
	char name[128];
	int xmax;
	int ymax;
	int zmax;

	int objsize;
	uint flags;
	int drawmode;
	int mousemode;
	int blacklevel;
	int whitelevel;
	
	float xoffset;
	float yoffset;
	float zoffset;

	float xscale;
	float yscale;
	float zscale;

	int object;
	int contour;
	int point;

	int res;
	int thresh;

	float pixsize;
	int units;

	int csum;
	float alpha;
	float beta;
	float gamma;
};

struct ImodObjectDataStructure
{
	char name[64];
	uint reserved[16];

	int contsize;
	uint flags;
	int axis;
	int drawmode;

	float red;
	float green;
	float blue;

	int pdrawsize;
	uchar symbol;
	uchar symsize;
	uchar linewidth2;
	uchar linewidth;
	uchar linesty;
	uchar symflags;

	uchar sympad;
	uchar trans;
	int meshsize;
	int surfsize;

};

struct ContourDataStructure
{
	int psize;
	uint flags;
	int time;
	int surf;
	std::vector<float> ptX;
	std::vector<float> ptY;
	std::vector<float> ptZ;
};

class ImodFiducialFile :
	public FileReader
{
public:
	ImodFiducialFile(string aFilename);
	~ImodFiducialFile();

	bool OpenAndRead();
	std::vector<PointF> GetMarkers(size_t projectionIdx);
	size_t GetProjectionCount();
	size_t GetMarkerCount();
	std::vector<float> GetTiltAngles();

	DataType_enum GetDataType();

private:
	ImodModelDataStructure _fileHeader;
	ImodObjectDataStructure _objHeader;
	std::vector<ContourDataStructure> _contours;
	std::vector<float> tiltAngles;
	std::vector<float> preAligX;
	std::vector<float> preAligY;
	int _binning;

	bool ReadTiltAngles();
	bool ReadPrealignment();
};

#endif // !IMODFIDUCIALFILE_H
