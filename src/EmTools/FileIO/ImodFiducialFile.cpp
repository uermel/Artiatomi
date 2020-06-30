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


#include "ImodFiducialFile.h"

ImodFiducialFile::ImodFiducialFile(string aFilename)
	: FileReader(aFilename, false)
{
}


ImodFiducialFile::~ImodFiducialFile()
{
}

bool ImodFiducialFile::OpenAndRead()
{
	bool res = FileReader::OpenRead();

	if (!res)
	{
		CloseRead();
		throw FileIOException(FileReader::mFileName, "Cannot open file for reading.");
	}

	uint test = ReadUI4LE();
	uint test2 = IMOD;
	res = (test == IMOD);
	res &= (ReadUI4LE() == V12);
	if (!res)
	{
		CloseRead();
		throw FileIOException(FileReader::mFileName, "This is not an IMOD fiducial file.");
	}

	Read((char*)_fileHeader.name, sizeof(_fileHeader.name));
	_fileHeader.xmax = ReadI4LE();
	_fileHeader.ymax = ReadI4LE();
	_fileHeader.zmax = ReadI4LE();

	_fileHeader.objsize = ReadI4LE();
	_fileHeader.flags = ReadUI4LE();

	_fileHeader.drawmode = ReadI4LE();
	_fileHeader.mousemode = ReadI4LE();
	_fileHeader.blacklevel = ReadI4LE();
	_fileHeader.whitelevel = ReadI4LE();

	_fileHeader.xoffset = ReadF4LE();
	_fileHeader.yoffset = ReadF4LE();
	_fileHeader.zoffset = ReadF4LE();

	_fileHeader.xscale = ReadF4LE();
	_fileHeader.yscale = ReadF4LE();
	_fileHeader.zscale = ReadF4LE();

	_fileHeader.object = ReadI4LE();
	_fileHeader.contour = ReadI4LE();
	_fileHeader.point = ReadI4LE();
	_fileHeader.res = ReadI4LE();
	_fileHeader.thresh = ReadI4LE();

	_fileHeader.pixsize = ReadF4LE();
	_fileHeader.units = ReadI4LE();
	_fileHeader.csum = ReadI4LE();

	_fileHeader.alpha = ReadF4LE();
	_fileHeader.beta = ReadF4LE();
	_fileHeader.gamma = ReadF4LE();

	res &= (ReadUI4LE() == OBJT);
	if (!res)
	{
		CloseRead();
		throw FileIOException(FileReader::mFileName, "This is not an IMOD fiducial file.");
	}

	Read((char*)&_objHeader, sizeof(_objHeader));
	Endian_swap(_objHeader.contsize);
	Endian_swap(_objHeader.flags);
	Endian_swap(_objHeader.axis);
	Endian_swap(_objHeader.drawmode);
	Endian_swap(_objHeader.red);
	Endian_swap(_objHeader.green);
	Endian_swap(_objHeader.blue);
	Endian_swap(_objHeader.pdrawsize);
	Endian_swap(_objHeader.meshsize);
	Endian_swap(_objHeader.surfsize);

	for (size_t c = 0; c < _objHeader.contsize; c++)
	{
		res = (ReadUI4LE() == CONT);
		if (!res)
		{
			CloseRead();
			throw FileIOException(FileReader::mFileName, "This is not an IMOD fiducial file.");
		}

		ContourDataStructure cont;
		cont.psize = ReadI4LE();
		cont.flags = ReadUI4LE();
		cont.time = ReadI4LE();
		cont.surf = ReadI4LE();

		for (size_t p = 0; p < cont.psize; p++)
		{
			cont.ptX.push_back(ReadF4LE());
			cont.ptY.push_back(ReadF4LE());
			cont.ptZ.push_back(ReadF4LE());
		}
		_contours.push_back(cont);
	}

	uint ID = ReadUI4LE();
	while (!(ID == IEOF || ID == 0x4D494E58)) //MINX
	{
		int size = ReadI4LE();
		Seek(size, ios::cur);
		ID = ReadUI4LE();
		if (!mFile->good())
			break;
	}

	float old[9];
	float newd[9];
	if (ID == 0x4D494E58) //MINX
	{
		int size = ReadI4LE();
		for (size_t i = 0; i < 9; i++)
		{
			old[i] = ReadF4LE();
		}
		for (size_t i = 0; i < 9; i++)
		{
			newd[i] = ReadF4LE();
		}
		_binning = (int)((newd[0] / newd[2]) + 0.5f); //needs to be checked!
		ID = ReadUI4LE();
	}

	while (!(ID == IEOF))
	{
		int size = ReadI4LE();
		Seek(size, ios::cur);
		ID = ReadUI4LE();
		if (!mFile->good())
			break;
	}
	bool ok1 = ReadTiltAngles();
	bool ok2 = ReadPrealignment();
	
	return ID == IEOF && ok1 && ok2;
}

std::vector<PointF> ImodFiducialFile::GetMarkers(size_t projectionIdx)
{
	std::vector<PointF> markers;

	if (projectionIdx < _fileHeader.zmax)
	{
		markers.resize(_objHeader.contsize);

		for (size_t i = 0; i < _objHeader.contsize; i++)
		{
			markers[i] = PointF(-1, -1);
		}

		for (size_t i = 0; i < _objHeader.contsize; i++)
		{
			ContourDataStructure c = _contours[i];
			bool found = false;
			int idx = -1;
			for (int pt = 0; pt < c.psize; pt++)
			{
				if ((int)(c.ptZ[pt] + 0.5f) == projectionIdx)
				{
					idx = pt;
					found = true;
				}
			}
			float x = -1;
			float y = -1;

			if (found)
			{
				//int idx = (int)(round(c.ptZ[projectionIdx]));
				x = c.ptX[idx] * _binning - preAligX[projectionIdx];
				y = c.ptY[idx] * _binning - preAligY[projectionIdx];
			}
			markers[i] = PointF(x, y);
		}
	}

	return markers;
}

size_t ImodFiducialFile::GetProjectionCount()
{
	return _fileHeader.zmax;
}

size_t ImodFiducialFile::GetMarkerCount()
{
	return _objHeader.contsize;
}

std::vector<float> ImodFiducialFile::GetTiltAngles()
{
	return tiltAngles;
}

DataType_enum ImodFiducialFile::GetDataType()
{
	return DataType_enum::DT_UNKNOWN;
}

bool ImodFiducialFile::ReadTiltAngles()
{
	string tiltFilename = mFileName;
	tiltFilename[tiltFilename.size() - 3] = 't';
	tiltFilename[tiltFilename.size() - 2] = 'l';
	tiltFilename[tiltFilename.size() - 1] = 't';

	ifstream tilt(tiltFilename);
	for (int i = 0; i < _fileHeader.zmax; i++)
	{
		float t;
		tilt >> t;
		tiltAngles.push_back(t);
	}
	bool res = tilt.good();
	tilt.close();
	return res;
}

bool ImodFiducialFile::ReadPrealignment()
{
	string tiltFilename = mFileName + "12";
	tiltFilename[tiltFilename.size() - 5] = 'p';
	tiltFilename[tiltFilename.size() - 4] = 'r';
	tiltFilename[tiltFilename.size() - 3] = 'e';
	tiltFilename[tiltFilename.size() - 2] = 'x';
	tiltFilename[tiltFilename.size() - 1] = 'g';

	ifstream alig(tiltFilename);
	for (int i = 0; i < _fileHeader.zmax; i++)
	{
		float t0, t1, t2, t3, t4, t5;
		alig >> t0 >> t1 >> t2 >> t3 >> t4 >> t5;
		preAligX.push_back(t4);
		preAligY.push_back(t5);
	}
	bool res = alig.good();
	alig.close();
	return res;
}
