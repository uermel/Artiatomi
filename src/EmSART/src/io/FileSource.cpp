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


#include "FileSource.h"

FileSource::FileSource(string aFilename)
	: ProjectionSource(), _filename(aFilename), _projectionCache(NULL), _ts(NULL)
{
	_ts = new TiltSeries(_filename, &FileLoadStatusUpdate);

	if (_ts->GetFileDataType() == DT_SHORT ||
		_ts->GetFileDataType() == DT_USHORT)
	{
		_projectionCache = new float* [_ts->GetImageCount()];
		for (uint i = 0; i < _ts->GetImageCount(); i++)
		{
			//We will convert all data to float, why we need more storage space for short/ushort data
			_projectionCache[i] = new float[(size_t)_ts->GetWidth() * (size_t)_ts->GetHeight()];
			memcpy(_projectionCache[i], _ts->GetData(i), (size_t)_ts->GetWidth() * (size_t)_ts->GetHeight() * sizeof(short));
		}
	}
}

FileSource::~FileSource()
{
	if (_projectionCache && _ts)
        for (uint i = 0; i < _ts->GetImageCount(); i++)
            if (_projectionCache[i])
                delete[] _projectionCache[i];
	delete[] _projectionCache;
	_projectionCache = NULL;

	if (_ts)
		delete _ts;
	_ts = NULL;
}


void FileSource::FileLoadStatusUpdate(FileReader::FileReaderStatus status)
{
	float progress = (float)status.bytesRead / (float)status.bytesToRead * 100.0f;

	printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
	printf("Loading projections: %.2f %%", progress);
	fflush(stdout);
}


DataType_enum FileSource::GetDataType()
{
	if (_ts)
		return _ts->GetFileDataType();
	return DT_FLOAT;
}

char* FileSource::GetProjection(uint aIndex)
{
	if (_ts == NULL)
		return NULL;

	if (_ts)
	{
		if (_ts->GetFileDataType() == DT_SHORT ||
			_ts->GetFileDataType() == DT_USHORT)
		{
			if (aIndex < _ts->GetImageCount())
				return (char*)_projectionCache[aIndex];
			else
				return NULL;
		}
		else
		{
			return (char*)_ts->GetData(aIndex);
		}
	}
}

float FileSource::GetPixelSize()
{
	if (_ts)
		return _ts->GetPixelSize();
	return 0;
}
int FileSource::GetWidth()
{
	if (_ts)
		return _ts->GetWidth();
	return 0;
}
int FileSource::GetHeight()
{
	if (_ts)
		return _ts->GetHeight();
	return 0;
}
int FileSource::GetProjectionCount()
{
	if (_ts)
		return _ts->GetImageCount();
	return 0;
}