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


// SER2MRC.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include "stdafx.h"
#include "../FileIO/FileIO.h"
#include "../FileIO/SingleFrame.h"
#include "../FileIO/TiltSeries.h"
#include "../FileIO/MovieStack.h"
#include "../FileIO/SERFile.h"
#include "../FileIO/EmFile.h"

using namespace std;

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		cout << "Usage: \nSER2MRC inputfile outputfile" << endl;
		exit(-1);
	}

	string inputFile(argv[1]);
	string outputFile(argv[2]);

	bool erg = SingleFrame::CanReadFile(inputFile);
	if (!erg)
	{
		cout << "Cannot read file '" << inputFile << "'. Aborting..." << endl;
		exit(-1);
	}
	SingleFrame sf(inputFile);

	float* dataOut = new float[sf.GetWidth() * sf.GetHeight()];
	if (sf.GetFileDataType() == DT_INT)
	{
		int* dataIn = (int*)sf.GetData();
		float maxi = 0;
		float mini = 10000000;
		for (size_t i = 0; i < sf.GetWidth() * sf.GetHeight(); i++)
		{
			int c = dataIn[i];
			maxi = std::fmaxf(maxi, (float)c);
			mini = std::fminf(mini, (float)c);
			dataOut[i] = (float)c;
		}
		rand();
	}
	else if (sf.GetFileDataType() == DT_UINT)
	{
		uint* dataIn = (uint*)sf.GetData();
		for (size_t i = 0; i < sf.GetWidth() * sf.GetHeight(); i++)
		{
			dataOut[i] = (float)dataIn[i];
		}
	}
	else if (sf.GetFileDataType() == DT_USHORT)
	{
		ushort* dataIn = (ushort*)sf.GetData();
		for (size_t i = 0; i < sf.GetWidth() * sf.GetHeight(); i++)
		{
			dataOut[i] = dataIn[i];
		}
	}
	else if (sf.GetFileDataType() == DT_SHORT)
	{
		short* dataIn = (short*)sf.GetData();
		for (size_t i = 0; i < sf.GetWidth() * sf.GetHeight(); i++)
		{
			dataOut[i] = dataIn[i];
		}
	}
	else if (sf.GetFileDataType() == DT_CHAR)
	{
		char* dataIn = (char*)sf.GetData();
		for (size_t i = 0; i < sf.GetWidth() * sf.GetHeight(); i++)
		{
			dataOut[i] = dataIn[i];
		}
	}
	else if (sf.GetFileDataType() == DT_UCHAR)
	{
		uchar* dataIn = (uchar*)sf.GetData();
		for (size_t i = 0; i < sf.GetWidth() * sf.GetHeight(); i++)
		{
			dataOut[i] = dataIn[i];
		}
	}
	else if (sf.GetFileDataType() == DT_FLOAT)
	{
		float* dataIn = (float*)sf.GetData();
		for (size_t i = 0; i < sf.GetWidth() * sf.GetHeight(); i++)
		{
			dataOut[i] = dataIn[i];
		}
	}

	MRCFile::InitHeaders(outputFile, sf.GetWidth(), sf.GetHeight(), sf.GetPixelSize(), DT_FLOAT, false);
	MRCFile::AddPlaneToMRCFile(outputFile, DT_FLOAT, dataOut, 0);

	delete[] dataOut;

    return 0;
}

