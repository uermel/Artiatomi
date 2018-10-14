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


#include "AlignmentOptions.h"
#include "../FileIO/MDocFile.h"
#include "../FileIO/SimpleFileList.h"

using namespace std;

AlignmentOptions::AlignmentOptions(int argc, char * argv[]) :
	LP(400), LPS(100), HP(0), HPS(0),
	PatchSize(800),
	MaxShift(1000),
	Algorithm(ALGO_YIFAN_CHENG),
	Interpolation(INTER_NEARESTNEIGHBOUR),
	EntireTiltSeries(false),
	DeviceID(0),
	PixelSize(1),
	DeadPixelThreshold(250),
	AssumeZeroShift(false),
	GroupStack(1)
{
	bool argsOK = argc >= 2;
	bool outputSet = false;

	for (int i = 1; i < argc; i++)
	{
		string str(argv[i]);
		if (str == "-a")
			EntireTiltSeries = true;
		else if (str == "-lp")
		{
			if (i >= argc - 1)
			{
				argsOK = false;
				break;
			}
			stringstream ss(argv[i + 1]);
			ss >> LP;
			i++;
		}
		else if (str == "-lps")
		{
			if (i >= argc - 1)
			{
				argsOK = false;
				break;
			}
			stringstream ss(argv[i + 1]);
			ss >> LPS;
			i++;
		}
		else if (str == "-hp")
		{
			if (i >= argc - 1)
			{
				argsOK = false;
				break;
			}
			stringstream ss(argv[i + 1]);
			ss >> HP;
			i++;
		}
		else if (str == "-hps")
		{
			if (i >= argc - 1)
			{
				argsOK = false;
				break;
			}
			stringstream ss(argv[i + 1]);
			ss >> HPS;
			i++;
		}
		else if (str == "-p")
		{
			if (i >= argc - 1)
			{
				argsOK = false;
				break;
			}
			stringstream ss(argv[i + 1]);
			ss >> PatchSize;
			i++;
		}
		else if (str == "-m")
		{
			if (i >= argc - 1)
			{
				argsOK = false;
				break;
			}
			stringstream ss(argv[i + 1]);
			ss >> MaxShift;
			i++;
		}
		else if (str == "-dp")
		{
			if (i >= argc - 1)
			{
				argsOK = false;
				break;
			}
			stringstream ss(argv[i + 1]);
			ss >> DeadPixelThreshold;
			i++;
		}
		else if (str == "-d")
		{
			if (i >= argc - 1)
			{
				argsOK = false;
				break;
			}
			stringstream ss(argv[i + 1]);
			ss >> DeviceID;
			i++;
		}
		else if (str == "-o")
		{
			if (i >= argc - 1)
			{
				argsOK = false;
				break;
			}
			string ss(argv[i + 1]);
			Output = ss;
			outputSet = true;
			i++;
		}
		else if (str == "-g")
		{
			if (i >= argc - 1)
			{
				argsOK = false;
				break;
			}
			stringstream ss(argv[i + 1]);
			ss >> GroupStack;
			i++;
		}
		else if (str == "-algo")
		{
			if (i >= argc - 1)
			{
				argsOK = false;
				break;
			}
			string str2(argv[i + 1]);
			if (str2 == "iterative" || str2 == "Iterative")
			{
				Algorithm = ALGO_ITERATIVE;
				i++;
			}
			if (str2 == "both" || str2 == "Both")
			{
				Algorithm = ALGO_BOTH;
				i++;
			}
			if (str2 == "matrix" || str2 == "Matrix")
			{
				Algorithm = ALGO_YIFAN_CHENG;
				i++;
			}
		}
		else if (str == "-zeroShift")
		{
			AssumeZeroShift = true;
		}
		else if (str == "-interpol")
		{
			if (i >= argc - 1)
			{
				argsOK = false;
				break;
			}
			string str2(argv[i + 1]);
			if (str2 == "nn" || str2 == "NN")
			{
				Interpolation = INTER_NEARESTNEIGHBOUR;
				i++;
			}
			else if (str2 == "bilinear" || str2 == "Bilinear")
			{
				Interpolation = INTER_BILINEAR;
				i++;
			}
			else if (str2 == "cubic" || str2 == "Cubic")
			{
				Interpolation = INTER_CUBIC;
				i++;
			}
		}

		if (!argsOK)
		{
			cout << "Usage:" << endl;
			cout << "ImageStackAlignator Inputfile [options]..." << endl;

			exit(-1);
		}


		if (EntireTiltSeries)
		{
			string str(argv[1]);
			if (str.substr(str.length() - 4) == "mdoc")
			{
				MDocFile mdoc(str);
				vector<MDocEntry> entries = mdoc.GetEntries();
				FileList.clear();
				TiltAngles.clear();
				for (size_t i = 0; i < entries.size(); i++)
				{
					FileList.push_back(entries[i].SubFramePath);
					TiltAngles.push_back(entries[i].TiltAngle);
					PixelSize = entries[i].PixelSpacing;
				}
				size_t last1 = str.find_last_of('\\');
				size_t last2 = str.find_last_of('/');

				if (last1 == string::npos && last2 == string::npos)
				{
					Path = "./";
				}
				else
				{
					if (last1 == string::npos)
					{
						Path = str.substr(0, last2+1);
					}
					else
					{
						Path = str.substr(0, last1+1);
					}
				}
				if (!outputSet)
					Output = str.substr(0, str.length() - 5) + "Alig.st";
			}
			else
			{
				SimpleFileList sfl(str);
				FileList = sfl.GetEntries();
				TiltAngles.push_back(0);
				PixelSize = 1;
				size_t last1 = str.find_last_of('\\');
				size_t last2 = str.find_last_of('/');

				if (last1 == string::npos && last2 == string::npos)
				{
					Path = "./";
				}
				else
				{
					if (last1 == string::npos)
					{
						Path = str.substr(0, last2 + 1);
					}
					else
					{
						Path = str.substr(0, last1 + 1);
					}
				}
				if (!outputSet)
					Output = str.substr(0, str.length() - 3) + "Alig.st";
			}
		}
		else
		{
			FileList.push_back(argv[1]);
			TiltAngles.push_back(0);
			size_t last1 = str.find_last_of('\\');
			size_t last2 = str.find_last_of('/');

			if (last1 == string::npos && last2 == string::npos)
			{
				Path = "./";
			}
			else
			{
				if (last1 == string::npos)
				{
					Path = str.substr(0, last2 + 1);
				}
				else
				{
					Path = str.substr(0, last1 + 1);
				}
			}
			if (!outputSet)
				Output = str.substr(0, str.length() - 3) + "Alig.st";
		}

	}
}
