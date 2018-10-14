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



#include "../FileIO/MarkerFile.h"
#include <stdio.h>


using namespace std;

static void status(int percent)
{
	cout << percent << "%" << endl;
}

int main(int argc, char* argv[])
{
	int dimX = 3710, dimY = 3838;
	float beamDeclination = 0;
	bool imageRotation = true;
	bool fixedImageRotation = false;
	bool tilts = false;
	bool alignBeamDeclination = false;
	bool magnification = false;
	bool normMin = false;
	bool normMinTilt = true;
	bool magsFirst = false;
	int iterSwitch = 0;
	int iterations = 5;
	float addZShift = 0;
	string markerfile;
	string markerOutFile;
	bool loadIMOD = false;
	float magAniso = 1;
	float magAnisoAngle = 0;
	int refMarker = 0;

	if (argc < 3)
	{
		cout << "Usage:" << endl;
		cout << "MarkerAlignator markerfile markerOutFile [options]" << endl;
		cout << "options are:" << endl;
		cout << " dimX value                  image width [3710]" << endl;
		cout << " dimY value                  image width [3838]" << endl;
		cout << " beamDeclination value       beam declination (phi) [0]" << endl;
		cout << " tilts value                 align tilt angles [false]" << endl;
		cout << " alignBeamDeclination value  align for beam declination [false]" << endl;
		cout << " magnification value         align for magnification [false]" << endl;
		cout << " normMin value               normalize magnifiaction on minimal value [false]" << endl;
		cout << " normMinTilt value           normalize magnifiaction on zero deg tilt [true]" << endl;
		cout << " magsFirst value             align first for magnification, then for tilts [false]" << endl;
		cout << " iterSwitch value            iteration when switching from magnificatino to tilt align [0]" << endl;
		cout << " iterations value            iteration to perform for alignment [5]" << endl;
		cout << " addZShift value             shift markers in Z by value pixels [0]" << endl;
		cout << " magAnisotropy factor angle  magnification anisotropy parameters [1 0]" << endl;
		cout << " refMarker value             reference marker (most central marker) [0]" << endl;
		return -1;
	}


	string filename(argv[1]);
	string substring = filename.substr(filename.length() - 4, 4);
	if (filename.substr(filename.length() - 4, 4) == ".fid")
	{
		loadIMOD = true;
	}
	markerOutFile = string(argv[2]);

	for (int i = 3; i < argc; i++)
	{
		string str(argv[i]);
		if (str == "dimX")
		{
			dimX = atoi(argv[i + 1]);
			i++;
			cout << "dimX set to " << dimX << endl;
		}
		else if (str == "dimY")
		{
			dimY = atoi(argv[i + 1]);
			i++;
			cout << "dimY set to " << dimY << endl;
		}
		else if (str == "beamDeclination")
		{
			beamDeclination = atof(argv[i + 1]);
			i++;
			cout << "beamDeclination set to " << beamDeclination << endl;
		}
		else if (str == "alignBeamDeclination")
		{
			alignBeamDeclination = string(argv[i + 1]) == "true";
			i++;
			cout << "alignBeamDeclination set to " << alignBeamDeclination << endl;
		}
		else if (str == "tilts")
		{
			tilts = string(argv[i + 1]) == "true";
			i++;
			cout << "tilts set to " << alignBeamDeclination << endl;
		}
		else if (str == "magnification")
		{
			magnification = string(argv[i + 1]) == "true";
			i++;
			cout << "magnification set to " << magnification << endl;
		}
		else if (str == "normMin")
		{
			normMin = string(argv[i + 1]) == "true";
			i++;
			cout << "normMin set to " << normMin << endl;
		}
		else if (str == "normMinTilt")
		{
			normMinTilt = string(argv[i + 1]) == "true";
			i++;
			cout << "normMinTilt set to " << normMinTilt << endl;
		}
		else if (str == "magsFirst")
		{
			magsFirst = string(argv[i + 1]) == "true";
			i++;
			cout << "magsFirst set to " << magsFirst << endl;
		}
		else if (str == "iterations")
		{
			iterations = atoi(argv[i + 1]);
			i++;
			cout << "iterations set to " << iterations << endl;
		}
		else if (str == "addZShift")
		{
			addZShift = atof(argv[i + 1]);
			i++;
			cout << "addZShift set to " << addZShift << endl;
		}
		else if (str == "refMarker")
		{
			refMarker = atoi(argv[i + 1]);
			i++;
			cout << "refMarker set to " << refMarker << endl;
		}
		else if (str == "magAnisotropy")
		{
			magAniso = atof(argv[i + 1]);
			magAnisoAngle = atof(argv[i + 2]);
			i++;
			i++;
			cout << "normMin set to " << magAniso << " ; " << magAnisoAngle << endl;
		}
	}
	MarkerFile* marker;
	if (loadIMOD)
	{
		marker = MarkerFile::ImportFromIMOD(filename);
	}
	else
	{
		marker = new MarkerFile(filename);
	}

	float error = 0;
	float* perProj = new float[marker->GetTotalProjectionCount()];
	float* perMarker = new float[marker->GetMarkerCount()];
	float* x = new float[marker->GetMarkerCount()];
	float* y = new float[marker->GetMarkerCount()];
	float* z = new float[marker->GetMarkerCount()];

	marker->SetMagAnisotropy(magAniso, magAnisoAngle, dimX,dimY);

	marker->Align3D(refMarker, dimX, dimY, error, beamDeclination, imageRotation, fixedImageRotation, tilts, alignBeamDeclination, magnification, normMin,
		normMinTilt, magsFirst, iterSwitch, iterations, addZShift, perProj, perMarker, x, y, z, &status);
	
	cout << "Alignment score: " << error << endl;
	
	marker->Save(markerOutFile);

    return 0;
}

