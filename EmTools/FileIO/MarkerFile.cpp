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


#include "MarkerFile.h"
#include "../Minimization/levmar.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>

#define MARKERFILE_X_DIMENSION 10
#define INVALID_MARKER_POSITION -1

MarkerFile::MarkerFile(string aFileName)
	: EmFile(aFileName), mMagAnisotropyAmount(1), mMagAnisotropyAngle(0), mMagAnisotropy(Matrix<float>(3,3))
{
	OpenAndRead();

	mMarkerXPos = new float[_fileHeader.DimZ];
	mMarkerYPos = new float[_fileHeader.DimZ];
	mMarkerZPos = new float[_fileHeader.DimZ];

	memset(mMarkerXPos, 0, _fileHeader.DimZ * sizeof(float));
	memset(mMarkerYPos, 0, _fileHeader.DimZ * sizeof(float));
	memset(mMarkerZPos, 0, _fileHeader.DimZ * sizeof(float));

	if (_fileHeader.IsNewHeaderFormat)
	{
		if (_fileHeader.MarkerOffset)
		{
			FileReader::OpenRead();
			FileReader::Seek(_fileHeader.MarkerOffset, ios::beg);
			for (size_t i = 0; i < _fileHeader.DimZ; i++)
			{
				mMarkerXPos[i] = FileReader::ReadF4LE();
				mMarkerYPos[i] = FileReader::ReadF4LE();
				mMarkerZPos[i] = FileReader::ReadF4LE();
			}
			FileReader::CloseRead();
		}
		SetMagAnisotropy(_fileHeader.MagAnisotropyFactor, _fileHeader.MagAnisotropyAngle, _fileHeader.ImageSizeX, _fileHeader.ImageSizeY);
	}
}

MarkerFile::MarkerFile(int aProjectionCount, float * aTiltAngles)
	: EmFile(""), mMagAnisotropyAmount(1), mMagAnisotropyAngle(0), mMagAnisotropy(Matrix<float>(3,3))
{
	SetHeaderData(_fileHeader, MARKERFILE_X_DIMENSION, aProjectionCount, 1, 0.0f, DT_FLOAT);
	_fileHeader.IsNewHeaderFormat = 1;
	_fileHeader.MagAnisotropyFactor = 1;
	_data = new float[MARKERFILE_X_DIMENSION * aProjectionCount];
	memset(_data, 0, MARKERFILE_X_DIMENSION * aProjectionCount * sizeof(float));

	for (int proj = 0; proj < aProjectionCount; proj++)
	{
		(*this)(MFI_TiltAngle, proj, 0) = aTiltAngles[proj];
		(*this)(MFI_X_Coordinate, proj, 0) = INVALID_MARKER_POSITION;
		(*this)(MFI_Y_Coordinate, proj, 0) = INVALID_MARKER_POSITION;
		(*this)(MFI_Magnifiaction, proj, 0) = 1.0f;
	}

	mMarkerXPos = new float[_fileHeader.DimZ];
	mMarkerYPos = new float[_fileHeader.DimZ];
	mMarkerZPos = new float[_fileHeader.DimZ];

	memset(mMarkerXPos, 0, _fileHeader.DimZ * sizeof(float));
	memset(mMarkerYPos, 0, _fileHeader.DimZ * sizeof(float));
	memset(mMarkerZPos, 0, _fileHeader.DimZ * sizeof(float));
}

MarkerFile::~MarkerFile()
{
	delete[] mMarkerXPos;
	delete[] mMarkerYPos;
	delete[] mMarkerZPos;
}

bool MarkerFile::CanReadAsMarkerfile(string aFilename)
{
	EmFile test(aFilename);
	bool ok = test.OpenAndReadHeader();

	if (!ok) return ok;

	if (test.GetFileHeader().DimX != MARKERFILE_X_DIMENSION) return false;

	if (test.GetFileHeader().DimY < 1) return false;

	if (test.GetFileHeader().DimZ < 1) return false;

	return true;
}

float* MarkerFile::GetData()
{
	return (float*)EmFile::GetData();
}

int MarkerFile::GetMarkerCount()
{
	return _fileHeader.DimZ;
}

int MarkerFile::GetProjectionCount()
{
	int count = 0;
	for (int i = 0; i < _fileHeader.DimY; i++)
		if (CheckIfProjIndexIsGood(i)) count++;
	return count;
}

int MarkerFile::GetTotalProjectionCount()
{
	return _fileHeader.DimY;
}

Marker & MarkerFile::operator()(const int aProjection, const int aMarker)
{
	Marker* fdata = (Marker*)_data;

	return fdata[aMarker * _fileHeader.DimY + aProjection];
}

bool MarkerFile::CheckIfProjIndexIsGood(const int index)
{
	int mRefMarker = 0;
	return ((*this)(MFI_X_Coordinate, index, mRefMarker) > INVALID_MARKER_POSITION && (*this)(MFI_Y_Coordinate, index, mRefMarker) > INVALID_MARKER_POSITION);
}

void MarkerFile::AddMarker()
{
	int count = GetMarkerCount();
	float* newData = new float[MARKERFILE_X_DIMENSION  * (count + 1) * GetTotalProjectionCount()];
	memset(newData, 0, MARKERFILE_X_DIMENSION  * (count + 1) * GetTotalProjectionCount() * sizeof(float));
	memcpy(newData, _data, MARKERFILE_X_DIMENSION  * (count)* GetTotalProjectionCount() * sizeof(float));
	delete[] _data;
	_data = newData;
	_fileHeader.DimZ++;
	for (int itilt = 0; itilt < GetTotalProjectionCount(); itilt++)
	{
		(*this)(MFI_TiltAngle, itilt, count) = (*this)(MFI_TiltAngle, itilt, count - 1);
		(*this)(MFI_X_Coordinate, itilt, count) = INVALID_MARKER_POSITION;
		(*this)(MFI_Y_Coordinate, itilt, count) = INVALID_MARKER_POSITION;
	}
	delete[] mMarkerXPos;
	delete[] mMarkerYPos;
	delete[] mMarkerZPos;

	mMarkerXPos = new float[_fileHeader.DimZ];
	mMarkerYPos = new float[_fileHeader.DimZ];
	mMarkerZPos = new float[_fileHeader.DimZ];

	memset(mMarkerXPos, 0, _fileHeader.DimZ * sizeof(float));
	memset(mMarkerYPos, 0, _fileHeader.DimZ * sizeof(float));
	memset(mMarkerZPos, 0, _fileHeader.DimZ * sizeof(float));
}

void MarkerFile::RemoveMarker(int idx)
{
	int count = GetMarkerCount();
	if (count < 2)
		return; //we can't delete the last marker!

	if (idx < 0 || idx >= count)
		return; //idx out of bounds

	float* newData = new float[MARKERFILE_X_DIMENSION  * (count - 1) * GetTotalProjectionCount()];
	Marker* m = (Marker*)newData;
	
	for (int imark = 0; imark < count; imark++)
	{
		for (int itilt = 0; itilt < GetTotalProjectionCount(); itilt++)
		{	
			if (imark < idx)
			{
				m[imark * GetTotalProjectionCount() + itilt] = (*this)(itilt, imark);
			}
			if (imark > idx)
			{
				m[(imark - 1) * GetTotalProjectionCount() + itilt] = (*this)(itilt, imark);
			}
		}
	}
    delete[] _data;
    _data = newData;
    _fileHeader.DimZ--;


	delete[] mMarkerXPos;
	delete[] mMarkerYPos;
	delete[] mMarkerZPos;

	mMarkerXPos = new float[_fileHeader.DimZ];
	mMarkerYPos = new float[_fileHeader.DimZ];
	mMarkerZPos = new float[_fileHeader.DimZ];

	memset(mMarkerXPos, 0, _fileHeader.DimZ * sizeof(float));
	memset(mMarkerYPos, 0, _fileHeader.DimZ * sizeof(float));
	memset(mMarkerZPos, 0, _fileHeader.DimZ * sizeof(float));
}

void MarkerFile::SetMarkerPosition(int aProjection, int aMarkerIdx, float aPositionX, float aPositionY)
{
	if (aProjection < 0 || aProjection >= GetTotalProjectionCount())
		return;

	if (aMarkerIdx < 0 || aMarkerIdx >= GetMarkerCount())
		return;

	(*this)(MFI_X_Coordinate, aProjection, aMarkerIdx) = aPositionX;
	(*this)(MFI_Y_Coordinate, aProjection, aMarkerIdx) = aPositionY;
}

void MarkerFile::SetTiltAngles(float * aTiltAngles)
{
	for (int proj = 0; proj < GetTotalProjectionCount(); proj++)
	{
		for (int imark = 0; imark < GetMarkerCount(); imark++)
		{
			(*this)(MFI_TiltAngle, proj, imark) = aTiltAngles[proj];
		}
	}
}

std::vector<PointF> MarkerFile::GetMarkersAt(int aProjection)
{
	if (aProjection < 0 || aProjection >= GetTotalProjectionCount())
		return std::vector<PointF>();

	std::vector<PointF> ret(GetMarkerCount());

	for (int i = 0; i < GetMarkerCount(); i++)
	{
		ret.push_back(PointF((*this)(MFI_X_Coordinate, aProjection, i), (*this)(MFI_Y_Coordinate, aProjection, i)));
	}
	return ret;
}

std::vector<PointF> MarkerFile::GetAlignedMarkersAt(int aProjection)
{
	if (aProjection < 0 || aProjection >= GetTotalProjectionCount())
		return std::vector<PointF>();

	std::vector<PointF> ret(GetMarkerCount());

	for (int i = 0; i < GetMarkerCount(); i++)
	{
		ret.push_back(PointF((*this)(MFI_ProjectedCoordinateX, aProjection, i), (*this)(MFI_ProjectedCoordinateY, aProjection, i)));
	}
	return ret;
}

void MarkerFile::SetImageShift(int aProjection, float aShiftX, float aShiftY)
{
	if (aProjection < 0 || aProjection >= GetTotalProjectionCount())
		return;

	for (int i = 0; i < GetMarkerCount(); i++)
	{
		(*this)(MFI_X_Shift, aProjection, i) = aShiftX;
		(*this)(MFI_Y_Shift, aProjection, i) = aShiftY;
	}
}

void MarkerFile::SetConstantImageRotation(float aAngleInDeg)
{
	while (aAngleInDeg <= -180.0f)
	{
		aAngleInDeg += 360.0f;
	}
	while (aAngleInDeg > 180.0f)
	{
		aAngleInDeg -= 360.0f;
	}

	for (int imark = 0; imark < GetMarkerCount(); imark++)
	{
		for (int itilt = 0; itilt < GetTotalProjectionCount(); itilt++)
		{
			(*this)(MFI_RotationPsi, itilt, imark) = aAngleInDeg;
		}
	}
}

void MarkerFile::AlignSimple(int aReferenceMarker, int imageWidth, int imageHeight, float& aErrorScore, float& aImageRotation, float addZShift)
{

	float* x = new float[GetMarkerCount()];
	float* y = new float[GetMarkerCount()];
	float* z = new float[GetMarkerCount()];

	AlignSimple(aReferenceMarker, imageWidth, imageHeight, aErrorScore, aImageRotation, x, y, z, addZShift);

	delete[] x;
	delete[] y;
	delete[] z;
}

void MarkerFile::Align3D(int aReferenceMarker, int width, int height, float& aErrorScore, float& aPhi, bool DoPsi, bool DoFixedPsi, bool DoTheta, bool DoPhi, bool DoMags, bool normMin, bool normZeroTilt,
	bool magsFirst, int iterSwitch, int iterations, float addZShift, float* projErrorList, float* markerErrorList, float* aX, float* aY, float* aZ, void(*updateprog)(int percent))
{
	int MarkerCount = GetMarkerCount();
	int ProjectionCount = GetTotalProjectionCount();
	_fileHeader.ImageSizeX = width;
	_fileHeader.ImageSizeY = height;
	
	float* x, *y, *z;
	x = mMarkerXPos;
	y = mMarkerYPos;
	z = mMarkerZPos;
	/*if (aX == NULL)
	{
		x = new float[GetMarkerCount()];
	}
	else
	{
		x = aX;
	}
	if (aY == NULL)
	{
		y = new float[GetMarkerCount()];
	}
	else
	{
		y = aY;
	}
	if (aZ == NULL)
	{
		z = new float[GetMarkerCount()];
	}
	else
	{
		z = aZ;
	}*/
	float rotation = 0;

	//start with simple old alignment to get good start values for minimization
	AlignSimple(aReferenceMarker, width, height, aErrorScore, rotation, x, y, z, addZShift);

	// newly used:
	float* shiftsX = new float[ProjectionCount];
	float* shiftsY = new float[ProjectionCount];
	float* psis = new float[ProjectionCount];
	float* thetas = new float[ProjectionCount];
	float* mags = new float[ProjectionCount];
	float phi[1] = { aPhi };


	float totalProgress = 1; //shifts are always "on"
	float progressCounter = 0;

	if (DoPsi)
	{
		totalProgress++;
	}
	if (DoMags | DoTheta)
	{
		totalProgress++;
	}
	if (DoPhi)
	{
		//phi[0] = 0.001f;
		totalProgress++;
	}
	totalProgress *= iterations;

	for (int proj = 0; proj < ProjectionCount; proj++)
	{
		for (int marker = 0; marker < 1; marker++)
		{
			shiftsX[proj] = (*this)(MFI_X_Shift, proj, marker);
			shiftsY[proj] = (*this)(MFI_Y_Shift, proj, marker);
			thetas[proj] = (*this)(MFI_TiltAngle, proj, marker);
			psis[proj] = (*this)(MFI_RotationPsi, proj, marker);
			(*this)(MFI_Magnifiaction, proj, marker) = 1;
			mags[proj] = (*this)(MFI_Magnifiaction, proj, marker);
		}
	}


	for (int iter = 0; iter < iterations; iter++)
	{
		bool doMagsLocal = DoMags;
		bool doThetaLocal = DoTheta;

		if (magsFirst)
		{
			if (iter <= iterSwitch)
			{
				doMagsLocal = DoMags;
				doThetaLocal = false;
			}
			else
			{
				doMagsLocal = false;
				doThetaLocal = DoTheta;
			}
		}
		else
		{
			if (iter <= iterSwitch)
			{
				doMagsLocal = false;
				doThetaLocal = DoTheta;
			}
			else
			{
				doMagsLocal = DoMags;
				doThetaLocal = false;
			}
		}

		//if only one of them is selected, don't care about a switch...
		if (DoMags ^ DoTheta)
		{
			doMagsLocal = DoMags;
			doThetaLocal = DoTheta;
		}

		//Console.WriteLine("Vor Shift");
		MinimizeShift(x, y, z, thetas, psis, mags, phi[0], shiftsX, shiftsY, width, height);
		//Console.WriteLine("Nach Shift");
		progressCounter++;
		if (updateprog)
			(*updateprog)((int)(progressCounter / totalProgress * 100.0f));
		if (DoPhi)
		{
			//Console.WriteLine("Vor Phi");
			MinimizePhi(x, y, z, thetas, psis, mags, phi, shiftsX, shiftsY, width, height);
			//Console.WriteLine("Nach Phi");
			progressCounter++;
			if (updateprog)
				(*updateprog)((int)(progressCounter / totalProgress * 100.0f));
		}
		if (DoPsi)
		{
			if (DoFixedPsi)
			{
				MinimizePsiFixed(x, y, z, thetas, psis, mags, phi[0], shiftsX, shiftsY, width, height);
			}
			else
			{
				MinimizePsis(x, y, z, thetas, psis, mags, phi[0], shiftsX, shiftsY, width, height);
			}
			//Console.WriteLine("Vor Psi");
			//Console.WriteLine("Nach Psi");
			progressCounter++;
			if (updateprog)
				(*updateprog)((int)(progressCounter / totalProgress * 100.0f));
		}
		if (doMagsLocal)
		{
			//Console.WriteLine("Vor Mag");
			MinimizeMags(x, y, z, thetas, psis, mags, phi[0], shiftsX, shiftsY, width, height);
			//Console.WriteLine("Nach Mag");

			if (normZeroTilt)
			{
				int zeroTilt = GetMinTiltIndex();
				float normVal = mags[zeroTilt];

				for (int i = 0; i < ProjectionCount; i++)
				{
					mags[i] /= normVal;
				}
			}
			else
			{
				float normVal;
				if (normMin)
					normVal = 100000;
				else
					normVal = -100000;

				for (int i = 0; i < ProjectionCount; i++)
				{
					if (normMin)
						normVal = std::min(normVal, mags[i]);
					else
						normVal = std::max(normVal, mags[i]);
				}

				for (int i = 0; i < ProjectionCount; i++)
				{
					mags[i] /= normVal;
				}
			}
			progressCounter++;
			if (updateprog)
				(*updateprog)((int)(progressCounter / totalProgress * 100.0f));
		}
		if (doThetaLocal)
		{
			//Console.WriteLine("Vor Theta");
			MinimizeThetas(x, y, z, thetas, psis, mags, phi[0], shiftsX, shiftsY, width, height);
			//Console.WriteLine("Nach Theta");
			progressCounter++;
			if (updateprog)
				(*updateprog)((int)(progressCounter / totalProgress * 100.0f));
		}
	}




	for (int proj = 0; proj < ProjectionCount; proj++)
	{
		for (int marker = 0; marker < MarkerCount; marker++)
		{
			(*this)(MFI_X_Shift, proj, marker) = shiftsX[proj];
			(*this)(MFI_Y_Shift, proj, marker) = shiftsY[proj];
			(*this)(MFI_TiltAngle, proj, marker) = thetas[proj];
			(*this)(MFI_RotationPsi, proj, marker) = psis[proj];
			(*this)(MFI_Magnifiaction, proj, marker) = mags[proj];
		}
	}

	float* diffs = new float[ProjectionCount * MarkerCount * 2];
	int* usedBeads = new int[ProjectionCount];

	float* diffsum = new float[ProjectionCount];
	float* projectedPos = new float[ProjectionCount*MarkerCount * 2];

	float rms = ComputeRMS(projectedPos, diffsum, diffs, usedBeads, x, y, z, thetas, psis, mags, phi[0], shiftsX, shiftsY, width, height, this);
	aErrorScore = (float)sqrt(rms);
	aPhi = phi[0];


	float* projBead = new float[ProjectionCount * MarkerCount];

	/*float* projErrorList = new float[ProjectionCount];
	float* markerErrorList = new float[MarkerCount];*/

	for (int projection = 0; projection < ProjectionCount; projection++)
	{
		for (int marker = 0; marker < MarkerCount; marker++)
		{
			projBead[projection * MarkerCount + marker] = -99999.0f;
			(*this)(MFI_ProjectedCoordinateX, projection, marker) = projectedPos[GetBeadIndex(projection, marker, 0, MarkerCount)];
			(*this)(MFI_ProjectedCoordinateY, projection, marker) = projectedPos[GetBeadIndex(projection, marker, 1, MarkerCount)];

			if ((*this)(MFI_X_Coordinate, projection, marker) > INVALID_MARKER_POSITION && (*this)(MFI_Y_Coordinate, projection, marker) > INVALID_MARKER_POSITION)
			{
				float x = (*this)(MFI_X_Coordinate, projection, marker);
				float y = (*this)(MFI_Y_Coordinate, projection, marker);

				MoveXYToMagDistort(x, y, mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, width, height);

				float diffX = x - projectedPos[GetBeadIndex(projection, marker, 0, MarkerCount)];
				float diffY = y - projectedPos[GetBeadIndex(projection, marker, 1, MarkerCount)];

				projBead[projection * MarkerCount + marker] = sqrtf(diffX * diffX + diffY * diffY);
			}
		}
	}

	if (projErrorList != NULL)
	{
		for (int projection = 0; projection < ProjectionCount; projection++)
		{
			double counter = 0;
			double sumError = 0;
			for (int marker = 0; marker < MarkerCount; marker++)
			{
				if (projBead[projection * MarkerCount + marker] != -99999.0f)
				{
					sumError += projBead[projection * MarkerCount + marker];
					counter++;
				}

			}
			if (counter > 0)
			{
				projErrorList[projection] = sumError / counter; //mean Error per projection
			}
			else
			{
				projErrorList[projection] = 0;
			}
		}
	}

	if (markerErrorList != NULL)
	{
		for (int marker = 0; marker < MarkerCount; marker++)
		{
			double counter = 0;
			double sumError = 0;
			for (int projection = 0; projection < ProjectionCount; projection++)
			{
				if (projBead[projection * MarkerCount + marker] != -99999.0f)
				{
					sumError += projBead[projection * MarkerCount + marker];
					counter++;
				}

			}
			if (counter > 0)
			{
				markerErrorList[marker] = sumError / counter; //mean Error per marker
			}
			else
			{
				markerErrorList[marker] = 0;
			}
		}
	}

	//projectedPos = diffs;
	//return ret;

	_fileHeader.BeamDeclination = aPhi;
	_fileHeader.AlignmentScore = aErrorScore;

	if (aX)
	{
		memcpy(aX, x, GetMarkerCount() * sizeof(float));
	}
	if (aY)
	{
		memcpy(aY, y, GetMarkerCount() * sizeof(float));
	}
	if (aZ)
	{
		memcpy(aZ, z, GetMarkerCount() * sizeof(float));
	}
	delete[] shiftsX;
	delete[] shiftsY;
	delete[] psis;
	delete[] thetas;
	delete[] mags;
	delete[] diffs;
	delete[] usedBeads;
	delete[] diffsum;
	delete[] projectedPos;
	delete[] projBead;/**/
}

void MarkerFile::AlignSimple(int aReferenceMarker, int imageWidth, int imageHeight, float& aErrorScore, float& aImageRotation, float* x, float* y, float* z, float addZShift)
{
	if (aReferenceMarker < 0 || aReferenceMarker >= GetMarkerCount())
		return;

	int imintilt = GetMinTiltIndex();
	float r[3];
	r[0] = (*this)(MFI_X_Coordinate, imintilt, aReferenceMarker);
	r[1] = (*this)(MFI_Y_Coordinate, imintilt, aReferenceMarker);
	r[2] = (imageWidth / 2.0f + 1.0f) + addZShift;

	MoveXYToMagDistort(r[0], r[1], mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imageWidth, imageHeight);

	float* meanx = new float[GetMarkerCount()];
	float* meany = new float[GetMarkerCount()];
	float* norm = new float[GetMarkerCount()];
	memset(meanx, 0, GetMarkerCount() * sizeof(float));
	memset(meany, 0, GetMarkerCount() * sizeof(float));
	memset(norm, 0, GetMarkerCount() * sizeof(float));

	for (int imark = 0; imark < GetMarkerCount(); imark++)
	{
		for (int itilt = 0; itilt < GetTotalProjectionCount(); itilt++)
		{
			if ((*this)(MFI_X_Coordinate, itilt, imark) > INVALID_MARKER_POSITION
				&& (*this)(MFI_Y_Coordinate, itilt, aReferenceMarker) > INVALID_MARKER_POSITION)
			{
				float x = (*this)(MFI_X_Coordinate, itilt, imark);
				float y = (*this)(MFI_Y_Coordinate, itilt, imark);
				float x2 = (*this)(MFI_X_Coordinate, itilt, aReferenceMarker);
				float y2 = (*this)(MFI_Y_Coordinate, itilt, aReferenceMarker);

				MoveXYToMagDistort(x, y, mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imageWidth, imageHeight);
				MoveXYToMagDistort(x2, y2, mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imageWidth, imageHeight);
				meanx[imark] += x - x2;
				meany[imark] += y - y2;
				norm[imark] += 1;
			}
		}
	}

	for (int imark = 0; imark < GetMarkerCount(); imark++)
	{
		meanx[imark] /= norm[imark];
		meany[imark] /= norm[imark];
	}

	float sumxx = 0;
	float sumyy = 0;
	float sumxy = 0;

	for (int imark = 0; imark < GetMarkerCount(); imark++)
	{
		for (int itilt = 0; itilt < GetTotalProjectionCount(); itilt++)
		{
			if ((*this)(MFI_X_Coordinate, itilt, imark) > INVALID_MARKER_POSITION
				&& (*this)(MFI_X_Coordinate, itilt, aReferenceMarker) > INVALID_MARKER_POSITION)
			{
				float x = (*this)(MFI_X_Coordinate, itilt, imark);
				float y = (*this)(MFI_Y_Coordinate, itilt, imark);
				float x2 = (*this)(MFI_X_Coordinate, itilt, aReferenceMarker);
				float y2 = (*this)(MFI_Y_Coordinate, itilt, aReferenceMarker);

				MoveXYToMagDistort(x, y, mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imageWidth, imageHeight);
				MoveXYToMagDistort(x2, y2, mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imageWidth, imageHeight);

				float tempX = x - x2 - meanx[imark];
				sumxx += tempX * tempX;

				float tempY = y - y2 - meany[imark];
				sumyy += tempY * tempY;

				sumxy += tempX * tempY;
			}
		}
	}

	double psi = 0.5 * atan(2.0 * sumxy / (sumxx - sumyy));
	double sign = signbit(psi) ? -1 : 1;
	if (sumxx > sumyy)
		psi = psi - 0.5 * M_PI * sign;

	aImageRotation = (float)(psi * 180 / M_PI);

	double cpsi = cos(psi);
	double spsi = sin(psi);

	double ndif = 0;
	sumxx = 0;

	for (int imark = 0; imark < GetMarkerCount(); imark++)
	{
		for (int itilt = 0; itilt < GetTotalProjectionCount(); itilt++)
		{
			if ((*this)(MFI_X_Coordinate, itilt, imark) > INVALID_MARKER_POSITION
				&& (*this)(MFI_X_Coordinate, itilt, aReferenceMarker) > INVALID_MARKER_POSITION)
			{
				if (imark != aReferenceMarker) ndif += 1;
				float x = (*this)(MFI_X_Coordinate, itilt, imark);
				float y = (*this)(MFI_Y_Coordinate, itilt, imark);
				float x2 = (*this)(MFI_X_Coordinate, itilt, aReferenceMarker);
				float y2 = (*this)(MFI_Y_Coordinate, itilt, aReferenceMarker);

				MoveXYToMagDistort(x, y, mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imageWidth, imageHeight);
				MoveXYToMagDistort(x2, y2, mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imageWidth, imageHeight);
				(*this)(MFI_DevOfMark, itilt, imark) = (float)
					((x - x2 - meanx[imark]) * cpsi +
					(y - y2 - meany[imark]) * spsi);
				sumxx += (*this)(MFI_DevOfMark, itilt, imark) * (*this)(MFI_DevOfMark, itilt, imark);
			}
		}
	}
	double sigma = sqrt(sumxx / (ndif - GetMarkerCount()));

	aErrorScore = (float)sigma;
		
	std::vector<float> wertung;
	std::vector<int> index;

	for (int i = 0; i < GetTotalProjectionCount(); i++)
	{
		wertung.push_back(0);
		for (int imark = 0; imark < GetMarkerCount(); imark++)
		{
			wertung[i] += fabs((*this)(MFI_DevOfMark, i, imark));
		}
		index.push_back(i);
	}


	for (int i = 0; i < GetTotalProjectionCount() - 1; i++)
	{
		for (int sort = 0; sort < GetTotalProjectionCount() - i - 1; sort++)
		{
			if (wertung[sort] > wertung[sort + 1])
			{
				int tempi = index[sort];
				float tempf = wertung[sort];

				index[sort] = index[sort + 1];
				wertung[sort] = wertung[sort + 1];
				index[sort + 1] = tempi;
				wertung[sort + 1] = tempf;
			}
		}
	}
	
	//2nd part: determination of shifts:
	float* theta = new float[GetTotalProjectionCount()];
	float* stheta = new float[GetTotalProjectionCount()];
	float* ctheta = new float[GetTotalProjectionCount()];
	
	for (int i = 0; i < GetTotalProjectionCount(); i++)
	{
		theta[i] = (float)((*this)(MFI_TiltAngle, i, 0) * M_PI / 180.0);
		stheta[i] = (float)sin(theta[i]);
		ctheta[i] = (float)cos(theta[i]);
	}
	
	for (int imark = 0; imark < GetMarkerCount(); imark++)
	{
		float P[3][3];
		sumxx = 0;
		sumyy = 0;
		sumxy = 0;
		float sumyx = 0;
		float salpsq = 0;
		float scalph = 0;
		norm[imark] = 0;
		float temp[3];

		for (int itilt = 0; itilt < GetTotalProjectionCount(); itilt++)
		{
			if ((*this)(MFI_X_Coordinate, itilt, imark) > INVALID_MARKER_POSITION
				&& (*this)(MFI_X_Coordinate, itilt, aReferenceMarker) > INVALID_MARKER_POSITION)
			{
				norm[imark] += 1;
				salpsq += stheta[itilt] * stheta[itilt];
				scalph += ctheta[itilt] * stheta[itilt];
				float x = (*this)(MFI_X_Coordinate, itilt, imark);
				float y = (*this)(MFI_Y_Coordinate, itilt, imark);
				float x2 = (*this)(MFI_X_Coordinate, itilt, aReferenceMarker);
				float y2 = (*this)(MFI_Y_Coordinate, itilt, aReferenceMarker);

				MoveXYToMagDistort(x, y, mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imageWidth, imageHeight);
				MoveXYToMagDistort(x2, y2, mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imageWidth, imageHeight);
				sumxx += (x - x2) * ctheta[itilt];
				sumyy += (y - y2) * ctheta[itilt];
				sumxy += (x - x2) * stheta[itilt];
				sumyx += (y - y2) * stheta[itilt];
			}
		}
		P[0][0] = norm[imark] - (float)(salpsq * spsi * spsi);
		P[0][1] = salpsq * (float)(cpsi * cpsi);
		P[0][2] = scalph * (float)spsi;
		P[1][0] = P[0][1];
		P[1][1] = norm[imark] - salpsq * (float)(cpsi * cpsi);
		P[1][2] = -scalph * (float)cpsi;
		P[2][0] = P[0][2];
		P[2][1] = P[1][2];
		P[2][2] = salpsq;
		float dt = det(P);

		temp[0] = (float)((sumxx * spsi - sumyy * cpsi) * spsi + (cpsi * meanx[imark] + spsi * meany[imark]) * cpsi * norm[imark]);
		temp[1] = (float)(-(sumxx * spsi - sumyy * cpsi) * cpsi + (cpsi * meanx[imark] + spsi * meany[imark]) * spsi * norm[imark]);
		temp[2] = (float)(sumxy * spsi - sumyx * cpsi);

		if (dt != 0)
		{
			//float[, ] P_t = copy(P);
			float P_t[3][3];
			memcpy(P_t, P, 9 * sizeof(float));
			P_t[0][0] = temp[0];
			P_t[1][0] = temp[1];
			P_t[2][0] = temp[2];
			x[imark] = det(P_t) / dt;

			//P_t = copy(P);
			memcpy(P_t, P, 9 * sizeof(float));
			P_t[0][1] = temp[0];
			P_t[1][1] = temp[1];
			P_t[2][1] = temp[2];
			y[imark] = det(P_t) / dt;

			//P_t = copy(P);
			memcpy(P_t, P, 9 * sizeof(float));
			P_t[0][2] = temp[0];
			P_t[1][2] = temp[1];
			P_t[2][2] = temp[2];
			z[imark] = det(P_t) / dt;

			x[imark] += r[0] - imageWidth / 2 + 1;
            y[imark] += r[1] - imageHeight / 2 + 1;
			z[imark] += r[2] - imageWidth / 2 + 1;
		}
	}

	for (int itilt = 0; itilt < GetTotalProjectionCount(); itilt++)
	{
		for (int imark = 0; imark < GetMarkerCount(); imark++)
		{
			float shiftX, shiftY;
			shiftX = (float)(x[imark] * (spsi * spsi * ctheta[itilt] + cpsi * cpsi) +
				y[imark] * spsi * cpsi * (1 - ctheta[itilt]) +
				z[imark] * spsi * stheta[itilt] + imageWidth / 2 + 1);

			shiftY = (float)(x[imark] * spsi * cpsi * (1 - ctheta[itilt]) +
				y[imark] * (cpsi * cpsi * ctheta[itilt] + spsi * spsi) -
                z[imark] * cpsi * stheta[itilt] + imageHeight / 2 + 1);
			float x = (*this)(MFI_X_Coordinate, itilt, imark);
			float y = (*this)(MFI_Y_Coordinate, itilt, imark);

			MoveXYToMagDistort(x, y, mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imageWidth, imageHeight);

            (*this)(MFI_X_Shift, itilt, imark) = x - shiftX;
            (*this)(MFI_Y_Shift, itilt, imark) = y - shiftY;
			(*this)(MFI_RotationPsi, itilt, imark) = aImageRotation;
		}
	}

	/*float XShiftMin = (*this)(MFI_X_Shift, GetMinTiltIndex(), aReferenceMarker);
	float YShiftMin = (*this)(MFI_Y_Shift, GetMinTiltIndex(), aReferenceMarker);*/

//        //Minimize shift X:
//        float meanXShift = 0;
//        for (int itilt = 0; itilt < GetTotalProjectionCount(); itilt++)
//        {
//                meanXShift += (*this)(MFI_X_Shift, itilt, aReferenceMarker);
//        }
//        meanXShift /= GetTotalProjectionCount();
        /*for (int imark = 0; imark < GetMarkerCount(); imark++)
        {
                x[imark] += XShiftMin;
        }
        for (int itilt = 0; itilt < GetTotalProjectionCount(); itilt++)
        {
                for (int imark = 0; imark < GetMarkerCount(); imark++)
                {
                        (*this)(MFI_X_Shift, itilt, imark) -= XShiftMin;
                }
        }*/
//        //Minimize shift Y:
//        float meanYShift = 0;
//        for (int itilt = 0; itilt < GetTotalProjectionCount(); itilt++)
//        {
//                meanYShift += (*this)(MFI_Y_Shift, itilt, aReferenceMarker);
//        }
//        meanYShift /= GetTotalProjectionCount();
        /*for (int imark = 0; imark < GetMarkerCount(); imark++)
        {
                y[imark] += YShiftMin;
        }
        for (int itilt = 0; itilt < GetTotalProjectionCount(); itilt++)
        {
                for (int imark = 0; imark < GetMarkerCount(); imark++)
                {
                        (*this)(MFI_Y_Shift, itilt, imark) -= YShiftMin;
                }
        }*/



    for (int imark = 0; imark < GetMarkerCount(); imark++)
    {
            z[imark] = -z[imark];
    }


	delete[] meanx;
	delete[] meany;
	delete[] norm;
	delete[] theta;
	delete[] stheta;
	delete[] ctheta;
}

int MarkerFile::GetMinTiltIndex()
{
	int mintiltIndex = -1;
	float mintilt = FLT_MAX;

	for (int i = 0; i < GetTotalProjectionCount(); i++)
	{
		if (fabs((*this)(MFI_TiltAngle, i, 0)) < mintilt)
		{
			mintiltIndex = i;
			mintilt = fabs((*this)(MFI_TiltAngle, i, 0));
		}
	}
	return mintiltIndex;
}

bool MarkerFile::Save(string aFileName)
{
	_fileHeader.IsNewHeaderFormat = 1;
	_fileHeader.MarkerOffset = sizeof(_fileHeader) + _fileHeader.DimX * _fileHeader.DimY * _fileHeader.DimZ * sizeof(float);
	FileWriter::mFileName = aFileName;

	bool check = EmFile::CanWriteAsEM(_fileHeader.DimX, _fileHeader.DimY, _fileHeader.DimZ, GetDataType());
	if (!check)
		return false;

	check &= InitHeader(aFileName, _fileHeader);
	check &= WriteRawData(aFileName, _data, GetDataSize());

	check &= OpenWrite(false);
	FileWriter::Seek(_fileHeader.MarkerOffset, ios::beg);
	for (size_t i = 0; i < GetMarkerCount(); i++)
	{
		WriteLE(mMarkerXPos[i]);
		WriteLE(mMarkerYPos[i]);
		WriteLE(mMarkerZPos[i]);
	}
	CloseWrite();
	return check;
}

void MarkerFile::SetMagAnisotropy(float aAmount, float angInDeg, float dimX, float dimY)
{
	_fileHeader.MagAnisotropyAngle = angInDeg;
	_fileHeader.MagAnisotropyFactor = aAmount;

	mMagAnisotropyAmount = aAmount;
	mMagAnisotropyAngle = angInDeg / 180.0f * M_PI;

	Matrix<float> shiftCenter(3, 3);
	Matrix<float> shiftBack(3, 3);
	Matrix<float> rotMat1 = Matrix<float>::GetRotationMatrix3DZ(mMagAnisotropyAngle);
	Matrix<float> rotMat2 = Matrix<float>::GetRotationMatrix3DZ(-mMagAnisotropyAngle);
	Matrix<float> stretch(3, 3);
	shiftCenter(0, 0) = 1;
	shiftCenter(0, 1) = 0;
	shiftCenter(0, 2) = -dimX / 2.0f;
	shiftCenter(1, 0) = 0;
	shiftCenter(1, 1) = 1;
	shiftCenter(1, 2) = -dimY / 2.0f;
	shiftCenter(2, 0) = 0;
	shiftCenter(2, 1) = 0;
	shiftCenter(2, 2) = 1;

	shiftBack(0, 0) = 1;
	shiftBack(0, 1) = 0;
	shiftBack(0, 2) = dimX / 2.0f;
	shiftBack(1, 0) = 0;
	shiftBack(1, 1) = 1;
	shiftBack(1, 2) = dimY / 2.0f;
	shiftBack(2, 0) = 0;
	shiftBack(2, 1) = 0;
	shiftBack(2, 2) = 1;

	stretch(0, 0) = aAmount;
	stretch(0, 1) = 0;
	stretch(0, 2) = 0;
	stretch(1, 0) = 0;
	stretch(1, 1) = 1;
	stretch(1, 2) = 0;
	stretch(2, 0) = 0;
	stretch(2, 1) = 0;
	stretch(2, 2) = 1;

	mMagAnisotropy = shiftBack * rotMat2 * stretch * rotMat1 * shiftCenter;
}

void MarkerFile::GetMagAnisotropy(float & aAmount, float & angInDeg)
{
	aAmount = 1.0f;
	angInDeg = 0;

	if (_fileHeader.IsNewHeaderFormat)
	{
		aAmount = _fileHeader.MagAnisotropyFactor;
		angInDeg = _fileHeader.MagAnisotropyAngle;
	}
}

void MarkerFile::GetBeamDeclination(float & aPhi)
{
	aPhi = 0;

	if (_fileHeader.IsNewHeaderFormat)
	{
		aPhi = _fileHeader.BeamDeclination;
	}
}

void MarkerFile::projectTestBeads(int imdim, int markerCount, float * x, float * y, float * z, int projectionCount, float * thetas, float * psis, float phi, float * mags, float * shiftX, float * shiftY, float magAniso, float ** posX, float ** posY)
{
	Matrix<float> shiftCenter(3, 3);
	Matrix<float> shiftBack(3, 3);
	Matrix<float> rotMat1 = Matrix<float>::GetRotationMatrix3DZ(43.0f / 180.0f * M_PI);
	Matrix<float> rotMat2 = Matrix<float>::GetRotationMatrix3DZ(-43.0f / 180.0f * M_PI);
	Matrix<float> stretch(3, 3);
	shiftCenter(0, 0) = 1;
	shiftCenter(0, 1) = 0;
	shiftCenter(0, 2) = -imdim / 2.0f;
	shiftCenter(1, 0) = 0;
	shiftCenter(1, 1) = 1;
	shiftCenter(1, 2) = -imdim / 2.0f;
	shiftCenter(2, 0) = 0;
	shiftCenter(2, 1) = 0;
	shiftCenter(2, 2) = 1;

	shiftBack(0, 0) = 1;
	shiftBack(0, 1) = 0;
	shiftBack(0, 2) = imdim / 2.0f;
	shiftBack(1, 0) = 0;
	shiftBack(1, 1) = 1;
	shiftBack(1, 2) = imdim / 2.0f;
	shiftBack(2, 0) = 0;
	shiftBack(2, 1) = 0;
	shiftBack(2, 2) = 1;

	stretch(0, 0) = magAniso;
	stretch(0, 1) = 0;
	stretch(0, 2) = 0;
	stretch(1, 0) = 0;
	stretch(1, 1) = 1;
	stretch(1, 2) = 0;
	stretch(2, 0) = 0;
	stretch(2, 1) = 0;
	stretch(2, 2) = 1;

	Matrix<float> magAnisotropy = shiftBack * rotMat2 * stretch * rotMat1 * shiftCenter;


	float* res = new float[markerCount * projectionCount * 2];
	projectBeads(x, y, z, thetas, psis, mags, phi, shiftX, shiftY, imdim, imdim, res, projectionCount, markerCount);

	for (size_t marker = 0; marker < markerCount; marker++)
	{
		for (size_t proj = 0; proj < projectionCount; proj++)
		{
			posX[proj][marker] = res[GetBeadIndex(proj, marker, 0, markerCount)];
			posY[proj][marker] = res[GetBeadIndex(proj, marker, 1, markerCount)];

			MoveXYToMagDistort(posX[proj][marker], posY[proj][marker], magAniso, 43.0f * M_PI / 180.0f, magAnisotropy, imdim, imdim);
		}
	}


}

string MarkerFile::GetFilename()
{
	return FileWriter::mFileName;
}

void MarkerFile::GetMarkerPositions(float * xPos, float * yPos, float * zPos)
{
	if (_fileHeader.IsNewHeaderFormat)
	{
		for (int i = 0; i < GetMarkerCount(); i++)
		{
			xPos[i] = mMarkerXPos[i];
			yPos[i] = mMarkerYPos[i];
			zPos[i] = mMarkerZPos[i];
		}
	}
	else
	{
		for (int i = 0; i < GetMarkerCount(); i++)
		{
			xPos[i] = 0;
			yPos[i] = 0;
			zPos[i] = 0;
		}
	}
}

void MarkerFile::DeleteAllMarkersInProjection(int idx)
{
	if (idx < 0 || idx >= GetTotalProjectionCount())
		return;

	for (int imark = 0; imark < GetMarkerCount(); imark++)
	{
		(*this)(MFI_X_Coordinate, idx, imark) = INVALID_MARKER_POSITION;
		(*this)(MFI_Y_Coordinate, idx, imark) = INVALID_MARKER_POSITION;
	}
}

MarkerFile * MarkerFile::ImportFromIMOD(string aFilename)
{
	ImodFiducialFile imod(aFilename);
	if (!imod.OpenAndRead())
		return NULL;

	int projCount = imod.GetProjectionCount();
	vector<float> tilts = imod.GetTiltAngles();

	MarkerFile* m = new MarkerFile(projCount, &tilts[0]);
	for (size_t i = 1; i < imod.GetMarkerCount(); i++) //first marker is already there...
	{
		m->AddMarker();
	}

	for (size_t p = 0; p < imod.GetProjectionCount(); p++)
	{
		vector<PointF> pos = imod.GetMarkers(p);
		for (size_t i = 0; i < imod.GetMarkerCount(); i++)
		{
			(*m)(MFI_X_Coordinate, p, i) = pos[i].x;
			(*m)(MFI_Y_Coordinate, p, i) = pos[i].y;
		}
	}
	
	return m;
}

float MarkerFile::det(float P[3][3])
{
	return P[0][0] * P[1][1] * P[2][2] + P[0][1] * P[1][2] * P[2][0] + P[0][2] * P[1][0] * P[2][1] - P[0][2] * P[1][1] * P[2][0] - P[0][1] * P[1][0] * P[2][2] - P[0][0] * P[1][2] * P[2][1];
}

float& MarkerFile::operator() (const MarkerFileItem_enum aItem, const int aProjection, const int aMarker)
{
	float* fdata = (float*)_data;

	return fdata[aMarker * _fileHeader.DimX * _fileHeader.DimY + aProjection * _fileHeader.DimX + aItem];
}

void MarkerFile::Pack(float* vars, float* x, float* y, float* z, float* shiftX, float* shiftY, int ProjectionCount, int MarkerCount)
{
	/*size_t MarkerCount = GetMarkerCount();
	size_t ProjectionCount = GetTotalProjectionCount();*/

	memcpy(vars + 0 * MarkerCount, x, MarkerCount * sizeof(float));
	memcpy(vars + 1 * MarkerCount, y, MarkerCount * sizeof(float));
	memcpy(vars + 2 * MarkerCount, z, MarkerCount * sizeof(float));
	memcpy(vars + 3 * MarkerCount, shiftX, ProjectionCount * sizeof(float));
	memcpy(vars + 3 * MarkerCount + ProjectionCount, shiftY, ProjectionCount * sizeof(float));
}

void MarkerFile::UnPack(float* vars, float* x, float* y, float* z, float* shiftX, float* shiftY, int ProjectionCount, int MarkerCount)
{
	/*size_t MarkerCount = GetMarkerCount();
	size_t ProjectionCount = GetTotalProjectionCount();*/

	memcpy(x, vars + 0 * MarkerCount, MarkerCount * sizeof(float));
	memcpy(y, vars + 1 * MarkerCount, MarkerCount * sizeof(float));
	memcpy(z, vars + 2 * MarkerCount, MarkerCount * sizeof(float));
	memcpy(shiftX, vars + 3 * MarkerCount, ProjectionCount * sizeof(float));
	memcpy(shiftY, vars + 3 * MarkerCount + ProjectionCount, ProjectionCount * sizeof(float));
}

void MarkerFile::Pack(float* vars, float* x, float* y, float* z, float* ang, int ProjectionCount, int MarkerCount)
{
	/*size_t MarkerCount = GetMarkerCount();
	size_t ProjectionCount = GetTotalProjectionCount();*/

	memcpy(vars + 0 * MarkerCount, x, MarkerCount * sizeof(float));
	memcpy(vars + 1 * MarkerCount, y, MarkerCount * sizeof(float));
	memcpy(vars + 2 * MarkerCount, z, MarkerCount * sizeof(float));
	memcpy(vars + 3 * MarkerCount, ang, ProjectionCount * sizeof(float));
}

void MarkerFile::UnPack(float* vars, float* x, float* y, float* z, float* ang, int ProjectionCount, int MarkerCount)
{
	/*size_t MarkerCount = GetMarkerCount();
	size_t ProjectionCount = GetTotalProjectionCount();*/

	memcpy(x, vars + 0 * MarkerCount, MarkerCount * sizeof(float));
	memcpy(y, vars + 1 * MarkerCount, MarkerCount * sizeof(float));
	memcpy(z, vars + 2 * MarkerCount, MarkerCount * sizeof(float));
	memcpy(ang, vars + 3 * MarkerCount, ProjectionCount * sizeof(float));
}

int MarkerFile::GetBeadIndex(int proj, int bead, int xy, int MarkerCount)
{
	return proj * 2 * MarkerCount + bead * 2 + xy;
}

void MarkerFile::projectBeads(float* x, float* y, float* z, float* thetas, float* psis, float* mags, float phi, float* shiftX, float* shiftY, float imdimX, float imdimY, float* ret, int ProjectionCount, int MarkerCount)
{
	//float* ret = new float[ProjectionCount*MarkerCount * 2];

	//Parallel.For(0, ProjectionCount, projection = > 

	/*size_t ProjectionCount = GetTotalProjectionCount();
	size_t MarkerCount = GetMarkerCount();*/


#pragma omp parallel for
	for (int projection = 0; projection < ProjectionCount; projection++)
	{
		float cpsi = (float)cosf(psis[projection] / 180.0 * M_PI);
		float spsi = (float)sinf(psis[projection] / 180.0 * M_PI);

		float ctheta = (float)cosf(thetas[projection] / 180.0 * M_PI);
		float stheta = (float)sinf(thetas[projection] / 180.0 * M_PI);

		float cphi = (float)cosf(phi / 180.0 * M_PI);
		float sphi = (float)sinf(phi / 180.0 * M_PI);

		float shrink = mags[projection];

		Matrix<float> m(2, 3);
		//Original from Matlab:
		//m = [[(cphi^2 * cpsi + sphi * ( sphi * cpsi * ctheta + spsi * stheta))/shrink (sphi * cpsi * stheta - spsi * ctheta)/shrink (-(sphi * cpsi * (ctheta - 1) + spsi * stheta) * cphi)/shrink];
		//     [(cphi^2 * spsi + sphi * ( sphi * spsi * ctheta - cpsi * stheta))/shrink (sphi * spsi * stheta + cpsi * ctheta)/shrink (-(sphi * spsi * (ctheta - 1) - cpsi * stheta) * cphi)/shrink]];

		//Simplified without phi and without shrink:
		//m = [[(1 * cpsi ) (- spsi * ctheta) (-(+ spsi * stheta) * 1)];
		//     [(1 * spsi ) (+ cpsi * ctheta) (-(- cpsi * stheta) * 1)]];

		//m[0,0] = cpsi; m[0,1] = -spsi * ctheta; m[0,2] = -spsi * stheta;
		//m[1,0] = spsi; m[1,1] = +cpsi * ctheta; m[1,2] = cpsi * stheta;

		m(0, 0) = (cphi * cphi * cpsi + sphi * (sphi * cpsi * ctheta + spsi * stheta)) / shrink;
		m(0, 1) = (sphi * cpsi * stheta - spsi * ctheta) / shrink;
		m(0, 2) = (-(sphi * cpsi * (ctheta - 1) + spsi * stheta) * cphi) / shrink;

		m(1, 0) = (cphi * cphi * spsi + sphi * (sphi * spsi * ctheta - cpsi * stheta)) / shrink;
		m(1, 1) = (sphi * spsi * stheta + cpsi * ctheta) / shrink;
		m(1, 2) = (-(sphi * spsi * (ctheta - 1) - cpsi * stheta) * cphi) / shrink;


		for (int marker = 0; marker < MarkerCount; marker++)
		{
			Matrix<float> pos(3, 1);
			pos(0, 0) = x[marker];
			pos(1, 0) = y[marker];
			pos(2, 0) = z[marker];

			Matrix<float> mul = m * pos;

			ret[GetBeadIndex(projection, marker, 0, MarkerCount)] = mul(0, 0);
			ret[GetBeadIndex(projection, marker, 1, MarkerCount)] = mul(1, 0);

			ret[GetBeadIndex(projection, marker, 0, MarkerCount)] += shiftX[projection] + imdimX * 0.5f;
			ret[GetBeadIndex(projection, marker, 1, MarkerCount)] += shiftY[projection] + imdimY * 0.5f;
		}
	}

	//return ret;
}

float MarkerFile::ComputeRMS(float* positions, float* diffsum, float* diffs, int* usedMarkers, float* x, float* y, float* z, float* thetas, float* psis, float* mags, float phi, float* shiftX, float* shiftY, float imdimX, float imdimY, MarkerFile* markerfile, bool minimizeZeroShift)
{
	int ProjectionCount = markerfile->GetTotalProjectionCount();
	int MarkerCount = markerfile->GetMarkerCount();
	int minTiltIndex = markerfile->GetMinTiltIndex();
	//float* positions = projectBeads(x, y, z, thetas, psis, mags, phi, shiftX, shiftY, imdimX, imdimY);
	projectBeads(x, y, z, thetas, psis, mags, phi, shiftX, shiftY, imdimX, imdimY, positions, ProjectionCount, MarkerCount);
	//float* diffsum = new float[ProjectionCount];


	float MinShiftX = 0;
	float MinShiftY = 0;
	if (minimizeZeroShift)
	{
		MinShiftX = fabs((*markerfile)(MFI_X_Shift, minTiltIndex, 0));
		MinShiftY = fabs((*markerfile)(MFI_Y_Shift, minTiltIndex, 0));
	}




#pragma omp parallel for
	for (int projection = 0; projection < ProjectionCount; projection++)
	{
		diffsum[projection] = -1;
		int usedMarker = 0;
		
		for (int marker = 0; marker < MarkerCount; marker++)
		{
			diffs[GetBeadIndex(projection, marker, 0, MarkerCount)] = 0;
			diffs[GetBeadIndex(projection, marker, 1, MarkerCount)] = 0;
			if ((*markerfile)(MFI_X_Coordinate, projection, marker) > INVALID_MARKER_POSITION && (*markerfile)(MFI_Y_Coordinate, projection, marker) > INVALID_MARKER_POSITION)
			{
				usedMarker++;
				float x = (*markerfile)(MFI_X_Coordinate, projection, marker);
				float y = (*markerfile)(MFI_Y_Coordinate, projection, marker);

				MoveXYToMagDistort(x, y, markerfile->mMagAnisotropyAmount, markerfile->mMagAnisotropyAngle, markerfile->mMagAnisotropy, imdimX, imdimY);
				float diffX = x - positions[GetBeadIndex(projection, marker, 0, MarkerCount)];
				float diffY = y - positions[GetBeadIndex(projection, marker, 1, MarkerCount)];
				diffsum[projection] += diffX * diffX + diffY * diffY;
				if (projection == minTiltIndex)
				{
					diffs[GetBeadIndex(projection, marker, 0, MarkerCount)] = positions[GetBeadIndex(projection, marker, 0, MarkerCount)];
					diffs[GetBeadIndex(projection, marker, 1, MarkerCount)] = positions[GetBeadIndex(projection, marker, 1, MarkerCount)];
				}
				else
				{
					diffs[GetBeadIndex(projection, marker, 0, MarkerCount)] = positions[GetBeadIndex(projection, marker, 0, MarkerCount)];
					diffs[GetBeadIndex(projection, marker, 1, MarkerCount)] = positions[GetBeadIndex(projection, marker, 1, MarkerCount)];
				}
			}
		}		
		usedMarkers[projection] = usedMarker;
	}

	int usedmarker = 0;
	float rms = 0;

	for (int i = 0; i < ProjectionCount; i++)
	{
		if (diffsum[i] > 0)
			rms += diffsum[i];
		usedmarker += usedMarkers[i];
	}

	return rms / (float)usedmarker;
}

void MarkerFile::computeError(float* p, float* hx, int m, int n, void* adata)
{
	MinimeData* dat = (MinimeData*)adata;

	if (dat->Shift)
	{
		float* x = dat->xTemp;
		float* y = dat->yTemp;
		float* z = dat->zTemp;

		float* sx = dat->shiftXTemp;
		float* sy = dat->shiftYTemp;
		int* ub = dat->usedBeads;

		UnPack(p, x, y, z, sx, sy, dat->ProjectionCount, dat->MarkerCount);

		ComputeRMS(dat->postions, dat->diffsum, hx, ub, x, y, z, dat->thetas, dat->psis, dat->magnitudes, dat->phi, sx, sy, dat->imdimX, dat->imdimY, dat->markerfile, true);
	}
	else if (dat->Psi)
	{
		float* x = dat->xTemp;
		float* y = dat->yTemp;
		float* z = dat->zTemp;

		float* psi = dat->angTemp;
		int* ub = dat->usedBeads;

		if (dat->PsiFixed)
		{
			UnPack(p, x, y, z, psi, 1, dat->MarkerCount);
			for (size_t i = 1; i < dat->ProjectionCount; i++)
			{
				psi[i] = psi[0];
			}
		}
		else
		{
			UnPack(p, x, y, z, psi, dat->ProjectionCount, dat->MarkerCount);
		}

		ComputeRMS(dat->postions, dat->diffsum, hx, ub, x, y, z, dat->thetas, psi, dat->magnitudes, dat->phi, dat->shiftX, dat->shiftY, dat->imdimX, dat->imdimY, dat->markerfile);
	}
	else if (dat->Theta)
	{
		float* x = dat->xTemp;
		float* y = dat->yTemp;
		float* z = dat->zTemp;

		float* theta = dat->angTemp;
		int* ub = dat->usedBeads;

		UnPack(p, x, y, z, theta, dat->ProjectionCount, dat->MarkerCount);

		ComputeRMS(dat->postions, dat->diffsum, hx, ub, x, y, z, theta, dat->psis, dat->magnitudes, dat->phi, dat->shiftX, dat->shiftY, dat->imdimX, dat->imdimY, dat->markerfile);
	}
	else if (dat->Phi)
	{
		float* x = dat->xTemp;
		float* y = dat->yTemp;
		float* z = dat->zTemp;

		float* phi = new float[1];
		int* ub = dat->usedBeads;

		UnPack(p, x, y, z, phi, 1, dat->MarkerCount);

		ComputeRMS(dat->postions, dat->diffsum, hx, ub, x, y, z, dat->thetas, dat->psis, dat->magnitudes, phi[0], dat->shiftX, dat->shiftY, dat->imdimX, dat->imdimY, dat->markerfile);
		delete[] phi;
	}
	else if (dat->Magnitude)
	{
		float* x = dat->xTemp;
		float* y = dat->yTemp;
		float* z = dat->zTemp;

		float* mags = dat->angTemp;
		int* ub = dat->usedBeads;

		UnPack(p, x, y, z, mags, dat->ProjectionCount, dat->MarkerCount);

		ComputeRMS(dat->postions, dat->diffsum, hx, ub, x, y, z, dat->thetas, dat->psis, mags, dat->phi, dat->shiftX, dat->shiftY, dat->imdimX, dat->imdimY, dat->markerfile);
	}
}

void MarkerFile::SetLevmarOptions(float options[5])
{
	options[0] = LM_INIT_MU;
	options[1] = options[2] = options[3] = LM_STOP_THRESH;
	options[4] = LM_DIFF_DELTA;
}

MarkerFile::MinimeData::MinimeData(int aProjectionCount, int aMarkerCount):
	ProjectionCount(aProjectionCount),
	MarkerCount(aMarkerCount),
	usedBeads(new int[aProjectionCount]),
	xTemp(new float[aMarkerCount]),
	yTemp(new float[aMarkerCount]),
	zTemp(new float[aMarkerCount]),
	shiftXTemp(new float[aProjectionCount]),
	shiftYTemp(new float[aProjectionCount]),
	angTemp(new float[aProjectionCount]),
	diffsum(new float[aProjectionCount]),
	postions(new float[aProjectionCount*aMarkerCount * 2])
{
}

MarkerFile::MinimeData::~MinimeData()
{
	delete[] usedBeads;
	delete[] xTemp;
	delete[] yTemp;
	delete[] zTemp;
	delete[] shiftXTemp;
	delete[] shiftYTemp;
	delete[] angTemp;
	delete[] diffsum;
	delete[] postions;
}


void MarkerFile::MinimizeShift(float* x, float* y, float* z, float* thetas, float* psis, float* mags, float phi, float* shiftX, float* shiftY, float imdimX, float imdimY)
{
	int MarkerCount = GetMarkerCount();
	int ProjectionCount = GetTotalProjectionCount();
	float* vars = new float[MarkerCount * 3 + ProjectionCount * 2];
	Pack(vars, x, y, z, shiftX, shiftY, ProjectionCount, MarkerCount);
	//float* xT = new float[x.Length];
	//float* yT = new float[y.Length];
	//float* zT = new float[z.Length];
	//float* shiftXT = new float[shiftX.Length];
	//float* shiftYT = new float[shiftY.Length];
	//UnPack(vars, x, y, z, shiftXT, shiftYT);


	//LevMarMinimzation.func_del del = computeError;
	float info[LM_INFO_SZ];
	float opt[5];
	SetLevmarOptions(opt);
	//LevMarMinimzation.ResultInfo info = new LevMarMinimzation.ResultInfo();
	//LevMarMinimzation.MinOptions opt = new LevMarMinimzation.MinOptions();

	MinimeData dat(ProjectionCount, MarkerCount);
	dat.imdimX = imdimX;
	dat.imdimY = imdimY;
	dat.psis = psis;
	dat.shiftX = shiftX;
	dat.shiftY = shiftY;
	dat.thetas = thetas;
	dat.magnitudes = mags;
	dat.phi = phi;
	dat.x = x;
	dat.y = y;
	dat.z = z;

	/*dat.usedBeads = new int[ProjectionCount];
	dat.xTemp = new float[MarkerCount];
	dat.yTemp = new float[MarkerCount];
	dat.zTemp = new float[MarkerCount];
	dat.shiftXTemp = new float[ProjectionCount];
	dat.shiftYTemp = new float[ProjectionCount];
	dat.angTemp = new float[ProjectionCount];*/
	dat.Shift = true;
	dat.Psi = false;
	dat.PsiFixed = false;
	dat.Theta = false;
	dat.Magnitude = false;
	dat.Phi = false;
	dat.markerfile = this;

	float* measurements = new float[MarkerCount * ProjectionCount * 2];
	for (int proj = 0; proj < ProjectionCount; proj++)
	{
		for (int marker = 0; marker < MarkerCount; marker++)
		{
			measurements[GetBeadIndex(proj, marker, 0, MarkerCount)] = (*this)(MFI_X_Coordinate, proj, marker);
			measurements[GetBeadIndex(proj, marker, 1, MarkerCount)] = (*this)(MFI_Y_Coordinate, proj, marker);
			MoveXYToMagDistort(measurements[GetBeadIndex(proj, marker, 0, MarkerCount)], measurements[GetBeadIndex(proj, marker, 1, MarkerCount)], mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imdimX, imdimY);
		}
	}

	LEVMAR_DIF(&computeError, vars, measurements, MarkerCount * 3 + ProjectionCount * 2, MarkerCount * ProjectionCount * 2, 1000, opt, info, NULL, NULL, &dat);
	//LevMarMinimzation.LevMar_DIF(del, vars, measurements, MarkerCount * 3 + ProjectionCount * 2, MarkerCount * ProjectionCount * 2, 1000, opt, info, null, dat);

	UnPack(vars, x, y, z, shiftX, shiftY, ProjectionCount, MarkerCount);
	delete[] vars;
	delete[] measurements;
}

void MarkerFile::MinimizePsis(float* x, float* y, float* z, float* thetas, float* psis, float* mags, float phi, float* shiftX, float* shiftY, float imdimX, float imdimY)
{
	int MarkerCount = GetMarkerCount();
	int ProjectionCount = GetTotalProjectionCount();
	float* vars = new float[MarkerCount * 3 + ProjectionCount];
	Pack(vars, x, y, z, psis, ProjectionCount, MarkerCount);


	//LevMarMinimzation.func_del del = computeError;
	float info[LM_INFO_SZ];
	float opt[5];
	SetLevmarOptions(opt);
	//LevMarMinimzation.ResultInfo info = new LevMarMinimzation.ResultInfo();
	//LevMarMinimzation.MinOptions opt = new LevMarMinimzation.MinOptions();

	MinimeData dat(ProjectionCount, MarkerCount);
	dat.imdimX = imdimX;
	dat.imdimY = imdimY;
	dat.psis = psis;
	dat.shiftX = shiftX;
	dat.shiftY = shiftY;
	dat.thetas = thetas;
	dat.magnitudes = mags;
	dat.phi = phi;
	dat.x = x;
	dat.y = y;
	dat.z = z;

	/*dat.usedBeads = new int[ProjectionCount];
	dat.xTemp = new float[MarkerCount];
	dat.yTemp = new float[MarkerCount];
	dat.zTemp = new float[MarkerCount];
	dat.shiftXTemp = new float[ProjectionCount];
	dat.shiftYTemp = new float[ProjectionCount];
	dat.angTemp = new float[ProjectionCount];*/
	dat.Shift = false;
	dat.Psi = true;
	dat.PsiFixed = false;
	dat.Theta = false;
	dat.Magnitude = false;
	dat.Phi = false;
	dat.markerfile = this;

	float* measurements = new float[MarkerCount * ProjectionCount * 2];
	for (int proj = 0; proj < ProjectionCount; proj++)
	{
		for (int marker = 0; marker < MarkerCount; marker++)
		{
			measurements[GetBeadIndex(proj, marker, 0, MarkerCount)] = (*this)(MFI_X_Coordinate, proj, marker);
			measurements[GetBeadIndex(proj, marker, 1, MarkerCount)] = (*this)(MFI_Y_Coordinate, proj, marker);
			MoveXYToMagDistort(measurements[GetBeadIndex(proj, marker, 0, MarkerCount)], measurements[GetBeadIndex(proj, marker, 1, MarkerCount)], mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imdimX, imdimY);
		}
	}

	//LevMarMinimzation.LevMar_DIF(del, vars, measurements, MarkerCount * 3 + ProjectionCount, MarkerCount * ProjectionCount * 2, 1000, opt, info, null, dat);
	LEVMAR_DIF(&computeError, vars, measurements, MarkerCount * 3 + ProjectionCount, MarkerCount * ProjectionCount * 2, 1000, opt, info, NULL, NULL, &dat);

	UnPack(vars, x, y, z, psis, ProjectionCount, MarkerCount);
	delete[] vars;
	delete[] measurements;
}

void MarkerFile::MinimizePsiFixed(float* x, float* y, float* z, float* thetas, float* psis, float* mags, float phi, float* shiftX, float* shiftY, float imdimX, float imdimY)
{
	int MarkerCount = GetMarkerCount();
	int ProjectionCount = GetTotalProjectionCount();
	float* vars = new float[MarkerCount * 3 + 1];
	Pack(vars, x, y, z, psis, 1, MarkerCount);

	float info[LM_INFO_SZ];
	float opt[5];
	SetLevmarOptions(opt);

	MinimeData dat(ProjectionCount, MarkerCount);
	dat.imdimX = imdimX;
	dat.imdimY = imdimY;
	dat.psis = psis;
	dat.shiftX = shiftX;
	dat.shiftY = shiftY;
	dat.thetas = thetas;
	dat.magnitudes = mags;
	dat.phi = phi;
	dat.x = x;
	dat.y = y;
	dat.z = z;

	dat.Shift = false;
	dat.Psi = true;
	dat.PsiFixed = true;
	dat.Theta = false;
	dat.Magnitude = false;
	dat.Phi = false;
	dat.markerfile = this;

	float* measurements = new float[MarkerCount * ProjectionCount * 2];
	for (int proj = 0; proj < ProjectionCount; proj++)
	{
		for (int marker = 0; marker < MarkerCount; marker++)
		{
			measurements[GetBeadIndex(proj, marker, 0, MarkerCount)] = (*this)(MFI_X_Coordinate, proj, marker);
			measurements[GetBeadIndex(proj, marker, 1, MarkerCount)] = (*this)(MFI_Y_Coordinate, proj, marker);
			MoveXYToMagDistort(measurements[GetBeadIndex(proj, marker, 0, MarkerCount)], measurements[GetBeadIndex(proj, marker, 1, MarkerCount)], mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imdimX, imdimY);
		}
	}

	LEVMAR_DIF(&computeError, vars, measurements, MarkerCount * 3 + 1, MarkerCount * ProjectionCount * 2, 1000, opt, info, NULL, NULL, &dat);

	UnPack(vars, x, y, z, psis, 1, MarkerCount);
	//extend first value to all projections
	for (size_t i = 1; i < ProjectionCount; i++)
	{
		psis[i] = psis[0];
	}

	delete[] vars;
	delete[] measurements;
}

void MarkerFile::MinimizeThetas(float* x, float* y, float* z, float* thetas, float* psis, float* mags, float phi, float* shiftX, float* shiftY, float imdimX, float imdimY)
{
	int MarkerCount = GetMarkerCount();
	int ProjectionCount = GetTotalProjectionCount();
	float* vars = new float[MarkerCount * 3 + ProjectionCount];
	Pack(vars, x, y, z, thetas, ProjectionCount, MarkerCount);


	//LevMarMinimzation.func_del del = computeError;
	float info[LM_INFO_SZ];
	float opt[5];
	SetLevmarOptions(opt);
	//LevMarMinimzation.ResultInfo info = new LevMarMinimzation.ResultInfo();
	//LevMarMinimzation.MinOptions opt = new LevMarMinimzation.MinOptions();

	MinimeData dat(ProjectionCount, MarkerCount);
	dat.imdimX = imdimX;
	dat.imdimY = imdimY;
	dat.psis = psis;
	dat.shiftX = shiftX;
	dat.shiftY = shiftY;
	dat.thetas = thetas;
	dat.magnitudes = mags;
	dat.phi = phi;
	dat.x = x;
	dat.y = y;
	dat.z = z;

	/*dat.usedBeads = new int[ProjectionCount];
	dat.xTemp = new float[MarkerCount];
	dat.yTemp = new float[MarkerCount];
	dat.zTemp = new float[MarkerCount];
	dat.shiftXTemp = new float[ProjectionCount];
	dat.shiftYTemp = new float[ProjectionCount];
	dat.angTemp = new float[ProjectionCount];*/
	dat.Shift = false;
	dat.Psi = false;
	dat.PsiFixed = false;
	dat.Theta = true;
	dat.Magnitude = false;
	dat.Phi = false;
	dat.markerfile = this;

	float* measurements = new float[MarkerCount * ProjectionCount * 2];
	for (int proj = 0; proj < ProjectionCount; proj++)
	{
		for (int marker = 0; marker < MarkerCount; marker++)
		{
			measurements[GetBeadIndex(proj, marker, 0, MarkerCount)] = (*this)(MFI_X_Coordinate, proj, marker);
			measurements[GetBeadIndex(proj, marker, 1, MarkerCount)] = (*this)(MFI_Y_Coordinate, proj, marker);
			MoveXYToMagDistort(measurements[GetBeadIndex(proj, marker, 0, MarkerCount)], measurements[GetBeadIndex(proj, marker, 1, MarkerCount)], mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imdimX, imdimY);
		}
	}

	//LevMarMinimzation.LevMar_DIF(del, vars, measurements, MarkerCount * 3 + ProjectionCount, MarkerCount * ProjectionCount * 2, 1000, opt, info, null, dat);
	LEVMAR_DIF(&computeError, vars, measurements, MarkerCount * 3 + ProjectionCount, MarkerCount * ProjectionCount * 2, 1000, opt, info, NULL, NULL, &dat);

	UnPack(vars, x, y, z, thetas, ProjectionCount, MarkerCount);
	delete[] vars;
	delete[] measurements;
}

void MarkerFile::MinimizeMags(float* x, float* y, float* z, float* thetas, float* psis, float* mags, float phi, float* shiftX, float* shiftY, float imdimX, float imdimY)
{
	int MarkerCount = GetMarkerCount();
	int ProjectionCount = GetTotalProjectionCount();
	float* vars = new float[MarkerCount * 3 + ProjectionCount];
	Pack(vars, x, y, z, mags, ProjectionCount, MarkerCount);


	//LevMarMinimzation.func_del del = computeError;
	float info[LM_INFO_SZ];
	float opt[5];
	SetLevmarOptions(opt);
	//LevMarMinimzation.ResultInfo info = new LevMarMinimzation.ResultInfo();
	//LevMarMinimzation.MinOptions opt = new LevMarMinimzation.MinOptions();

	MinimeData dat(ProjectionCount, MarkerCount);
	dat.imdimX = imdimX;
	dat.imdimY = imdimY;
	dat.psis = psis;
	dat.shiftX = shiftX;
	dat.shiftY = shiftY;
	dat.thetas = thetas;
	dat.magnitudes = mags;
	dat.phi = phi;
	dat.x = x;
	dat.y = y;
	dat.z = z;

	/*dat.usedBeads = new int[ProjectionCount];
	dat.xTemp = new float[MarkerCount];
	dat.yTemp = new float[MarkerCount];
	dat.zTemp = new float[MarkerCount];
	dat.shiftXTemp = new float[ProjectionCount];
	dat.shiftYTemp = new float[ProjectionCount];
	dat.angTemp = new float[ProjectionCount];*/
	dat.Shift = false;
	dat.Psi = false;
	dat.PsiFixed = false;
	dat.Theta = false;
	dat.Magnitude = true;
	dat.Phi = false;
	dat.markerfile = this;

	float* measurements = new float[MarkerCount * ProjectionCount * 2];
	for (int proj = 0; proj < ProjectionCount; proj++)
	{
		for (int marker = 0; marker < MarkerCount; marker++)
		{
			measurements[GetBeadIndex(proj, marker, 0, MarkerCount)] = (*this)(MFI_X_Coordinate, proj, marker);
			measurements[GetBeadIndex(proj, marker, 1, MarkerCount)] = (*this)(MFI_Y_Coordinate, proj, marker);
			MoveXYToMagDistort(measurements[GetBeadIndex(proj, marker, 0, MarkerCount)], measurements[GetBeadIndex(proj, marker, 1, MarkerCount)], mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imdimX, imdimY);
		}
	}

	//LevMarMinimzation.LevMar_DIF(del, vars, measurements, MarkerCount * 3 + ProjectionCount, MarkerCount * ProjectionCount * 2, 1000, opt, info, null, dat);
	LEVMAR_DIF(&computeError, vars, measurements, MarkerCount * 3 + ProjectionCount, MarkerCount * ProjectionCount * 2, 1000, opt, info, NULL, NULL, &dat);

	UnPack(vars, x, y, z, mags, ProjectionCount, MarkerCount);
	delete[] vars;
	delete[] measurements;
}

void MarkerFile::MinimizePhi(float* x, float* y, float* z, float* thetas, float* psis, float* mags, float* phi, float* shiftX, float* shiftY, float imdimX, float imdimY)
{
	int MarkerCount = GetMarkerCount();
	int ProjectionCount = GetTotalProjectionCount();
	float* vars = new float[MarkerCount * 3 + 1];
	Pack(vars, x, y, z, phi, 1, MarkerCount);


	float info[LM_INFO_SZ];
	float opt[5];
	SetLevmarOptions(opt);

	MinimeData dat(ProjectionCount, MarkerCount);
	dat.imdimX = imdimX;
	dat.imdimY = imdimY;
	dat.psis = psis;
	dat.shiftX = shiftX;
	dat.shiftY = shiftY;
	dat.thetas = thetas;
	dat.magnitudes = mags;
	dat.phi = phi[0];
	dat.x = x;
	dat.y = y;
	dat.z = z;

	dat.Shift = false;
	dat.Psi = false;
	dat.PsiFixed = false;
	dat.Theta = false;
	dat.Magnitude = false;
	dat.Phi = true;
	dat.markerfile = this;

	float* measurements = new float[MarkerCount * ProjectionCount * 2];
	for (int proj = 0; proj < ProjectionCount; proj++)
	{
		for (int marker = 0; marker < MarkerCount; marker++)
		{
			measurements[GetBeadIndex(proj, marker, 0, MarkerCount)] = (*this)(MFI_X_Coordinate, proj, marker);
			measurements[GetBeadIndex(proj, marker, 1, MarkerCount)] = (*this)(MFI_Y_Coordinate, proj, marker);
			MoveXYToMagDistort(measurements[GetBeadIndex(proj, marker, 0, MarkerCount)], measurements[GetBeadIndex(proj, marker, 1, MarkerCount)], mMagAnisotropyAmount, mMagAnisotropyAngle, mMagAnisotropy, imdimX, imdimY);
		}
	}

	//LevMarMinimzation.LevMar_DIF(del, vars, measurements, MarkerCount * 3 + 1, MarkerCount * ProjectionCount * 2, 1000, opt, info, null, dat);
	LEVMAR_DIF(&computeError, vars, measurements, MarkerCount * 3 + 1, MarkerCount * ProjectionCount * 2, 1000, opt, info, NULL, NULL, &dat);

	UnPack(vars, x, y, z, phi, 1, MarkerCount);
	delete[] vars;
	delete[] measurements;
}

void MarkerFile::MoveXYToMagDistort(float & x, float & y, float amplitude, float angleInRad, Matrix<float>& m, float dimX, float dimY)
{
	if (amplitude == 1)
		return;

    Matrix<float> vec(3, 1);
	vec(0, 0) = x;
	vec(1, 0) = y;
	vec(2, 0) = 1;
	
    Matrix<float> erg = m * vec;
	x = erg(0, 0);
	y = erg(0, 1);
}
