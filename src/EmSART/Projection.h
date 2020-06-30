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


#ifndef PROJECTION_H
#define PROJECTION_H

#include "EmSartDefault.h"
#include "io/ProjectionSource.h"
//#include "io/MRCFile.h"
//#include "io/Dm4FileStack.h"
#include <MarkerFile.h>
//#include "io/MarkerFile.h"
#include <Matrix.h>
//#include "utils/Matrix.h"
#include "cuda_kernels/Constants.h"
#include "Volume.h"

enum ProjectionListType
{
	PLT_NORMAL,
	PLT_RANDOM,
	PLT_RANDOM_START_ZERO_TILT,
	PLT_RANDOM_MIDDLE_PROJ_TWICE
};

class Projection
{
private:
	Matrix<float> float3ToMatrix(float3 val);
	float3 matrixToFloat3(Matrix<float>& val);
	bool compensateImageRotation;

protected:
	ProjectionSource* ps;
	MarkerFile* markers;
	float2* extraShifts;

public:
	Projection(ProjectionSource* aPs, MarkerFile* aMarkers, bool aCompensateImageRotation);
	~Projection();

	dim3 GetDimension();
	int GetWidth();
	int GetHeight();
	int GetMaxDimension();
	float3 GetPosition(uint aIndex);
	float3 GetPixelUPitch(uint aIndex);  //c_zPitch
	float3 GetPixelVPitch(uint aIndex);  //c_yPitch
	float3 GetNormalVector(uint aIndex);
	void GetDetectorMatrix(uint aIndex, float aMatrix[16], float os);
	int GetMinimumTiltIndex();
	float2 GetMinimumTiltShift();
	/*float2 GetMeanShift();
	float2 GetMedianShift();
	float GetMean(float* data);
	float GetMean(int index);
	void Normalize(float* data, float mean);*/
	Matrix<float> RotateMatrix(uint aIndex, Matrix<float>& matrix);
	void CreateProjectionIndexList(ProjectionListType type, int* projectionCount, int** indexList);
	float GetPixelSize();

	void ComputeHitPoints(Volume<unsigned short>& vol, uint index, int2& pA, int2& pB, int2& pC, int2& pD);
	void ComputeHitPoints(Volume<float>& vol, uint index, int2& pA, int2& pB, int2& pC, int2& pD);
	void ComputeHitPoint(float posX, float posY, float posZ, uint index, int2& pA);

	float2 GetExtraShift(size_t index);
	void SetExtraShift(size_t index, float2 extraShift);
	void AddExtraShift(size_t index, float2 extraShift);

	float GetImageRotationToCompensate(uint index);
};

#endif
