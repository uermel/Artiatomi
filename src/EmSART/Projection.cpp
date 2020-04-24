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


#include "Projection.h"
#include "utils/Config.h"
#include <algorithm>

Projection::Projection(ProjectionSource* aPs, MarkerFile* aMarkers, bool aCompensateImageRotation)
	: ps(aPs), markers(aMarkers), extraShifts(new float2[aPs->GetProjectionCount()]), compensateImageRotation(aCompensateImageRotation)
{
	float2 zero;
	zero.x = 0;
	zero.y = 0;
	for (size_t i = 0; i < aPs->GetProjectionCount(); i++)
	{
		extraShifts[i] = zero;
	}
}

Projection::~Projection()
{
	if (extraShifts)
	{
		delete[] extraShifts;
		extraShifts = NULL;
	}
}

dim3 Projection::GetDimension()
{
	dim3 dim;
	dim.x = ps->GetWidth();
	dim.y = ps->GetHeight();
	dim.z = 1;
	return dim;
}

int Projection::GetWidth()
{
	return ps->GetWidth();
}

int Projection::GetHeight()
{
	return ps->GetHeight();
}

int Projection::GetMaxDimension()
{
	return max(ps->GetWidth(), ps->GetHeight());
}

float Projection::GetPixelSize()
{
    //return 0.282f;
    //if (index > ps->GetProjectionCount() || index < 0) return 0;
	return ps->GetPixelSize(); // --> already in nm! / 10.0f;  //Angstrom to nm
}

float Projection::GetImageRotationToCompensate(uint aIndex)
{
	if (!compensateImageRotation)
	{
		return 0;
	}

	double psiAngle = -(*markers)(MFI_RotationPsi, aIndex, 0) / 180.0 * (double)M_PI;
	if (Configuration::Config::GetConfig().UseFixPsiAngle)
		psiAngle = -Configuration::Config::GetConfig().PsiAngle / 180.0 * (double)M_PI;

	return (float)psiAngle;
}

Matrix<float> Projection::RotateMatrix(uint aIndex, Matrix<float>& matrix)
{
	double tiltAngle = ((double)(*markers)(MFI_TiltAngle, aIndex, 0) + (double)Configuration::Config::GetConfig().AddTiltAngle) / 180.0 * (double)M_PI;
	double tiltXAngle = (Configuration::Config::GetConfig().AddTiltXAngle) / 180.0 * (double)M_PI;
	double psiAngle  = -(*markers)(MFI_RotationPsi, aIndex, 0) / 180.0 * (double)M_PI;
	if (Configuration::Config::GetConfig().UseFixPsiAngle)
        psiAngle = -Configuration::Config::GetConfig().PsiAngle / 180.0 * (double)M_PI;
	double phiAngle = Configuration::Config::GetConfig().PhiAngle / 180.0 * (double)M_PI;

	if (compensateImageRotation)
	{
		psiAngle = 0;
	}
	
	Matrix<double> matrixD(3,1);
	Matrix<double> mPsi(3,3);
	Matrix<double> mPhi(3,3);
	Matrix<double> mTheta(3,3);
	Matrix<double> mAdjust(3,3);
	Matrix<double> mXTilt(3,3);
	Matrix<double> resD(3,1);
	Matrix<float> resF(3,1);
	
	matrixD(0,0) = matrix(0,0);// matrixD(0,1) = matrix(0,1); matrixD(0,2) = matrix(0,2);
	matrixD(1,0) = matrix(1,0); //matrixD(1,1) = matrix(1,1); matrixD(1,2) = matrix(1,2);
	matrixD(2,0) = matrix(2,0); //matrixD(2,1) = matrix(2,1); matrixD(2,2) = matrix(2,2);


	mAdjust(0,0) = 0; mAdjust(0,1) = -1; mAdjust(0,2) = 0;
	mAdjust(1,0) = 1; mAdjust(1,1) = 0;  mAdjust(1,2) = 0;
	mAdjust(2,0) = 0; mAdjust(2,1) = 0;  mAdjust(2,2) = -1;

	mPsi(0,0) = cos(psiAngle); mPsi(0,1) = -sin(psiAngle); mPsi(0,2) = 0;
	mPsi(1,0) = sin(psiAngle); mPsi(1,1) =  cos(psiAngle); mPsi(1,2) = 0;
	mPsi(2,0) = 0;             mPsi(2,1) = 0;              mPsi(2,2) = 1;

	mPhi(0,0) = 1; mPhi(0,1) = 0;             mPhi(0,2) = 0;
	mPhi(1,0) = 0; mPhi(1,1) = cos(phiAngle); mPhi(1,2) = -sin(phiAngle);
	mPhi(2,0) = 0; mPhi(2,1) = sin(phiAngle); mPhi(2,2) =  cos(phiAngle);

	mTheta(0,0) = cos(tiltAngle);  mTheta(0,1) = 0; mTheta(0,2) = sin(tiltAngle);
	mTheta(1,0) = 0;               mTheta(1,1) = 1; mTheta(1,2) = 0;
	mTheta(2,0) = -sin(tiltAngle); mTheta(2,1) = 0; mTheta(2,2) = cos(tiltAngle);

	mXTilt(0,0) = 1; mXTilt(0,1) = 0;               mXTilt(0,2) =  0;
	mXTilt(1,0) = 0; mXTilt(1,1) = cos(tiltXAngle); mXTilt(1,2) = -sin(tiltXAngle);
	mXTilt(2,0) = 0; mXTilt(2,1) = sin(tiltXAngle); mXTilt(2,2) =  cos(tiltXAngle);

    //Matrix<float> t = mPhi.Transpose();
    mTheta = (mPhi * (mTheta * mPhi.Transpose()));
	
    resD = mXTilt * (mTheta*(mPsi * ((mAdjust * matrixD))));
	resF(0,0) = resD(0,0);// resF(0,1) = resD(0,1); resF(0,2) = resD(0,2);
	resF(1,0) = resD(1,0);// resF(1,1) = resD(1,1); resF(1,2) = resD(1,2);
	resF(2,0) = resD(2,0);// resF(2,1) = resD(2,1); resF(2,2) = resD(2,2);
	
	/*printf("\n");
	for(int y = 0; y < 1; y++)
	{
		for(int x = 0; x < 3; x++)
		{
			printf("%f ", resF(x,y));
		}
		printf("\n");
	}
	printf("\n");*/
	return resF;
	//return (mPsi * (mAdjust.Transpose() * (mPhi.Transpose() * (mTheta * (mPhi * (mAdjust * matrix))))));
}

Matrix<float> Projection::float3ToMatrix(float3 val)
{
	Matrix<float> ret(3, 1); //column vector

	float* values = ret.GetData();
	values[0] = val.x;
	values[1] = val.y;
	values[2] = val.z;

	return ret;
}
float3 Projection::matrixToFloat3(Matrix<float>& val)
{
	float3 ret;
	float* values = val.GetData();
	ret.x = values[0];
	ret.y = values[1];
	ret.z = values[2];

	return ret;
}

float3 Projection::GetPosition(uint aIndex)
{
	float3 pos;
	if ((*markers)(MFI_Magnifiaction, aIndex, 0) > 0.8f && 
		(*markers)(MFI_Magnifiaction, aIndex, 0) > 0.8f)
	{
		pos.x = -0.5f * ps->GetWidth() * (*markers)(MFI_Magnifiaction, aIndex, 0);
		pos.y = -0.5f * ps->GetHeight() * (*markers)(MFI_Magnifiaction, aIndex, 0);
		pos.z = DIST;
	}
	else
	{
		pos.x = -0.5f * ps->GetWidth();
		pos.y = -0.5f * ps->GetHeight();
		pos.z = DIST;
	}
	float shiftX = (*markers)(MFI_X_Shift, aIndex, 0) + extraShifts[aIndex].x;
	float shiftY = (*markers)(MFI_Y_Shift, aIndex, 0) + extraShifts[aIndex].y;

	if (compensateImageRotation)
	{
		float rotAngle = GetImageRotationToCompensate(aIndex);
		float cosAngle = cos(rotAngle); 
		float sinAngle = sin(rotAngle);

		float rotatedShiftX = cosAngle * shiftX - sinAngle * shiftY;
		float rotatedShiftY = sinAngle * shiftX + cosAngle * shiftY;

		shiftX = rotatedShiftX;
		shiftY = rotatedShiftY;
	}

	Matrix<float> vec = float3ToMatrix(pos);

	Matrix<float> posRot = RotateMatrix(aIndex, vec);

	pos = matrixToFloat3(posRot);

	float3 up = GetPixelUPitch(aIndex);
	float3 vp = GetPixelVPitch(aIndex);
	
	return pos - up * shiftX - vp * shiftY;

}

float3 Projection::GetPixelUPitch(uint aIndex)
{
	float3 u;
	
	if ((*markers)(MFI_Magnifiaction, aIndex, 0) > 0.8f && 
		(*markers)(MFI_Magnifiaction, aIndex, 0) > 0.8f)
	{
		u.x = 1 * (*markers)(MFI_Magnifiaction, aIndex, 0);	
	}
	else
	{
		u.x = 1;
	}
	u.y = 0;
	u.z = 0;

	Matrix<float> vec = float3ToMatrix(u);

	Matrix<float> posRot = RotateMatrix(aIndex, vec);

	u = matrixToFloat3(posRot);
	return u;

}

float3 Projection::GetPixelVPitch(uint aIndex)
{
	float3 v;
	v.x = 0;
	if ((*markers)(MFI_Magnifiaction, aIndex, 0) > 0.8f && 
		(*markers)(MFI_Magnifiaction, aIndex, 0) > 0.8f)
	{
		v.y = 1 * (*markers)(MFI_Magnifiaction, aIndex, 0);
	}
	else
	{
		v.y = 1;
	}
	v.z = 0;

	Matrix<float> vec = float3ToMatrix(v);

	Matrix<float> posRot = RotateMatrix(aIndex, vec);

	v = matrixToFloat3(posRot);

	return v;
}

float3 Projection::GetNormalVector(uint aIndex)
{
	/*float phiAngle = Configuration::Config::GetConfig().PhiAngle / 180.0f * (float)M_PI;
	float cphi = cos(phiAngle);
	float sphi = sin(phiAngle);*/

	//U and V pitch are not normalized due to magnifaction factors! 
	float3 u = normalize(GetPixelUPitch(aIndex));
	float3 v = normalize(GetPixelVPitch(aIndex));
	//float3 n = v;
	float3 norm; //normal vec point from projection to center
	//norm.x = u.y * v.z - u.z * v.y;
	//norm.y = u.z * v.x - u.x * v.z;
	//norm.z = u.x * v.y - u.y * v.x;
	norm.x = u.z * v.y - u.y * v.z;
	norm.y = u.x * v.z - u.z * v.x;
	norm.z = u.y * v.x - u.x * v.y;

	norm = normalize(norm);

	/*Matrix<float> mPhi(3,3);

	mPhi(0,0) = n.x * n.x * (1.0f - cphi) + cphi;
	mPhi(0,1) = n.x * n.y * (1.0f - cphi) - n.z * sphi;
	mPhi(0,2) = n.x * n.z * (1.0f - cphi) + n.y * sphi;

	mPhi(1,0) = n.y * n.x * (1.0f - cphi) + n.z * sphi;
	mPhi(1,1) = n.y * n.y * (1.0f - cphi) + cphi;
	mPhi(1,2) = n.y * n.z * (1.0f - cphi) - n.x * sphi;

	mPhi(2,0) = n.z * n.x * (1.0f - cphi) - n.y * sphi;
	mPhi(2,1) = n.z * n.y * (1.0f - cphi) + n.x * sphi;
	mPhi(2,2) = n.z * n.z * (1.0f - cphi) + cphi;

	Matrix<float> vec(norm);

    vec = mPhi * vec;*/

	return norm;//vec.GetAsFloat3();
}

void Projection::GetDetectorMatrix(uint aIndex, float aMatrix2[16], float os)
{
	////////////////////////////////////////////////////////////////////////////////
	//from http://www.geometrictools.com//LibFoundation/Mathematics/Wm4Matrix4.inl:
	////////////////////////////////////////////////////////////////////////////////
	double3 uPitch;
	float3 uPitchf = GetPixelUPitch(aIndex);
	uPitch.x = (double)uPitchf.x / (double)os;
	uPitch.y = (double)uPitchf.y / (double)os;
	uPitch.z = (double)uPitchf.z / (double)os;
	double3 vPitch;
	float3 vPitchf = GetPixelVPitch(aIndex);
	vPitch.x = (double)vPitchf.x / (double)os;
	vPitch.y = (double)vPitchf.y / (double)os;
	vPitch.z = (double)vPitchf.z / (double)os;
	float3 detector = GetPosition(aIndex);
	float3 helper = GetNormalVector(aIndex);// make_float3(
		//vPitch.y * uPitch.z - vPitch.z * uPitch.y,
		//vPitch.z * uPitch.x - vPitch.x * uPitch.z,
		//vPitch.x * uPitch.y - vPitch.y * uPitch.x );

	double m_afEntry[16];
	double aMatrix[16];
	m_afEntry[0] = uPitch.x;
	m_afEntry[1] = uPitch.y;
	m_afEntry[2] = uPitch.z;
	m_afEntry[3] = 0.f;
	m_afEntry[4] = vPitch.x;
	m_afEntry[5] = vPitch.y;
	m_afEntry[6] = vPitch.z;
	m_afEntry[7] = 0.f;
	m_afEntry[8] = helper.x;
	m_afEntry[9] = helper.y;
	m_afEntry[10] = helper.z;
	m_afEntry[11] = 0.f;
	m_afEntry[12] = detector.x;
	m_afEntry[13] = detector.y;
	m_afEntry[14] = detector.z;
	m_afEntry[15] = 1.f;


	double fA0 = m_afEntry[ 0]*m_afEntry[ 5] - m_afEntry[ 1]*m_afEntry[ 4];
	double fA1 = m_afEntry[ 0]*m_afEntry[ 6] - m_afEntry[ 2]*m_afEntry[ 4];
	double fA2 = m_afEntry[ 0]*m_afEntry[ 7] - m_afEntry[ 3]*m_afEntry[ 4];
	double fA3 = m_afEntry[ 1]*m_afEntry[ 6] - m_afEntry[ 2]*m_afEntry[ 5];
	double fA4 = m_afEntry[ 1]*m_afEntry[ 7] - m_afEntry[ 3]*m_afEntry[ 5];
	double fA5 = m_afEntry[ 2]*m_afEntry[ 7] - m_afEntry[ 3]*m_afEntry[ 6];
	double fB0 = m_afEntry[ 8]*m_afEntry[13] - m_afEntry[ 9]*m_afEntry[12];
	double fB1 = m_afEntry[ 8]*m_afEntry[14] - m_afEntry[10]*m_afEntry[12];
	double fB2 = m_afEntry[ 8]*m_afEntry[15] - m_afEntry[11]*m_afEntry[12];
	double fB3 = m_afEntry[ 9]*m_afEntry[14] - m_afEntry[10]*m_afEntry[13];
	double fB4 = m_afEntry[ 9]*m_afEntry[15] - m_afEntry[11]*m_afEntry[13];
	double fB5 = m_afEntry[10]*m_afEntry[15] - m_afEntry[11]*m_afEntry[14];

	double fDet = fA0*fB5-fA1*fB4+fA2*fB3+fA3*fB2-fA4*fB1+fA5*fB0;
	if (fDet == 0)
	{
		printf("Determinante of detector matrix is not 0! Can't inverse matrix.\n");
		exit(1);
	}

	aMatrix[ 0] =
		+ m_afEntry[ 5]*fB5 - m_afEntry[ 6]*fB4 + m_afEntry[ 7]*fB3;
	aMatrix[ 1] =
		- m_afEntry[ 4]*fB5 + m_afEntry[ 6]*fB2 - m_afEntry[ 7]*fB1;
	aMatrix[ 2] =
		+ m_afEntry[ 4]*fB4 - m_afEntry[ 5]*fB2 + m_afEntry[ 7]*fB0;
	aMatrix[ 3] =
		- m_afEntry[ 4]*fB3 + m_afEntry[ 5]*fB1 - m_afEntry[ 6]*fB0;
	aMatrix[ 4] =
		- m_afEntry[ 1]*fB5 + m_afEntry[ 2]*fB4 - m_afEntry[ 3]*fB3;
	aMatrix[ 5] =
		+ m_afEntry[ 0]*fB5 - m_afEntry[ 2]*fB2 + m_afEntry[ 3]*fB1;
	aMatrix[ 6] =
		- m_afEntry[ 0]*fB4 + m_afEntry[ 1]*fB2 - m_afEntry[ 3]*fB0;
	aMatrix[ 7] =
		+ m_afEntry[ 0]*fB3 - m_afEntry[ 1]*fB1 + m_afEntry[ 2]*fB0;
	aMatrix[ 8] =
		+ m_afEntry[13]*fA5 - m_afEntry[14]*fA4 + m_afEntry[15]*fA3;
	aMatrix[ 9] =
		- m_afEntry[12]*fA5 + m_afEntry[14]*fA2 - m_afEntry[15]*fA1;
	aMatrix[10] =
		+ m_afEntry[12]*fA4 - m_afEntry[13]*fA2 + m_afEntry[15]*fA0;
	aMatrix[11] =
		- m_afEntry[12]*fA3 + m_afEntry[13]*fA1 - m_afEntry[14]*fA0;
	aMatrix[12] =
		- m_afEntry[ 9]*fA5 + m_afEntry[10]*fA4 - m_afEntry[11]*fA3;
	aMatrix[13] =
		+ m_afEntry[ 8]*fA5 - m_afEntry[10]*fA2 + m_afEntry[11]*fA1;
	aMatrix[14] =
		- m_afEntry[ 8]*fA4 + m_afEntry[ 9]*fA2 - m_afEntry[11]*fA0;
	aMatrix[15] =
		+ m_afEntry[ 8]*fA3 - m_afEntry[ 9]*fA1 + m_afEntry[10]*fA0;

	double fInvDet = (1.0f)/fDet;
	aMatrix[ 0] *= fInvDet;
	aMatrix[ 1] *= fInvDet;
	aMatrix[ 2] *= fInvDet;
	aMatrix[ 3] *= fInvDet;
	aMatrix[ 4] *= fInvDet;
	aMatrix[ 5] *= fInvDet;
	aMatrix[ 6] *= fInvDet;
	aMatrix[ 7] *= fInvDet;
	aMatrix[ 8] *= fInvDet;
	aMatrix[ 9] *= fInvDet;
	aMatrix[10] *= fInvDet;
	aMatrix[11] *= fInvDet;
	aMatrix[12] *= fInvDet;
	aMatrix[13] *= fInvDet;
	aMatrix[14] *= fInvDet;
	aMatrix[15] *= fInvDet;

	for (int i = 0; i < 16; i++)
		aMatrix2[i] = (float)aMatrix[i];
	/////////////////////////////////////////////////////////////////////////
}


int Projection::GetMinimumTiltIndex()
{
	float minTilt = FLT_MAX;
	int minTiltIndex = 0;
	int projCount = markers->GetTotalProjectionCount();

	for (int i = 0; i < projCount; i++)
	{
		if (fabs((*markers)(MFI_TiltAngle, i, 0)) < minTilt)
		{
			minTilt = fabs((*markers)(MFI_TiltAngle, i, 0));
			minTiltIndex = i;
		}
	}

	return minTiltIndex;
}


float2 Projection::GetMinimumTiltShift()
{
	float2 shift;
	/*int minTiltIndex = GetMinimumTiltIndex();
    shift.x = -GetPixelUPitch(minTiltIndex).x * (*markers)(MFI_X_Shift, minTiltIndex, 0) - GetPixelVPitch(minTiltIndex).x * (*markers)(MFI_X_Shift, minTiltIndex, 0);
	shift.y = -GetPixelUPitch(minTiltIndex).y * (*markers)(MFI_Y_Shift, minTiltIndex, 0) - GetPixelVPitch(minTiltIndex).y * (*markers)(MFI_Y_Shift, minTiltIndex, 0);*/


	shift.x = 0;//(*markers)(MFI_Y_Shift, minTiltIndex, 0);
	shift.y = 0;//-(*markers)(MFI_X_Shift, minTiltIndex, 0);

	return shift;
}


//float2 Projection::GetMeanShift()
//{
//	float2 shift;
//	shift.x = 0;
//	shift.y = 0;
//
//    int tiltCount = markers->GetTotalProjectionCount();
//    for (int i = 0; i < tiltCount; i++)
//    {
//        shift.x += (*markers)(MFI_X_Shift, i, 0);
//        shift.y += (*markers)(MFI_Y_Shift, i, 0);
//    }
//    shift.x /= (float)tiltCount;
//    shift.y /= (float)tiltCount;
//	return shift;
//}
//
//
//float2 Projection::GetMedianShift()
//{
//	float2 shift;
//    int tiltCount = markers->GetTotalProjectionCount();
//    float* x = new float[tiltCount];
//    float* y = new float[tiltCount];
//
//    for (int i = 0; i < tiltCount; i++)
//    {
//        x[i] = (*markers)(MFI_X_Shift, i, 0);
//        y[i] = (*markers)(MFI_Y_Shift, i, 0);
//    }
//    std::sort(x, x + tiltCount);
//    std::sort(y, y + tiltCount);
//
//    shift.x = x[tiltCount/2];
//    shift.y = y[tiltCount/2];
//	return shift;
//}

//float Projection::GetMean(float* data)
//{
//	int width = GetWidth();
//	int height = GetHeight();
//	float pixelCount = width * height;
//	float mean = 0;
//
//	for (int i = 0; i < width; i++)
//		for (int j = 0; j < height; j++)
//		{
//			mean += data[j * width + i];
//		}
//
//	return mean / pixelCount;
//}

//float Projection::GetMean(int index)
//{
//	float* data = ps->GetProjectionFloat(index);
//	float mean = GetMean(data);
//	delete[] data;
//	return mean;
//}

//void Projection::Normalize(float* data, float mean)
//{
//	int width = GetWidth();
//	int height = GetHeight();
//    float normVal = 1.0f;//Configuration::Config::GetConfig().ProjNormVal;
//	//subtract mean and norm to mean * 0.8f - 1.6f
//	for (int i = 0; i < width; i++)
//		for (int j = 0; j < height; j++)
//		{
//			data[j * width + i] = ((data[j * width + i] - mean) / mean * normVal);
//		}
//}

void Projection::CreateProjectionIndexList(ProjectionListType type, int* projectionCount, int** indexList)
{
	if (ps == NULL) return;

	int counter;
	int temp;
	int minTiltIndex;
	switch (type)
	{
	case PLT_NORMAL:
		*projectionCount = markers->GetProjectionCount();
		*indexList = new int[*projectionCount];
		counter = 0;
		for (int i = 0; i < ps->GetProjectionCount(); i++)
		{
			if (markers->CheckIfProjIndexIsGood(i))
			{
				(*indexList)[counter] = i;
				counter++;
			}
		}
		break;
	case PLT_RANDOM:
		// initialize seed "randomly"
        srand(0);
		*projectionCount = markers->GetProjectionCount();
		(*indexList) = new int[*projectionCount];
		counter = 0;
		for (int i = 0; i < ps->GetProjectionCount(); i++)
		{
			if (markers->CheckIfProjIndexIsGood(i))
			{
				(*indexList)[counter] = i;
				counter++;
			}
		}

		for (int i = 0; i < *projectionCount - 1; i++)
		{
			int r = i + (rand() % (*projectionCount - i));
			temp = (*indexList)[i];
			(*indexList)[i] = (*indexList)[r];
			(*indexList)[r] = temp;
		}
		break;
	case PLT_RANDOM_START_ZERO_TILT:
		// initialize seed "randomly"
        srand(0);
		*projectionCount = markers->GetProjectionCount();
		(*indexList) = new int[*projectionCount];
		counter = 0;
		for (int i = 0; i < ps->GetProjectionCount(); i++)
		{
			if (markers->CheckIfProjIndexIsGood(i))
			{
				(*indexList)[counter] = i;
				counter++;
			}
		}

		for (int i = 0; i < *projectionCount - 1; i++)
		{
			int r = i + (rand() % (*projectionCount - i));
			temp = (*indexList)[i];
			(*indexList)[i] = (*indexList)[r];
			(*indexList)[r] = temp;
		}
		minTiltIndex = GetMinimumTiltIndex();

		for (int i = 0; i < *projectionCount; i++)
		{
			if ((*indexList)[i] == minTiltIndex)
			{
                temp = (*indexList)[0];
                (*indexList)[0] = (*indexList)[i];
                (*indexList)[i] = temp;
                break;
            }
		}

		break;
	case PLT_RANDOM_MIDDLE_PROJ_TWICE:
		// initialize seed "randomly"
        srand(0);
		*projectionCount = markers->GetProjectionCount();
		*projectionCount += *projectionCount / 2;
		(*indexList) = new int[*projectionCount];
		counter = 0;
		for (int i = 0; i < ps->GetProjectionCount(); i++)
		{
			if (markers->CheckIfProjIndexIsGood(i))
			{
				(*indexList)[counter] = i;
				counter++;
			}
		}
		int start2 = markers->GetProjectionCount() / 4;
		while (counter < *projectionCount)
		{
			(*indexList)[counter] = (*indexList)[start2];
			start2++;
			counter++;
		}

		for (int i = 0; i < *projectionCount - 1; i++)
		{
			int r = i + (rand() % (*projectionCount - i));
			temp = (*indexList)[i];
			(*indexList)[i] = (*indexList)[r];
			(*indexList)[r] = temp;
		}

		int minTiltIndex = GetMinimumTiltIndex();
		temp = (*indexList)[minTiltIndex];
		(*indexList)[minTiltIndex] = (*indexList)[0];
		(*indexList)[0] = temp;

		break;
	}
}

void MatrixVector3Mul(float4* M, float3* v)
{
	float3 erg;
	erg.x = M[0].x * v->x + M[0].y * v->y + M[0].z * v->z + 1.f * M[0].w;
	erg.y = M[1].x * v->x + M[1].y * v->y + M[1].z * v->z + 1.f * M[1].w;
	erg.z = M[2].x * v->x + M[2].y * v->y + M[2].z * v->z + 1.f * M[2].w;
	*v = erg;
}




void Projection::ComputeHitPoints(Volume<unsigned short>& vol, uint index, int2& pA, int2& pB, int2& pC, int2& pD)
{
	///Project volume on detector to get shadowed surface
	
	float3 MC_bBoxMin = vol.GetVolumeBBoxMin();
	float3 MC_bBoxMax = vol.GetVolumeBBoxMax();

	float t;
	float3 hitPoint;
	float3 c_detektor = GetPosition(index);
	float3 c_projNorm = GetNormalVector(index);

	float3 borderMin = make_float3(FLT_MAX);
	float3 borderMax = make_float3(-FLT_MAX);

	float3 p1, p2, p3, p4, p5, p6, p7, p8;

	//first corner
	t = (c_projNorm.x * MC_bBoxMin.x + c_projNorm.y * MC_bBoxMin.y + c_projNorm.z * MC_bBoxMin.z);
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);

	hitPoint.x = t * (-c_projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-c_projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-c_projNorm.z) + MC_bBoxMin.z;

	p1 = hitPoint;

	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/


	//second corner
	t = (c_projNorm.x * MC_bBoxMin.x + c_projNorm.y * MC_bBoxMin.y + c_projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-c_projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));
	
	p2 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//third corner
	t = (c_projNorm.x * MC_bBoxMin.x + c_projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + c_projNorm.z * MC_bBoxMin.z);
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-c_projNorm.z) + MC_bBoxMin.z;
	
	p3 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//fourth corner
	t = (c_projNorm.x * MC_bBoxMin.x + c_projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + c_projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));
	
	p4 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//fifth corner
	t = (c_projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + c_projNorm.y * MC_bBoxMin.y + c_projNorm.z * MC_bBoxMin.z);
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-c_projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-c_projNorm.z) + MC_bBoxMin.z;
	
	p5 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//sixth corner
	t = (c_projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + c_projNorm.y * MC_bBoxMin.y + c_projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-c_projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));
	
	p6 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//seventh corner
	t = (c_projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + c_projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + c_projNorm.z * MC_bBoxMin.z);
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-c_projNorm.z) + MC_bBoxMin.z;
	
	p7 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//eighth corner
	t = (c_projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + c_projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + c_projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));
	
	p8 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//get largest area
	/*MC_bBoxMin = fminf(borderMin, borderMax);
	MC_bBoxMax = fmaxf(borderMin, borderMax);*/

	/*if (c_projNorm.x <= 0)
	{
		float temp = MC_bBoxMin.x;
		MC_bBoxMin.x = MC_bBoxMax.x;
		MC_bBoxMax.x = temp;
	}*/

	float matrix[16];
	GetDetectorMatrix(index, matrix, 1);

	MatrixVector3Mul((float4*)matrix, &p1);
	MatrixVector3Mul((float4*)matrix, &p2);
	MatrixVector3Mul((float4*)matrix, &p3);
	MatrixVector3Mul((float4*)matrix, &p4);

	if  (fabs(p1.x) < fabs(p2.x))
		pA.x = p1.x;
	else
		pA.x = p2.x;
	if  (fabs(p1.y) < fabs(p2.y))
		pA.y = p1.y;
	else
		pA.y = p2.y;
	
	if  (fabs(p3.x) < fabs(p4.x))
		pB.x = p3.x;
	else
		pB.x = p4.x;
	if  (fabs(p3.y) < fabs(p4.y))
		pB.y = p3.y;
	else
		pB.y = p4.y;

	
	MatrixVector3Mul((float4*)matrix, &p5);
	MatrixVector3Mul((float4*)matrix, &p6);
	MatrixVector3Mul((float4*)matrix, &p7);
	MatrixVector3Mul((float4*)matrix, &p8);

	
	if  (fabs(p5.x) < fabs(p6.x))
		pC.x = p5.x;
	else
		pC.x = p6.x;
	if  (fabs(p5.y) < fabs(p6.y))
		pC.y = p5.y;
	else
		pC.y = p6.y;
	
	if  (fabs(p7.x) < fabs(p8.x))
		pD.x = p7.x;
	else
		pD.x = p8.x;
	if  (fabs(p7.y) < fabs(p8.y))
		pD.y = p7.y;
	else
		pD.y = p8.y;



	//hitPoint = fminf(MC_bBoxMin, MC_bBoxMax);
	////--> pixelBorders.x = x.min; pixelBorders.z = y.min;
	//pixelBordersA.x = floor(hitPoint.x);
	//pixelBordersA.y = floor(hitPoint.y);

	////--> pixelBorders.y = x.max; pixelBorders.v = y.max
	//hitPoint = fmaxf(MC_bBoxMin, MC_bBoxMax);
	//pixelBordersB.x = ceil(hitPoint.x);
	//pixelBordersB.y = ceil(hitPoint.y);

	//if (pixelBordersA.x < 0) pixelBordersA.x = 0;
	//if (pixelBordersA.y < 0) pixelBordersA.y = 0;
	//if (pixelBordersB.x < 0) pixelBordersB.x = 0;
	//if (pixelBordersB.y < 0) pixelBordersB.y = 0;

	//if (pixelBordersA.x >= GetWidth()) pixelBordersA.x = GetWidth() - 1;
	//if (pixelBordersA.y >= GetHeight()) pixelBordersA.y = GetHeight() - 1;
	//if (pixelBordersB.x >= GetWidth()) pixelBordersB.x = GetWidth() - 1;
	//if (pixelBordersB.y >= GetHeight()) pixelBordersB.y = GetHeight() - 1;
}

void Projection::ComputeHitPoints(Volume<float>& vol, uint index, int2& pA, int2& pB, int2& pC, int2& pD)
{
	///Project volume on detector to get shadowed surface
	
	float3 MC_bBoxMin = vol.GetVolumeBBoxMin();
	float3 MC_bBoxMax = vol.GetVolumeBBoxMax();

	float t;
	float3 hitPoint;
	float3 c_detektor = GetPosition(index);
	float3 c_projNorm = GetNormalVector(index);

	float3 borderMin = make_float3(FLT_MAX);
	float3 borderMax = make_float3(-FLT_MAX);

	float3 p1, p2, p3, p4, p5, p6, p7, p8;

	//first corner
	t = (c_projNorm.x * MC_bBoxMin.x + c_projNorm.y * MC_bBoxMin.y + c_projNorm.z * MC_bBoxMin.z);
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);

	hitPoint.x = t * (-c_projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-c_projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-c_projNorm.z) + MC_bBoxMin.z;

	p1 = hitPoint;

	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/


	//second corner
	t = (c_projNorm.x * MC_bBoxMin.x + c_projNorm.y * MC_bBoxMin.y + c_projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-c_projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));
	
	p2 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//third corner
	t = (c_projNorm.x * MC_bBoxMin.x + c_projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + c_projNorm.z * MC_bBoxMin.z);
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-c_projNorm.z) + MC_bBoxMin.z;
	
	p3 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//fourth corner
	t = (c_projNorm.x * MC_bBoxMin.x + c_projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + c_projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));
	
	p4 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//fifth corner
	t = (c_projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + c_projNorm.y * MC_bBoxMin.y + c_projNorm.z * MC_bBoxMin.z);
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-c_projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-c_projNorm.z) + MC_bBoxMin.z;
	
	p5 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//sixth corner
	t = (c_projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + c_projNorm.y * MC_bBoxMin.y + c_projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-c_projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));
	
	p6 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//seventh corner
	t = (c_projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + c_projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + c_projNorm.z * MC_bBoxMin.z);
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-c_projNorm.z) + MC_bBoxMin.z;
	
	p7 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//eighth corner
	t = (c_projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + c_projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + c_projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);
	hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));
	
	p8 = hitPoint;
	/*borderMin = fminf(hitPoint, borderMin);
	borderMax = fmaxf(hitPoint, borderMax);*/

	//get largest area
	/*MC_bBoxMin = fminf(borderMin, borderMax);
	MC_bBoxMax = fmaxf(borderMin, borderMax);*/

	/*if (c_projNorm.x <= 0)
	{
		float temp = MC_bBoxMin.x;
		MC_bBoxMin.x = MC_bBoxMax.x;
		MC_bBoxMax.x = temp;
	}*/

	float matrix[16];
	GetDetectorMatrix(index, matrix, 1);

	MatrixVector3Mul((float4*)matrix, &p1);
	MatrixVector3Mul((float4*)matrix, &p2);
	MatrixVector3Mul((float4*)matrix, &p3);
	MatrixVector3Mul((float4*)matrix, &p4);

	if  (fabs(p1.x) < fabs(p2.x))
		pA.x = p1.x;
	else
		pA.x = p2.x;
	if  (fabs(p1.y) < fabs(p2.y))
		pA.y = p1.y;
	else
		pA.y = p2.y;
	
	if  (fabs(p3.x) < fabs(p4.x))
		pB.x = p3.x;
	else
		pB.x = p4.x;
	if  (fabs(p3.y) < fabs(p4.y))
		pB.y = p3.y;
	else
		pB.y = p4.y;

	
	MatrixVector3Mul((float4*)matrix, &p5);
	MatrixVector3Mul((float4*)matrix, &p6);
	MatrixVector3Mul((float4*)matrix, &p7);
	MatrixVector3Mul((float4*)matrix, &p8);

	
	if  (fabs(p5.x) < fabs(p6.x))
		pC.x = p5.x;
	else
		pC.x = p6.x;
	if  (fabs(p5.y) < fabs(p6.y))
		pC.y = p5.y;
	else
		pC.y = p6.y;
	
	if  (fabs(p7.x) < fabs(p8.x))
		pD.x = p7.x;
	else
		pD.x = p8.x;
	if  (fabs(p7.y) < fabs(p8.y))
		pD.y = p7.y;
	else
		pD.y = p8.y;



	//hitPoint = fminf(MC_bBoxMin, MC_bBoxMax);
	////--> pixelBorders.x = x.min; pixelBorders.z = y.min;
	//pixelBordersA.x = floor(hitPoint.x);
	//pixelBordersA.y = floor(hitPoint.y);

	////--> pixelBorders.y = x.max; pixelBorders.v = y.max
	//hitPoint = fmaxf(MC_bBoxMin, MC_bBoxMax);
	//pixelBordersB.x = ceil(hitPoint.x);
	//pixelBordersB.y = ceil(hitPoint.y);

	//if (pixelBordersA.x < 0) pixelBordersA.x = 0;
	//if (pixelBordersA.y < 0) pixelBordersA.y = 0;
	//if (pixelBordersB.x < 0) pixelBordersB.x = 0;
	//if (pixelBordersB.y < 0) pixelBordersB.y = 0;

	//if (pixelBordersA.x >= GetWidth()) pixelBordersA.x = GetWidth() - 1;
	//if (pixelBordersA.y >= GetHeight()) pixelBordersA.y = GetHeight() - 1;
	//if (pixelBordersB.x >= GetWidth()) pixelBordersB.x = GetWidth() - 1;
	//if (pixelBordersB.y >= GetHeight()) pixelBordersB.y = GetHeight() - 1;
}

void Projection::ComputeHitPoint(float posX, float posY, float posZ, uint index, int2 & pA)
{
	//Add 0.5 to get to the center of the voxel
	float3 posInVol = make_float3(posX, posY, posZ);
	//printf("PosInVol1: %f, %f, %f\n", posX, posY, posZ);

	float t;
	float3 hitPoint;
	float3 c_detektor = GetPosition(index);
	float3 c_projNorm = GetNormalVector(index);

	float3 borderMin = make_float3(FLT_MAX);
	float3 borderMax = make_float3(-FLT_MAX);

	float3 p1;

	//center:
	t = (c_projNorm.x * posInVol.x + c_projNorm.y * posInVol.y + c_projNorm.z * posInVol.z);
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = fabs(t);

	hitPoint.x = t * (-c_projNorm.x) + posInVol.x;
	hitPoint.y = t * (-c_projNorm.y) + posInVol.y;
	hitPoint.z = t * (-c_projNorm.z) + posInVol.z;

	p1 = hitPoint;

	float matrix[16];
	GetDetectorMatrix(index, matrix, 1);

	MatrixVector3Mul((float4*)matrix, &p1);

	pA.x = p1.x;
	pA.y = p1.y;
}

float2 Projection::GetExtraShift(size_t index)
{
	if (index < ps->GetProjectionCount())
	{
		return extraShifts[index];
	}
	else
	{
		float2 erg;
		erg.x = 0;
		erg.y = 0;
		return erg;
	}
}

void Projection::SetExtraShift(size_t index, float2 extraShift)
{
	if (index < ps->GetProjectionCount())
	{
		extraShifts[index] = extraShift;
	}
}

void Projection::AddExtraShift(size_t index, float2 extraShift)
{
	if (index < ps->GetProjectionCount())
	{
		extraShifts[index].x += extraShift.x*0.5f;
		extraShifts[index].y += extraShift.y*0.5f;
	}
}
