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


#include "CrossCorrelator.h"
#include "../FileIO/EmFile.h"
#include <cmath>

void CrossCorrelator::setup(int x, int y)
{
	if (mDimX == x && mDimY == y)
		return;

	cleanup();
	int fftWidth = x / 2 + 1;

	mDimX = x;
	mDimY = y;

	mGaussRadius = -1;
	mReference = fftwf_alloc_real(mDimX * mDimY);
	mReferenceCplx = fftwf_alloc_complex(fftWidth * mDimY);
	mImage = fftwf_alloc_real(mDimX * mDimY);
	mTemp = fftwf_alloc_complex(fftWidth * mDimY);

	mPlanReferenceFwd = fftwf_plan_dft_r2c_2d(mDimY, mDimX, mReference, mReferenceCplx, FFTW_ESTIMATE);
	mPlanImageFwd = fftwf_plan_dft_r2c_2d(mDimY, mDimX, mImage, mTemp, FFTW_ESTIMATE);
	mPlanReferenceBkwd = fftwf_plan_dft_c2r_2d(mDimY, mDimX, mReferenceCplx, mReference, FFTW_ESTIMATE);
	mPlanImageBkwd = fftwf_plan_dft_c2r_2d(mDimY, mDimX, mTemp, mImage, FFTW_ESTIMATE);
}

void CrossCorrelator::cleanup()
{
	if (mReferenceCplx)
		fftwf_free(mReferenceCplx);
	mReferenceCplx = NULL;

	if (mTemp)
		fftwf_free(mTemp);
	mTemp = NULL;

	if (mImage)
		fftwf_free(mImage);
	mImage = NULL;

	if (mReference)
		fftwf_free(mReference);
	mReference = NULL;

	if (mPlanReferenceFwd)
		fftwf_destroy_plan(mPlanReferenceFwd);
	mPlanReferenceFwd = 0;

	if (mPlanImageFwd)
		fftwf_destroy_plan(mPlanImageFwd);
	mPlanImageFwd = 0;

	if (mPlanReferenceBkwd)
		fftwf_destroy_plan(mPlanReferenceBkwd);
	mPlanReferenceBkwd = 0;

	if (mPlanImageBkwd)
		fftwf_destroy_plan(mPlanImageBkwd);
	mPlanImageBkwd = 0;
}

CrossCorrelator::CrossCorrelator() :
	mDimX(-1), mDimY(-1),
	mGaussRadius(-1),
	mReferenceCplx(NULL),
	mTemp(NULL),
	mImage(NULL),
	mReference(NULL),
	mPlanReferenceFwd(0),
	mPlanImageFwd(0),
	mPlanReferenceBkwd(0),
	mPlanImageBkwd(0)
{
}

CrossCorrelator::~CrossCorrelator()
{
	cleanup();
}

FilterPoint2D CrossCorrelator::GetShiftGauss(float * aImage, unsigned char * aCC, int aDimX, int aDimY, int aRadius)
{
	if (aDimX != mDimX || aDimY != mDimY || mGaussRadius != aRadius)
	{
		setup(aDimX, aDimY);
		SetGaussBlobAsReference(aDimX, aDimY, aRadius);
	}

	//EmFile::InitHeader("VorFilter.em", aDimX, aDimY, 1, 0, DT_FLOAT);
	//EmFile::WriteRawData("VorFilter.em", aImage, aDimX * aDimY * sizeof(float));
	Smooth(aImage, aDimX, aDimY);
	//EmFile::InitHeader("NachFilter.em", aDimX, aDimY, 1, 0, DT_FLOAT);
	//EmFile::WriteRawData("NachFilter.em", aImage, aDimX * aDimY * sizeof(float));
	//EmFile::InitHeader("Reference.em", aDimX, aDimY, 1, 0, DT_FLOAT);
	//EmFile::WriteRawData("Reference.em", mReference, aDimX * aDimY * sizeof(float));
	//memcpy(mImage, aImage, mDimX * mDimY * sizeof(float));

	fftwf_execute(mPlanImageFwd);
	int fftWidth = mDimX / 2 + 1;

	for (int y = 0; y < mDimY; y++)
	{
		for (int x = 0; x < mDimX / 2 + 1; x++)
		{
			fftwf_complex a;
			a[0] = mTemp[y * fftWidth + x][0];
			a[1] = -mTemp[y * fftWidth + x][1]; //conj complex
			fftwf_complex b;
			b[0] = mReferenceCplx[y * fftWidth + x][0];
			b[1] = mReferenceCplx[y * fftWidth + x][1];
			mTemp[y * fftWidth + x][0] = a[0] * b[0] - a[1] * b[1];
			mTemp[y * fftWidth + x][1] = a[0] * b[1] + a[1] * b[0];
		}
	}

	fftwf_execute(mPlanImageBkwd);
	
	int minX = -1, minY = - 1;
	float minVal = FLT_MAX; //Reference is contrast inverted: search for minimum...
	float maxVal = -FLT_MAX;
	//peak is at image border, so ignore the center...
	float searchRange = 4;
	float minValVisu = FLT_MAX;
	float maxValVisu = -FLT_MAX;

	for (int y = 0; y < mDimY; y++)
	{
		for (int x = 0; x < mDimX; x++)
		{
			float len = sqrtf((x - mDimX / 2.0f) * (x - mDimX / 2.0f) + (y - mDimY / 2.0f) * (y - mDimY / 2.0f));

			if (len > searchRange)
			{
				if (minVal > mImage[y * mDimX + x])
				{
					minVal = mImage[y * mDimX + x];
					minX = x;
					minY = y;
				}
				if (maxVal < mImage[y * mDimX + x])
					maxVal = mImage[y * mDimX + x];
			}

			if (aImage[y * mDimX + x] < minValVisu)
				minValVisu = aImage[y * mDimX + x];

			if (aImage[y * mDimX + x] > maxValVisu)
				maxValVisu = aImage[y * mDimX + x];
		}
	}

	//normalize filtered image for visu
	for (int y = 0; y < mDimY; y++)
	{
		for (int x = 0; x < mDimX; x++)
		{
			aImage[y * mDimX + x] = (aImage[y * mDimX + x] - minValVisu) / (maxValVisu - minValVisu);
		}
	}

	if (aCC != NULL)
	{
		//copy CC-map to input array
		for (int y = 0; y < mDimY; y++)
		{
			for (int x = 0; x < mDimX; x++)
			{
				int nx = x + mDimX / 2 - 1;
				int ny = y + mDimY / 2 - 1;

				if (nx >= mDimX)
					nx = nx - mDimX;
				if (ny >= mDimY)
					ny = ny - mDimY;

				aCC[(mDimY - ny - 1) * mDimX + (mDimX - nx - 1)] = (uchar)(255.0f * (1.0f - (mImage[y * mDimX + x] - minVal) / (maxVal - minVal)));
			}
		}
	}
	//EmFile::InitHeader("CC.em", aDimX, aDimY, 1, 0, DT_FLOAT);
	//EmFile::WriteRawData("CC.em", mImage, aDimX * aDimY * sizeof(float));

	if (minX < 0)
		return FilterPoint2D();

	if (minX > mDimX / 2)
		minX = minX - mDimX;

	if (minY > mDimY / 2)
		minY = minY - mDimY;

	return FilterPoint2D(minX, minY);
}

void CrossCorrelator::SetReference(float * aReference, int aDimX, int aDimY)
{
	//realloc arrays if necessary
	setup(aDimX, aDimY);

	memcpy(mReference, aReference, mDimX * mDimY * sizeof(float));

	fftwf_execute(mPlanReferenceFwd);
}

void CrossCorrelator::SetGaussBlobAsReference(int aDimX, int aDimY, int aRadius)
{
	//realloc arrays if necessary
	setup(aDimX, aDimY);

	//recreate the Gauss-blob if radius changed
	if (mGaussRadius != aRadius)
	{
		mGaussRadius = aRadius;
		float lp = 3;
		float lps = aRadius - 3;
		

		for (int y = 0; y < mDimY; y++)
		{
			for (int x = 0; x < mDimX; x++)
			{
				float len = sqrtf((x - mDimX / 2.0f) * (x - mDimX / 2.0f) + (y - mDimY / 2.0f) * (y - mDimY / 2.0f));
				float fil = 0;
				//Low pass
				if (len <= lp) fil = 1;

				float fil2;
				if (len <= lp) fil2 = 1;
				else fil2 = 0;

				fil2 = (-fil + 1.0f) * expf(-((len - lp) * (len - lp) / (2 * lps * lps)));
				if (fil2 > 0.001f)
					fil = fil2;

				mReference[y * mDimX + x] = fil;
				/*if (len < aRadius)
				{
					mReference[y * mDimX + x] = 1;
				}
				else
				{
					mReference[y * mDimX + x] = 0;
				}*/
			}
		}

		/*Smooth(mReference, mDimX, mDimY);


		for (int y = 0; y < mDimY; y++)
		{
			for (int x = 0; x < mDimX; x++)
			{
				mReference[y * mDimX + x] /= mDimX * mDimY;
				if (mReference[y * mDimX + x] > 1.0f) mReference[y * mDimX + x] = 1;
				if (mReference[y * mDimX + x] < 0.0f) mReference[y * mDimX + x] = 0;
			}
		}*/

		fftwf_execute(mPlanReferenceFwd);
	}
}

void CrossCorrelator::Smooth(float * aImage, int aDimX, int aDimY)
{
	setup(aDimX, aDimY);
	if (mImage != aImage)
		memcpy(mImage, aImage, mDimX * mDimY * sizeof(float));

	float lp = aDimX / 8.0f;
	float lps = lp / 4.0f;
	lp = lp - lps;
	float hp = 0;
	float hps = 0;

	fftwf_execute(mPlanImageFwd);

	int fftWidth = mDimX / 2 + 1;

	for (int y = 0; y < mDimY; y++)
	{
		for (int x = 0; x < mDimX / 2 + 1; x++)
		{
			//FFT Shift
			float realX = x + mDimX / 2;
			float realY = y + mDimY / 2;
			if (realX >= mDimX) realX -= mDimX;
			if (realY >= mDimY) realY -= mDimY;
			
			float len = sqrtf((realX - mDimX / 2.0f) * (realX - mDimX / 2.0f) + (realY - mDimY / 2.0f) * (realY - mDimY / 2.0f));
			
			float fil = 0;
			//Low pass
			if (len <= lp) fil = 1;
						
			float fil2;
			if (len < lp) fil2 = 1;
			else fil2 = 0;

			fil2 = (-len + 1.0f) * expf(-((len - lp) * (len - lp) / (2 * lps * lps)));
			if (fil2 > 0.001f)
				fil = fil2;

			mTemp[y * fftWidth + x][0] *= fil;
			mTemp[y * fftWidth + x][1] *= fil;
		}
	}
	fftwf_execute(mPlanImageBkwd);

	if (mImage != aImage)
		memcpy(aImage, mImage, mDimX * mDimY * sizeof(float));
}

bool CrossCorrelator::CopyPatch(float * aImageSource, float * aPatch, int aDimImageX, int aDimImageY, int aPatchSize, int aX, int aY)
{
	int px = aX - aPatchSize / 2;
	int py = aY - aPatchSize / 2;

	if (px < 0 || py < 0 || px + aPatchSize >= aDimImageX || py + aPatchSize >= aDimImageY)
	{
		return false;
	}

	//pitched and ROI based copy:
	for (int y = py; y < py + aPatchSize; y++)
	{
		memcpy((char*)aPatch + (y-py) * aPatchSize * sizeof(float) + 0,
			(char*)aImageSource + y * aDimImageX * sizeof(float) + px * sizeof(float),
			aPatchSize * sizeof(float));
	}
	return true;
}
