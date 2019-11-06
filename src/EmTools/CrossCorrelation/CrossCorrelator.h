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


#ifndef CROSSCORRELATOR_H
#define CROSSCORRELATOR_H

#include "../Basics/Default.h"
#include "../FilterGraph/FilterPoint2D.h"
#include <fftw3.h>

class CrossCorrelator
{
private:
	int mDimX, mDimY;
	int mGaussRadius;
	fftwf_complex* mReferenceCplx;
	fftwf_complex* mTemp;
	float* mImage;
	float* mReference;
	fftwf_plan mPlanReferenceFwd;
	fftwf_plan mPlanImageFwd;
	fftwf_plan mPlanReferenceBkwd;
	fftwf_plan mPlanImageBkwd;

	void setup(int x, int y);
	void cleanup();

public:
	CrossCorrelator();
	~CrossCorrelator();

	FilterPoint2D GetShiftGauss(float* aImage, unsigned char * aCC, int aDimX, int aDimY, int aRadius);
	void SetReference(float* aReference, int aDimX, int aDimY);
	void SetGaussBlobAsReference(int aDimX, int aDimY, int aRadius);
	void Smooth(float* aImage, int aDimX, int aDimY);
	bool CopyPatch(float* aImageSource, float* aPatch, int aDimImageX, int aDimImageY, int aPatchSize, int aX, int aY);


};

#endif