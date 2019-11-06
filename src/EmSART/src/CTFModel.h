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


#ifndef CTFMODEL_H
#define CTFMODEL_H

#include "default.h"
class CTFModel
{
    private:
        float _defocus;
		float _openingAngle;
		float _ampContrast;
		float _phaseContrast;
		float _pixelsize;
		float _pixelcount;
		float _maxFreq;
		float _freqStepSize;
		float* _CTFImage;
		float* Xvalues;
		float* Yvalues;
		float* Xaxis;
		float* _envelope;
		float applyScatteringProfile;
		float applyEnvelopeFunction;

		const static float _voltage = 300;
		const static float h = 6.63E-34; //Planck's quantum
		const static float c = 3.00E+08; //Light speed
		const static float Cs = 2.7 * 0.001;
		const static float Cc = 2.7 * 0.001;

		const static float PhaseShift = 0;
		const static float EnergySpread = 0.7; //eV
		const static float E0 = 511; //keV
		float RelativisticCorrectionFactor;// = (1 + _voltage / (E0 * 1000))/(1 + ((_voltage*1000) / (2 * E0 * 1000)));
		float H;// = (Cc * EnergySpread * RelativisticCorrectionFactor) / (_voltage * 1000);

		const static float a1 = 1.494; //Scat.Profile Carbon Amplitude 1
		const static float a2 = 0.937; //Scat.Profile Carbon Amplitude 2
		const static float b1 = 23.22 * 1E-20; //Scat.Profile Carbon Halfwidth 1
        const static float b2 = 3.79 * 1E-20;  //Scat.Profile Carbon Halfwidth 2
        float lambda;
        bool _absolut;

public:
		CTFModel(float defocus, float pixelsize, float pixelcount, float openingAngle, float ampContrast);

		float* GetCTF();

		float* GetCTFImage();

		void SetDefocus(float value);

};

#endif // CTFMODEL_H
