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


#include "CTFModel.h"
#define _USE_MATH_DEFINES
#include <math.h>

		CTFModel::CTFModel(float defocus, float pixelsize, float pixelcount, float openingAngle, float ampContrast)
		{
            RelativisticCorrectionFactor = (1 + _voltage / (E0 * 1000))/(1 + ((_voltage*1000) / (2 * E0 * 1000)));
            H = (Cc * EnergySpread * RelativisticCorrectionFactor) / (_voltage * 1000);
            _absolut = false;

			_defocus = defocus * 0.000000001f;
			_openingAngle = openingAngle / 1000.0f;
			_ampContrast = ampContrast;
			_pixelsize = pixelsize * powf(10, -9);
			_pixelcount = pixelcount;
			_maxFreq = 1.0 / (_pixelsize * 2.0f);
			_freqStepSize = _maxFreq / (_pixelcount / 2.0f);
			Xvalues = new float[(int)pixelcount / 2];
			Yvalues = new float[(int)pixelcount / 2];
			Xaxis = new float[(int)pixelcount / 2];
            _envelope = new float[(int)pixelcount / 2];

			for (int i = 0; i < pixelcount / 2; i++)
			{
				Xvalues[i] = i * _freqStepSize;
                Xaxis[i] = 1.0 / ((i * _freqStepSize) * 0.000000001f);
			}
            Xaxis[0] = 0;
			_phaseContrast = sqrt(1 - _ampContrast * _ampContrast);

            lambda = (h * c) / sqrtf(((2 * E0 * _voltage * 1000.0f * 1000.0f) + (_voltage * _voltage * 1000.0f * 1000.0f)) * 1.602E-19 * 1.602E-19);

            _CTFImage = new float[(int)pixelcount * (int)pixelcount];
		}

		float* CTFModel::GetCTF()
        {
            for (int i = 0; i < _pixelcount / 2; i++)
            {
                float o = expf(-14.238829f * (_openingAngle * _openingAngle * ((Cs * lambda * lambda * Xvalues[i] * Xvalues[i] * Xvalues[i] - _defocus * Xvalues[i]) * (Cs * lambda * lambda * Xvalues[i] * Xvalues[i] * Xvalues[i] - _defocus * Xvalues[i]))));
                float p = expf(-((0.943359f * lambda * Xvalues[i] * Xvalues[i] * H) * (0.943359f * lambda * Xvalues[i] * Xvalues[i] * H)));
                float q = (a1 * expf(-b1 * (Xvalues[i] * Xvalues[i])) + a2 * expf(-b2 * (Xvalues[i] * Xvalues[i]))) / 2.431f;

                float m = -PhaseShift + ((float)M_PI / 2.0f) * (Cs * lambda * lambda * lambda * Xvalues[i] * Xvalues[i] * Xvalues[i] * Xvalues[i] - 2 * _defocus * lambda * Xvalues[i] * Xvalues[i]);
                float n = _phaseContrast * sinf(m) + _ampContrast * cosf(m);

                float r = (o * p * ((1 - applyScatteringProfile) + (q * applyScatteringProfile))) * applyEnvelopeFunction + (1 - applyEnvelopeFunction);
                _envelope[i] = r;
                if (_absolut)
                    Yvalues[i] = abs(r * n);
                else
                    Yvalues[i] = r * n;
            }
            return Yvalues;

		}

		float* CTFModel::GetCTFImage()
        {
            for (int y = 0; y < _pixelcount; y++)
                for (int x = 0; x < _pixelcount; x++)
                {
                    float length = sqrtf((x-_pixelcount/2) * (x-_pixelcount/2) + (y-_pixelcount/2) * (y-_pixelcount/2));
                    length *= _freqStepSize;

                    float o = expf(-14.238829f * (_openingAngle * _openingAngle * ((Cs * lambda * lambda * length * length * length - _defocus * length) * (Cs * lambda * lambda * length * length * length - _defocus * length))));
                    float p = expf(-((0.943359f * lambda * length * length * H) * (0.943359f * lambda * length * length * H)));
                    float q = (a1 * expf(-b1 * (length * length)) + a2 * expf(-b2 * (length * length))) / 2.431f;

                    float m = -PhaseShift + ((float)M_PI / 2.0f) * (Cs * lambda * lambda * lambda * length * length * length * length - 2 * _defocus * lambda * length * length);
                    float n = _phaseContrast * sinf(m) + _ampContrast * cosf(m);

                    float r = (o * p * ((1 - applyScatteringProfile) + (q * applyScatteringProfile))) * applyEnvelopeFunction + (1 - applyEnvelopeFunction);

                    if (_absolut)
                        _CTFImage[y * (int)_pixelcount + x] = abs(r * n);
                    else
                        _CTFImage[y * (int)_pixelcount + x] = r * n;
                }

            return _CTFImage;

		}

		void CTFModel::SetDefocus(float value)
		{
		    _defocus = value * 0.000000001f;
		}
