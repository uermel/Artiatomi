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


#include "AvgProcess.h"
#include "io/EMFile.h"

#include <time.h>

void maxVals_t::getXYZ(int size, int& x, int& y, int&z)
{
	z = index / size /size;
	y = (index - z * size * size) / size;
	x = index - z * size * size - y * size;
	
	x -= size/2;
	y -= size/2;
	z -= size/2;
}

int maxVals_t::getIndex(int size, int x, int y, int z)
{
	x += size / 2;
	y += size / 2;
	z += size / 2;

	return z * size * size + y * size + x;
}

AvgProcess::AvgProcess(size_t _sizeVol, CUstream _stream, CudaContext* _ctx, float* _mask, float* _ref, float* _ccMask, float aPhiAngIter, float aPhiAngInc, float aAngIter, float aAngIncr, 
	bool aBinarizeMask, bool aRotateMaskCC, bool aUseFilterVolume, bool linearInterpolation)
	: sizeVol(_sizeVol), sizeTot(_sizeVol * _sizeVol * _sizeVol), stream(_stream), binarizeMask(aBinarizeMask), rotateMaskCC(aRotateMaskCC), useFilterVolume(aUseFilterVolume),
	  ctx(_ctx), 
	  rot((int)_sizeVol, _stream, _ctx, linearInterpolation), 
	  rotMask((int)_sizeVol, _stream, _ctx, linearInterpolation),
	  rotMaskCC((int)_sizeVol, _stream, _ctx, linearInterpolation),
	  reduce((int)_sizeVol * (int)_sizeVol * (int)_sizeVol, _stream, _ctx),
	  sub((int)_sizeVol, _stream, _ctx),
	  makecplx((int)_sizeVol, _stream, _ctx),
	  binarize((int)_sizeVol, _stream, _ctx),
	  mul((int)_sizeVol, _stream, _ctx),
	  fft((int)_sizeVol, _stream, _ctx),
	  max(_stream, _ctx), 
	  mask(_mask),
	  ref(_ref),
	  //wedges(_wedges),
	  ccMask(_ccMask),
	  d_ffttemp(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	  d_particle(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	  d_wedge(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	  d_filter(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	  d_reference_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	  d_mask_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	  d_reference(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	  d_mask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	  d_ccMask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	  d_ccMask_Orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	  d_particleCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	  d_particleSqrCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	  d_particleCplx_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	  d_particleSqrCplx_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	  d_referenceCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	  d_maskCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	  d_buffer(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)), //should be sufficient for everything...
	  d_index(_sizeVol * _sizeVol * _sizeVol * sizeof(int)), //should be sufficient for everything...
	  nVox(_sizeVol * _sizeVol * _sizeVol * sizeof(float)), //should be sufficient for everything...
	  sum(_sizeVol * _sizeVol * _sizeVol * sizeof(float)), //should be sufficient for everything...
	  sumSqr(_sizeVol * _sizeVol * _sizeVol * sizeof(float)), //should be sufficient for everything...
	  maxVals(sizeof(maxVals_t)), //should be sufficient for everything...
	  phi_angiter(aPhiAngIter),
	  phi_angincr(aPhiAngInc),
	  angiter(aAngIter),
	  angincr(aAngIncr)
{
	cudaSafeCall(cuMemAllocHost((void**)&index, sizeof(int)));
	cudaSafeCall(cuMemAllocHost((void**)&sum_h, sizeof(float)));
	cudaSafeCall(cuMemAllocHost((void**)&sumCplx, sizeof(float2)));

	int n[] = { (int)sizeVol, (int)sizeVol, (int)sizeVol };
	cufftSafeCall(cufftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 1));
	cufftSafeCall(cufftSetStream(ffthandle, stream));

	d_reference.CopyHostToDevice(ref);

	reduce.Sum(d_reference, d_buffer);
	float sum = 0;
	d_buffer.CopyDeviceToHost(&sum, sizeof(float));
	sum = sum / sizeTot;
	sub.Sub(d_reference, d_reference_orig, sum);

	d_mask_orig.CopyHostToDevice(mask);

	d_ccMask_Orig.CopyHostToDevice(ccMask);
	d_ccMask.CopyHostToDevice(ccMask);

	/*for (auto iter : wedges)
	{
		d_particle.CopyHostToDevice(iter.second->GetData());
		fft.FFTShiftReal(d_particle, d_wedge);
		d_wedge.CopyDeviceToHost(iter.second->GetData());
	}	*/
}

AvgProcess::~AvgProcess()
{
	cufftDestroy(ffthandle);
}

maxVals_t AvgProcess::execute(float* _data, float* wedge, float* filter, float oldphi, float oldpsi, float oldtheta, float rDown, float rUp, float smooth, float3 oldShift, bool couplePhiToPsi, bool computeCCValOnly, int oldIndex)
{
	int oldWedge = -1;
	maxVals_t m;
	m.index = 0;
	m.ccVal = -10000;
	m.rphi = 0;
	m.rpsi = 0;
	m.rthe = 0;
	cudaSafeCall(cuStreamSynchronize(stream));
	maxVals.CopyHostToDeviceAsync(stream, &m);
	/*
	makecplx.MakeCplxWithSub(d_reference, d_referenceCplx, 0);
	cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_FORWARD));
	fft.BandpassFFTShift(d_referenceCplx, rDown, rUp, smooth);
	cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_INVERSE));
	mul.Mul(1.0f / sizeTot, d_referenceCplx);
	makecplx.MakeReal(d_referenceCplx, d_reference_orig);
	
*/
	d_particle.CopyHostToDevice(wedge);
	fft.FFTShiftReal(d_particle, d_wedge);

	if (useFilterVolume)
	{
		d_particle.CopyHostToDevice(filter);
		fft.FFTShiftReal(d_particle, d_filter);
	}
	
	d_particle.CopyHostToDeviceAsync(stream, _data);
	makecplx.MakeCplxWithSub(d_particle, d_particleCplx_orig, 0);
	cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), CUFFT_FORWARD));

	mul.MulVol(d_wedge, d_particleCplx_orig);
	
	if (useFilterVolume)
	{
		mul.MulVol(d_filter, d_particleCplx_orig);
	}
	else
	{
		fft.BandpassFFTShift(d_particleCplx_orig, rDown, rUp, smooth);
	}

	cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), CUFFT_INVERSE));
	mul.Mul(1.0f / sizeTot, d_particleCplx_orig);
	makecplx.MakeReal(d_particleCplx_orig, d_particle);

	
	reduce.Sum(d_particle, d_buffer);

	d_buffer.CopyDeviceToHostAsync(stream, sum_h, sizeof(float));

	cudaSafeCall(cuStreamSynchronize(stream));
	
	makecplx.MakeCplxWithSub(d_particle, d_particleCplx_orig, *sum_h / sizeTot);
	makecplx.MakeCplxWithSqrSub(d_particle, d_particleSqrCplx_orig, *sum_h / sizeTot);
	
	
	cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), CUFFT_FORWARD));
	cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleSqrCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleSqrCplx_orig.GetDevicePtr(), CUFFT_FORWARD));	

	mul.MulVol(d_wedge, d_particleCplx_orig);
	
	if (useFilterVolume)
	{
		mul.MulVol(d_filter, d_particleCplx_orig);
	}
	else
	{
		fft.BandpassFFTShift(d_particleCplx_orig, rDown, rUp, smooth);
	}

	/*float* check = new float[128*64*64];
	d_particleCplx_orig.CopyDeviceToHost(check);
	emwrite("c:\\users\\kunz\\Desktop\\check.em", check, 128, 64, 64);
	exit(0);*/

	rot.SetTexture(d_reference_orig);
	rotMask.SetTexture(d_mask_orig);

	if (rotateMaskCC)
	{
		rotMaskCC.SetTexture(d_ccMask_Orig);
		rotMaskCC.SetOldAngles(oldphi, oldpsi, oldtheta);
	}
	//rotMaskCC.SetTextureShift(d_ccMask_Orig);

	rot.SetOldAngles(oldphi, oldpsi, oldtheta);
	rotMask.SetOldAngles(oldphi, oldpsi, oldtheta);
	//rotMaskCC.SetOldAngles(oldphi, oldpsi, oldtheta);
	//rotMaskCC.SetOldAngles(0, 0, 0);

	//for angle...
	float rthe = 0;
	float rpsi = 0;
	float rphi = 0;
	float maxthe = 0;
	float maxpsi = 0;
	float maxphi = 0;
	
	
	float npsi, dpsi;

	int counter = 0;
	
	double diff1 = 0;
	double diff2 = 0;
	double diff3 = 0;
	double diff4 = 0;
	
	float maxTest = -1000;
	int maxindex = -1000;

	for (int iterPhi = 0; iterPhi < 2 * phi_angiter + 1; ++iterPhi)
	{
        rphi = phi_angincr * (iterPhi - phi_angiter);
        for (int iterThe = (int)0; iterThe < angiter+1; ++iterThe)
		{
            if (iterThe == 0) 
			{
                npsi=1;
                dpsi=360;
            }
            else
			{
                dpsi=angincr / sinf(iterThe * angincr * (float)M_PI /180.0f);
                npsi = ceilf(360.0f / dpsi);
            }
            rthe=iterThe * angincr;
            for (int iterPsi = 0; iterPsi< npsi; ++iterPsi) 
			{
                rpsi=iterPsi * dpsi;

				if (couplePhiToPsi)
				{
					rphi = phi_angincr * (iterPhi - phi_angiter) - rpsi;
				}
				else
				{
					rphi = phi_angincr * (iterPhi - phi_angiter);
				}

				d_particleCplx.CopyDeviceToDeviceAsync(stream, d_particleCplx_orig);
				d_particleSqrCplx.CopyDeviceToDeviceAsync(stream, d_particleSqrCplx_orig);
				
				rot.Rot(d_reference, rphi, rpsi, rthe);
				rotMask.Rot(d_mask, rphi, rpsi, rthe);
				
				if (rotateMaskCC)
				{
					d_ccMask.Memset(0);
					rotMaskCC.Rot(d_ccMask, rphi, rpsi, rthe);
				}
				//rotMaskCC.Rot(d_ccMask, 0, 0, 0);
				//rotMaskCC.Shift(d_ccMask, make_float3(oldShift.x, oldShift.y, oldShift.z) );
				/*cout << rotateMaskCC << endl;
	float* check = new float[64*64*64];
	d_ccMask.CopyDeviceToHost(check);
	emwrite("check.em", check, 64, 64, 64);
	exit(0);*/

				if (binarizeMask)
				{
					binarize.Binarize(d_mask, d_mask);
				}
				reduce.Sum(d_mask, nVox);
	
				makecplx.MakeCplxWithSub(d_reference, d_referenceCplx, 0);
				makecplx.MakeCplxWithSub(d_mask, d_maskCplx, 0);
					
				cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_FORWARD));

				mul.MulVol(d_wedge, d_referenceCplx);

				cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_maskCplx.GetDevicePtr(), (cufftComplex*)d_maskCplx.GetDevicePtr(), CUFFT_FORWARD));
		
				if (useFilterVolume)
				{
					mul.MulVol(d_filter, d_referenceCplx);
				}
				else
				{
					fft.BandpassFFTShift(d_referenceCplx, rDown, rUp, smooth);
				}

				cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_INVERSE));

				mul.Mul(1.0f / sizeTot, d_referenceCplx);
				mul.MulVol(d_mask, d_referenceCplx);
	
				reduce.SumCplx(d_referenceCplx, sum);
	/*cudaSafeCall(cuStreamSynchronize(stream));
				float summe = 0;
				sum.CopyDeviceToHost(&summe, 4);
				cout << "Summe: " << summe << endl;*/
	
				sub.SubCplx(d_referenceCplx, d_referenceCplx, sum, nVox);
				mul.MulVol(d_mask, d_referenceCplx);
	
				reduce.SumSqrCplx(d_referenceCplx, sumSqr);	
	
				cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_FORWARD));

				fft.Correl(d_particleCplx, d_referenceCplx);

				fft.Conv(d_maskCplx, d_particleCplx);
				fft.Conv(d_maskCplx, d_particleSqrCplx);
	
				cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_INVERSE));
				mul.Mul(1.0f / sizeTot, d_referenceCplx);
	
				cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx.GetDevicePtr(), (cufftComplex*)d_particleCplx.GetDevicePtr(), CUFFT_INVERSE));
				mul.Mul(1.0f / sizeTot, d_particleCplx);
	
				cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleSqrCplx.GetDevicePtr(), (cufftComplex*)d_particleSqrCplx.GetDevicePtr(), CUFFT_INVERSE));
				mul.Mul(1.0f / sizeTot, d_particleSqrCplx);
				
				fft.EnergyNorm(d_particleCplx, d_particleSqrCplx, d_referenceCplx, sumSqr, nVox);
				
				fft.FFTShift2(d_referenceCplx, d_ffttemp);
				
	/*float* check = new float[d_ffttemp.GetSize()];
	d_ffttemp.CopyDeviceToHost(check);
	emwrite("c:\\users\\kunz\\Desktop\\check.em", check, 400, 200, 200);
	exit(0);*/

				mul.MulVol(d_ccMask, d_ffttemp);
				counter++;

				if (computeCCValOnly)
				{
					//only read out the CC value at the old shift position and store it in d_buffer
					d_index.CopyHostToDevice(&oldIndex);					
					cudaSafeCall(cuMemcpy(d_buffer.GetDevicePtr(), d_ffttemp.GetDevicePtr() + oldIndex, sizeof(float)));
				}
				else
				{
					//find new Maximum value and store position and value
					reduce.MaxIndexCplx(d_ffttemp, d_buffer, d_index);
				}

				max.Max(maxVals, d_index, d_buffer, rphi, rpsi, rthe);

			}
		}
	}
	
	cudaSafeCall(cuStreamSynchronize(stream));
	maxVals.CopyDeviceToHost(&m);
	cudaSafeCall(cuStreamSynchronize(stream));

	return m;
}


maxVals_t AvgProcess::executePhaseCorrelation(float* _data, float* wedge, float* filter, float oldphi, float oldpsi, float oldtheta, float rDown, float rUp, float smooth, float3 oldShift, bool couplePhiToPsi, bool computeCCValOnly, int oldIndex, int certaintyDistance)
{
	int oldWedge = -1;
	maxVals_t m;
	m.index = 0;
	m.ccVal = -10000;
	m.rphi = 0;
	m.rpsi = 0;
	m.rthe = 0;


	d_particle.CopyHostToDevice(wedge);
	fft.FFTShiftReal(d_particle, d_wedge);

	//we need to know what the maximum correlation value would be. inverse Fourier transfrom an ideal correlation result to get the normalization factor
	d_particle.Memset(0);
	makecplx.MakeCplxWithSub(d_particle, d_particleCplx_orig, -1); //flat spectrum everywhere +1
	mul.MulVol(d_wedge, d_particleCplx_orig);

	//apply fourier filter: put more weight on "good" frequencies
	if (useFilterVolume)
	{
		mul.MulVol(d_filter, d_particleCplx_orig);
	}
	else
	{
		fft.BandpassFFTShift(d_particleCplx_orig, rDown, rUp, smooth);
	}
	cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), CUFFT_INVERSE));
	//FFT normalization
	mul.Mul(1.0f / sizeTot, d_particleCplx_orig);

	fft.FFTShift2(d_particleCplx_orig, d_ffttemp);
	//find new Maximum value and store position and value
	reduce.MaxIndexCplx(d_ffttemp, d_buffer, d_index);
	float maxPCCValue;
	d_buffer.CopyDeviceToHost(&maxPCCValue, sizeof(float));





	cudaSafeCall(cuStreamSynchronize(stream));
	maxVals.CopyHostToDeviceAsync(stream, &m);	


	if (useFilterVolume)
	{
		d_particle.CopyHostToDevice(filter);
		fft.FFTShiftReal(d_particle, d_filter);
	}

	d_particle.CopyHostToDeviceAsync(stream, _data);
	makecplx.MakeCplxWithSub(d_particle, d_particleCplx_orig, 0);
	

	rot.SetTexture(d_reference_orig);
	rotMask.SetTexture(d_mask_orig);

	if (rotateMaskCC)
	{
		rotMaskCC.SetTexture(d_ccMask_Orig);
		rotMaskCC.SetOldAngles(oldphi, oldpsi, oldtheta);
	}
	//rotMaskCC.SetTextureShift(d_ccMask_Orig);

	rot.SetOldAngles(oldphi, oldpsi, oldtheta);
	rotMask.SetOldAngles(oldphi, oldpsi, oldtheta);
	//rotMaskCC.SetOldAngles(oldphi, oldpsi, oldtheta);
	//rotMaskCC.SetOldAngles(0, 0, 0);

	//for angle...
	float rthe = 0;
	float rpsi = 0;
	float rphi = 0;
	float maxthe = 0;
	float maxpsi = 0;
	float maxphi = 0;


	float npsi, dpsi;

	int counter = 0;

	double diff1 = 0;
	double diff2 = 0;
	double diff3 = 0;
	double diff4 = 0;

	float maxTest = -1000;
	int maxindex = -1000;

	for (int iterPhi = 0; iterPhi < 2 * phi_angiter + 1; ++iterPhi)
	{
		rphi = phi_angincr * (iterPhi - phi_angiter);
		for (int iterThe = (int)0; iterThe < angiter + 1; ++iterThe)
		{
			if (iterThe == 0)
			{
				npsi = 1;
				dpsi = 360;
			}
			else
			{
				dpsi = angincr / sinf(iterThe * angincr * (float)M_PI / 180.0f);
				npsi = ceilf(360.0f / dpsi);
			}
			rthe = iterThe * angincr;
			for (int iterPsi = 0; iterPsi < npsi; ++iterPsi)
			{
				rpsi = iterPsi * dpsi;

				if (couplePhiToPsi)
				{
					rphi = phi_angincr * (iterPhi - phi_angiter) - rpsi;
				}
				else
				{
					rphi = phi_angincr * (iterPhi - phi_angiter);
				}

				d_particleCplx.CopyDeviceToDeviceAsync(stream, d_particleCplx_orig);

				rot.Rot(d_reference, rphi, rpsi, rthe);
				rotMask.Rot(d_mask, rphi, rpsi, rthe);

				if (rotateMaskCC)
				{
					d_ccMask.Memset(0);
					rotMaskCC.Rot(d_ccMask, rphi, rpsi, rthe);
				}

				makecplx.MakeCplxWithSub(d_reference, d_referenceCplx, 0);
				mul.MulVol(d_mask, d_referenceCplx);
				mul.MulVol(d_mask, d_particleCplx);

				cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_FORWARD));
				cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx.GetDevicePtr(), (cufftComplex*)d_particleCplx.GetDevicePtr(), CUFFT_FORWARD));

				fft.PhaseCorrel(d_particleCplx, d_referenceCplx);
				//d_referenceCplx contains phase correlation result

				//apply dose weighting: Put more weight on the phases we can trust
				mul.MulVol(d_wedge, d_referenceCplx);
				
				//apply fourier filter: put more weight on "good" frequencies
				if (useFilterVolume)
				{
					mul.MulVol(d_filter, d_referenceCplx);
				}
				else
				{
					fft.BandpassFFTShift(d_referenceCplx, rDown, rUp, smooth);
				}

				if (certaintyDistance >= 0)
				{
					fft.SplitDataset(d_referenceCplx, d_particleCplx, d_particleSqrCplx);
					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx.GetDevicePtr(), (cufftComplex*)d_particleCplx.GetDevicePtr(), CUFFT_INVERSE));
					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleSqrCplx.GetDevicePtr(), (cufftComplex*)d_particleSqrCplx.GetDevicePtr(), CUFFT_INVERSE));
					//FFT normalization
					mul.Mul(1.0f / sizeTot / maxPCCValue, d_particleCplx);
					mul.Mul(1.0f / sizeTot / maxPCCValue, d_particleSqrCplx);


					fft.FFTShift2(d_particleCplx, d_ffttemp);
					mul.MulVol(d_ccMask, d_ffttemp);
					reduce.MaxIndexCplx(d_ffttemp, d_buffer, sum);

					/*float* test2 = new float[sizeTot * 2];
					d_ffttemp.CopyDeviceToHost(test2, sizeTot * 2 * 4);
					emwrite("C:\\Users\\kunz_\\Desktop\\Data\\Average\\checkA.em", test2, 96 * 2, 96, 96);*/

					fft.FFTShift2(d_particleSqrCplx, d_ffttemp);
					mul.MulVol(d_ccMask, d_ffttemp);
					reduce.MaxIndexCplx(d_ffttemp, d_buffer, sumSqr);

					/*d_ffttemp.CopyDeviceToHost(test2, sizeTot * 2 * 4);
					emwrite("C:\\Users\\kunz_\\Desktop\\Data\\Average\\checkB.em", test2, 96 * 2, 96, 96);
					delete[] test2;*/
				}

				/*float* test2 = new float[sizeTot * 2];
				d_referenceCplx.CopyDeviceToHost(test2, sizeTot * 2 * 4);

				emwrite("C:\\Users\\kunz_\\Desktop\\Data\\Average\\check2.em", test2, 96 * 2, 96, 96);
				delete[] test2;*/

				cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_INVERSE));
				//FFT normalization
				mul.Mul(1.0f / sizeTot / maxPCCValue, d_referenceCplx);

				fft.FFTShift2(d_referenceCplx, d_ffttemp);

				/*float* test = new float[sizeTot * 2];
				d_ffttemp.CopyDeviceToHost(test, sizeTot * 2 * 4);
				
				emwrite("C:\\Users\\kunz_\\Desktop\\Data\\Average\\check.em", test, 96*2, 96, 96);
				delete[] test;*/
	

				mul.MulVol(d_ccMask, d_ffttemp);
				counter++;

				if (computeCCValOnly)
				{
					//only read out the CC value at the old shift position and store it in d_buffer
					d_index.CopyHostToDevice(&oldIndex);
					cudaSafeCall(cuMemcpy(d_buffer.GetDevicePtr(), d_ffttemp.GetDevicePtr() + oldIndex, sizeof(float)));
				}
				else
				{
					//find new Maximum value and store position and value
					reduce.MaxIndexCplx(d_ffttemp, d_buffer, d_index);
				}

				if (certaintyDistance >= 0)
				{
					max.MaxWithCertainty(maxVals, d_index, d_buffer, sum, sumSqr, rphi, rpsi, rthe, sizeVol, certaintyDistance);
				}
				else
				{
					max.Max(maxVals, d_index, d_buffer, rphi, rpsi, rthe);
				}

			}
		}
	}

	cudaSafeCall(cuStreamSynchronize(stream));
	maxVals.CopyDeviceToHost(&m);
	cudaSafeCall(cuStreamSynchronize(stream));

	return m;
}


