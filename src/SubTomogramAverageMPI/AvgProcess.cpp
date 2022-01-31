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

AvgProcess::AvgProcess(size_t _sizeVol,
                       CUstream _stream,
                       CudaContext* _ctx,
                       float* _mask,
                       float* _ref,
                       float* _ccMask,
                       bool aBinarizeMask,
                       bool aRotateMaskCC,
                       bool aUseFilterVolume,
                       bool linearInterpolation)

	: sizeVol(_sizeVol),
      sizeTot(_sizeVol * _sizeVol * _sizeVol),
      stream(_stream),
      binarizeMask(aBinarizeMask),
      rotateMaskCC(aRotateMaskCC),
      useFilterVolume(aUseFilterVolume),
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
      // Temp storage
	  d_ffttemp(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	  d_buffer(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)), //should be sufficient for everything...
	  d_index(_sizeVol * _sizeVol * _sizeVol * sizeof(int)), //should be sufficient for everything...
	  d_sum(_sizeVol * _sizeVol * _sizeVol * sizeof(float)), //should be sufficient for everything...
	  d_maxVals(sizeof(maxVals_t)), //should be sufficient for everything...
      d_real_tmp(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      // Freq space filters
      d_real_cov_wedge(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_real_ovl_wedge(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_real_filter(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_real_ccMask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_real_ccMask_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      // Particle image
      d_real_f2(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_cplx_F2(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_cplx_F2sqr(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_cplx_F2_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_cplx_F2sqr_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_cplx_NCCNum(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      // Mask image
      d_real_mask1(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_real_mask1_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_real_maskNorm(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_cplx_M1(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      // Reference image
      d_real_f1(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_real_f1_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_cplx_F1(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_cplx_f1sqr(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_real_NCCDen1(_sizeVol * _sizeVol * _sizeVol * sizeof(float))
{
	cudaSafeCall(cuMemAllocHost((void**)&index, sizeof(int)));
	cudaSafeCall(cuMemAllocHost((void**)&sum_h, sizeof(float)));
	cudaSafeCall(cuMemAllocHost((void**)&sumCplx, sizeof(float2)));

	int n[] = { (int)sizeVol, (int)sizeVol, (int)sizeVol };
	cufftSafeCall(cufftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 1));
	cufftSafeCall(cufftSetStream(ffthandle, stream));

	d_real_f1.CopyHostToDevice(ref);

	reduce.Sum(d_real_f1, d_buffer);
	float refsum = 0.f;
	d_buffer.CopyDeviceToHost(&refsum, sizeof(float));
    refsum = refsum / (float)sizeTot;
	sub.Sub(d_real_f1, d_real_f1_orig, refsum);

	d_real_mask1_orig.CopyHostToDevice(mask);

	d_real_ccMask_orig.CopyHostToDevice(ccMask);
	d_real_ccMask.CopyHostToDevice(ccMask);
}

AvgProcess::~AvgProcess()
{
	cufftDestroy(ffthandle);
}

void AvgProcess::planAngularSampling(float aPhiAngIter,
                                     float aPhiAngInc,
                                     float aAngIter,
                                     float aAngIncr,
                                     bool aCouplePhiToPsi)
{
    float rphi, npsi, dpsi, rpsi, rthe;

    for (int iterPhi = 0; iterPhi < 2 * (int)aPhiAngIter + 1; ++iterPhi) {
        rphi = aPhiAngInc * ((float)iterPhi - aPhiAngIter);

        for (int iterThe = (int) 0; iterThe < (int)aAngIter + 1; ++iterThe) {
            rthe = (float)iterThe * aAngIncr;

            if (iterThe == 0) {
                npsi = 1;
                dpsi = 360;
            } else {
                dpsi = aAngIncr / sinf((float)iterThe * aAngIncr * (float) M_PI / 180.0f);
                npsi = ceilf(360.0f / dpsi);
            }

            for (int iterPsi = 0; iterPsi < (int)npsi; ++iterPsi) {
                rpsi = (float)iterPsi * dpsi;

                if (aCouplePhiToPsi) {
                    rphi = aPhiAngInc * ((float)iterPhi - aPhiAngIter) - rpsi;
                } else {
                    rphi = aPhiAngInc * ((float)iterPhi - aPhiAngIter);
                }

                vector<float> angles = {rphi, rpsi, rthe};
                angleList.push_back(angles);
            }
        }
    }

//    printf("ANGLES Planned: \n");
//    auto angles = new float[angleList.size()*3];
//    for (int trpIdx = 0; trpIdx < angleList.size(); trpIdx++)
//    {
//        printf("phi: %f psi: %f theta: %f\n", angleList[trpIdx][0], angleList[trpIdx][1], angleList[trpIdx][2]);
//        angles[trpIdx*3+0] = angleList[trpIdx][0];
//        angles[trpIdx*3+1] = angleList[trpIdx][1];
//        angles[trpIdx*3+2] = angleList[trpIdx][2];
//    }
//
//    stringstream ss;
//    ss << "testangles.em";
//    emwrite(ss.str(), angles, 3, angleList.size());
//    delete[] angles;

}

void AvgProcess::setAngularSampling(const float* customAngles,
                                    int customAngleNum)
{
    for (int trpIdx = 0; trpIdx < customAngleNum; trpIdx++){
        float rphi = customAngles[trpIdx*3+0];
        float rpsi = customAngles[trpIdx*3+1];
        float rthe = customAngles[trpIdx*3+2];

        vector<float> angles = {rphi, rpsi, rthe};
        angleList.push_back(angles);
    }

    printf("ANGLES Custom: \n");
    printf("size: %i\n", angleList.size());
    for (int trpIdx = 0; trpIdx < angleList.size(); trpIdx++)
    {
        printf("phi: %f psi: %f theta: %f\n", angleList[trpIdx][0], angleList[trpIdx][1], angleList[trpIdx][2]);
    }
}

maxVals_t AvgProcess::executePadfield(float* _data,
                                      float* coverageWedge,
                                      float* overlapWedge,
                                      float* filter,
                                      float oldphi,
                                      float oldpsi,
                                      float oldtheta,
                                      float rDown,
                                      float rUp,
                                      float smooth,
                                      float3 oldShift,
                                      bool computeCCValOnly,
                                      int oldIndex)
{
    int oldWedge = -1;
    maxVals_t m;
    m.index = 0;
    m.ccVal = -10000;
    m.rphi = 0;
    m.rpsi = 0;
    m.rthe = 0;
    cudaSafeCall(cuStreamSynchronize(stream));
    d_maxVals.CopyHostToDeviceAsync(stream, &m);

    ////////////////////
    /// Prep Filters ///
    ////////////////////
    // FFTshift Coverage Wedge
    d_real_tmp.CopyHostToDevice(coverageWedge);
    fft.FFTShiftReal(d_real_tmp, d_real_cov_wedge);

    d_real_tmp.CopyHostToDevice(overlapWedge);
    fft.FFTShiftReal(d_real_tmp, d_real_ovl_wedge);

    // FFTshift Filter
    if (useFilterVolume)
    {
        d_real_tmp.CopyHostToDevice(filter);
        fft.FFTShiftReal(d_real_tmp, d_real_filter);
    }

    ///////////////////////////
    /// Prep Particle image ///
    ///////////////////////////
    // FFT of particle
    d_real_tmp.CopyHostToDeviceAsync(stream, _data);
    makecplx.MakeCplxWithSub(d_real_tmp, d_cplx_F2, 0);
    cufftSafeCall(cufftExecC2C(ffthandle,
                               (cufftComplex*)d_cplx_F2.GetDevicePtr(),
                               (cufftComplex*)d_cplx_F2.GetDevicePtr(),
                               CUFFT_FORWARD));

    // Particle * wedge (coverage)
    //TODO:change back
    //mul.MulVol(d_real_cov_wedge, d_cplx_F2);
    fft.ParticleWiener(d_cplx_F2, d_real_ovl_wedge, d_real_cov_wedge, 1.f);

    // Filter particle
    if (useFilterVolume)
    {
        mul.MulVol(d_real_filter, d_cplx_F2);
    }
    else
    {
        fft.BandpassFFTShift(d_cplx_F2, rDown, rUp, smooth);
    }

    // IFFT particle
    cufftSafeCall(cufftExecC2C(ffthandle,
                               (cufftComplex*)d_cplx_F2.GetDevicePtr(),
                               (cufftComplex*)d_cplx_F2.GetDevicePtr(),
                               CUFFT_INVERSE));

    mul.Mul(1.0f / sizeTot, d_cplx_F2);
    makecplx.MakeReal(d_cplx_F2, d_real_f2);

    // Particle sum
    reduce.Sum(d_real_f2, d_buffer);
    d_buffer.CopyDeviceToHostAsync(stream, sum_h, sizeof(float));

    cudaSafeCall(cuStreamSynchronize(stream));

    // Subtract mean, FFT of particle, FFT of particle^2
    makecplx.MakeCplxWithSub(d_real_f2, d_cplx_F2_orig, *sum_h / (float)sizeTot);
    makecplx.MakeCplxWithSqrSub(d_real_f2, d_cplx_F2sqr_orig, *sum_h / (float)sizeTot);

    // F2 and F2^2
    cufftSafeCall(cufftExecC2C(ffthandle,
                               (cufftComplex*)d_cplx_F2_orig.GetDevicePtr(),
                               (cufftComplex*)d_cplx_F2_orig.GetDevicePtr(),
                               CUFFT_FORWARD));

    cufftSafeCall(cufftExecC2C(ffthandle,
                               (cufftComplex*)d_cplx_F2sqr_orig.GetDevicePtr(),
                               (cufftComplex*)d_cplx_F2sqr_orig.GetDevicePtr(),
                               CUFFT_FORWARD));


    /////////////////////
    /// Prep Rotation ///
    /////////////////////
    // Setup rotation of ref/mask/maskCC
    rot.SetTexture(d_real_f1_orig);
    rotMask.SetTexture(d_real_mask1_orig);

    rot.SetOldAngles(oldphi, oldpsi, oldtheta);
    rotMask.SetOldAngles(oldphi, oldpsi, oldtheta);

    if (rotateMaskCC)
    {
        rotMaskCC.SetTexture(d_real_ccMask_orig);
        rotMaskCC.SetOldAngles(oldphi, oldpsi, oldtheta);
    }

    //for angle...
    float rthe = 0;
    float rpsi = 0;
    float rphi = 0;

    int counter = 0;

    for (int trpIdx = 0; trpIdx < angleList.size(); trpIdx++)
    {
        vector<float> angles = angleList[trpIdx];
        rphi = angles[0];
        rpsi = angles[1];
        rthe = angles[2];

        // Get pre-processed particle
        d_cplx_F2.CopyDeviceToDeviceAsync(stream, d_cplx_F2_orig);
        d_cplx_NCCNum.CopyDeviceToDeviceAsync(stream, d_cplx_F2_orig);
        d_cplx_F2sqr.CopyDeviceToDeviceAsync(stream, d_cplx_F2sqr_orig);

        ///////////////////////////////
        /// Process reference image ///
        ///////////////////////////////

        // Get rotated ref/mask/maskCC
        rot.Rot(d_real_f1, rphi, rpsi, rthe);
        rotMask.Rot(d_real_mask1, rphi, rpsi, rthe);

        if (rotateMaskCC)
        {
            d_real_ccMask.Memset(0);
            rotMaskCC.Rot(d_real_ccMask, rphi, rpsi, rthe);
        }

        // Binarize mask if desired
        if (binarizeMask)
        {
            binarize.Binarize(d_real_mask1, d_real_mask1);
        }

        // Sum of mask
        reduce.Sum(d_real_mask1, d_real_maskNorm);

        // FFT of ref/mask
        makecplx.MakeCplxWithSub(d_real_f1, d_cplx_F1, 0);
        makecplx.MakeCplxWithSub(d_real_mask1, d_cplx_M1, 0);

        cufftSafeCall(cufftExecC2C(ffthandle,
                                   (cufftComplex*)d_cplx_F1.GetDevicePtr(),
                                   (cufftComplex*)d_cplx_F1.GetDevicePtr(),
                                   CUFFT_FORWARD));

        cufftSafeCall(cufftExecC2C(ffthandle,
                                   (cufftComplex*)d_cplx_M1.GetDevicePtr(),
                                   (cufftComplex*)d_cplx_M1.GetDevicePtr(),
                                   CUFFT_FORWARD));

        // Wedge * ref (overlap wedge)
        //TODO: change to ovl
        mul.MulVol(d_real_cov_wedge, d_cplx_F1);

        // Filter ref
        if (useFilterVolume)
        {
            mul.MulVol(d_real_filter, d_cplx_F1);
        }
        else
        {
            fft.BandpassFFTShift(d_cplx_F1, rDown, rUp, smooth);
        }

        // IFFT of ref
        cufftSafeCall(cufftExecC2C(ffthandle,
                                   (cufftComplex*)d_cplx_F1.GetDevicePtr(),
                                   (cufftComplex*)d_cplx_F1.GetDevicePtr(),
                                   CUFFT_INVERSE));

        mul.Mul(1.0f / sizeTot, d_cplx_F1);

//        {
//            auto temp = new float2[sizeTot];
//            auto tempx = new float[sizeTot];
//            auto tempy = new float[sizeTot];
//
//            d_cplx_F1.CopyDeviceToHost(temp);
//
//            for (int i = 0; i<sizeTot; i++){
//                tempx[i] = temp[i].x;
//                tempy[i] = temp[i].y;
//            }
//
//            stringstream ss;
//            ss << "F1_before_mask_real.em";
//            emwrite(ss.str(), tempx, sizeVol, sizeVol, sizeVol);
//
//            stringstream ss2;
//            ss2 << "F1_before_mask_imag.em";
//            emwrite(ss2.str(), tempy, sizeVol, sizeVol, sizeVol);
//
//            delete[] temp;
//            delete[] tempx;
//            delete[] tempy;
//        }

        // Sum of masked ref (real space)
        reduce.MaskedSumCplx(d_cplx_F1, d_real_mask1, d_sum);

//        {
//            float tmp1 = 0;
//            d_sum.CopyDeviceToHost(&tmp1, sizeof(float));
//
//            printf("\n RefSum: %f\n", tmp1);
//        }

        // Apply mask, generate masked ref, and masked, squared ref
        mul.MulMaskMeanFreeCplx(d_cplx_F1, d_cplx_f1sqr, d_real_mask1, d_sum, d_real_maskNorm);

//        {
//            auto temp = new float2[sizeTot];
//            auto tempx = new float[sizeTot];
//            auto tempy = new float[sizeTot];
//
//            d_cplx_F1.CopyDeviceToHost(temp);
//
//            for (int i = 0; i<sizeTot; i++){
//                tempx[i] = temp[i].x;
//                tempy[i] = temp[i].y;
//            }
//
//            stringstream ss;
//            ss << "F1_real.em";
//            emwrite(ss.str(), tempx, sizeVol, sizeVol, sizeVol);
//
//            stringstream ss2;
//            ss2 << "F1_imag.em";
//            emwrite(ss2.str(), tempy, sizeVol, sizeVol, sizeVol);
//
//            delete[] temp;
//            delete[] tempx;
//            delete[] tempy;
//        }

//        {
//            auto temp = new float2[sizeTot];
//            auto tempx = new float[sizeTot];
//            auto tempy = new float[sizeTot];
//
//            d_cplx_f1sqr.CopyDeviceToHost(temp);
//
//            for (int i = 0; i<sizeTot; i++){
//                tempx[i] = temp[i].x;
//                tempy[i] = temp[i].y;
//            }
//
//            stringstream ss;
//            ss << "F1sqr_real.em";
//            emwrite(ss.str(), tempx, sizeVol, sizeVol, sizeVol);
//
//            stringstream ss2;
//            ss2 << "F1sqr_imag.em";
//            emwrite(ss2.str(), tempy, sizeVol, sizeVol, sizeVol);
//
//            delete[] temp;
//            delete[] tempx;
//            delete[] tempy;
//        }

        // Sum of squared ref
        reduce.SumCplx(d_cplx_f1sqr, d_real_NCCDen1);

        // FFT of ref
        cufftSafeCall(cufftExecC2C(ffthandle,
                                   (cufftComplex*)d_cplx_F1.GetDevicePtr(),
                                   (cufftComplex*)d_cplx_F1.GetDevicePtr(),
                                   CUFFT_FORWARD));

        // NCCNum
        fft.Correl(d_cplx_F1, d_cplx_NCCNum);

        // NCCDen2 part 1 and 2
        fft.Correl(d_cplx_M1, d_cplx_F2);
        fft.Correl(d_cplx_M1, d_cplx_F2sqr);

        // IFFT NCCNum
        cufftSafeCall(cufftExecC2C(ffthandle,
                                   (cufftComplex*)d_cplx_NCCNum.GetDevicePtr(),
                                   (cufftComplex*)d_cplx_NCCNum.GetDevicePtr(),
                                   CUFFT_INVERSE));
        mul.Mul(1.0f / (float)sizeTot, d_cplx_NCCNum);

        // IFFT NCCDen2 part 1
        cufftSafeCall(cufftExecC2C(ffthandle,
                                   (cufftComplex*)d_cplx_F2.GetDevicePtr(),
                                   (cufftComplex*)d_cplx_F2.GetDevicePtr(),
                                   CUFFT_INVERSE));
        mul.Mul(1.0f / (float)sizeTot, d_cplx_F2);

        // IFFT NCCDen2 part 2
        cufftSafeCall(cufftExecC2C(ffthandle,
                                   (cufftComplex*)d_cplx_F2sqr.GetDevicePtr(),
                                   (cufftComplex*)d_cplx_F2sqr.GetDevicePtr(),
                                   CUFFT_INVERSE));
        mul.Mul(1.0f / (float)sizeTot, d_cplx_F2sqr);

//        {
//            auto temp = new float2[sizeTot];
//            auto tempx = new float[sizeTot];
//            auto tempy = new float[sizeTot];
//
//            d_cplx_NCCNum.CopyDeviceToHost(temp);
//
//            for (int i = 0; i<sizeTot; i++){
//                tempx[i] = temp[i].x;
//                tempy[i] = temp[i].y;
//            }
//
//            stringstream ss;
//            ss << "nccnum_real.em";
//            emwrite(ss.str(), tempx, sizeVol, sizeVol, sizeVol);
//
//            stringstream ss2;
//            ss2 << "nccnum_imag.em";
//            emwrite(ss2.str(), tempy, sizeVol, sizeVol, sizeVol);
//
//            delete[] temp;
//            delete[] tempx;
//            delete[] tempy;
//        }

//        {
//            auto temp = new float2[sizeTot];
//            auto tempx = new float[sizeTot];
//            auto tempy = new float[sizeTot];
//
//            d_cplx_F2sqr.CopyDeviceToHost(temp);
//
//            for (int i = 0; i<sizeTot; i++){
//                tempx[i] = temp[i].x;
//                tempy[i] = temp[i].y;
//            }
//
//            stringstream ss;
//            ss << "f2sqr_real.em";
//            emwrite(ss.str(), tempx, sizeVol, sizeVol, sizeVol);
//
//            stringstream ss2;
//            ss2 << "f2sqr_imag.em";
//            emwrite(ss2.str(), tempy, sizeVol, sizeVol, sizeVol);
//
//            delete[] temp;
//            delete[] tempx;
//            delete[] tempy;
//        }

//        {
//            auto temp = new float2[sizeTot];
//            auto tempx = new float[sizeTot];
//            auto tempy = new float[sizeTot];
//
//            d_cplx_F2.CopyDeviceToHost(temp);
//
//            for (int i = 0; i<sizeTot; i++){
//                tempx[i] = temp[i].x;
//                tempy[i] = temp[i].y;
//            }
//
//            stringstream ss;
//            ss << "f2_real.em";
//            emwrite(ss.str(), tempx, sizeVol, sizeVol, sizeVol);
//
//            stringstream ss2;
//            ss2 << "f2_imag.em";
//            emwrite(ss2.str(), tempy, sizeVol, sizeVol, sizeVol);
//
//            delete[] temp;
//            delete[] tempx;
//            delete[] tempy;
//        }

//        {
//            float tmp1 = 0;
//            float tmp2 = 0;
//            d_real_NCCDen1.CopyDeviceToHost(&tmp1, sizeof(float));
//            d_real_maskNorm.CopyDeviceToHost(&tmp2, sizeof(float));
//
//            printf("\n Den1: %f\n maskNorm: %f\n", tmp1, tmp2);
//        }

        // Normalize for mask
        //fft.EnergyNorm(d_cplx_NCCNum, d_cplx_F2sqr, d_cplx_F2, d_real_NCCDen1, d_real_maskNorm);
        fft.EnergyNormPadfield(d_cplx_NCCNum, d_cplx_F2sqr, d_cplx_F2, d_real_NCCDen1, d_real_maskNorm);

        // FFTshift normalized cc-result
        fft.FFTShift2(d_cplx_NCCNum, d_ffttemp);

//        {
//            auto temp = new float2[sizeTot];
//            auto tempx = new float[sizeTot];
//            auto tempy = new float[sizeTot];
//
//            d_ffttemp.CopyDeviceToHost(temp);
//
//            for (int i = 0; i<sizeTot; i++){
//                tempx[i] = temp[i].x;
//                tempy[i] = temp[i].y;
//            }
//
//            stringstream ss;
//            ss << "ccmap_real.em";
//            emwrite(ss.str(), tempx, sizeVol, sizeVol, sizeVol);
//
//            stringstream ss2;
//            ss2 << "ccmap_imag.em";
//            emwrite(ss2.str(), tempy, sizeVol, sizeVol, sizeVol);
//
//            delete[] temp;
//            delete[] tempx;
//            delete[] tempy;
//        }

        // Apply cc-mask
        mul.MulVol(d_real_ccMask, d_ffttemp);
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
//            {
//                auto tmp = new float[3];
//                d_index.
//                printf("INDEX");
//            }
        }

        // Get maximum
        max.Max(d_maxVals, d_index, d_buffer, rphi, rpsi, rthe);
    }

    cudaSafeCall(cuStreamSynchronize(stream));
    d_maxVals.CopyDeviceToHost(&m);
    cudaSafeCall(cuStreamSynchronize(stream));

    return m;
}

//maxVals_t AvgProcess::execute(float* _data,
//                              float* wedge,
//                              float* filter,
//                              float oldphi,
//                              float oldpsi,
//                              float oldtheta,
//                              float rDown,
//                              float rUp,
//                              float smooth,
//                              float3 oldShift,
//                              bool computeCCValOnly,
//                              int oldIndex)
//{
//	int oldWedge = -1;
//	maxVals_t m;
//	m.index = 0;
//	m.ccVal = -10000;
//	m.rphi = 0;
//	m.rpsi = 0;
//	m.rthe = 0;
//	cudaSafeCall(cuStreamSynchronize(stream));
//	maxVals.CopyHostToDeviceAsync(stream, &m);
//
//    // FFTshift Wedge
//	d_particle.CopyHostToDevice(wedge);
//	fft.FFTShiftReal(d_particle, d_wedge);
//
//    // FFTshift Filter
//	if (useFilterVolume)
//	{
//		d_particle.CopyHostToDevice(filter);
//		fft.FFTShiftReal(d_particle, d_filter);
//	}
//
//    // FFT of particle
//	d_particle.CopyHostToDeviceAsync(stream, _data);
//	makecplx.MakeCplxWithSub(d_particle, d_particleCplx_orig, 0);
//	cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), CUFFT_FORWARD));
//
//    // Particle * wedge (coverage)
//	mul.MulVol(d_wedge, d_particleCplx_orig);
//
//    // Filter particle
//	if (useFilterVolume)
//	{
//		mul.MulVol(d_filter, d_particleCplx_orig);
//	}
//	else
//	{
//		fft.BandpassFFTShift(d_particleCplx_orig, rDown, rUp, smooth);
//	}
//
//    // IFFT particle
//	cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), CUFFT_INVERSE));
//	mul.Mul(1.0f / sizeTot, d_particleCplx_orig);
//	makecplx.MakeReal(d_particleCplx_orig, d_particle);
//
//	// Particle sum
//	reduce.Sum(d_particle, d_buffer);
//	d_buffer.CopyDeviceToHostAsync(stream, sum_h, sizeof(float));
//
//	cudaSafeCall(cuStreamSynchronize(stream));
//
//    // Subtract mean, FFT of particle, FFT of particle^2
//	makecplx.MakeCplxWithSub(d_particle, d_particleCplx_orig, *sum_h / sizeTot);
//	makecplx.MakeCplxWithSqrSub(d_particle, d_particleSqrCplx_orig, *sum_h / sizeTot);
//
//	cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), CUFFT_FORWARD));
//	cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleSqrCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleSqrCplx_orig.GetDevicePtr(), CUFFT_FORWARD));
//
//    // Particle * Wedge
//	mul.MulVol(d_wedge, d_particleCplx_orig);
//
//    // Filter particle
//	if (useFilterVolume)
//	{
//		mul.MulVol(d_filter, d_particleCplx_orig);
//	}
//	else
//	{
//		fft.BandpassFFTShift(d_particleCplx_orig, rDown, rUp, smooth);
//	}
//
//    // Setup rotation of ref/mask/maskCC
//	rot.SetTexture(d_reference_orig);
//	rotMask.SetTexture(d_mask_orig);
//
//	if (rotateMaskCC)
//	{
//		rotMaskCC.SetTexture(d_ccMask_Orig);
//		rotMaskCC.SetOldAngles(oldphi, oldpsi, oldtheta);
//	}
//
//	rot.SetOldAngles(oldphi, oldpsi, oldtheta);
//	rotMask.SetOldAngles(oldphi, oldpsi, oldtheta);
//
//	//for angle...
//	float rthe = 0;
//	float rpsi = 0;
//	float rphi = 0;
//
//	int counter = 0;
//
//    for (int trpIdx = 0; trpIdx < angleList.size(); trpIdx++)
//    {
//        vector<float> angles = angleList[trpIdx];
//        rphi = angles[0];
//        rpsi = angles[1];
//        rthe = angles[2];
//
//        // Get pre-processed particle
//        d_particleCplx.CopyDeviceToDeviceAsync(stream, d_particleCplx_orig);
//        d_particleSqrCplx.CopyDeviceToDeviceAsync(stream, d_particleSqrCplx_orig);
//
//        // Get rotated ref/mask/maskCC
//        rot.Rot(d_reference, rphi, rpsi, rthe);
//        rotMask.Rot(d_mask, rphi, rpsi, rthe);
//
//        if (rotateMaskCC)
//        {
//            d_ccMask.Memset(0);
//            rotMaskCC.Rot(d_ccMask, rphi, rpsi, rthe);
//        }
//
//        // Binarize mask if desired
//        if (binarizeMask)
//        {
//            binarize.Binarize(d_mask, d_mask);
//        }
//
//        // Sum of mask
//        reduce.Sum(d_mask, nVox);
//
//        // FFT of ref/mask
//        makecplx.MakeCplxWithSub(d_reference, d_referenceCplx, 0);
//        makecplx.MakeCplxWithSub(d_mask, d_maskCplx, 0);
//
//        cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_FORWARD));
//        cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_maskCplx.GetDevicePtr(), (cufftComplex*)d_maskCplx.GetDevicePtr(), CUFFT_FORWARD));
//
//        // Wedge * ref (weight wedge)
//        mul.MulVol(d_wedge, d_referenceCplx);
//
//        // Filter ref
//        if (useFilterVolume)
//        {
//            mul.MulVol(d_filter, d_referenceCplx);
//        }
//        else
//        {
//            fft.BandpassFFTShift(d_referenceCplx, rDown, rUp, smooth);
//        }
//
//        // IFFT of ref
//        cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_INVERSE));
//        mul.Mul(1.0f / sizeTot, d_referenceCplx);
//
//        // Mask ref (real space)
//        mul.MulVol(d_mask, d_referenceCplx);
//
//        // Sum of ref (real space)
//        reduce.SumCplx(d_referenceCplx, sum);
//
//        // Subtract mean from ref
//        sub.SubCplx(d_referenceCplx, d_referenceCplx, sum, nVox);
//
//        // Multiply mask with ref
//        mul.MulVol(d_mask, d_referenceCplx);
//
//        // Sum of squared ref
//        reduce.SumSqrCplx(d_referenceCplx, sumSqr);
//
//        // FFT of ref
//        cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_FORWARD));
//
//        // X-Corr
//        fft.Correl(d_particleCplx, d_referenceCplx);
//
//        // Convolution with mask
//        fft.Conv(d_maskCplx, d_particleCplx);
//        fft.Conv(d_maskCplx, d_particleSqrCplx);
//
//        // IFFT cc-result
//        cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_INVERSE));
//        mul.Mul(1.0f / sizeTot, d_referenceCplx);
//
//        // IFFT particle
//        cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx.GetDevicePtr(), (cufftComplex*)d_particleCplx.GetDevicePtr(), CUFFT_INVERSE));
//        mul.Mul(1.0f / sizeTot, d_particleCplx);
//
//        // IFFT particle^2
//        cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleSqrCplx.GetDevicePtr(), (cufftComplex*)d_particleSqrCplx.GetDevicePtr(), CUFFT_INVERSE));
//        mul.Mul(1.0f / sizeTot, d_particleSqrCplx);
//
//        // Normalize for mask
//        fft.EnergyNorm(d_particleCplx, d_particleSqrCplx, d_referenceCplx, sumSqr, nVox);
//
//        // FFTshift cc-result
//        fft.FFTShift2(d_referenceCplx, d_ffttemp);
//
//        // Apply cc-mask
//        mul.MulVol(d_ccMask, d_ffttemp);
//        counter++;
//
//        if (computeCCValOnly)
//        {
//            //only read out the CC value at the old shift position and store it in d_buffer
//            d_index.CopyHostToDevice(&oldIndex);
//            cudaSafeCall(cuMemcpy(d_buffer.GetDevicePtr(), d_ffttemp.GetDevicePtr() + oldIndex, sizeof(float)));
//        }
//        else
//        {
//            //find new Maximum value and store position and value
//            reduce.MaxIndexCplx(d_ffttemp, d_buffer, d_index);
//        }
//
//        // Get maximum
//        max.Max(maxVals, d_index, d_buffer, rphi, rpsi, rthe);
//	}
//
//	cudaSafeCall(cuStreamSynchronize(stream));
//	maxVals.CopyDeviceToHost(&m);
//	cudaSafeCall(cuStreamSynchronize(stream));
//
//	return m;
//}


maxVals_t AvgProcess::executePhaseCorrelation(float* _data,
                                              float* wedge,
                                              float* filter,
                                              float oldphi,
                                              float oldpsi,
                                              float oldtheta,
                                              float rDown,
                                              float rUp,
                                              float smooth,
                                              float3 oldShift,
                                              bool computeCCValOnly,
                                              int oldIndex,
                                              int certaintyDistance)
{
//	int oldWedge = -1;
	maxVals_t m;
//	m.index = 0;
//	m.ccVal = -10000;
//	m.rphi = 0;
//	m.rpsi = 0;
//	m.rthe = 0;
//
//
//	d_particle.CopyHostToDevice(wedge);
//	fft.FFTShiftReal(d_particle, d_wedge);
//
//	//we need to know what the maximum correlation value would be. inverse Fourier transfrom an ideal correlation result to get the normalization factor
//	d_particle.Memset(0);
//	makecplx.MakeCplxWithSub(d_particle, d_particleCplx_orig, -1); //flat spectrum everywhere +1
//	mul.MulVol(d_wedge, d_particleCplx_orig);
//
//	//apply fourier filter: put more weight on "good" frequencies
//	if (useFilterVolume)
//	{
//		mul.MulVol(d_filter, d_particleCplx_orig);
//	}
//	else
//	{
//		fft.BandpassFFTShift(d_particleCplx_orig, rDown, rUp, smooth);
//	}
//	cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), CUFFT_INVERSE));
//	//FFT normalization
//	mul.Mul(1.0f / sizeTot, d_particleCplx_orig);
//
//	fft.FFTShift2(d_particleCplx_orig, d_ffttemp);
//	//find new Maximum value and store position and value
//	reduce.MaxIndexCplx(d_ffttemp, d_buffer, d_index);
//	float maxPCCValue;
//	d_buffer.CopyDeviceToHost(&maxPCCValue, sizeof(float));
//
//
//	cudaSafeCall(cuStreamSynchronize(stream));
//	maxVals.CopyHostToDeviceAsync(stream, &m);
//
//
//	if (useFilterVolume)
//	{
//		d_particle.CopyHostToDevice(filter);
//		fft.FFTShiftReal(d_particle, d_filter);
//	}
//
//	d_particle.CopyHostToDeviceAsync(stream, _data);
//	makecplx.MakeCplxWithSub(d_particle, d_particleCplx_orig, 0);
//
//
//	rot.SetTexture(d_reference_orig);
//	rotMask.SetTexture(d_mask_orig);
//
//    rot.SetOldAngles(oldphi, oldpsi, oldtheta);
//    rotMask.SetOldAngles(oldphi, oldpsi, oldtheta);
//
//	if (rotateMaskCC)
//	{
//		rotMaskCC.SetTexture(d_ccMask_Orig);
//		rotMaskCC.SetOldAngles(oldphi, oldpsi, oldtheta);
//	}
//
//
//
//	//for angle...
//	float rthe = 0;
//	float rpsi = 0;
//	float rphi = 0;
//
//	int counter = 0;
//
//    for(int trpIdx = 0; trpIdx < angleList.size(); trpIdx++){
//
//        vector<float> angles = angleList[trpIdx];
//        rphi = angles[0];
//        rpsi = angles[1];
//        rthe = angles[2];
//
//        d_particleCplx.CopyDeviceToDeviceAsync(stream, d_particleCplx_orig);
//
//        rot.Rot(d_reference, rphi, rpsi, rthe);
//        rotMask.Rot(d_mask, rphi, rpsi, rthe);
//
//        if (rotateMaskCC)
//        {
//            d_ccMask.Memset(0);
//            rotMaskCC.Rot(d_ccMask, rphi, rpsi, rthe);
//        }
//
//        makecplx.MakeCplxWithSub(d_reference, d_referenceCplx, 0);
//        mul.MulVol(d_mask, d_referenceCplx);
//        mul.MulVol(d_mask, d_particleCplx);
//
//        cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_FORWARD));
//        cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx.GetDevicePtr(), (cufftComplex*)d_particleCplx.GetDevicePtr(), CUFFT_FORWARD));
//
//        fft.PhaseCorrel(d_particleCplx, d_referenceCplx);
//        //d_referenceCplx contains phase correlation result
//
//        //apply dose weighting: Put more weight on the phases we can trust
//        mul.MulVol(d_wedge, d_referenceCplx);
//
//        //apply fourier filter: put more weight on "good" frequencies
//        if (useFilterVolume)
//        {
//            mul.MulVol(d_filter, d_referenceCplx);
//        }
//        else
//        {
//            fft.BandpassFFTShift(d_referenceCplx, rDown, rUp, smooth);
//        }
//
//        if (certaintyDistance >= 0)
//        {
//            fft.SplitDataset(d_referenceCplx, d_particleCplx, d_particleSqrCplx);
//            cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx.GetDevicePtr(), (cufftComplex*)d_particleCplx.GetDevicePtr(), CUFFT_INVERSE));
//            cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)d_particleSqrCplx.GetDevicePtr(), (cufftComplex*)d_particleSqrCplx.GetDevicePtr(), CUFFT_INVERSE));
//            //FFT normalization
//            mul.Mul(1.0f / sizeTot / maxPCCValue, d_particleCplx);
//            mul.Mul(1.0f / sizeTot / maxPCCValue, d_particleSqrCplx);
//
//
//            fft.FFTShift2(d_particleCplx, d_ffttemp);
//            mul.MulVol(d_ccMask, d_ffttemp);
//            reduce.MaxIndexCplx(d_ffttemp, d_buffer, sum);
//
//            fft.FFTShift2(d_particleSqrCplx, d_ffttemp);
//            mul.MulVol(d_ccMask, d_ffttemp);
//            reduce.MaxIndexCplx(d_ffttemp, d_buffer, sumSqr);
//        }
//
//        mul.MulVol(d_ccMask, d_ffttemp);
//        counter++;
//
//        if (computeCCValOnly)
//        {
//            //only read out the CC value at the old shift position and store it in d_buffer
//            d_index.CopyHostToDevice(&oldIndex);
//            cudaSafeCall(cuMemcpy(d_buffer.GetDevicePtr(), d_ffttemp.GetDevicePtr() + oldIndex, sizeof(float)));
//        }
//        else
//        {
//            //find new Maximum value and store position and value
//            reduce.MaxIndexCplx(d_ffttemp, d_buffer, d_index);
//        }
//
//        if (certaintyDistance >= 0)
//        {
//            max.MaxWithCertainty(maxVals, d_index, d_buffer, sum, sumSqr, rphi, rpsi, rthe, sizeVol, certaintyDistance);
//        }
//        else
//        {
//            max.Max(maxVals, d_index, d_buffer, rphi, rpsi, rthe);
//        }
//	}
//
//	cudaSafeCall(cuStreamSynchronize(stream));
//	maxVals.CopyDeviceToHost(&m);
//	cudaSafeCall(cuStreamSynchronize(stream));

	return m;
}


