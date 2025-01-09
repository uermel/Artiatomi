//
// Created by uermel on 2/14/22.
//

#include "AdaptiveRegularizer.h"

AdaptiveRegularizer::AdaptiveRegularizer(size_t _sizeVol,
                       CUstream _stream,
                       CudaContext* _ctx)

        : sizeVol(_sizeVol),
          sizeTot(_sizeVol * _sizeVol * _sizeVol),
          stream(_stream),
          ctx(_ctx),
          // Kernels
          reduce((int)_sizeVol * (int)_sizeVol * (int)_sizeVol, _stream, _ctx),
          sub((int)_sizeVol, _stream, _ctx),
          makecplx((int)_sizeVol, _stream, _ctx),
          mul((int)_sizeVol, _stream, _ctx),
          fft((int)_sizeVol, _stream, _ctx),
          max(_stream, _ctx),
          mask(_sizeVol, _stream, _ctx),
          cmp(_sizeVol, _stream, _ctx),
          // Temp storage
          d_sum(_sizeVol * _sizeVol * _sizeVol * sizeof(float)), //should be sufficient for everything...
          // Images
          d_real_im1(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
          d_real_im2(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
          d_cplx_im1(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
          d_cplx_im2(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
          d_cplx_im1filt(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
          d_cplx_im2filt(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
          d_cplx_error_full(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
          d_cplx_error_tmp(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
          // Results
          d_current_min(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
          d_min_idx(_sizeVol * _sizeVol * _sizeVol * sizeof(int)),
          d_min_window(_sizeVol * _sizeVol * _sizeVol * sizeof(int)),
          d_cplx_res(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
          // Filter
          d_cplx_window(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
          d_real_butter(_sizeVol * _sizeVol * _sizeVol * sizeof(float))
{
    // Host stores
    cudaSafeCall(cuMemAllocHost((void**)&sum_h, sizeof(float)));

    // Plan FFT
    int n[] = { (int)sizeVol, (int)sizeVol, (int)sizeVol };
    cufftSafeCall(cufftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 1));
    cufftSafeCall(cufftSetStream(ffthandle, stream));
}

AdaptiveRegularizer::~AdaptiveRegularizer()
{
    cufftDestroy(ffthandle);
}

void AdaptiveRegularizer::findRegParams(float* aIm1,
                                        float* aIm2,
                                        int aOrder,
                                        float aAWF,
                                        int aSteps,
                                        int* aRegParams)
{
    // Get images on device and prep for FFT
    d_real_im1.CopyHostToDevice(aIm1);
    d_real_im2.CopyHostToDevice(aIm2);

    d_cplx_im1.Memset(0);
    d_cplx_im2.Memset(0);

    makecplx.MakeCplxWithSub(d_real_im1, d_cplx_im1, 0.f);
    makecplx.MakeCplxWithSub(d_real_im2, d_cplx_im2, 0.f);

    // FFT
    cufftSafeCall(cufftExecC2C(ffthandle,
                               (cufftComplex*)d_cplx_im1.GetDevicePtr(),
                               (cufftComplex*)d_cplx_im1.GetDevicePtr(),
                               CUFFT_FORWARD));
    cufftSafeCall(cufftExecC2C(ffthandle,
                               (cufftComplex*)d_cplx_im2.GetDevicePtr(),
                               (cufftComplex*)d_cplx_im2.GetDevicePtr(),
                               CUFFT_FORWARD));

    // Filter Params
    float step_size = 0.5f/(float)aSteps;
    auto cutoffs = new float[aSteps];
    auto theta = new float[aSteps];
    auto min_rho = new float[aSteps];

    for (int i=0; i<aSteps; i++)
    {
        cutoffs[i] = step_size + step_size * (float)i;
        theta[i] = 1.f/cutoffs[i];
        min_rho[i] = aAWF * theta[i];
        printf("Cutoff: %f  Extent: %f  Window: %f\n", cutoffs[i], theta[i], min_rho[i]);
    }

    // Init everything
    sub.Set(d_current_min, FLT_MAX);
    d_min_idx.Memset(0);
    d_min_window.Memset(0);

    // Main param search
    for (int filtIdx = 0; filtIdx < aSteps; filtIdx++)
    {
        // Filter
        fft.ButterFilter(d_cplx_im1, d_cplx_im1filt, aOrder, cutoffs[filtIdx]);
        fft.ButterFilter(d_cplx_im2, d_cplx_im2filt, aOrder, cutoffs[filtIdx]);
        fft.ButterTest(d_real_butter, aOrder, cutoffs[filtIdx]);
        printf("Cutoff: %f  Index: %i\n", cutoffs[filtIdx], filtIdx);
//        {
//            auto tmp = new float[sizeTot];
//            stringstream ss1;
//            d_real_butter.CopyDeviceToHost(tmp);
//            ss1 << "butterfilt_" << filtIdx << ".em";
//            emwrite(ss1.str(), tmp, sizeVol, sizeVol, sizeVol);
//            delete[] tmp;
//        }

        // Inverse Transform Filtered images
        cufftSafeCall(cufftExecC2C(ffthandle,
                                   (cufftComplex*)d_cplx_im1filt.GetDevicePtr(),
                                   (cufftComplex*)d_cplx_im1filt.GetDevicePtr(),
                                   CUFFT_INVERSE));
        mul.Mul(1.0f / (float)sizeTot, d_cplx_im1filt);

        cufftSafeCall(cufftExecC2C(ffthandle,
                                   (cufftComplex*)d_cplx_im2filt.GetDevicePtr(),
                                   (cufftComplex*)d_cplx_im2filt.GetDevicePtr(),
                                   CUFFT_INVERSE));
        mul.Mul(1.0f / (float)sizeTot, d_cplx_im2filt);

        // Difference
        sub.RegError(d_real_im1, d_real_im2, d_cplx_im1filt, d_cplx_im2filt, d_cplx_error_full);

//        {
//            auto tmp1 = new float[sizeTot];
//            auto tmp2 = new float[sizeTot];
//            auto tmp3 = new float2[sizeTot];
//            d_cplx_error_full.CopyDeviceToHost(tmp3);
//            for (int i = 0; i < sizeTot; i++)
//            {
//                tmp1[i] = tmp3[i].x;
//                tmp2[i] = tmp3[i].y;
//            }
//            stringstream ss1;
//            ss1 << "error_real_" << filtIdx << ".em";
//            emwrite(ss1.str(), tmp1, sizeVol, sizeVol, sizeVol);
//            delete[] tmp1;
//
//            stringstream ss2;
//            ss2 << "error_imag_" << filtIdx << ".em";
//            emwrite(ss2.str(), tmp2, sizeVol, sizeVol, sizeVol);
//            delete[] tmp2;
//
//            delete[] tmp3;
//        }

        // Forward transform error
        cufftSafeCall(cufftExecC2C(ffthandle,
                                   (cufftComplex*)d_cplx_error_full.GetDevicePtr(),
                                   (cufftComplex*)d_cplx_error_full.GetDevicePtr(),
                                   CUFFT_FORWARD));

        for (int windIdx = 0; windIdx < 50; windIdx++)
        {
            // Get error image
            d_cplx_error_tmp.CopyDeviceToDevice(d_cplx_error_full);

            // Compute window
            mask.SphericalMaskCosineCplx(d_cplx_window, min_rho[filtIdx]+(float)windIdx, 0.f, make_float3(0.f, 0.f, 0.f));

            printf("Radius: %f\n", min_rho[filtIdx]+(float)windIdx);

//            {
//                auto tmp1 = new float[sizeTot];
//                auto tmp2 = new float[sizeTot];
//                auto tmp3 = new float2[sizeTot];
//                d_cplx_window.CopyDeviceToHost(tmp3);
//                for (int i = 0; i < sizeTot; i++)
//                {
//                    tmp1[i] = tmp3[i].x;
//                    tmp2[i] = tmp3[i].y;
//                }
//                stringstream ss1;
//                ss1 << "window_real_" << filtIdx << "_" << filtIdx << "_" << windIdx << ".em";
//                emwrite(ss1.str(), tmp1, sizeVol, sizeVol, sizeVol);
//                delete[] tmp1;
//
//                stringstream ss2;
//                ss2 << "window_imag_" << filtIdx << "_" << filtIdx << "_" << windIdx << ".em";
//                emwrite(ss2.str(), tmp2, sizeVol, sizeVol, sizeVol);
//                delete[] tmp2;
//
//                delete[] tmp3;
//            }

            // Sum of window for norm
            sum_h[0] = 0.f;
            reduce.SumCplx(d_cplx_window, d_sum);
            d_sum.CopyDeviceToHost(sum_h, sizeof(float));
            printf("Window sum: %f\n", sum_h[0]);

            // FFT of window
            cufftSafeCall(cufftExecC2C(ffthandle,
                                       (cufftComplex*)d_cplx_window.GetDevicePtr(),
                                       (cufftComplex*)d_cplx_window.GetDevicePtr(),
                                       CUFFT_FORWARD));

            // Convolution with window
            fft.Conv(d_cplx_window, d_cplx_error_tmp);

            // Inverse transform
            cufftSafeCall(cufftExecC2C(ffthandle,
                                       (cufftComplex*)d_cplx_error_tmp.GetDevicePtr(),
                                       (cufftComplex*)d_cplx_error_tmp.GetDevicePtr(),
                                       CUFFT_INVERSE));
            // Norm by size and window sum
            mul.Mul((1.0f / (float)sizeTot) * (1.f/sum_h[0]), d_cplx_error_tmp);

            fft.FFTShift2(d_cplx_error_tmp, d_cplx_res);

//            {
//                auto tmp1 = new float[sizeTot];
//                auto tmp2 = new float[sizeTot];
//                auto tmp3 = new float2[sizeTot];
//                d_cplx_res.CopyDeviceToHost(tmp3);
//                for (int i = 0; i < sizeTot; i++)
//                {
//                    tmp1[i] = tmp3[i].x;
//                    tmp2[i] = tmp3[i].y;
//                }
//                stringstream ss1;
//                ss1 << "result_real_" << filtIdx << "_" << filtIdx << "_" << windIdx << ".em";
//                emwrite(ss1.str(), tmp1, sizeVol, sizeVol, sizeVol);
//                delete[] tmp1;
//
//                stringstream ss2;
//                ss2 << "result_imag_" << filtIdx << "_" << filtIdx << "_" <<windIdx << ".em";
//                emwrite(ss2.str(), tmp2, sizeVol, sizeVol, sizeVol);
//                delete[] tmp2;
//
//                delete[] tmp3;
//            }

            // Is it lower than previous?
            cmp.MinIdx(d_cplx_res, d_current_min, d_min_idx, d_min_window, filtIdx, windIdx);
        }
    }

    // Store result
    d_min_idx.CopyDeviceToHost(aRegParams);
}

void AdaptiveRegularizer::regularize(float *aIm,
                                     int aOrder,
                                     int aSteps,
                                     int *aRegParams,
                                     float* aRecon)
{
    // Get image on device and prep for FFT (im2 will contain the reconstruction)
    d_real_im1.CopyHostToDevice(aIm);
    d_real_im2.Memset(0);

    d_cplx_im1.Memset(0);

    makecplx.MakeCplxWithSub(d_real_im1, d_cplx_im1, 0.f);

    // FFT
    cufftSafeCall(cufftExecC2C(ffthandle,
                               (cufftComplex*)d_cplx_im1.GetDevicePtr(),
                               (cufftComplex*)d_cplx_im1.GetDevicePtr(),
                               CUFFT_FORWARD));

    // Filter Params
    float step_size = 0.5f/(float)aSteps;
    auto cutoffs = new float[aSteps];
    auto theta = new float[aSteps];

    for (int i=0; i<aSteps; i++)
    {
        cutoffs[i] = step_size + step_size * (float)i;
        theta[i] = 1.f/cutoffs[i];
    }

    // Get reg params
    d_min_idx.CopyHostToDevice(aRegParams);

    // Reconstruct
    for (int filtIdx = 0; filtIdx < aSteps; filtIdx++)
    {
        // Filter
        fft.ButterFilter(d_cplx_im1, d_cplx_im1filt, aOrder, cutoffs[filtIdx]);

        // Inverse Transform filtered image
        cufftSafeCall(cufftExecC2C(ffthandle,
                                   (cufftComplex*)d_cplx_im1filt.GetDevicePtr(),
                                   (cufftComplex*)d_cplx_im1filt.GetDevicePtr(),
                                   CUFFT_INVERSE));
        mul.Mul(1.0f / (float)sizeTot, d_cplx_im1filt);

        // Assign voxels in reconstruction
        cmp.SelIdx(d_cplx_im1filt, d_min_idx, d_real_im2, filtIdx);
    }

    d_real_im2.CopyDeviceToHost(aRecon);
}