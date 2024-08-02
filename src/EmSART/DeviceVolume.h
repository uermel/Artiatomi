//
// Created by uermel on 9/19/23.
//

#ifndef ARTIATOMI_DEVICEVOLUME_H
#define ARTIATOMI_DEVICEVOLUME_H


#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include "Volume.h"
#include "KernelModules.h"
#include "npps.h"
#include "kernels.h"

class DeviceVolume {
protected:
    int signalLength = 0;
    uint3 volDim;

    Cuda::CudaSurfaceObject3D surf_dual;
    Cuda::CudaSurfaceObject3D surf_card;
    Cuda::CudaTextureObject3D text_dual;

    Cuda::CudaArray3D arr_card;
    Cuda::CudaArray3D arr_dual;

    KernelModules& kernelModules;

private:
    Set3DKernel set3D;
public:
    explicit DeviceVolume(uint3 aVolDim, KernelModules& modules);
    ~DeviceVolume();

    uint3 GetDim();
    size_t GetSize();

    Cuda::CudaSurfaceObject3D& surface_dual();
    Cuda::CudaSurfaceObject3D& surface_card();

    Cuda::CudaTextureObject3D& texture_dual();

    Cuda::CudaArray3D& array_dual();
    Cuda::CudaArray3D& array_card();

    virtual void reset();

    virtual void HostToCard(Volume<float>* volume, int part = 0);
    virtual void HostToCard(float* volume);

    void CardToHost(Volume<float>* volume, int part = 0);
    void CardToHost(float* volume);

    void DualToHost(Volume<float>* volume, int part = 0);
    void DualToHost(float* volume);

    void CardToDual();
    void DualToCard();

    void Set(float aValue);
};

enum SplineFilterMode {
    CARD_TO_CARD,
    DUAL_TO_DUAL,
    DUAL_TO_CARD,
    CARD_TO_DUAL
};

class DeviceVolumeBuf : public DeviceVolume {
protected:
    using DeviceVolume::signalLength;
    using DeviceVolume::volDim;

    using DeviceVolume::surf_dual;
    using DeviceVolume::surf_card;
    using DeviceVolume::text_dual;

    using DeviceVolume::arr_dual;
    using DeviceVolume::arr_card;

    using DeviceVolume::kernelModules;

    Cuda::CudaDeviceVariable dev_var;
    Cuda::CudaDeviceVariable dev_work_buffer;
    Cuda::CudaDeviceVariable dev_mean;
    Cuda::CudaDeviceVariable dev_std;

    SplinePrefilter3DXSurf filter3DX;
    SplinePrefilter3DYSurf filter3DY;
    SplinePrefilter3DZSurf filter3DZ;
    Div3DMaskKernel div3Dmask;

public:
    explicit DeviceVolumeBuf(uint3 aVolDim, KernelModules& aModules);
    ~DeviceVolumeBuf();

    Cuda::CudaDeviceVariable& device_var();

    void reset() override;

    void HostToCard(Volume<float>* aVolume, int aPart) override;
    void HostToCard(float* aVolume) override;

    void HostToVar(Volume<float>* aVolume, int aPart = 0);
    void HostToVar(float* aVolume);

    void VarToHost(Volume<float>* aVolume, int aPart = 0);
    void VarToHost(float* aVolume);

    void CardToVar();
    void CardToVar(Cuda::CudaDeviceVariable& aVar);
    void CardToVar(DeviceVolumeBuf& aVol);

    void VarToCard();
    void VarToCard(Cuda::CudaDeviceVariable& aVar);
    void VarToCard(DeviceVolumeBuf& aVol);

    void DualToVar();
    void DualToVar(Cuda::CudaDeviceVariable& aVar);
    void DualToVar(DeviceVolumeBuf& aVol);

    void VarToDual();
    void VarToDual(Cuda::CudaDeviceVariable& aVar);
    void VarToDual(DeviceVolumeBuf& aVol);

    // Pointwise multiply
    void Mul(DeviceVolumeBuf& aVolume);

    // Multiply constant
    void MulC(float aValue);

    // Subtract constant from signal
    void SubC(float aValue);

    // Subtract signal from constant
    void SubCRev(float aValue);

    // Square root of signal
    void Sqrt();

    // Divide by constant
    void DivC(float aValue);

    // Divide by constant * mask where mask is non-zero
    void DivCMask(DeviceVolume& mask, float aValue);

    // Reductive sum
    float Sum();

    // Mean and Std for entire volume
    void MeanStd(float& meanVal, float& stdVal);

    // Min and Max for entire volume
    void MinMax(float& minVal, float& maxVal);

    // Norm to zero mean
    void ZeroMean();

    // Norm to zero mean and std 1
    void NormVol();

    // Threshold signal (less than value == thresh)
    void Threshold_LTVal(float thresh);

    // Spline prefilter
    void SplinePrefilter(SplineFilterMode mode);

};

class DeviceVolumeFFT : public DeviceVolumeBuf {
protected:
    int fftLength = 0;
    uint3 volFFTDim = {0, 0, 0};
    int maxDim = 0;
    int numShells = 0;
    float3 asymCorrFac = {};

private:
    Cuda::CudaDeviceVariable dev_var_comp;

    cufftHandle handleR2C;
    cufftHandle handleC2R;
    Cuda::CudaDeviceVariable dev_fft_buf;

    MultNorm3DcompKernel multNorm3Dcomp;
    MultNorm1DKernel multNorm1D;
    RadialSum3DKernel sphAvgReal;
    RadialSumAbs3DKernel sphAvg;
    FreqSampleKernel freqSample;
    FSC3DKernel fsc3D;
    FSCNorm1DKernel fscNorm;

    Cuda::CudaDeviceVariable dev_avg;
    Cuda::CudaDeviceVariable dev_multiplicity;
    Cuda::CudaDeviceVariable dev_amp1;
    Cuda::CudaDeviceVariable dev_amp2;
    Cuda::CudaDeviceVariable dev_ampd;

    Cuda::CudaArray1D arr_avg;
    Cuda::CudaTextureObject1D tex_avg;


public:
    explicit DeviceVolumeFFT(uint3 aVolDim, KernelModules& aModules);
    ~DeviceVolumeFFT();

    uint3 GetFFTDim();
    size_t GetFFTSize();
    int GetMaxDim();
    int GetNumShells();

    Cuda::CudaDeviceVariable& device_var_comp();

    void reset() override;

    void R2C();
    void C2R();

    void MultNorm(DeviceVolumeBuf& multiplicity, float threshold);
    void MultNormSqrt(DeviceVolumeBuf& multiplicity, float threshold);

    void SSNR(Cuda::CudaDeviceVariable& aSSNR,
              uint length,
              uint nyquist,
              float voxelSizeOut,
              float voxelSizeVol);
    void SSNRmasked(Cuda::CudaDeviceVariable& aSSNR,
                    DeviceVolumeBuf& devMask,
                    uint length,
                    uint nyquist,
                    float voxelSizeOut,
                    float voxelSizeVol);

    void FourierContribution(DeviceVolumeBuf& multiplicity,
                             Cuda::CudaDeviceVariable& aOccupancy,
                             uint length,
                             uint nyquist,
                             float voxelSizeOut,
                             float voxelSizeVol);

    void FSC(DeviceVolumeFFT& vol2,
             Cuda::CudaDeviceVariable* aFSC,
             uint length, uint nyquist, float voxelSizeOut, float voxelSizeVol,
             Cuda::CudaDeviceVariable* aMean = nullptr,
             Cuda::CudaDeviceVariable* aRMSD = nullptr);

    void FSCmasked(DeviceVolumeFFT& vol2,
                   DeviceVolumeBuf& devMask,
                   Cuda::CudaDeviceVariable* aFSC,
                   uint length, uint nyquist, float voxelSizeOut, float voxelSizeVol,
                   Cuda::CudaDeviceVariable* aMean = nullptr,
                   Cuda::CudaDeviceVariable* aRMSD = nullptr);
};




#endif //ARTIATOMI_DEVICEVOLUME_H
