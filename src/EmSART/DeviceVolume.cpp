//
// Created by uermel on 9/19/23.
//

#include "DeviceVolume.h"

using namespace Cuda;

DeviceVolume::DeviceVolume(uint3 aVolDim, KernelModules& aModules) :
    volDim(aVolDim),
    kernelModules(aModules),
    set3D(kernelModules.modFPLUT)
{
    signalLength = (int)(aVolDim.x * aVolDim.y * aVolDim.z);
    arr_card.Alloc(CU_AD_FORMAT_FLOAT,
                   aVolDim.x,
                   aVolDim.y,
                   aVolDim.z,
                   1, 2);
    arr_dual.Alloc(CU_AD_FORMAT_FLOAT,
                   aVolDim.x,
                   aVolDim.y,
                   aVolDim.z,
                   1, 2);

    text_dual.Bind(CU_TR_ADDRESS_MODE_CLAMP,
                   CU_TR_ADDRESS_MODE_CLAMP,
                   CU_TR_ADDRESS_MODE_CLAMP,
                   CU_TR_FILTER_MODE_LINEAR,
                   0, &arr_dual);

    surf_dual.Bind(&arr_dual);
    surf_card.Bind(&arr_card);

    set3D.SetComputeSize(aVolDim);
}

DeviceVolume::~DeviceVolume(){}

uint3 DeviceVolume::GetDim(){
    return volDim;
}

size_t DeviceVolume::GetSize(){
    return signalLength;
}

Cuda::CudaSurfaceObject3D& DeviceVolume::surface_card()
{
    return surf_card;
}

Cuda::CudaSurfaceObject3D& DeviceVolume::surface_dual()
{
    return surf_dual;
}

Cuda::CudaTextureObject3D& DeviceVolume::texture_dual()
{
    return text_dual;
}

Cuda::CudaArray3D& DeviceVolume::array_card()
{
    return arr_card;
}

Cuda::CudaArray3D& DeviceVolume::array_dual()
{
    return arr_dual;
}

void DeviceVolume::HostToCard(Volume<float>* volume, int part)
{
    arr_card.CopyFromHostToArray(volume->GetPtrToSubVolume(part));
}

void DeviceVolume::HostToCard(float* volume)
{
    arr_card.CopyFromHostToArray(volume);
}

void DeviceVolume::CardToHost(Volume<float>* volume, int part)
{
    arr_card.CopyFromArrayToHost(volume->GetPtrToSubVolume(part));
}

void DeviceVolume::CardToHost(float* volume)
{
    arr_card.CopyFromArrayToHost(volume);
}

void DeviceVolume::DualToHost(Volume<float>* volume, int part)
{
    arr_dual.CopyFromArrayToHost(volume->GetPtrToSubVolume(part));
}

void DeviceVolume::DualToHost(float* volume)
{
    arr_dual.CopyFromArrayToHost(volume);
}

void DeviceVolume::CardToDual(){
    arr_dual.CopyFromArrayToArray(arr_card);
}

void DeviceVolume::DualToCard(){
    arr_card.CopyFromArrayToArray(arr_dual);
}

void DeviceVolume::reset()
{
    set3D(surf_card, 0, volDim);
    set3D(surf_dual, 0, volDim);
}

void DeviceVolume::Set(float aValue){
    set3D(surf_card, aValue, volDim);
    set3D(surf_dual, aValue, volDim);
}

////////////////////////////////////////////////////

DeviceVolumeBuf::DeviceVolumeBuf(uint3 aVolDim, KernelModules& aModules) :
    DeviceVolume(aVolDim, aModules),
    filter3DX(kernelModules.modSplines, 3),
    filter3DY(kernelModules.modSplines, 3),
    filter3DZ(kernelModules.modSplines, 3),
    div3Dmask(kernelModules.modFPLUT)
{
    // Linear device buffer
    dev_var.Alloc(signalLength * sizeof(float));

    // Plan Reductions
    int size_sum = 0;
    int size_meanstd = 0;
    int size_minmax = 0;

    nppSafeCall(nppsSumGetBufferSize_32f(signalLength, &size_sum));
    nppSafeCall(nppsMeanStdDevGetBufferSize_32f(signalLength, &size_meanstd));
    nppSafeCall(nppsMinMaxGetBufferSize_32f(signalLength, &size_minmax));

    int maxsize = max({size_sum, size_meanstd, size_minmax});
    dev_work_buffer.Alloc(maxsize);

    dev_mean.Alloc(2 * sizeof(float));
    dev_std.Alloc(2 * sizeof(float));

    // Compute Dims filter
    div3Dmask.SetComputeSize(volDim);
}

DeviceVolumeBuf::~DeviceVolumeBuf() {}

Cuda::CudaDeviceVariable& DeviceVolumeBuf::device_var(){
    return dev_var;
}

void DeviceVolumeBuf::reset()
{
    DeviceVolume::reset();
    dev_var.Memset(0);
}

// Host to Card
void DeviceVolumeBuf::HostToCard(Volume<float>* aVolume, int aPart)
{
    DeviceVolume::HostToCard(aVolume, aPart);
    arr_card.CopyFromArrayToDevice(dev_var);
}
void DeviceVolumeBuf::HostToCard(float* aVolume)
{
    DeviceVolume::HostToCard(aVolume);
    arr_card.CopyFromArrayToDevice(dev_var);
}

// Host to Var
void DeviceVolumeBuf::HostToVar(Volume<float>* aVolume, int aPart)
{
    dev_var.CopyHostToDevice(aVolume->GetPtrToSubVolume(aPart));
}
void DeviceVolumeBuf::HostToVar(float* aVolume)
{
    dev_var.CopyHostToDevice(aVolume);
}

// Var to Host
void DeviceVolumeBuf::VarToHost(Volume<float>* aVolume, int aPart)
{
    dev_var.CopyDeviceToHost(aVolume->GetPtrToSubVolume(aPart));
}
void DeviceVolumeBuf::VarToHost(float* aVolume)
{
    dev_var.CopyDeviceToHost(aVolume);
}

void DeviceVolumeBuf::CardToVar(){
    arr_card.CopyFromArrayToDevice(dev_var);
}
void DeviceVolumeBuf::CardToVar(CudaDeviceVariable& aVar){
    arr_card.CopyFromArrayToDevice(aVar);
}
void DeviceVolumeBuf::CardToVar(DeviceVolumeBuf& aVol){
    arr_card.CopyFromArrayToDevice(aVol.device_var());
}

void DeviceVolumeBuf::VarToCard(){
    arr_card.CopyFromDeviceToArray(dev_var);
}
void DeviceVolumeBuf::VarToCard(CudaDeviceVariable& aVar){
    arr_card.CopyFromDeviceToArray(aVar);
}
void DeviceVolumeBuf::VarToCard(DeviceVolumeBuf& aVol){
    arr_card.CopyFromDeviceToArray(aVol.device_var());
}

void DeviceVolumeBuf::DualToVar(){
    arr_dual.CopyFromArrayToDevice(dev_var);
}
void DeviceVolumeBuf::DualToVar(CudaDeviceVariable& aVar){
    arr_dual.CopyFromArrayToDevice(aVar);
}
void DeviceVolumeBuf::DualToVar(DeviceVolumeBuf& aVol){
    arr_dual.CopyFromArrayToDevice(aVol.device_var());
}

void DeviceVolumeBuf::VarToDual(){
    arr_dual.CopyFromDeviceToArray(dev_var);
}
void DeviceVolumeBuf::VarToDual(CudaDeviceVariable& aVar){
    arr_dual.CopyFromDeviceToArray(aVar);
}
void DeviceVolumeBuf::VarToDual(DeviceVolumeBuf& aVol){
    arr_dual.CopyFromDeviceToArray(aVol.device_var());
}

void DeviceVolumeBuf::Mul(DeviceVolumeBuf& aVolume){
    nppSafeCall(nppsMul_32f_I((Npp32f*)aVolume.device_var().GetDevicePtr(),
                              (Npp32f*) dev_var.GetDevicePtr(),
                              signalLength));
}

void DeviceVolumeBuf::MulC(float aValue){

    nppSafeCall(nppsMulC_32f_I(aValue, (Npp32f*) dev_var.GetDevicePtr(), signalLength));
}

void DeviceVolumeBuf::SubC(float aValue){

    nppSafeCall(nppsSubC_32f_I(aValue, (Npp32f*) dev_var.GetDevicePtr(), signalLength));
}

void DeviceVolumeBuf::DivC(float aValue){

    nppSafeCall(nppsDivC_32f_I(aValue, (Npp32f*) dev_var.GetDevicePtr(), signalLength));
}

void DeviceVolumeBuf::DivCMask(DeviceVolume& mask, float aValue)
{
    div3Dmask(surf_dual,
              surf_dual,
              mask.surface_card(),
              aValue,
              volDim);
}

void DeviceVolumeBuf::SubCRev(float aValue){

    nppSafeCall(nppsSubCRev_32f_I(aValue, (Npp32f*) dev_var.GetDevicePtr(), signalLength));
}

void DeviceVolumeBuf::Sqrt(){

    nppSafeCall(nppsSqrt_32f_I((Npp32f*) dev_var.GetDevicePtr(), signalLength));
}

float DeviceVolumeBuf::Sum(){

    nppSafeCall(nppsSum_32f((Npp32f*) dev_var.GetDevicePtr(),
                            signalLength,
                            (Npp32f*) dev_mean.GetDevicePtr(),
                            (Npp8u*) dev_work_buffer.GetDevicePtr()));

    float ret = 0;
    dev_mean.CopyDeviceToHost(&ret, sizeof(float));

    return ret;
}

void DeviceVolumeBuf::MeanStd(float& meanVal, float& stdVal){

    nppSafeCall(nppsMeanStdDev_32f((Npp32f*) dev_var.GetDevicePtr(),
                            signalLength,
                            (Npp32f*) dev_mean.GetDevicePtr(),
                            (Npp32f*) dev_std.GetDevicePtr(),
                            (Npp8u*) dev_work_buffer.GetDevicePtr()));

    dev_mean.CopyDeviceToHost(&meanVal, sizeof(float));
    dev_std.CopyDeviceToHost(&stdVal, sizeof(float));
}

void DeviceVolumeBuf::MinMax(float& minVal, float& maxVal){

    nppSafeCall(nppsMinMax_32f((Npp32f*) dev_var.GetDevicePtr(),
                                   signalLength,
                                   (Npp32f*) dev_mean.GetDevicePtr(),
                                   (Npp32f*) dev_std.GetDevicePtr(),
                                   (Npp8u*) dev_work_buffer.GetDevicePtr()));

    dev_mean.CopyDeviceToHost(&minVal, sizeof(float));
    dev_std.CopyDeviceToHost(&maxVal, sizeof(float));
}

void DeviceVolumeBuf::ZeroMean() {
    CardToVar();

    float meanval = 0;
    float stdval = 0;

    MeanStd(meanval, stdval);

    SubC(meanval);
    VarToCard();
}

void DeviceVolumeBuf::NormVol() {
    CardToVar();

    float meanval = 0;
    float stdval = 0;

    MeanStd(meanval, stdval);

    SubC(meanval);
    DivC(stdval);

    VarToCard();
}

void DeviceVolumeBuf::Threshold_LTVal(float thresh)
{
    nppSafeCall(nppsThreshold_LTVal_32f_I((Npp32f*)dev_var.GetDevicePtr(),
                                          signalLength,
                                          thresh,
                                          thresh));
}

void DeviceVolumeBuf::SplinePrefilter(SplineFilterMode mode)
{
    switch (mode) {
        case CARD_TO_CARD:
            filter3DX(surf_card, surf_card, (int)volDim.x, (int)volDim.y, (int)volDim.z);
            filter3DY(surf_card, surf_card, (int)volDim.x, (int)volDim.y, (int)volDim.z);
            filter3DZ(surf_card, surf_card, (int)volDim.x, (int)volDim.y, (int)volDim.z);
            break;
        case DUAL_TO_DUAL:
            filter3DX(surf_dual, surf_dual, (int)volDim.x, (int)volDim.y, (int)volDim.z);
            filter3DY(surf_dual, surf_dual, (int)volDim.x, (int)volDim.y, (int)volDim.z);
            filter3DZ(surf_dual, surf_dual, (int)volDim.x, (int)volDim.y, (int)volDim.z);
            break;
        case DUAL_TO_CARD:
            filter3DX(surf_dual, surf_card, (int)volDim.x, (int)volDim.y, (int)volDim.z);
            filter3DY(surf_card, surf_card, (int)volDim.x, (int)volDim.y, (int)volDim.z);
            filter3DZ(surf_card, surf_card, (int)volDim.x, (int)volDim.y, (int)volDim.z);
            break;
        case CARD_TO_DUAL:
            filter3DX(surf_card, surf_dual, (int)volDim.x, (int)volDim.y, (int)volDim.z);
            filter3DY(surf_dual, surf_dual, (int)volDim.x, (int)volDim.y, (int)volDim.z);
            filter3DZ(surf_dual, surf_dual, (int)volDim.x, (int)volDim.y, (int)volDim.z);
            break;
    }
}


/////////////////////////////////////////////////////////

DeviceVolumeFFT::DeviceVolumeFFT(uint3 aVolDim, KernelModules& aModules) :
    DeviceVolumeBuf(aVolDim, aModules),
    handleR2C(0),
    handleC2R(0),
    multNorm3Dcomp(kernelModules.modBPLUT),
    multNorm1D(kernelModules.modBPLUT),
    sphAvgReal(kernelModules.modCTF),
    sphAvg(kernelModules.modCTF),
    freqSample(kernelModules.modBPLUT),
    fsc3D(kernelModules.modCTF),
    fscNorm(kernelModules.modCTF)
{
    volFFTDim = make_uint3((aVolDim.x/2+1), aVolDim.y, aVolDim.z);
    fftLength = (int)((aVolDim.x/2+1) * aVolDim.y * aVolDim.z);

    maxDim = max({(int)aVolDim.x, (int)aVolDim.y, (int)aVolDim.z});
    numShells = maxDim/2;
    asymCorrFac = make_float3((float)maxDim/(float)aVolDim.x,
                              (float)maxDim/(float)aVolDim.y,
                              (float)maxDim/(float)aVolDim.z);


    // Linear device buffer for FFT
    dev_var_comp.Alloc(fftLength * sizeof(float2));

    // Buffer for spherical avg
    dev_avg.Alloc(numShells * sizeof(float));
    dev_multiplicity.Alloc(numShells * sizeof(float));

    dev_amp1.Alloc(numShells * sizeof(float));
    dev_amp2.Alloc(numShells * sizeof(float));
    dev_ampd.Alloc(numShells * sizeof(float));

    arr_avg.Alloc(CU_AD_FORMAT_FLOAT,
                  numShells, 1);

    tex_avg.Bind(CU_TR_ADDRESS_MODE_CLAMP,
                 CU_TR_FILTER_MODE_LINEAR,
                 0, &arr_avg, CU_AD_FORMAT_FLOAT, 1);

    // Plan FFT
    cufftSafeCall(cufftCreate(&handleR2C));
    cufftSafeCall(cufftCreate(&handleC2R));

    cufftSafeCall(cufftSetAutoAllocation(handleR2C, false));
    cufftSafeCall(cufftSetAutoAllocation(handleC2R, false));

    cufftSafeCall(cufftPlan3d(&handleR2C,
                              aVolDim.z,
                              aVolDim.y,
                              aVolDim.x,
                              CUFFT_R2C));

    cufftSafeCall(cufftPlan3d(&handleC2R,
                              aVolDim.z,
                              aVolDim.y,
                              aVolDim.x,
                              CUFFT_C2R));

    size_t sz_proj_r2c = 0;
    size_t sz_proj_c2r = 0;
    cufftSafeCall(cufftGetSize(handleR2C, &sz_proj_r2c));
    cufftSafeCall(cufftGetSize(handleC2R, &sz_proj_c2r));

    size_t fftsize = max(sz_proj_r2c, sz_proj_c2r);
    dev_fft_buf.Alloc(fftsize);

    cufftSetWorkArea(handleR2C, (void*)dev_fft_buf.GetDevicePtr());
    cufftSetWorkArea(handleC2R, (void*)dev_fft_buf.GetDevicePtr());

    multNorm3Dcomp.SetComputeSize(volFFTDim);
    sphAvgReal.SetComputeSize(volFFTDim);
    sphAvg.SetComputeSize(volFFTDim);
    fsc3D.SetComputeSize(volFFTDim);
    multNorm1D.SetComputeSize((uint)numShells, 1, 1);
    fscNorm.SetComputeSize((uint)numShells, 1, 1);
}

DeviceVolumeFFT::~DeviceVolumeFFT()
{
    //cufftSafeCall(cufftDestroy(handleR2C));
    //cufftSafeCall(cufftDestroy(handleC2R));
};

uint3 DeviceVolumeFFT::GetFFTDim(){
    return volFFTDim;
}

size_t DeviceVolumeFFT::GetFFTSize(){
    return fftLength;
}

int DeviceVolumeFFT::GetMaxDim(){
    return maxDim;
}

int DeviceVolumeFFT::GetNumShells(){
    return numShells;
}

Cuda::CudaDeviceVariable& DeviceVolumeFFT::device_var_comp(){
    return dev_var_comp;
}

void DeviceVolumeFFT::reset()
{
    DeviceVolumeBuf::reset();
    dev_var_comp.Memset(0);
}

void DeviceVolumeFFT::R2C(){
    cufftSafeCall(cufftExecR2C(handleR2C,
                               (cufftReal*) dev_var.GetDevicePtr(),
                               (cufftComplex*) dev_var_comp.GetDevicePtr()));
}

void DeviceVolumeFFT::C2R(){
    cufftSafeCall(cufftExecC2R(handleC2R,
                               (cufftComplex*) dev_var_comp.GetDevicePtr(),
                               (cufftReal*) dev_var.GetDevicePtr()));

    nppSafeCall(nppsDivC_32f_I((float)signalLength, (Npp32f*) dev_var.GetDevicePtr(), signalLength));
}

void DeviceVolumeFFT::MultNorm(DeviceVolumeBuf& multiplicity,
                               float threshold){

    // Normalize for occupancy in fourier space
    R2C();
    multNorm3Dcomp(dev_var_comp,
                   multiplicity.device_var(),
                   volFFTDim,
                   threshold);

    dev_avg.Memset(0);
    dev_multiplicity.Memset(0);
    C2R();
}

void DeviceVolumeFFT::MultNormSqrt(DeviceVolumeBuf& multiplicity,
                                   float threshold){

    // Normalize noise for multiplicity
    // First divide by sqrt of multiplicity in FSpace
    multiplicity.Sqrt();
    float min, max;
    multiplicity.MinMax(min, max);

    R2C();
    multNorm3Dcomp(dev_var_comp,
                   multiplicity.device_var(),
                   volFFTDim,
                   threshold);
    C2R();

    // Then divide by sqrt(N) in real space
    DivC(max);
}

void DeviceVolumeFFT::FourierContribution(DeviceVolumeBuf& multiplicity,
                                          CudaDeviceVariable& aOccupancy,
                                          uint length,
                                          uint nyquist,
                                          float voxelSizeOut,
                                          float voxelSizeVol){

    // Compute how many images contributed to the average per shell, so we can later scale the averaged
    // noise appropriately
    sphAvgReal(multiplicity.device_var(),
               dev_avg,
               dev_multiplicity,
               volFFTDim,
               volDim,
               asymCorrFac,
               1.f,
               numShells);

    // dev_avg contains the summed multiplicity per shell
    // dev_multiplicity contains the number of fourier components per shell

    // Compute the average number of images per shell
    multNorm1D(dev_avg,
               dev_multiplicity,
               numShells,
               1.f);

    // dev_avg now contains the average number of images per shell
    arr_avg.CopyFromDeviceToArray(dev_avg);

    // Upsample to unbinned image
    freqSample.SetComputeSize(length, 1, 1);
    freqSample(aOccupancy,
               tex_avg,
               length,
               nyquist,
               numShells,
               voxelSizeOut,
               voxelSizeVol);
}

void DeviceVolumeFFT::SSNR(CudaDeviceVariable& aSSNR,
                           uint length,
                           uint nyquist,
                           float voxelSizeOut,
                           float voxelSizeVol){

    dev_avg.Memset(0);
    dev_multiplicity.Memset(0);

    CardToVar();
    R2C();

    sphAvg(dev_var_comp,
           dev_avg,
           dev_multiplicity,
           volFFTDim,
           volDim,
           asymCorrFac,
           1.f,
           numShells);

    multNorm1D(dev_avg,
               dev_multiplicity,
               numShells,
               1.f);

    arr_avg.CopyFromDeviceToArray(dev_avg);

    freqSample.SetComputeSize(length, 1, 1);
    freqSample(aSSNR,
               tex_avg,
               length,
               nyquist,
               numShells,
               voxelSizeOut,
               voxelSizeVol);
}


void DeviceVolumeFFT::SSNRmasked(CudaDeviceVariable& aSSNR,
                                 DeviceVolumeBuf& devMask,
                                 uint length,
                                 uint nyquist,
                                 float voxelSizeOut,
                                 float voxelSizeVol){

    dev_avg.Memset(0);
    dev_multiplicity.Memset(0);

    CardToVar();
    Mul(devMask);

    R2C();

    sphAvg(dev_var_comp,
           dev_avg,
           dev_multiplicity,
           volFFTDim,
           volDim,
           asymCorrFac,
           1.f,
           numShells);

    multNorm1D(dev_avg,
               dev_multiplicity,
               numShells,
               1.f);

    arr_avg.CopyFromDeviceToArray(dev_avg);

    freqSample.SetComputeSize(length, 1, 1);
    freqSample(aSSNR,
               tex_avg,
               length,
               nyquist,
               numShells,
               voxelSizeOut,
               voxelSizeVol);
}


void DeviceVolumeFFT::FSC(DeviceVolumeFFT& vol2,
                          CudaDeviceVariable* aFSC,
                          uint length, uint nyquist, float voxelSizeOut, float voxelSizeVol,
                          CudaDeviceVariable* aMean,
                          CudaDeviceVariable* aRMSD){

    dev_amp1.Memset(0);
    dev_amp2.Memset(0);
    dev_ampd.Memset(0);
    dev_multiplicity.Memset(0);

    // Self FFT
    CardToVar();
    R2C();

    // Target FFT
    vol2.CardToVar();
    vol2.R2C();

    fsc3D(dev_var_comp,
          vol2.device_var_comp(),
          dev_amp1,
          dev_amp2,
          dev_ampd,
          dev_multiplicity,
          volFFTDim,
          volDim,
          asymCorrFac,
          1.f,
          numShells);

    fscNorm(dev_amp1,
            dev_amp2,
            dev_ampd,
            dev_multiplicity,
            numShells,
            1.f);

    arr_avg.CopyFromDeviceToArray(dev_amp1);
    freqSample.SetComputeSize(length, 1, 1);
    freqSample(*aFSC,
               tex_avg,
               length,
               nyquist,
               numShells,
               voxelSizeOut,
               voxelSizeVol);

    if (aMean != nullptr){
        arr_avg.CopyFromDeviceToArray(dev_amp2);
        freqSample.SetComputeSize(length, 1, 1);
        freqSample(*aMean,
                   tex_avg,
                   length,
                   nyquist,
                   numShells,
                   voxelSizeOut,
                   voxelSizeVol);
    }

    if (aRMSD != nullptr){
        arr_avg.CopyFromDeviceToArray(dev_ampd);
        freqSample.SetComputeSize(length, 1, 1);
        freqSample(*aRMSD,
                   tex_avg,
                   length,
                   nyquist,
                   numShells,
                   voxelSizeOut,
                   voxelSizeVol);
    }
}

void DeviceVolumeFFT::FSCmasked(DeviceVolumeFFT& vol2,
                                DeviceVolumeBuf& devMask,
                                CudaDeviceVariable* aFSC,
                                uint length, uint nyquist, float voxelSizeOut, float voxelSizeVol,
                                CudaDeviceVariable* aMean,
                                CudaDeviceVariable* aRMSD){

    dev_amp1.Memset(0);
    dev_amp2.Memset(0);
    dev_ampd.Memset(0);
    dev_multiplicity.Memset(0);

    // Self FFT
    CardToVar();
    Mul(devMask);
    R2C();

    // Target FFT
    vol2.CardToVar();
    vol2.Mul(devMask);
    vol2.R2C();

    fsc3D(dev_var_comp,
          vol2.device_var_comp(),
          dev_amp1,
          dev_amp2,
          dev_ampd,
          dev_multiplicity,
          volFFTDim,
          volDim,
          asymCorrFac,
          1.f,
          numShells);

    fscNorm(dev_amp1,
            dev_amp2,
            dev_ampd,
            dev_multiplicity,
            numShells,
            1.f);

    arr_avg.CopyFromDeviceToArray(dev_amp1);
    freqSample.SetComputeSize(length, 1, 1);
    freqSample(*aFSC,
               tex_avg,
               length,
               nyquist,
               numShells,
               voxelSizeOut,
               voxelSizeVol);

    if (aMean != nullptr){
        arr_avg.CopyFromDeviceToArray(dev_amp2);
        freqSample.SetComputeSize(length, 1, 1);
        freqSample(*aMean,
                   tex_avg,
                   length,
                   nyquist,
                   numShells,
                   voxelSizeOut,
                   voxelSizeVol);
    }

    if (aRMSD != nullptr){
        arr_avg.CopyFromDeviceToArray(dev_ampd);
        freqSample.SetComputeSize(length, 1, 1);
        freqSample(*aRMSD,
                   tex_avg,
                   length,
                   nyquist,
                   numShells,
                   voxelSizeOut,
                   voxelSizeVol);
    }
}