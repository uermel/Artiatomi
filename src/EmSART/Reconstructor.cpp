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


#include "Reconstructor.h"
#include <typeinfo>
#include "cuda_profiler_api.h"

//#define DEBUG_IMAGES

using namespace std;
using namespace Cuda;

void Reconstructor::MatrixScalarMul(float4x4 &M, float v)
{
    for (int i=0; i<4; i++) {
        M.m[i].x = M.m[i].x * v;
        M.m[i].y = M.m[i].y * v;
        M.m[i].z = M.m[i].z * v;
        M.m[i].w = M.m[i].w * v;
    }
}

void Reconstructor::MatrixVector3Mul(float4x4 M, float3* v)
{
	float3 erg;
	erg.x = M.m[0].x * v->x + M.m[0].y * v->y + M.m[0].z * v->z + 1.f * M.m[0].w;
	erg.y = M.m[1].x * v->x + M.m[1].y * v->y + M.m[1].z * v->z + 1.f * M.m[1].w;
	erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
	*v = erg;
}

void Reconstructor::MatrixVector3Mul(float3x3& M, float xIn, float yIn, float& xOut, float& yOut)
{
    xOut = M.m[0].x * xIn + M.m[0].y * yIn + M.m[0].z * 1.f;
    yOut = M.m[1].x * xIn + M.m[1].y * yIn + M.m[1].z * 1.f;
}

void Reconstructor::MatrixVector3Mul(float3x3& M, float2& val)
{
    val.x = M.m[0].x * val.x + M.m[0].y * val.y + M.m[0].z * 1.f;
    val.y = M.m[1].x * val.x + M.m[1].y * val.y + M.m[1].z * 1.f;
}

template<class TVol>
void Reconstructor::GetDefocusDistances(float & t_in, float & t_out, int index, Volume<TVol>* vol)
{
	//Shoot ray from center of volume:
	float3 c_projNorm = proj.GetNormalVector(index);
	float3 c_detektor = proj.GetPosition(index);
	float3 MC_bBoxMin;
	float3 MC_bBoxMax;
	MC_bBoxMin = vol->GetVolumeBBoxMin();
	MC_bBoxMax = vol->GetVolumeBBoxMax();
	float3 volDim = vol->GetDimension();
	float3 hitPoint;
	float t;

	t = (c_projNorm.x * (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f)) + 
		 c_projNorm.y * (MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f)) + 
		 c_projNorm.z * (MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f)));
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = abs(t);
	
	hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f));
	hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f));
	hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f));

	float4x4 c_DetectorMatrix;
	
	proj.GetDetectorMatrix(index, (float*)&c_DetectorMatrix, 1);
	MatrixVector3Mul(c_DetectorMatrix, &hitPoint);

	//--> pixelBorders.x = x.min; pixelBorders.z = y.min;
	float hitX = round(hitPoint.x);
	float hitY = round(hitPoint.y);

	//printf("HitX: %d, HitY: %d\n", hitX, hitY);

	//Shoot ray from hit point on projection towards volume to get the distance to entry and exit point
	//float3 pos = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitY * proj.GetPixelVPitch(index);
	float3 pos = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitY * proj.GetPixelVPitch(index);
	hitX = (float)proj.GetWidth() * 0.5f;
	hitY = (float)proj.GetHeight() * 0.5f;
	float3 pos2 = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitX * proj.GetPixelVPitch(index);
	float3 nvec = proj.GetNormalVector(index);

	/*float3 MC_bBoxMin;
	float3 MC_bBoxMax;*/

	

	t_in = 2*-DIST;
	t_out = 2*DIST;

	for (int x = 0; x <= 1; x++)
		for (int y = 0; y <= 1; y++)
			for (int z = 0; z <= 1; z++)
			{
				//float t;

				t = (nvec.x * (MC_bBoxMin.x + x * (MC_bBoxMax.x - MC_bBoxMin.x))
					+ nvec.y * (MC_bBoxMin.y + y * (MC_bBoxMax.y - MC_bBoxMin.y))
					+ nvec.z * (MC_bBoxMin.z + z * (MC_bBoxMax.z - MC_bBoxMin.z)));
				t += (-nvec.x * pos.x - nvec.y * pos.y - nvec.z * pos.z);

				if (t < t_in) t_in = t;
				if (t > t_out) t_out = t;
			}
}
template void Reconstructor::GetDefocusDistances(float & t_in, float & t_out, int index, Volume<unsigned short>* vol);
template void Reconstructor::GetDefocusDistances(float & t_in, float & t_out, int index, Volume<float>* vol);


void Reconstructor::GetDefocusMinMax(float ray, int index, float & defocusMin, float & defocusMax)
{
	defocusMin = defocus.GetMinDefocus(index);
	defocusMax = defocus.GetMaxDefocus(index);
	float tiltAngle = (markers(MFI_TiltAngle, index, 0) + config.AddTiltAngle) / 180.0f * (float)M_PI;

	float distanceTo0 = ray + DIST; //in pixel
	if (config.IgnoreZShiftForCTF)
	{
		distanceTo0 = (round(distanceTo0 * proj.GetPixelSize() * config.CTFSliceThickness) / config.CTFSliceThickness) + config.CTFSliceThickness / 2.0f;
	}
	else
	{
		distanceTo0 = (round(distanceTo0 * proj.GetPixelSize() * config.CTFSliceThickness) / config.CTFSliceThickness) + config.CTFSliceThickness / 2.0f - (config.VolumeShift.z * proj.GetPixelSize() * cosf(tiltAngle)); //in nm
	}
	if (config.SwitchCTFDirectionForIMOD)
	{
		distanceTo0 *= -1; //IMOD inverses the logic...
	}
	

	defocusMin = defocusMin + distanceTo0;
	defocusMax = defocusMax + distanceTo0;
}

Reconstructor::Reconstructor(Configuration::Config & aConfig,
                             Projection & aProj,
                             ProjectionSource* aProjectionSource,
                             MarkerFile& aMarkers,
                             CtfFile& aDefocus,
                             KernelModules& modules,
                             int aMpi_part,
                             int aMpi_size,
                             bool doHalfsets)
	: 
	proj(aProj),
	projSource(aProjectionSource),
    ctfHandler(aConfig, aProj, aDefocus, aMpi_part),
	fpKernel(modules.modFP),
	slicerKernel(modules.modSlicer),
	volTravLenKernel(modules.modVolTravLen),
	wbp(modules.modWBP),
	fourFilterKernel(modules.modWBP),
	doseWeightingKernel(modules.modWBP),
	conjKernel(modules.modWBP),
    pcKernel(modules.modWBP),
	maxShiftKernel(modules.modWBP),
	compKernel(modules.modComp),
	subEKernel(modules.modComp),
	cropKernel(modules.modComp),
    cropInvKernel(modules.modComp),
	cropSlicesKernel(modules.modComp),
    cropSlicesInvKernel(modules.modComp),
	bpKernel(modules.modBP, aConfig.FP16Volume),
	convVolKernel(modules.modBP),
	convVol3DKernel(modules.modBP),
	ctf(modules.modCTF),
    postFilterSum(modules.modCTF),
	cts(modules.modCTS),
	r2ss(modules.modCTS),
    ss2rs(modules.modCTS),
    rs2ss(modules.modCTS),
    ss2r(modules.modCTS),
	dimBordersKernel(modules.modComp),
    prefilter3DX(modules.modSplines, 3),
    prefilter3DY(modules.modSplines, 3),
    prefilter3DZ(modules.modSplines, 3),
    postfilter3DX(modules.modSplines, 3),
    postfilter3DY(modules.modSplines, 3),
    postfilter3DZ(modules.modSplines, 3),
    fpOrthoKernel(modules.modFPLUT),
    fpOrthoSSKernel(modules.modFPLUT),
    fpOrthoOVKernel(modules.modFPLUT),
    distOrthoKernel(modules.modFPLUT),
    preFilterBox(modules.modSplines),
    postFilterBox(modules.modSplines),
    sample2D(modules.modFPLUT),
    sample3D(modules.modFPLUT),
    copyToPitched(modules.modCTS),
    copyFromPitched(modules.modCTS),
    addToPitched(modules.modCTS),
    postFilter(modules.modCTF),
    slicesToArrays(modules.modCTS),
    maskedSlicesToArrays(modules.modCTS),
    bpOrthoKernel(modules.modBPLUT),
    bpOrthoAdd(modules.modBPLUT),
    bpOrthoAddSS(modules.modBPLUT),
    add3D(modules.modFPLUT),
    set3D(modules.modFPLUT),
    add3Dmasked(modules.modFPLUT),
    mask3D(modules.modFPLUT),
    radialSum(modules.modCTF),
    preFilterSpreadAdHoc(modules.modCTF),
    preFilterSpreadSNR(modules.modCTF),
    compSpecialKernel(modules.modComp),
    mask3DTransform(modules.modFPLUT),
    add3DTransform(modules.modFPLUT),
    norm3Doverlap(modules.modFPLUT),
    multiplicity3D(modules.modBPLUT),
#ifdef REFINE_MODE
	rotKernel(modules.modWBP, aConfig.SizeSubVol),
	maxShiftWeightedKernel(modules.modWBP),
	findPeakKernel(modules.modWBP),
#endif
	defocus(aDefocus),
	markers(aMarkers),
	config(aConfig),
	mpi_part(aMpi_part),
	mpi_size(aMpi_size),
	skipFilter(aConfig.SkipFilter),
	squareBorderSizeX(0),
	squareBorderSizeY(0),
	squarePointerShift(0),
	magAnisotropy(GetMagAnistropyMatrix(aConfig.MagAnisotropyAmount, aConfig.MagAnisotropyAngleInDeg, (float)proj.GetWidth(), (float)proj.GetHeight())),
	magAnisotropyInv(GetMagAnistropyMatrix(1.0f / aConfig.MagAnisotropyAmount, aConfig.MagAnisotropyAngleInDeg, (float)proj.GetWidth(), (float)proj.GetHeight()))
{
    // START Utility Vars
    // Dims
    projDim = aProj.GetDim<uint2>();
    projDimF = aProj.GetDim<float2>();

    fftDim = aProj.GetFFTDim<uint2>();
    fftDimF = aProj.GetFFTDim<float2>();
    asymCorrFac = aProj.GetAsymCorrFactor();
    snrShells = aProj.GetMaxShells();

    // Sizes
    projSize = projDim.x * projDim.y;
    projSizeF32 = projSize * sizeof(float);

    fftSize = fftDim.x * fftDim.y;
    fftSizeFC32 = fftSize * sizeof(cuComplex);

    // Steps
    projPitchChar = projDim.x * sizeof(char);
    projPitchFloat = projDim.x * sizeof(float);

    fftPitchComplex = fftDim.x * sizeof(cuComplex);
    // END Utility Vars

	// BEGIN Kernel Dimensions //
    // 2D
	fpKernel.SetComputeSize(projDim);
	slicerKernel.SetComputeSize(projDim);
	volTravLenKernel.SetComputeSize(projDim);
	compKernel.SetComputeSize(projDim);
	subEKernel.SetComputeSize(projDim);
	cropKernel.SetComputeSize(projDim);
    cropInvKernel.SetComputeSize(projDim);
    sample2D.SetComputeSize(projDim);

	wbp.SetComputeSize(fftDim);
	ctf.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);
	fourFilterKernel.SetComputeSize(fftDim);
	doseWeightingKernel.SetComputeSize(fftDim);
	conjKernel.SetComputeSize(fftDim);
    pcKernel.SetComputeSize(fftDim);
	cts.SetComputeSize(projDim);
	//maxShiftKernel.SetComputeSize(proj.GetMaxDimension(), proj.GetMaxDimension(), 1);
	//convVolKernel.SetComputeSize(config.RecDimensions.x, config.RecDimensions.y, 1);
	dimBordersKernel.SetComputeSize(projDim);
	//osKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);
    copyToPitched.SetComputeSize(projDim);
    copyFromPitched.SetComputeSize(projDim);
    addToPitched.SetComputeSize(projDim);
    // END Kernel Dimensions //

	// BEGIN Device variables //
	// Real space
    realprojUS_d.Alloc(proj.GetWidth() * sizeof(int), proj.GetHeight(), sizeof(int));
	proj_d.Alloc(projPitchFloat, projDim.y, sizeof(float));
    proj_children_d.Alloc(projPitchFloat, projDim.y, sizeof(float));
	realproj_d.Alloc(projPitchFloat, projDim.y, sizeof(float));
	dist_d.Alloc(projPitchFloat, projDim.y, sizeof(float));
    dist_children_d.Alloc(projPitchFloat, projDim.y, sizeof(float));
    filterImage_d.Alloc(projPitchFloat, projDim.y, sizeof(float));
    proj_dv_d.Alloc(projSizeF32);
    // END Device variables //

    // BEGIN Projection order //
    int projCount = 0;
    int* indexList;
    proj.CreateProjectionIndexList(PLT_RANDOM_START_ZERO_TILT, &projCount, &indexList);
    // END Projection order //


    // BEGIN LUT computation //
    // LUT Kernels
    preFilterBox.SetComputeSize(proj.GetWidth() / 2 + 1, proj.GetHeight(), 1);
    postFilterBox.SetComputeSize(proj.GetWidth() / 2 + 1, proj.GetHeight(), 1);
    postFilter.SetComputeSize(proj.GetWidth() / 2 + 1, proj.GetHeight(), 1);


    // Device vars for computation
    d_prefilter_fft.Alloc(fftSizeFC32);
    // END LUT computation //

    // START SNR computation
    // Alloc 1D arrays
    signal_power_d.Alloc(snrShells * sizeof(float));
    noise_power_d.Alloc(snrShells * sizeof(float));
    multiplicity_d.Alloc(snrShells * sizeof(float));
    fourier_contrib_d.Alloc(snrShells * sizeof(float));

    signal_power_arr_d.Alloc(CU_AD_FORMAT_FLOAT,
                             snrShells, 1);
    noise_power_arr_d.Alloc(CU_AD_FORMAT_FLOAT,
                            snrShells, 1);

    texSP.Bind(CU_TR_ADDRESS_MODE_CLAMP,
               CU_TR_FILTER_MODE_LINEAR,
               0, &signal_power_arr_d,
               CU_AD_FORMAT_FLOAT, 1);
    texNP.Bind(CU_TR_ADDRESS_MODE_CLAMP,
               CU_TR_FILTER_MODE_LINEAR,
               0, &noise_power_arr_d,
               CU_AD_FORMAT_FLOAT, 1);

    // Kernel Compute size
    radialSum.SetComputeSize(fftDim);

    // Host arrays
    sp_stack = new float[snrShells * proj.GetProjCount()];
    np_stack = new float[snrShells * proj.GetProjCount()];
    std::memset(sp_stack, 0, snrShells * proj.GetProjCount() * sizeof(float));
    std::memset(np_stack, 0, snrShells * proj.GetProjCount() * sizeof(float));

    // Halfset setup
    if (doHalfsets) {
        for (int half = 0; half < 2; half++) {
            signal_power_arr_d_HS[half].Alloc(CU_AD_FORMAT_FLOAT,
                                              snrShells, 1);

            noise_power_arr_d_HS[half].Alloc(CU_AD_FORMAT_FLOAT,
                                             snrShells, 1);

            texSP_HS[half].Bind(CU_TR_ADDRESS_MODE_CLAMP,
                                CU_TR_FILTER_MODE_LINEAR,
                                0, &signal_power_arr_d_HS[half],
                                CU_AD_FORMAT_FLOAT, 1);

            texSP_HS[half].Bind(CU_TR_ADDRESS_MODE_CLAMP,
                                CU_TR_FILTER_MODE_LINEAR,
                                0, &noise_power_arr_d_HS[half],
                                CU_AD_FORMAT_FLOAT, 1);

            sp_stack_HS[half] = new float[snrShells * proj.GetProjCount()];
            np_stack_HS[half] = new float[snrShells * proj.GetProjCount()];
            std::memset(sp_stack_HS[half], 0, snrShells * proj.GetProjCount() * sizeof(float));
            std::memset(np_stack_HS[half], 0, snrShells * proj.GetProjCount() * sizeof(float));
        }
    }
    // END SNR computation

    //START Exact Filter computation aide
    det_mats_d.Alloc(projSource->GetProjectionCount() * sizeof(float3x3));
    //END Exact Filter computation aide

    //START Distance buffer
    proj_distance = new float[proj.GetProjCount()];
    //END Distance buffer

#ifdef REFINE_MODE
	maxShiftWeightedKernel.SetComputeSize(proj.GetMaxDimension(), proj.GetMaxDimension(), 1);
	findPeakKernel.SetComputeSize(proj.GetMaxDimension(), proj.GetMaxDimension(), 1);


	projSubVols_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));
	ccMap = new float[aConfig.MaxShift * 4 * aConfig.MaxShift * 4];
	ccMapMulti = new float[aConfig.MaxShift * 4 * aConfig.MaxShift * 4];
	ccMap_d.Alloc(4 * aConfig.MaxShift * sizeof(float), 4 * aConfig.MaxShift, sizeof(float));
	ccMap_d.Memset(0);

	roiCC1.x = 0;
	roiCC1.y = 0;
	roiCC1.width = aConfig.MaxShift * 2;
	roiCC1.height = aConfig.MaxShift * 2;

	roiCC2.x = proj.GetMaxDimension() - aConfig.MaxShift * 2;
	roiCC2.y = 0;
	roiCC2.width = aConfig.MaxShift * 2;
	roiCC2.height = aConfig.MaxShift * 2;

	roiCC3.x = 0;
	roiCC3.y = proj.GetMaxDimension() - aConfig.MaxShift * 2;
	roiCC3.width = aConfig.MaxShift * 2;
	roiCC3.height = aConfig.MaxShift * 2;

	roiCC4.x = proj.GetMaxDimension() - aConfig.MaxShift * 2;
	roiCC4.y = proj.GetMaxDimension() - aConfig.MaxShift * 2;
	roiCC4.width = aConfig.MaxShift * 2;
	roiCC4.height = aConfig.MaxShift * 2;

	roiDestCC4.x = 0;
	roiDestCC4.y = 0;
	roiDestCC1.width = aConfig.MaxShift * 2;
	roiDestCC1.height = aConfig.MaxShift * 2;

	roiDestCC3.x = aConfig.MaxShift * 4 - aConfig.MaxShift * 2;
	roiDestCC3.y = 0;
	roiDestCC2.width = aConfig.MaxShift * 2;
	roiDestCC2.height = aConfig.MaxShift * 2;

	roiDestCC2.x = 0;
	roiDestCC2.y = aConfig.MaxShift * 4 - aConfig.MaxShift * 2;
	roiDestCC3.width = aConfig.MaxShift * 2;
	roiDestCC3.height = aConfig.MaxShift * 2;

	roiDestCC1.x = aConfig.MaxShift * 4 - aConfig.MaxShift * 2;
	roiDestCC1.y = aConfig.MaxShift * 4 - aConfig.MaxShift * 2;
	roiDestCC4.width = aConfig.MaxShift * 2;
	roiDestCC4.height = aConfig.MaxShift * 2;
	projSquare2_d.Alloc(proj.GetMaxDimension() * sizeof(float) * proj.GetMaxDimension());
#endif

	// Bind back projection image to texref in BP Kernel
	if (aConfig.CtfMode == Configuration::Config::CTFM_YES)
	{
		texImage.Bind(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &dist_d, CU_AD_FORMAT_FLOAT, 1);
	}
	else
	{
		texImage.Bind(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &proj_d, CU_AD_FORMAT_FLOAT, 1);
        //texOS.Bind(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_POINT, 0, &osproj_v, CU_AD_FORMAT_FLOAT, 1);
	}

	ctf_d.Alloc((proj.GetWidth() / 2 + 1) * sizeof(cuComplex), proj.GetHeight(), sizeof(cuComplex));
	fft_d.Alloc(fftSizeFC32);
    fft_d2.Alloc(fftSizeFC32);

    projSquare_d.Alloc(proj.GetMaxDimension() * sizeof(float) * proj.GetMaxDimension());
	badPixelMask_d.Alloc(projPitchChar, projDim.y, 4 * sizeof(char));

	int bufferSize = 0;
	size_t squarePointerShiftX = ((proj.GetMaxDimension() - proj.GetWidth()) / 2);
	size_t squarePointerShiftY = ((proj.GetMaxDimension() - proj.GetHeight()) / 2) * proj.GetMaxDimension();
	squarePointerShift = squarePointerShiftX + squarePointerShiftY;
	squareBorderSizeX = (proj.GetMaxDimension() - proj.GetWidth()) / 2;
	squareBorderSizeY = (proj.GetMaxDimension() - proj.GetHeight()) / 2;
	//roiBorderSquare.width = squareBorderSize;
	//roiBorderSquare.height = proj.GetHeight();
	roiSquare.width = proj.GetMaxDimension();
	roiSquare.height = proj.GetMaxDimension();

    // TODO: check this is correct
	roiAll.width = proj.GetWidth();
	roiAll.height = proj.GetHeight();


	roiFFT.width = proj.GetMaxDimension() / 2 + 1;
	roiFFT.height = proj.GetHeight();
	nppiMeanStdDevGetBufferHostSize_32f_C1R(roiAll, &bufferSize);
	int bufferSize2;
	nppiMaxIndxGetBufferHostSize_32f_C1R(roiSquare, &bufferSize2);
	if (bufferSize2 > bufferSize)
		bufferSize = bufferSize2;
	nppiMeanGetBufferHostSize_32f_C1R(roiSquare, &bufferSize2);
	if (bufferSize2 > bufferSize)
		bufferSize = bufferSize2;
	nppiSumGetBufferHostSize_32f_C1R(roiSquare, &bufferSize2);
	if (bufferSize2 > bufferSize)
		bufferSize = bufferSize2;

	if (markers.GetProjectionCount() * sizeof(float) > bufferSize)
	{
		bufferSize = markers.GetProjectionCount() * sizeof(float); //for exact WBP filter
	}

	meanbuffer.Alloc(bufferSize * 10);
	meanval.Alloc(sizeof(double));
	stdval.Alloc(sizeof(double));

	MPIBuffer = new float[proj.GetWidth() * proj.GetHeight()];
	SetConstantValues(ctf, proj, 0, config.Cs, config.Voltage);

	ResetProjectionsDevice();
}

Reconstructor::~Reconstructor()
{
	if (MPIBuffer)
	{
		delete[] MPIBuffer;
		MPIBuffer = NULL;
	}

    delete[] np_stack;
    delete[] sp_stack;
    delete[] proj_distance;

    for (int half=0; half < 2; half++) {
        delete[] sp_stack_HS[half];
        delete[] np_stack_HS[half];
    }

    sp_stack_HS.clear();
    np_stack_HS.clear();
	//cufftSafeCall(cufftDestroy(handleR2C));
	//cufftSafeCall(cufftDestroy(handleC2R));
    //cufftSafeCall(cufftDestroy(FFThandleR2Call));
    //cufftSafeCall(cufftDestroy(FFThandleC2Rall));
}

Matrix<float> Reconstructor::GetMagAnistropyMatrix(float aAmount, float angleInDeg, float dimX, float dimY)
{
	float angle = (float)(angleInDeg / 180.0 * M_PI);

	Matrix<float> shiftCenter(3, 3);
	Matrix<float> shiftBack(3, 3);
	Matrix<float> rotMat1 = Matrix<float>::GetRotationMatrix3DZ(angle);
	Matrix<float> rotMat2 = Matrix<float>::GetRotationMatrix3DZ(-angle);
	Matrix<float> stretch(3, 3);
	shiftCenter(0, 0) = 1;
	shiftCenter(0, 1) = 0;
	shiftCenter(0, 2) = -dimX / 2.0f;
	shiftCenter(1, 0) = 0;
	shiftCenter(1, 1) = 1;
	shiftCenter(1, 2) = -dimY / 2.0f;
	shiftCenter(2, 0) = 0;
	shiftCenter(2, 1) = 0;
	shiftCenter(2, 2) = 1;

	shiftBack(0, 0) = 1;
	shiftBack(0, 1) = 0;
	shiftBack(0, 2) = dimX / 2.0f;
	shiftBack(1, 0) = 0;
	shiftBack(1, 1) = 1;
	shiftBack(1, 2) = dimY / 2.0f;
	shiftBack(2, 0) = 0;
	shiftBack(2, 1) = 0;
	shiftBack(2, 2) = 1;

	stretch(0, 0) = aAmount;
	stretch(0, 1) = 0;
	stretch(0, 2) = 0;
	stretch(1, 0) = 0;
	stretch(1, 1) = 1;
	stretch(1, 2) = 0;
	stretch(2, 0) = 0;
	stretch(2, 1) = 0;
	stretch(2, 2) = 1;

	return shiftBack * rotMat2 * stretch * rotMat1 * shiftCenter;
}

void Reconstructor::PlanCTFCorrection(Volume<float>* aVol, int goodProjNumber, int fullProjNumber, const int* indexList)
{
    // Advanced Data layout cuFFT for 2D transform
    // Our X is Cuda's Y
    int inputSize[2] = {(int)projDim.y, (int)projDim.x};
    int inembedR2C[2] = {(int)projDim.y, (int)projDim.x};
    int istrideR2C = 1;
    int idistR2C = (int)projSize;
    int onembedR2C[2] = {(int)fftDim.y, (int)fftDim.x};
    int ostrideR2C = 1;
    int odistR2C = (int)fftSize;

    ctfHandler.PlanSlices(aVol);

    maxSliceNumber = ctfHandler.GetMaxSliceNumber();
    sliceBatchSize = config.CTFSliceBatch;

    // Real Space sliced stack
    CTFbuffer_realRect.Alloc(projSizeF32 * sliceBatchSize);

    // Fourier Space sliced stacks
    CTFbuffer_comp1.Alloc(fftSizeFC32 * sliceBatchSize);

    // Correction
    postFilterSum.AllocOffsets(maxSliceNumber);
    preFilterSpreadAdHoc.AllocOffsets(maxSliceNumber);
    preFilterSpreadSNR.AllocOffsets(maxSliceNumber);

    // All FFT Plans
    size_t sz_slices_r2c = 0;
    size_t sz_slices_c2r = 0;
    size_t sz_proj_c2r = 0;
    size_t sz_proj_r2c = 0;

    // All FFTs in same buffer, so don't allocate while planning
    cufftSafeCall(cufftCreate(&FFThandleR2Call));
    cufftSafeCall(cufftCreate(&FFThandleC2Rall));
    cufftSafeCall(cufftCreate(&handleR2C));
    cufftSafeCall(cufftCreate(&handleC2R));
    cufftSafeCall(cufftSetAutoAllocation(FFThandleR2Call, false));
    cufftSafeCall(cufftSetAutoAllocation(FFThandleC2Rall, false));
    cufftSafeCall(cufftSetAutoAllocation(handleR2C, false));
    cufftSafeCall(cufftSetAutoAllocation(handleC2R, false));

    // For us, x is fastest changing, so confusing setup
    cufftSafeCall(cufftPlan2d(&handleR2C,
                              proj.GetHeight(),
                              proj.GetWidth(),
                              CUFFT_R2C));

    cufftSafeCall(cufftPlan2d(&handleC2R,
                              proj.GetHeight(),
                              proj.GetWidth(),
                              CUFFT_C2R));

    cufftSafeCall(cufftPlanMany(&FFThandleR2Call, 2,
                                inputSize,
                                inembedR2C,
                                istrideR2C,
                                idistR2C,
                                onembedR2C,
                                ostrideR2C,
                                odistR2C,
                                CUFFT_R2C,
                                sliceBatchSize));

    cufftSafeCall(cufftPlanMany(&FFThandleC2Rall, 2,
                                inputSize,
                                onembedR2C,
                                ostrideR2C,
                                odistR2C,
                                inembedR2C,
                                istrideR2C,
                                idistR2C,
                                CUFFT_C2R,
                                sliceBatchSize));

    cufftSafeCall(cufftGetSize(handleR2C, &sz_proj_r2c));
    cufftSafeCall(cufftGetSize(handleC2R, &sz_proj_c2r));
    cufftSafeCall(cufftGetSize(FFThandleR2Call, &sz_slices_r2c));
    cufftSafeCall(cufftGetSize(FFThandleC2Rall, &sz_slices_c2r));

    size_t fftsize = max({sz_proj_r2c, sz_proj_c2r, sz_slices_r2c, sz_slices_c2r});

    // Work area for all FFTs
    CTF_compute_buffer.Alloc(fftsize);
    cufftSetWorkArea(handleR2C, (void*)CTF_compute_buffer.GetDevicePtr());
    cufftSetWorkArea(handleC2R, (void*)CTF_compute_buffer.GetDevicePtr());
    cufftSetWorkArea(FFThandleR2Call, (void*)CTF_compute_buffer.GetDevicePtr());
    cufftSetWorkArea(FFThandleC2Rall, (void*)CTF_compute_buffer.GetDevicePtr());

    // Arrays and Textures for back projection
    for (int sliceIdx=0; sliceIdx<sliceBatchSize; sliceIdx++){
        auto sliceArray = new CudaArray2D(CU_AD_FORMAT_FLOAT,
                                          projDim.x,
                                          projDim.y,
                                          1);
        auto sliceSurf = new CudaSurfaceObject2D(sliceArray);
        auto sliceTex = new CudaTextureObject2D(CU_TR_ADDRESS_MODE_CLAMP,
                                                CU_TR_ADDRESS_MODE_CLAMP,
                                                CU_TR_FILTER_MODE_LINEAR,
                                                0, sliceArray,
                                                CU_AD_FORMAT_FLOAT, 1);

        BPArrays.push_back(sliceArray);
        BPSurfaces.push_back(sliceSurf);
        BPTextures.push_back(sliceTex);
    }

    slicesToArrays.setSlices(BPSurfaces, sliceBatchSize);
    slicesToArrays.SetComputeSize(projDim.x, projDim.y, sliceBatchSize);

    maskedSlicesToArrays.setSlices(BPSurfaces, sliceBatchSize);
    maskedSlicesToArrays.SetComputeSize(projDim.x, projDim.y, sliceBatchSize);

    bpOrthoKernel.setSlices(BPTextures, sliceBatchSize);
    bpOrthoAdd.setSlices(BPTextures, sliceBatchSize);
    bpOrthoAddSS.setSlices(BPTextures, sliceBatchSize);

    if (config.CtfMode == Configuration::Config::CTFM_NO){
        printf("\n Not performing CTF correction.\n");
    } else {
        printf("Plan for CTF correction: \n");
        printf("\t- Batch Size: %i\n", config.CTFSliceBatch);
        printf("\t- Maximum slice number: %i\n", maxSliceNumber);
    }
}


void Reconstructor::BackProjectionChildren(Volume<float>* parentVol,
                                           std::vector<Volume<float>*>&  childVols,
                                           DeviceVolumeFFT& childDevVol,
                                           DeviceVolumeBuf& maskDevVol,
                                           DeviceVolumeBuf& multDevVol,
                                           int stackIdx, float SIRTCount, int iter,
                                           stringstream& output, bool useSNR)
{
    float runtime;

    // Volume dimensions
    uint3 childDim;
    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    uint3 childFFTDim;
    childFFTDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x/2+1;
    childFFTDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childFFTDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Box spline prefilter
    computePreFilter(stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    if (config.WBP_NoSART)
    {
        magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
        magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
    }

    // FFT f --> F
    copyFromPitched(proj_d, proj_dv_d, projDim);

#ifdef WRITEDEBUG
    {
        auto img = new float[proj.GetWidth()*proj.GetHeight()];
        proj_dv_d.CopyDeviceToHost(img, proj.GetWidth()*proj.GetHeight()*sizeof(float));
        stringstream DP;
        DP << "after_bppitchcopy_" << stackIdx << "_" << iter << ".em";
        emwrite(DP.str(), img, proj.GetWidth(),
                proj.GetHeight());
        delete[] img;
    }
#endif

    // Mask image f --> f_m
    cropKernel(proj_dv_d,
               projDim,
               config.CutLength,
               config.DimLength,
               corners,
               normVals);



    // FFT f_m -> F_M
    cufftSafeCall(cufftExecR2C(handleR2C,
                               (cufftReal*)proj_dv_d.GetDevicePtr(),
                               (cufftComplex*)fft_d.GetDevicePtr()));

    // Zero out particle and multiplicity
    set3D.SetComputeSize(childDim.x, childDim.y, childDim.z);
    set3D(childDevVol.surface_dual(), 0, childDim);

    // Batched CTF correction
    int batchCount = ctfHandler.GetBatchCount(stackIdx);
    for (int ctfBatch = 0; ctfBatch < batchCount; ctfBatch++) {

        // Min/Max slice and effective batch size
        int2 minmax = ctfHandler.GetMinMaxSlice(stackIdx, ctfBatch);
        int batchSize = ctfHandler.GetBatchSize(stackIdx, ctfBatch);

        // Progress
        cout << "\r\e[K" << flush;
        cout << output.str() << "BP " << stackIdx << " | CTF batch " <<  ctfBatch + 1 << "/" << batchCount << flush;


        // Compute [F_m * Q]
        if (config.CtfMode == Configuration::Config::CTFM_YES) {
            if (useSNR) {
                preFilterSpreadSNR.SetComputeSize(fftDim.x, fftDim.y, batchSize);
                preFilterSpreadSNR(fft_d,
                                   d_prefilter_fft,
                                   CTFbuffer_comp1,
                                   texSP,
                                   texNP,
                                   fftDim,
                                   ctfHandler.GetGlobalConstants(),
                                   ctfHandler.GetImageConstants(stackIdx),
                                   ctfHandler.GetDefocusOffsets(stackIdx),
                                   minmax,
                                   config.DeconvStrength);
            } else {
                preFilterSpreadAdHoc.SetComputeSize(fftDim.x, fftDim.y, batchSize);
                preFilterSpreadAdHoc(fft_d,
                                     d_prefilter_fft,
                                     CTFbuffer_comp1,
                                     fftDim,
                                     ctfHandler.GetGlobalConstants(),
                                     ctfHandler.GetImageConstants(stackIdx),
                                     ctfHandler.GetDefocusOffsets(stackIdx),
                                     minmax);
            }
        } else {
            postFilter(fft_d,
                       d_prefilter_fft,
                       CTFbuffer_comp1,
                       fftDim,
                       (float)projSize);
        }

        // Batched IFFT --> F' -> f'
        cufftSafeCall(cufftExecC2R(FFThandleC2Rall,
                                   (cufftComplex *) CTFbuffer_comp1.GetDevicePtr(),
                                   (cufftReal *) CTFbuffer_realRect.GetDevicePtr()));

        // Copy to CUDA arrays and apply particle mask
        slicesToArrays(CTFbuffer_realRect, projDim);

        // We have now prepared everything for BP of the children.
        // Now just backproject all of 'em.
        int c = 0;
        for (auto child: childVols) {
            c++;
            // System Matrix
            float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, child->VolumeMatrix());

            // BP
            runtime = bpOrthoAdd(projDim,
                                 childDim,
                                 config.Lambda / SIRTCount,
                                 ctfHandler.GetImageConstants(stackIdx),
                                 systemMatrix,
                                 childDevVol.surface_dual(),
                                 minmax);
        }
    }

    childDevVol.SplinePrefilter(DUAL_TO_DUAL);

    add3D.SetComputeSize(childDim.x, childDim.y, childDim.z);
    add3D(childDevVol.surface_dual(),
          childDevVol.surface_card(),
          childDevVol.GetDim(),
          1.f / (float) childVols.size());

}

void Reconstructor::BackProjectionChildrenHS(Volume<float>* parentVol,
                                             std::vector<Volume<float>*>&  childVols,
                                             DeviceVolumeFFT& childDevVol,
                                             std::vector<DeviceVolumeFFT*>& childDevHalfVols,
                                             DeviceVolumeBuf& maskDevVol,
                                             DeviceVolumeBuf& multDevVol,
                                             int stackIdx, float SIRTCount, int iter,
                                             stringstream& output, bool useSNR)
{
    float runtime;

    // Volume dimensions
    uint3 childDim;
    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    uint3 childFFTDim;
    childFFTDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x/2+1;
    childFFTDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childFFTDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Box spline prefilter
    computePreFilter(stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    if (config.WBP_NoSART)
    {
        magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
        magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
    }

    // FFT f --> F
    copyFromPitched(proj_d, proj_dv_d, projDim);

#ifdef WRITEDEBUG
    {
        auto img = new float[proj.GetWidth()*proj.GetHeight()];
        proj_dv_d.CopyDeviceToHost(img, proj.GetWidth()*proj.GetHeight()*sizeof(float));
        stringstream DP;
        DP << "after_bppitchcopy_" << stackIdx << "_" << iter << ".em";
        emwrite(DP.str(), img, proj.GetWidth(),
                proj.GetHeight());
        delete[] img;
    }
#endif

    // Mask image f --> f_m
    cropKernel(proj_dv_d,
               projDim,
               config.CutLength,
               config.DimLength,
               corners,
               normVals);

    // FFT f_m -> F_M
    cufftSafeCall(cufftExecR2C(handleR2C,
                               (cufftReal*)proj_dv_d.GetDevicePtr(),
                               (cufftComplex*)fft_d.GetDevicePtr()));

    // Zero out particles
    set3D.SetComputeSize(childDim.x, childDim.y, childDim.z);
    set3D(childDevVol.surface_dual(), 0.f, childDim);
    for (auto childHalf : childDevHalfVols){
        set3D(childHalf->surface_dual(), 0.f, childDim);
    }

    // Batched CTF correction
    int batchCount = ctfHandler.GetBatchCount(stackIdx);
    for (int ctfBatch = 0; ctfBatch < batchCount; ctfBatch++) {

        // Min/Max slice and effective batch size
        int2 minmax = ctfHandler.GetMinMaxSlice(stackIdx, ctfBatch);
        int batchSize = ctfHandler.GetBatchSize(stackIdx, ctfBatch);

        // Progress
        cout << "\r\e[K" << flush;
        cout << output.str() << "BP " << stackIdx << " | CTF batch " <<  ctfBatch + 1 << "/" << batchCount << flush;


        // Compute [F_m * Q]
        if (config.CtfMode == Configuration::Config::CTFM_YES) {
            if (useSNR) {
                preFilterSpreadSNR.SetComputeSize(fftDim.x, fftDim.y, batchSize);
                preFilterSpreadSNR(fft_d,
                                   d_prefilter_fft,
                                   CTFbuffer_comp1,
                                   texSP,
                                   texNP,
                                   fftDim,
                                   ctfHandler.GetGlobalConstants(),
                                   ctfHandler.GetImageConstants(stackIdx),
                                   ctfHandler.GetDefocusOffsets(stackIdx),
                                   minmax,
                                   config.DeconvStrength);
            } else {
                preFilterSpreadAdHoc.SetComputeSize(fftDim.x, fftDim.y, batchSize);
                preFilterSpreadAdHoc(fft_d,
                                     d_prefilter_fft,
                                     CTFbuffer_comp1,
                                     fftDim,
                                     ctfHandler.GetGlobalConstants(),
                                     ctfHandler.GetImageConstants(stackIdx),
                                     ctfHandler.GetDefocusOffsets(stackIdx),
                                     minmax);
            }
        } else {
            postFilter(fft_d,
                       d_prefilter_fft,
                       CTFbuffer_comp1,
                       fftDim,
                       (float)projSize);
        }

        // Batched IFFT --> F' -> f'
        cufftSafeCall(cufftExecC2R(FFThandleC2Rall,
                                   (cufftComplex *) CTFbuffer_comp1.GetDevicePtr(),
                                   (cufftReal *) CTFbuffer_realRect.GetDevicePtr()));

        // Copy to CUDA arrays and apply particle mask
        slicesToArrays(CTFbuffer_realRect, projDim);

        // We have now prepared everything for BP of the children.
        // Now just backproject all of 'em.
        int c = 0;
        for (auto child: childVols) {
            c++;
            // System Matrix
            float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, child->VolumeMatrix());

            // BP halfset
            runtime = bpOrthoAdd(projDim,
                                 childDim,
                                 config.Lambda / SIRTCount,
                                 ctfHandler.GetImageConstants(stackIdx),
                                 systemMatrix,
                                 childDevHalfVols[child->GetHalfSet()]->surface_dual(),
                                 minmax);
        }
    }

    int c = 0;
    for (auto childHalf : childDevHalfVols) {
        childHalf->SplinePrefilter(DUAL_TO_DUAL);

//        multDevVol.reset();
//        multiplicity3D.SetComputeSize(multDevVol.GetDim());
//        for (auto child : childVols){
//            if (child->GetHalfSet() != c) continue;
//
//            float3x3 systemMatrix = proj.SystemMatrix3x3(stackIdx, child->VolumeMatrixNorm());
//            multiplicity3D(multDevVol.device_var(), multDevVol.GetDim(), systemMatrix);
//        }
//
//        childHalf->DualToVar();
//        childHalf->MultNorm(multDevVol, 1.f);
//        childHalf->VarToDual();

        // Add to halfset
        add3D.SetComputeSize(childDim.x, childDim.y, childDim.z);
        add3D(childHalf->surface_dual(),
              childHalf->surface_card(),
              childHalf->GetDim(),
              1.f / ((float) childVols.size() * 0.5f));

        // Add to full recon
        add3D(childHalf->surface_dual(),
              childDevVol.surface_card(),
              childHalf->GetDim(),
              1.f / ((float) childVols.size() * 0.5f));

        c++;
    }
}

void Reconstructor::BackProjectionOrphans(Volume<float>* parentVol,
                                          std::vector<Volume<float>*>&  childVols,
                                          DeviceVolumeFFT& childDevVol,
                                          DeviceVolumeBuf& maskDevVol,
                                          DeviceVolumeBuf& multDevVol,
                                          int stackIdx, float SIRTCount, int iter,
                                          stringstream& output, bool useSNR)
{
    float runtime;

    // Volume dimensions
    uint3 childDim;
    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    uint3 childFFTDim;
    childFFTDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x/2+1;
    childFFTDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childFFTDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Box spline prefilter
    computePreFilter(stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    if (config.WBP_NoSART)
    {
        magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
        magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
    }

    // FFT f --> F
    copyFromPitched(proj_d, proj_dv_d, projDim);

    // Mask image f --> f_m
    cropKernel(proj_dv_d,
               projDim,
               config.CutLength,
               config.DimLength,
               corners,
               normVals);

    // FFT f_m -> F_M
    cufftSafeCall(cufftExecR2C(handleR2C,
                               (cufftReal*)proj_dv_d.GetDevicePtr(),
                               (cufftComplex*)fft_d.GetDevicePtr()));

    // Zero out noise vol
    set3D.SetComputeSize(childDim.x, childDim.y, childDim.z);
    set3D(childDevVol.surface_dual(), 0, childDim);

    // No CTF correction, single slice back projection of noise
    int batchCount = 1;

    // Progress
    cout << "\r\e[K" << flush;
    cout << output.str() << "Bn " << stackIdx << " | CTF batch " <<  1 << "/" << batchCount << flush;

    postFilter(fft_d,
               d_prefilter_fft,
               CTFbuffer_comp1,
               fftDim,
               (float)projSize);

    // IFFT --> F' -> f' (not batched, as only the first slice has signal)
    cufftSafeCall(cufftExecC2R(handleC2R,
                               (cufftComplex *) CTFbuffer_comp1.GetDevicePtr(),
                               (cufftReal *) CTFbuffer_realRect.GetDevicePtr()));

    // Copy to CUDA arrays
    slicesToArrays(CTFbuffer_realRect, projDim);

    // We have now prepared everything for BP of the children.
    // Now just backproject all of 'em.
    int c = 0;
    for (auto child: childVols) {
        c++;
        // System Matrix
        float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, child->VolumeMatrix());

        // BP
        runtime = bpOrthoAddSS(projDim,
                             childDim,
                             1.f,
                             ctfHandler.GetImageConstants(stackIdx),
                             systemMatrix,
                             childDevVol.surface_dual());
    }

    childDevVol.SplinePrefilter(DUAL_TO_DUAL);

    childDevVol.DualToVar();
    childDevVol.DivC((float)childVols.size());
    childDevVol.VarToCard();
}

void Reconstructor::BackProjectionOrphansHS(Volume<float>* parentVol,
                                            std::vector<Volume<float>*>&  childVols,
                                            DeviceVolumeFFT& childDevVol,
                                            std::vector<DeviceVolumeFFT*>& childDevHalfVols,
                                            DeviceVolumeBuf& maskDevVol,
                                            DeviceVolumeBuf& multDevVol,
                                            int stackIdx, float SIRTCount, int iter,
                                            stringstream& output, bool useSNR)
{
    float runtime;

    // Volume dimensions
    uint3 childDim;
    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    uint3 childFFTDim;
    childFFTDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x/2+1;
    childFFTDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childFFTDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Box spline prefilter
    computePreFilter(stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    if (config.WBP_NoSART)
    {
        magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
        magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
    }

    // FFT f --> F
    copyFromPitched(proj_d, proj_dv_d, projDim);

    // Mask image f --> f_m
    cropKernel(proj_dv_d,
               projDim,
               config.CutLength,
               config.DimLength,
               corners,
               normVals);

    // FFT f_m -> F_M
    cufftSafeCall(cufftExecR2C(handleR2C,
                               (cufftReal*)proj_dv_d.GetDevicePtr(),
                               (cufftComplex*)fft_d.GetDevicePtr()));

    // Zero out noise vol
    set3D.SetComputeSize(childDim.x, childDim.y, childDim.z);
    set3D(childDevVol.surface_dual(), 0.f, childDim);
    for (auto childHalf : childDevHalfVols){
        set3D(childHalf->surface_dual(), 0.f, childDim);
    }

    // No CTF correction, single slice back projection of noise
    int batchCount = 1;

    // Progress
    cout << "\r\e[K" << flush;
    cout << output.str() << "Bn " << stackIdx << " | CTF batch " <<  1 << "/" << batchCount << flush;

    postFilter(fft_d,
               d_prefilter_fft,
               CTFbuffer_comp1,
               fftDim,
               (float)projSize);

    // IFFT --> F' -> f' (not batched, as only the first slice has signal)
    cufftSafeCall(cufftExecC2R(handleC2R,
                               (cufftComplex *) CTFbuffer_comp1.GetDevicePtr(),
                               (cufftReal *) CTFbuffer_realRect.GetDevicePtr()));

    // Copy to CUDA arrays
    slicesToArrays(CTFbuffer_realRect, projDim);

    // We have now prepared everything for BP of the children.
    // Now just backproject all of 'em.
    int c = 0;
    for (auto child: childVols) {
        c++;
        // System Matrix
        float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, child->VolumeMatrix());

        // BP
        runtime = bpOrthoAddSS(projDim,
                               childDim,
                               1.f,
                               ctfHandler.GetImageConstants(stackIdx),
                               systemMatrix,
                               childDevHalfVols[child->GetHalfSet()]->surface_dual());
    }

    c = 0;
    for (auto childHalf : childDevHalfVols) {
        childHalf->SplinePrefilter(DUAL_TO_DUAL);

//        multDevVol.reset();
//        multiplicity3D.SetComputeSize(multDevVol.GetDim());
//        for (auto child : childVols){
//            if (child->GetHalfSet() != c) continue;
//
//            float3x3 systemMatrix = proj.SystemMatrix3x3(stackIdx, child->VolumeMatrixNorm());
//            multiplicity3D(multDevVol.device_var(), multDevVol.GetDim(), systemMatrix);
//        }
//
//        childHalf->DualToVar();
//        childHalf->MultNorm(multDevVol, 1.f);
//        childHalf->VarToDual();


        childHalf->DualToVar();
        childHalf->DivC((float) childVols.size() * 0.5f);
        childHalf->VarToCard();

        add3D.SetComputeSize(childDim.x, childDim.y, childDim.z);
        add3D(childHalf->surface_card(),
              childDevVol.surface_card(),
              childHalf->GetDim(),
              1.f);

        c++;
    }
}


void Reconstructor::ForwardProjectionChildren(Volume<float>* parentVol,
                                              std::vector<Volume<float>*>& childVols,
                                              DeviceVolumeFFT& childDevVol,
                                              DeviceVolumeBuf& maskDevVol,
                                              int stackIdx, bool volumeIsEmpty, int iter, bool noSync,
                                              stringstream& output)
{
    float runtime;

    // Volume dimensions
    uint3 childDim;
    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    uint3 parentDim;
    parentDim.x = (uint) parentVol->GetSubVolumeDimension(0).x;
    parentDim.y = (uint) parentVol->GetSubVolumeDimension(0).y;
    parentDim.z = (uint) parentVol->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Compute lookup table
    computePostFilter(childVols[0], stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    // Mask and Prefilter volume
    mask3D.SetComputeSize(childDim);
    mask3D(childDevVol.surface_card(),
           childDevVol.surface_dual(),
           maskDevVol.surface_card(),
           childDim);

    childDevVol.SplinePrefilter(DUAL_TO_DUAL);

    // Batched CTF correction
    fft_d.Memset(0);
    fft_d2.Memset(0);
    int batchCount = ctfHandler.GetBatchCount(stackIdx);
    for (int ctfBatch = 0; ctfBatch < batchCount; ctfBatch++) {
        // Min/Max slice and effective batch size
        int2 minmax = ctfHandler.GetMinMaxSlice(stackIdx, ctfBatch);
        int batchSize = ctfHandler.GetBatchSize(stackIdx, ctfBatch);

        // Progress
        cout << "\r\e[K" << flush;
        cout << output.str() << "FP " << stackIdx << " | CTF batch " <<  ctfBatch + 1 << "/" << batchCount << flush;

        // Reset projection and FFT buffer
        CTFbuffer_realRect.Memset(0);
        CTFbuffer_comp1.Memset(0);

        // Project all particles
        for (auto child: childVols) {
            // System Matrix
            float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, child->VolumeMatrix());

            runtime = fpOrthoKernel(projDim,
                                    childDim,
                                    ctfHandler.GetImageConstants(stackIdx),
                                    systemMatrix,
                                    CTFbuffer_realRect,
                                    childDevVol.surface_dual(),
                                    minmax);
        }

        // Mask forward projection: f_m = f .* mask
        cropSlicesKernel.SetComputeSize(projDim.x, projDim.y, (uint) batchSize);
        cropSlicesKernel(CTFbuffer_realRect,
                         projDim,
                         (uint) batchSize,
                         config.CutLength,
                         config.DimLength,
                         corners,
                         normVals);

        // Batched FFT of forward projection: f_m -> F_m
        cufftSafeCall(cufftExecR2C(FFThandleR2Call,
                                   (cufftReal *) CTFbuffer_realRect.GetDevicePtr(),
                                   (cufftComplex *) CTFbuffer_comp1.GetDevicePtr()));

        // Compute sum([F_m * Q * CTF], defSlices)
        if (config.CtfMode == Configuration::Config::CTFM_YES) {
            postFilterSum.SetComputeSize(fftDim.x, fftDim.y, (uint) batchSize);
            postFilterSum(CTFbuffer_comp1,
                          d_prefilter_fft,
                          fft_d,
                          fftDim,
                          ctfHandler.GetGlobalConstants(),
                          ctfHandler.GetImageConstants(stackIdx),
                          ctfHandler.GetDefocusOffsets(stackIdx),
                          minmax);
        } else {
            postFilter(CTFbuffer_comp1,
                       d_prefilter_fft,
                       fft_d,
                       fftDim,
                       (float) projSize);
        }
    }

    // IFFT of the summed transform
    cufftSafeCall(cufftExecC2R(handleC2R,
                               (cufftComplex*) fft_d.GetDevicePtr(),
                               (cufftReal*) proj_dv_d.GetDevicePtr()));

    //addToPitched(proj_dv_d, proj_d, projDim);
    copyToPitched(proj_dv_d, proj_d, projDim);
}

void Reconstructor::ForwardProjectionChildrenHS(Volume<float>* parentVol,
                                                std::vector<Volume<float>*>& childVols,
                                                std::vector<DeviceVolumeFFT*>& childDevVols,
                                                DeviceVolumeBuf& maskDevVol,
                                                int stackIdx, bool volumeIsEmpty, int iter, bool noSync,
                                                stringstream& output)
{
    float runtime;

    // Volume dimensions
    uint3 childDim;
    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    uint3 parentDim;
    parentDim.x = (uint) parentVol->GetSubVolumeDimension(0).x;
    parentDim.y = (uint) parentVol->GetSubVolumeDimension(0).y;
    parentDim.z = (uint) parentVol->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Compute lookup table
    computePostFilter(childVols[0], stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    // Mask and Prefilter volumes
    for (auto childHalf : childDevVols) {
        mask3D.SetComputeSize(childDim);
        mask3D(childHalf->surface_card(),
               childHalf->surface_dual(),
               maskDevVol.surface_card(),
               childDim);

        childHalf->SplinePrefilter(DUAL_TO_DUAL);
    }

    // Batched CTF correction
    fft_d.Memset(0);
    fft_d2.Memset(0);
    int batchCount = ctfHandler.GetBatchCount(stackIdx);
    for (int ctfBatch = 0; ctfBatch < batchCount; ctfBatch++) {
        // Min/Max slice and effective batch size
        int2 minmax = ctfHandler.GetMinMaxSlice(stackIdx, ctfBatch);
        int batchSize = ctfHandler.GetBatchSize(stackIdx, ctfBatch);

        // Progress
        cout << "\r\e[K" << flush;
        cout << output.str() << "FP " << stackIdx << " | CTF batch " <<  ctfBatch + 1 << "/" << batchCount << flush;

        // Reset projection and FFT buffer
        CTFbuffer_realRect.Memset(0);
        CTFbuffer_comp1.Memset(0);

        // Project all particles
        for (auto child: childVols) {
            // System Matrix
            float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, child->VolumeMatrix());

            runtime = fpOrthoKernel(projDim,
                                    childDim,
                                    ctfHandler.GetImageConstants(stackIdx),
                                    systemMatrix,
                                    CTFbuffer_realRect,
                                    childDevVols[child->GetHalfSet()]->surface_dual(),
                                    minmax);
        }

        // Mask forward projection: f_m = f .* mask
        cropSlicesKernel.SetComputeSize(projDim.x, projDim.y, (uint) batchSize);
        cropSlicesKernel(CTFbuffer_realRect,
                         projDim,
                         (uint) batchSize,
                         config.CutLength,
                         config.DimLength,
                         corners,
                         normVals);

        // Batched FFT of forward projection: f_m -> F_m
        cufftSafeCall(cufftExecR2C(FFThandleR2Call,
                                   (cufftReal *) CTFbuffer_realRect.GetDevicePtr(),
                                   (cufftComplex *) CTFbuffer_comp1.GetDevicePtr()));

        // Compute sum([F_m * Q * CTF], defSlices)
        if (config.CtfMode == Configuration::Config::CTFM_YES) {
            postFilterSum.SetComputeSize(fftDim.x, fftDim.y, (uint) batchSize);
            postFilterSum(CTFbuffer_comp1,
                          d_prefilter_fft,
                          fft_d,
                          fftDim,
                          ctfHandler.GetGlobalConstants(),
                          ctfHandler.GetImageConstants(stackIdx),
                          ctfHandler.GetDefocusOffsets(stackIdx),
                          minmax);
        } else {
            postFilter(CTFbuffer_comp1,
                       d_prefilter_fft,
                       fft_d,
                       fftDim,
                       (float) projSize);
        }
    }

    // IFFT of the summed transform
    cufftSafeCall(cufftExecC2R(handleC2R,
                               (cufftComplex*) fft_d.GetDevicePtr(),
                               (cufftReal*) proj_dv_d.GetDevicePtr()));

    //addToPitched(proj_dv_d, proj_d, projDim);
    copyToPitched(proj_dv_d, proj_d, projDim);
}

void Reconstructor::ForwardProjectionConjoinedChildren(Volume<float>* parentVol,
                                                       std::vector<Volume<float>*>& childVols,
                                                       DeviceVolume& overlapDevVol,
                                                       DeviceVolumeFFT& childDevVol,
                                                       DeviceVolumeBuf& maskDevVol,
                                                       int stackIdx, bool volumeIsEmpty, int iter, bool noSync,
                                                       stringstream& output)
{
    float runtime;

    // Volume dimensions
    uint3 childDim;
    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    uint3 parentDim;
    parentDim.x = (uint) parentVol->GetSubVolumeDimension(0).x;
    parentDim.y = (uint) parentVol->GetSubVolumeDimension(0).y;
    parentDim.z = (uint) parentVol->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Compute lookup table
    computePostFilter(childVols[0], stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    // Mask and Prefilter volume
    mask3D.SetComputeSize(childDim);
    mask3D(childDevVol.surface_card(),
           childDevVol.surface_dual(),
           maskDevVol.surface_card(),
           childDim);

    childDevVol.SplinePrefilter(DUAL_TO_DUAL);

    // Batched CTF correction
    fft_d.Memset(0);
    fft_d2.Memset(0);
    int batchCount = ctfHandler.GetBatchCount(stackIdx);
    for (int ctfBatch = 0; ctfBatch < batchCount; ctfBatch++) {
        // Min/Max slice and effective batch size
        int2 minmax = ctfHandler.GetMinMaxSlice(stackIdx, ctfBatch);
        int batchSize = ctfHandler.GetBatchSize(stackIdx, ctfBatch);

        // Progress
        cout << "\r\e[K" << flush;
        cout << output.str() << "FP " << stackIdx << " | CTF batch " <<  ctfBatch + 1 << "/" << batchCount << flush;

        // Reset projection and FFT buffer
        CTFbuffer_realRect.Memset(0);
        CTFbuffer_comp1.Memset(0);

        // Inv volume matrix parent (for overlap)
        Matrix<double> parentMatrixInv = parentVol->VolumeMatrixInv();

        // Project all particles
        for (auto child: childVols) {
            // System Matrix (projection
            float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, child->VolumeMatrix());

            // Volume Matrix child (for overlap)
            Matrix<double> childMatrix = child->VolumeMatrix();

            // Child -> Parent (for overlap)
            Matrix<double> transformInv = parentMatrixInv * childMatrix;
            float4x4 childToParent = MatrixTo4x4(transformInv);

            runtime = fpOrthoOVKernel(projDim,
                                      childDim,
                                      ctfHandler.GetImageConstants(stackIdx),
                                      systemMatrix,
                                      CTFbuffer_realRect,
                                      childDevVol.surface_dual(),
                                      minmax,
                                      overlapDevVol.texture_dual(),
                                      childToParent);
        }

        // Mask forward projection: f_m = f .* mask
        cropSlicesKernel.SetComputeSize(projDim.x, projDim.y, (uint) batchSize);
        cropSlicesKernel(CTFbuffer_realRect,
                         projDim,
                         (uint) batchSize,
                         config.CutLength,
                         config.DimLength,
                         corners,
                         normVals);

        // Batched FFT of forward projection: f_m -> F_m
        cufftSafeCall(cufftExecR2C(FFThandleR2Call,
                                   (cufftReal *) CTFbuffer_realRect.GetDevicePtr(),
                                   (cufftComplex *) CTFbuffer_comp1.GetDevicePtr()));

        // Compute sum([F_m * Q * CTF], defSlices)
        if (config.CtfMode == Configuration::Config::CTFM_YES) {
            postFilterSum.SetComputeSize(fftDim.x, fftDim.y, (uint) batchSize);
            postFilterSum(CTFbuffer_comp1,
                          d_prefilter_fft,
                          fft_d,
                          fftDim,
                          ctfHandler.GetGlobalConstants(),
                          ctfHandler.GetImageConstants(stackIdx),
                          ctfHandler.GetDefocusOffsets(stackIdx),
                          minmax);
        } else {
            postFilter(CTFbuffer_comp1,
                       d_prefilter_fft,
                       fft_d,
                       fftDim,
                       (float) projSize);
        }
    }

    // IFFT of the summed transform
    cufftSafeCall(cufftExecC2R(handleC2R,
                               (cufftComplex*) fft_d.GetDevicePtr(),
                               (cufftReal*) proj_dv_d.GetDevicePtr()));

    //addToPitched(proj_dv_d, proj_d, projDim);
    copyToPitched(proj_dv_d, proj_d, projDim);
}

void Reconstructor::ForwardProjectionConjoinedChildrenHS(Volume<float>* parentVol,
                                                         std::vector<Volume<float>*>& childVols,
                                                         DeviceVolume& overlapDevVol,
                                                         std::vector<DeviceVolumeFFT*>& childDevVols,
                                                         DeviceVolumeBuf& maskDevVol,
                                                         int stackIdx, bool volumeIsEmpty, int iter, bool noSync,
                                                         stringstream& output)
{
    float runtime;

    // Volume dimensions
    uint3 childDim;
    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    uint3 parentDim;
    parentDim.x = (uint) parentVol->GetSubVolumeDimension(0).x;
    parentDim.y = (uint) parentVol->GetSubVolumeDimension(0).y;
    parentDim.z = (uint) parentVol->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Compute lookup table
    computePostFilter(childVols[0], stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    // Mask and Prefilter volumes
    for (auto childHalf : childDevVols) {
        mask3D.SetComputeSize(childDim);
        mask3D(childHalf->surface_card(),
               childHalf->surface_dual(),
               maskDevVol.surface_card(),
               childDim);

        childHalf->SplinePrefilter(DUAL_TO_DUAL);
    }

    // Batched CTF correction
    fft_d.Memset(0);
    fft_d2.Memset(0);
    int batchCount = ctfHandler.GetBatchCount(stackIdx);
    for (int ctfBatch = 0; ctfBatch < batchCount; ctfBatch++) {
        // Min/Max slice and effective batch size
        int2 minmax = ctfHandler.GetMinMaxSlice(stackIdx, ctfBatch);
        int batchSize = ctfHandler.GetBatchSize(stackIdx, ctfBatch);

        // Progress
        cout << "\r\e[K" << flush;
        cout << output.str() << "FP " << stackIdx << " | CTF batch " <<  ctfBatch + 1 << "/" << batchCount << flush;

        // Reset projection and FFT buffer
        CTFbuffer_realRect.Memset(0);
        CTFbuffer_comp1.Memset(0);

        // Inv volume matrix parent (for overlap)
        Matrix<double> parentMatrixInv = parentVol->VolumeMatrixInv();

        // Project all particles
        for (auto child: childVols) {
            // System Matrix (projection
            float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, child->VolumeMatrix());

            // Volume Matrix child (for overlap)
            Matrix<double> childMatrix = child->VolumeMatrix();

            // Child -> Parent (for overlap)
            Matrix<double> transformInv = parentMatrixInv * childMatrix;
            float4x4 childToParent = MatrixTo4x4(transformInv);

            runtime = fpOrthoOVKernel(projDim,
                                      childDim,
                                      ctfHandler.GetImageConstants(stackIdx),
                                      systemMatrix,
                                      CTFbuffer_realRect,
                                      childDevVols[child->GetHalfSet()]->surface_dual(),
                                      minmax,
                                      overlapDevVol.texture_dual(),
                                      childToParent);
        }

        // Mask forward projection: f_m = f .* mask
        cropSlicesKernel.SetComputeSize(projDim.x, projDim.y, (uint) batchSize);
        cropSlicesKernel(CTFbuffer_realRect,
                         projDim,
                         (uint) batchSize,
                         config.CutLength,
                         config.DimLength,
                         corners,
                         normVals);

        // Batched FFT of forward projection: f_m -> F_m
        cufftSafeCall(cufftExecR2C(FFThandleR2Call,
                                   (cufftReal *) CTFbuffer_realRect.GetDevicePtr(),
                                   (cufftComplex *) CTFbuffer_comp1.GetDevicePtr()));

        // Compute sum([F_m * Q * CTF], defSlices)
        if (config.CtfMode == Configuration::Config::CTFM_YES) {
            postFilterSum.SetComputeSize(fftDim.x, fftDim.y, (uint) batchSize);
            postFilterSum(CTFbuffer_comp1,
                          d_prefilter_fft,
                          fft_d,
                          fftDim,
                          ctfHandler.GetGlobalConstants(),
                          ctfHandler.GetImageConstants(stackIdx),
                          ctfHandler.GetDefocusOffsets(stackIdx),
                          minmax);
        } else {
            postFilter(CTFbuffer_comp1,
                       d_prefilter_fft,
                       fft_d,
                       fftDim,
                       (float) projSize);
        }
    }

    // IFFT of the summed transform
    cufftSafeCall(cufftExecC2R(handleC2R,
                               (cufftComplex*) fft_d.GetDevicePtr(),
                               (cufftReal*) proj_dv_d.GetDevicePtr()));

    //addToPitched(proj_dv_d, proj_d, projDim);
    copyToPitched(proj_dv_d, proj_d, projDim);
}

void Reconstructor::BackProjectionSiblings(Volume<float>* parentVol,
                                           std::vector<Volume<float>*>&  childVols,
                                           std::vector<DeviceVolumeBuf*>& childDevVols,
                                           DeviceVolumeBuf& maskDevVol,
                                           DeviceVolumeBuf& multDevVol,
                                           int stackIdx, float SIRTCount, int iter, stringstream& output, bool useSNR)
{
    float runtime;

    // Volume dimensions
    uint3 childDim;
    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    uint3 childFFTDim;
    childFFTDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x/2+1;
    childFFTDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childFFTDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Box spline prefilter
    computePreFilter(stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    if (config.WBP_NoSART)
    {
        magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
        magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
    }

    // FFT f --> F
    copyFromPitched(proj_children_d, proj_dv_d, projDim);

    // Mask image f --> f_m
    cropKernel(proj_dv_d,
               projDim,
               config.CutLength,
               config.DimLength,
               corners,
               normVals);

    // FFT f_m -> F_M
    cufftSafeCall(cufftExecR2C(handleR2C,
                               (cufftReal*)proj_dv_d.GetDevicePtr(),
                               (cufftComplex*)fft_d.GetDevicePtr()));

    // Batched CTF correction
    int batchCount = ctfHandler.GetBatchCount(stackIdx);
    for (int ctfBatch = 0; ctfBatch < batchCount; ctfBatch++) {

        // Min/Max slice and effective batch size
        int2 minmax = ctfHandler.GetMinMaxSlice(stackIdx, ctfBatch);
        int batchSize = ctfHandler.GetBatchSize(stackIdx, ctfBatch);

        // Progress
        cout << "\r\e[K" << flush;
        cout << output.str() << "BP " << stackIdx << " | CTF batch " <<  ctfBatch + 1 << "/" << batchCount << flush;

        // Compute [F_m * Q]
        if (config.CtfMode == Configuration::Config::CTFM_YES) {
            if (useSNR) {
                preFilterSpreadSNR.SetComputeSize(fftDim.x, fftDim.y, batchSize);
                preFilterSpreadSNR(fft_d,
                                   d_prefilter_fft,
                                   CTFbuffer_comp1,
                                   texSP,
                                   texNP,
                                   fftDim,
                                   ctfHandler.GetGlobalConstants(),
                                   ctfHandler.GetImageConstants(stackIdx),
                                   ctfHandler.GetDefocusOffsets(stackIdx),
                                   minmax,
                                   config.DeconvStrength);
            } else {
                preFilterSpreadAdHoc.SetComputeSize(fftDim.x, fftDim.y, batchSize);
                preFilterSpreadAdHoc(fft_d,
                                     d_prefilter_fft,
                                     CTFbuffer_comp1,
                                     fftDim,
                                     ctfHandler.GetGlobalConstants(),
                                     ctfHandler.GetImageConstants(stackIdx),
                                     ctfHandler.GetDefocusOffsets(stackIdx),
                                     minmax);
            }
        } else {
            postFilter(fft_d,
                       d_prefilter_fft,
                       CTFbuffer_comp1,
                       fftDim,
                       (float)projSize);
        }

        // Batched IFFT --> F' -> f'
        cufftSafeCall(cufftExecC2R(FFThandleC2Rall,
                                   (cufftComplex *) CTFbuffer_comp1.GetDevicePtr(),
                                   (cufftReal *) CTFbuffer_realRect.GetDevicePtr()));

        // Copy to CUDA arrays
        slicesToArrays(CTFbuffer_realRect, projDim);

        // We have now prepared everything for BP of the children.
        // Now just backproject all of 'em.
        int c = 0;
        for (auto child : childVols) {
            // System Matrix
            float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, child->VolumeMatrix());

            // BP
            runtime = bpOrthoAdd(projDim,
                                 childDim,
                                 1.f,
                                 ctfHandler.GetImageConstants(stackIdx),
                                 systemMatrix,
                                 childDevVols[c]->surface_dual(),
                                 minmax);

            c++;
        }
    }
}

void Reconstructor::BackProjectionParent(Volume<float>* parentVol,
                                         DeviceVolume& parentDevVol,
                                         int stackIdx, float SIRTCount,
                                         int iter, stringstream& output, bool useSNR)
{
    float runtime;

    // Volume dimensions
    uint3 parentDim;
    parentDim.x = (uint) parentVol->GetSubVolumeDimension(0).x;
    parentDim.y = (uint) parentVol->GetSubVolumeDimension(0).y;
    parentDim.z = (uint) parentVol->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Box spline prefilter
    computePreFilter(stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims for parent volume
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    if (config.WBP_NoSART)
    {
        magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
        magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
    }

    // Get from pitched
    copyFromPitched(proj_d, proj_dv_d, projDim);

    // Mask image f --> f_m
    cropKernel(proj_dv_d,
                  projDim,
                  config.CutLength,
                  config.DimLength,
                  corners,
                  normVals);

    // FFT f --> F
    cufftSafeCall(cufftExecR2C(handleR2C,
                               (cufftReal*)proj_dv_d.GetDevicePtr(),
                               (cufftComplex*)fft_d.GetDevicePtr()));

    // Batched CTF correction
    int batchCount = ctfHandler.GetBatchCount(stackIdx);
    for (int ctfBatch = 0; ctfBatch < batchCount; ctfBatch++) {

        // Min/Max slice and effective batch size
        int2 minmax = ctfHandler.GetMinMaxSlice(stackIdx, ctfBatch);
        int batchSize = ctfHandler.GetBatchSize(stackIdx, ctfBatch);

        // Progress
        cout << "\r\e[K" << flush;
        cout << output.str() << "BP " << stackIdx << " | CTF batch " <<  ctfBatch + 1 << "/" << batchCount << flush;

        // Compute [F_m * Q]
        if (config.CtfMode == Configuration::Config::CTFM_YES) {
            if (useSNR) {
                preFilterSpreadSNR.SetComputeSize(fftDim.x,fftDim.y,batchSize);
                preFilterSpreadSNR(fft_d,
                                   d_prefilter_fft,
                                   CTFbuffer_comp1,
                                   texSP,
                                   texNP,
                                   fftDim,
                                   ctfHandler.GetGlobalConstants(),
                                   ctfHandler.GetImageConstants(stackIdx),
                                   ctfHandler.GetDefocusOffsets(stackIdx),
                                   minmax,
                                   config.DeconvStrength);
            } else {
                preFilterSpreadAdHoc.SetComputeSize(fftDim.x, fftDim.y, batchSize);
                preFilterSpreadAdHoc(fft_d,
                                     d_prefilter_fft,
                                     CTFbuffer_comp1,
                                     fftDim,
                                     ctfHandler.GetGlobalConstants(),
                                     ctfHandler.GetImageConstants(stackIdx),
                                     ctfHandler.GetDefocusOffsets(stackIdx),
                                     minmax);
            }
        } else {
            postFilter(fft_d,
                       d_prefilter_fft,
                       CTFbuffer_comp1,
                       fftDim,
                       (float) projSize);
        }

        // Batched IFFT --> F' -> f'
        cufftSafeCall(cufftExecC2R(FFThandleC2Rall,
                                   (cufftComplex *) CTFbuffer_comp1.GetDevicePtr(),
                                   (cufftReal *) CTFbuffer_realRect.GetDevicePtr()));

        // Copy to CUDA arrays
        slicesToArrays(CTFbuffer_realRect, projDim);

        // We have now prepared everything for BP of the entire family.
        // Backproject Parent
        // System Matrix
        float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, parentVol->VolumeMatrix());

        // Actual BP
        runtime = bpOrthoKernel(projDim,
                                parentDim,
                                config.Lambda / SIRTCount,
                                ctfHandler.GetImageConstants(stackIdx),
                                systemMatrix,
                                parentDevVol.surface_dual(),
                                minmax);
    }

    // Postfilter and add to reconstruction
    postfilter3DX(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int) parentDim.x, (int) parentDim.y, (int) parentDim.z);
    postfilter3DY(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int) parentDim.x, (int) parentDim.y, (int) parentDim.z);
    postfilter3DZ(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int) parentDim.x, (int) parentDim.y, (int) parentDim.z);

    add3D.SetComputeSize(parentDim.x, parentDim.y, parentDim.z);
    add3D(parentDevVol.surface_dual(), parentDevVol.surface_card(), parentDevVol.GetDim());
}

void Reconstructor::ForwardProjectionParent(Volume<float>* parentVol,
                                            DeviceVolume& parentDevVol,
                                            int stackIdx, bool volumeIsEmpty, int iter, stringstream& output)
{
    float runtime;

    uint3 parentDim;
    parentDim.x = (uint) parentVol->GetSubVolumeDimension(0).x;
    parentDim.y = (uint) parentVol->GetSubVolumeDimension(0).y;
    parentDim.z = (uint) parentVol->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Compute lookup table
    computePostFilter(parentVol, stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    // Prefilter parent
    prefilter3DX(parentDevVol.surface_card(), parentDevVol.surface_dual(), (int)parentDim.x, (int)parentDim.y, (int)parentDim.z);
    prefilter3DY(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int)parentDim.x, (int)parentDim.y, (int)parentDim.z);
    prefilter3DZ(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int)parentDim.x, (int)parentDim.y, (int)parentDim.z);

    // System Matrix
    float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, parentVol->VolumeMatrix());

    // Batched CTF correction
    fft_d.Memset(0);
    fft_d2.Memset(0);
    int batchCount = ctfHandler.GetBatchCount(stackIdx);
    for (int ctfBatch = 0; ctfBatch < batchCount; ctfBatch++) {

        // Min/Max slice and effective batch size
        int2 minmax = ctfHandler.GetMinMaxSlice(stackIdx, ctfBatch);
        int batchSize = ctfHandler.GetBatchSize(stackIdx, ctfBatch);
//        printf("CTF batch %i/%i from %i to %i of %i\n", ctfBatch + 1, batchCount, minmax.x + 1, minmax.y + 1,
//               sliceNumber);

        cout << "\r\e[K" << flush;
        cout << output.str() << "FP " << stackIdx << " | CTF batch " <<  ctfBatch + 1 << "/" << batchCount << flush;

        // Reset projection and FFT buffer
        CTFbuffer_realRect.Memset(0);
        CTFbuffer_comp1.Memset(0);

        // FP parent
        runtime = fpOrthoKernel(projDim,
                                parentDim,
                                ctfHandler.GetImageConstants(stackIdx),
                                systemMatrix,
                                CTFbuffer_realRect,
                                parentDevVol.surface_dual(),
                                minmax);

        // Mask forward projection: f_m = f .* mask
        cropSlicesKernel.SetComputeSize(projDim.x, projDim.y, batchSize);
        cropSlicesKernel(CTFbuffer_realRect,
                         projDim,
                         batchSize,
                         config.CutLength,
                         config.DimLength,
                         corners,
                         normVals);

        // Batched FFT of forward projection: f_m -> F_m
        cufftSafeCall(cufftExecR2C(FFThandleR2Call,
                                   (cufftReal *) CTFbuffer_realRect.GetDevicePtr(),
                                   (cufftComplex *) CTFbuffer_comp1.GetDevicePtr()));

        // Compute sum([F_m * Q * CTF], defSlices)
        if (config.CtfMode == Configuration::Config::CTFM_YES) {
            postFilterSum.SetComputeSize(fftDim.x, fftDim.y, batchSize);
            postFilterSum(CTFbuffer_comp1,
                          d_prefilter_fft,
                          fft_d,
                          fftDim,
                          ctfHandler.GetGlobalConstants(),
                          ctfHandler.GetImageConstants(stackIdx),
                          ctfHandler.GetDefocusOffsets(stackIdx),
                          minmax);
        } else {
            postFilter(CTFbuffer_comp1,
                       d_prefilter_fft,
                       fft_d,
                       fftDim,
                       (float) projSize);
        }
    }

    // IFFT of the summed transform
    cufftSafeCall(cufftExecC2R(handleC2R,
                               (cufftComplex*) fft_d.GetDevicePtr(),
                               (cufftReal*) proj_dv_d.GetDevicePtr()));

    copyToPitched(proj_dv_d, proj_d, projDim);
}

void Reconstructor::ConjoinChildren(Volume<float>* parentVol,
                                    std::vector<Volume<float>*>& childVols,
                                    DeviceVolume& parentDevVol,
                                    DeviceVolumeBuf& maskDevVol)
{
    // Volume dimensions
    uint3 childDim;
    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    uint3 parentDim;
    parentDim.x = (uint) parentVol->GetSubVolumeDimension(0).x;
    parentDim.y = (uint) parentVol->GetSubVolumeDimension(0).y;
    parentDim.z = (uint) parentVol->GetSubVolumeDimension(0).z;

    // Mask parent
    //parentDevVol.CardToDual();
    maskDevVol.CardToDual();

    Matrix<double> parentMatrix = parentVol->VolumeMatrix();
    Matrix<double> parentMatrixInv = parentVol->VolumeMatrixInv();

    for (auto child : childVols) {
        // Volume Matrix child
        Matrix<double> childMatrix = child->VolumeMatrix();
        Matrix<double> childMatrixInv = child->VolumeMatrixInv();

        // Parent -> Child
        Matrix<double> transform = childMatrixInv * parentMatrix;
        float4x4 transform4x4 = MatrixTo4x4(transform);

        // Child -> Parent
        Matrix<double> transformInv = parentMatrixInv * childMatrix;

        // Center in Parent
        Matrix<double> pos(4, 1);
        pos(0, 0) = (double)(childDim.x/2);
        pos(1, 0) = (double)(childDim.y/2);
        pos(2, 0) = (double)(childDim.z/2);
        pos(3, 0) = 1;
        pos = transformInv * pos;

        // Offset in Parent
        uint3 offset;
        offset.x = max(0, floor(pos(0, 0) - sqrt(3) * childDim.x * 0.5));
        offset.y = max(0, floor(pos(1, 0) - sqrt(3) * childDim.x * 0.5));
        offset.z = max(0, floor(pos(2, 0) - sqrt(3) * childDim.x * 0.5));

        // Compute size
        uint3 size;
        size.x = ceil(sqrt(3) * childDim.x);
        size.y = ceil(sqrt(3) * childDim.x);
        size.z = ceil(sqrt(3) * childDim.x);

        add3DTransform.SetComputeSize(size);
        add3DTransform(parentDevVol.surface_dual(),
                       maskDevVol.texture_dual(),
                        transform4x4,
                        parentDim,
                        offset);
    }

    norm3Doverlap.SetComputeSize(parentDim);
    norm3Doverlap(parentDevVol.surface_dual(),
                  parentDim);

}

void Reconstructor::ForwardProjectionParentMasked(Volume<float>* parentVol,
                                                  std::vector<Volume<float>*>& childVols,
                                                  DeviceVolume& parentDevVol,
                                                  DeviceVolumeBuf& maskInvDevVol,
                                                  int stackIdx, bool volumeIsEmpty, int iter, bool noSync,
                                                  stringstream& output)
{
    float runtime;

    // Volume dimensions
    uint3 childDim;
    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    uint3 parentDim;
    parentDim.x = (uint) parentVol->GetSubVolumeDimension(0).x;
    parentDim.y = (uint) parentVol->GetSubVolumeDimension(0).y;
    parentDim.z = (uint) parentVol->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Compute lookup table
    computePostFilter(parentVol, stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    // Mask parent
    parentDevVol.CardToDual();
    maskInvDevVol.CardToDual();

    Matrix<double> parentMatrix = parentVol->VolumeMatrix();
    Matrix<double> parentMatrixInv = parentVol->VolumeMatrixInv();

    for (auto child : childVols) {
        // Volume Matrix child
        Matrix<double> childMatrix = child->VolumeMatrix();
        Matrix<double> childMatrixInv = child->VolumeMatrixInv();

        // Parent -> Child
        Matrix<double> transform = childMatrixInv * parentMatrix;
        float4x4 transform4x4 = MatrixTo4x4(transform);

        // Child -> Parent
        Matrix<double> transformInv = parentMatrixInv * childMatrix;

        // Center in Parent
        Matrix<double> pos(4, 1);
        pos(0, 0) = (double)(childDim.x/2);
        pos(1, 0) = (double)(childDim.y/2);
        pos(2, 0) = (double)(childDim.z/2);
        pos(3, 0) = 1;
        pos = transformInv * pos;

        // Offset in Parent
        uint3 offset;
        offset.x = max(0, floor(pos(0, 0) - sqrt(3) * childDim.x * 0.5));
        offset.y = max(0, floor(pos(1, 0) - sqrt(3) * childDim.x * 0.5));
        offset.z = max(0, floor(pos(2, 0) - sqrt(3) * childDim.x * 0.5));

        // Compute size
        uint3 size;
        size.x = ceil(sqrt(3) * childDim.x);
        size.y = ceil(sqrt(3) * childDim.x);
        size.z = ceil(sqrt(3) * childDim.x);

        mask3DTransform.SetComputeSize(size);
        mask3DTransform(parentDevVol.surface_dual(),
                        maskInvDevVol.texture_dual(),
                        transform4x4,
                        parentDim,
                        offset);
    }

    // Prefilter parent
    prefilter3DX(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int)parentDim.x, (int)parentDim.y, (int)parentDim.z);
    prefilter3DY(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int)parentDim.x, (int)parentDim.y, (int)parentDim.z);
    prefilter3DZ(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int)parentDim.x, (int)parentDim.y, (int)parentDim.z);

    // System Matrix
    float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, parentVol->VolumeMatrix());

    // Batched CTF correction
    fft_d.Memset(0);
    int batchCount = ctfHandler.GetBatchCount(stackIdx);
    for (int ctfBatch = 0; ctfBatch < batchCount; ctfBatch++) {

        // Min/Max slice and effective batch size
        int2 minmax = ctfHandler.GetMinMaxSlice(stackIdx, ctfBatch);
        int batchSize = ctfHandler.GetBatchSize(stackIdx, ctfBatch);

        // Progress
        cout << "\r\e[K" << flush;
        cout << output.str() << "FP " << stackIdx << " | CTF batch " <<  ctfBatch + 1 << "/" << batchCount << flush;

        // Reset projection and FFT buffer
        CTFbuffer_realRect.Memset(0);
        CTFbuffer_comp1.Memset(0);

        // FP parent
        runtime = fpOrthoKernel(projDim,
                                parentDim,
                                ctfHandler.GetImageConstants(stackIdx),
                                systemMatrix,
                                CTFbuffer_realRect,
                                parentDevVol.surface_dual(),
                                minmax);

        // Mask forward projection: f_m = f .* mask
        cropSlicesKernel.SetComputeSize(projDim.x, projDim.y, batchSize);
        cropSlicesKernel(CTFbuffer_realRect,
                         projDim,
                         batchSize,
                         config.CutLength,
                         config.DimLength,
                         corners,
                         normVals);

        // Batched FFT of forward projection: f_m -> F_m
        cufftSafeCall(cufftExecR2C(FFThandleR2Call,
                                   (cufftReal *) CTFbuffer_realRect.GetDevicePtr(),
                                   (cufftComplex *) CTFbuffer_comp1.GetDevicePtr()));

        // Compute sum([F_m * Q * CTF], defSlices)
        if (config.CtfMode == Configuration::Config::CTFM_YES) {
            postFilterSum.SetComputeSize(fftDim.x, fftDim.y, batchSize);
            postFilterSum(CTFbuffer_comp1,
                          d_prefilter_fft,
                          fft_d,
                          fftDim,
                          ctfHandler.GetGlobalConstants(),
                          ctfHandler.GetImageConstants(stackIdx),
                          ctfHandler.GetDefocusOffsets(stackIdx),
                          minmax);
        } else {
            postFilter(CTFbuffer_comp1,
                       d_prefilter_fft,
                       fft_d,
                       fftDim,
                       (float) projSize);
        }
    }

//    // Radial Average of signal power for SNR, no normalization for FFT as this
//    // already happened in postFilterSum
//    multiplicity_d.Memset(0);
//    signal_power_d.Memset(0);
//    radialSum(fft_d,
//              signal_power_d,
//              multiplicity_d,
//              fftDim,
//              projDimF,
//              asymCorrFac,
//              projDimF.x * projDimF.y);
//    nppSafeCall(nppsDiv_32f_I((Npp32f*)multiplicity_d.GetDevicePtr(),
//                              (Npp32f*)signal_power_d.GetDevicePtr(),
//                              snrShells));

    // IFFT of the summed transform
    cufftSafeCall(cufftExecC2R(handleC2R,
                               (cufftComplex*) fft_d.GetDevicePtr(),
                               (cufftReal*) proj_dv_d.GetDevicePtr()));
    copyToPitched(proj_dv_d, proj_d, projDim);
}

//void Reconstructor::BackProjectionFamily(Volume<float>* parentVol,
//                                         std::vector<Volume<float>*>& childVols,
//                                         DeviceVolume& childDevVol,
//                                         DeviceVolume& maskDevVol,
//                                         DeviceVolume& parentDevVol,
//                                         int stackIdx, float SIRTCount, int iter, bool useSNR)
//{
//    printf("\n BACK PROJECTION FAMILY %i iter %i ==============================================\n", stackIdx, iter);
//    float runtime;
//
//    // Number of ctf slices
//    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;
//
//    // Box spline prefilter
//    computePreFilter(stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);
//
//    // Crop Dims for parent volume
//    vector<float2> corners;
//    vector<float> normVals;
//    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);
//
//    if (config.WBP_NoSART)
//    {
//        magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
//        magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)stackIdx) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
//    }
//
//    // Get from pitched
//    copyFromPitched(proj_d, proj_dv_d, projDim);
//
//    // Mask image f --> f_m
//    cropKernel(proj_dv_d,
//               projDim,
//               config.CutLength,
//               config.DimLength,
//               corners,
//               normVals);
//
//    // FFT f --> F
//    cufftSafeCall(cufftExecR2C(handleR2C,
//                               (cufftReal*)proj_dv_d.GetDevicePtr(),
//                               (cufftComplex*)fft_d.GetDevicePtr()));
//
//    // Compute [F * Q] - [F_m * Q]
//    if (config.CtfMode == Configuration::Config::CTFM_YES){
//        if (useSNR) {
//            preFilterSpreadSNR.SetComputeSize(fftDim.x, fftDim.y, sliceNumber);
//            preFilterSpreadSNR(fft_d,
//                               d_prefilter_fft,
//                               CTFbuffer_comp1,
//                               texSNR,
//                               fftDim,
//                               ctfHandler.GetGlobalConstants(),
//                               ctfHandler.GetImageConstants(stackIdx),
//                               ctfHandler.GetDefocusOffsets(stackIdx));
//        } else {
//            preFilterSpreadAdHoc.SetComputeSize(fftDim.x, fftDim.y, sliceNumber);
//            preFilterSpreadAdHoc(fft_d,
//                                 d_prefilter_fft,
//                                 CTFbuffer_comp1,
//                                 fftDim,
//                                 ctfHandler.GetGlobalConstants(),
//                                 ctfHandler.GetImageConstants(stackIdx),
//                                 ctfHandler.GetDefocusOffsets(stackIdx));
//        }
//    } else {
//        postFilter(fft_d,
//                   d_prefilter_fft,
//                   CTFbuffer_comp1,
//                   (proj.GetWidth() / 2 + 1), proj.GetHeight(),
//                   (float) proj.GetWidth() * (float) proj.GetHeight());
//    }
//
//    // Batched IFFT --> F' -> f'
//    cufftSafeCall(cufftExecC2R(FFThandleC2Rall,
//                               (cufftComplex*) CTFbuffer_comp1.GetDevicePtr(),
//                               (cufftReal*)CTFbuffer_realRect.GetDevicePtr()));
//
//    // Copy to CUDA arrays
//    slicesToArrays(CTFbuffer_realRect, projDim);
//
//    // We have now prepared everything for BP of the entire family.
//    // Backproject children.
//    for (auto child : childVols) {
//
//        // Volume dimensions
//        uint3 childDim;
//        childDim.x = (uint) child->GetSubVolumeDimension(0).x;
//        childDim.y = (uint) child->GetSubVolumeDimension(0).y;
//        childDim.z = (uint) child->GetSubVolumeDimension(0).z;
//
//        // System Matrix
//        float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, child->VolumeMatrix());
//
//        // Actual BP
//        runtime = bpOrthoKernel(projDim,
//                                childDim,
//                                config.Lambda / SIRTCount,
//                                ctfHandler.GetImageConstants(stackIdx),
//                                systemMatrix,
//                                childDevVol.surface_dual());
//
//        postfilter3DX(childDevVol.surface_dual(), childDevVol.surface_dual(), (int) childDim.x, (int) childDim.y, (int) childDim.z);
//        postfilter3DY(childDevVol.surface_dual(), childDevVol.surface_dual(), (int) childDim.x, (int) childDim.y, (int) childDim.z);
//        postfilter3DZ(childDevVol.surface_dual(), childDevVol.surface_dual(), (int) childDim.x, (int) childDim.y, (int) childDim.z);
//
//        // Scale by number of particles so range is ok
//        add3D.SetComputeSize(childDim.x, childDim.y, childDim.z);
//        add3D(childDevVol.surface_dual(),
//              childDevVol.surface_card(),
//              child, 1.f/(float)childVols.size());
//    }
//
//    // Backproject parent.
//    // Volume dimensions
//    uint3 parentDim;
//    parentDim.x = (uint) parentVol->GetSubVolumeDimension(0).x;
//    parentDim.y = (uint) parentVol->GetSubVolumeDimension(0).y;
//    parentDim.z = (uint) parentVol->GetSubVolumeDimension(0).z;
//
//    // System Matrix
//    float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, parentVol->VolumeMatrix());
//
//    // Actual BP
//    runtime = bpOrthoKernel(projDim,
//                            parentDim,
//                            config.Lambda / SIRTCount,
//                            ctfHandler.GetImageConstants(stackIdx),
//                            systemMatrix,
//                            parentDevVol.surface_dual());
//
//    // Add to recon
//    postfilter3DX(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int) parentDim.x, (int) parentDim.y, (int) parentDim.z);
//    postfilter3DY(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int) parentDim.x, (int) parentDim.y, (int) parentDim.z);
//    postfilter3DZ(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int) parentDim.x, (int) parentDim.y, (int) parentDim.z);
//
//    add3D.SetComputeSize(parentDim.x, parentDim.y, parentDim.z);
//    add3D(parentDevVol.surface_dual(), parentDevVol.surface_card(), parentVol);
//}


//void Reconstructor::ForwardProjectionFamily(Volume<float>* parentVol,
//                                            std::vector<Volume<float>*>& childVols,
//                                            DeviceVolume& childDevVol,
//                                            DeviceVolume& maskDevVol,
//                                            DeviceVolume& maskInvDevVol,
//                                            DeviceVolume& parentDevVol,
//                                            int stackIdx, bool volumeIsEmpty, int iter, bool noSync)
//{
//    printf("\n FORWARD PROJECTION FAMILY %i iter %i ==============================================\n", stackIdx, iter);
//
//    float runtime;
//
//    // Volume dimensions
//    uint3 childDim;
//    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
//    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
//    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;
//
//    uint3 parentDim;
//    parentDim.x = (uint) parentVol->GetSubVolumeDimension(0).x;
//    parentDim.y = (uint) parentVol->GetSubVolumeDimension(0).y;
//    parentDim.z = (uint) parentVol->GetSubVolumeDimension(0).z;
//
//    // Number of ctf slices
//    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;
//
//    // Compute lookup table
//    computePostFilter(stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);
//
//    // Crop Dims
//    vector<float2> corners;
//    vector<float> normVals;
//    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);
//
//    // Mask and prefilter particle volume
//    mask3D.SetComputeSize(childDim);
//    mask3D(childDevVol.surface_card(),
//           childDevVol.surface_dual(),
//           maskDevVol.surface_card(),
//           childDim);
//
//    // Prefilter child
//    prefilter3DX(childDevVol.surface_dual(), childDevVol.surface_dual(), (int)childDim.x, (int)childDim.y, (int)childDim.z);
//    prefilter3DY(childDevVol.surface_dual(), childDevVol.surface_dual(), (int)childDim.x, (int)childDim.y, (int)childDim.z);
//    prefilter3DZ(childDevVol.surface_dual(), childDevVol.surface_dual(), (int)childDim.x, (int)childDim.y, (int)childDim.z);
//
//    // Project all children
//    for (auto child : childVols) {
//        // System Matrix
//        float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, child->VolumeMatrix());
//
//        runtime = fpOrthoKernel(projDim,
//                                childDim,
//                                ctfHandler.GetImageConstants(stackIdx),
//                                systemMatrix,
//                                CTFbuffer_realRect,
//                                childDevVol.surface_dual());
//
//    }
//
//    // Mask parent
//    parentDevVol.CardToDual();
//    maskInvDevVol.CardToDual();
//
////    {
////        auto img = new float[parentVol->GetSubVolumeSizeInVoxels(0)];
////        parentDevVol.array_dual().CopyFromArrayToHost(img);
////        for (int i=0; i<parentVol->GetSubVolumeSizeInVoxels(0); i++){
////            img[i] = img[i] * -1.f;
////        }
////        stringstream DP;
////        DP << "before_masking_"  << iter << ".em";
////        emwrite(DP.str(), img, parentVol->GetDimension().x,
////                parentVol->GetDimension().y, parentVol->GetDimension().z);
////        delete[] img;
////    }
//
//    Matrix<double> parentMatrix = parentVol->VolumeMatrix();
//    Matrix<double> parentMatrixInv = parentVol->VolumeMatrixInv();
//
//    for (auto child : childVols) {
//        // Volume Matrix child
//        Matrix<double> childMatrix = child->VolumeMatrix();
//        Matrix<double> childMatrixInv = child->VolumeMatrixInv();
//
//        // Parent -> Child
//        Matrix<double> transform = childMatrixInv * parentMatrix;
//        float4x4 transform4x4 = MatrixTo4x4(transform);
//
//        // Child -> Parent
//        Matrix<double> transformInv = parentMatrixInv * childMatrix;
//
//        // Center in Parent
//        Matrix<double> pos(4, 1);
//        pos(0, 0) = (double)(childDim.x/2);
//        pos(1, 0) = (double)(childDim.y/2);
//        pos(2, 0) = (double)(childDim.z/2);
//        pos(3, 0) = 1;
//        pos = transformInv * pos;
//
//        // Offset in Parent
//        uint3 offset;
//        offset.x = max(0, floor(pos(0, 0) - sqrt(3) * childDim.x * 0.5));
//        offset.y = max(0, floor(pos(1, 0) - sqrt(3) * childDim.x * 0.5));
//        offset.z = max(0, floor(pos(2, 0) - sqrt(3) * childDim.x * 0.5));
//
//        // Compute size
//        uint3 size;
//        size.x = ceil(sqrt(3) * childDim.x);
//        size.y = ceil(sqrt(3) * childDim.x);
//        size.z = ceil(sqrt(3) * childDim.x);
//
//        //printf("compute ok \n");
//
//        mask3DTransform.SetComputeSize(size);
//        mask3DTransform(parentDevVol.surface_dual(),
//                        maskInvDevVol.texture_dual(),
//                        transform4x4,
//                        parentDim,
//                        offset);
//    }
//
////    {
////        auto img = new float[parentVol->GetSubVolumeSizeInVoxels(0)];
////        parentDevVol.array_dual().CopyFromArrayToHost(img);
////        for (int i=0; i<parentVol->GetSubVolumeSizeInVoxels(0); i++){
////            img[i] = img[i] * -1.f;
////        }
////        stringstream DP;
////        DP << "after_masking_"  << iter << ".em";
////        emwrite(DP.str(), img, parentVol->GetDimension().x,
////                parentVol->GetDimension().y, parentVol->GetDimension().z);
////        delete[] img;
////    }
//
//    // Prefilter parent (could be concurrent)
//    prefilter3DX(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int)parentDim.x, (int)parentDim.y, (int)parentDim.z);
//    prefilter3DY(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int)parentDim.x, (int)parentDim.y, (int)parentDim.z);
//    prefilter3DZ(parentDevVol.surface_dual(), parentDevVol.surface_dual(), (int)parentDim.x, (int)parentDim.y, (int)parentDim.z);
//
//    // System Matrix
//    float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, parentVol->VolumeMatrix());
//
//    // FP parent
//    runtime = fpOrthoKernel(projDim,
//                            parentDim,
//                            ctfHandler.GetImageConstants(stackIdx),
//                            systemMatrix,
//                            CTFbuffer_realRect,
//                            parentDevVol.surface_dual());
//
//    // Mask forward projection: f_m = f .* mask
//    cropSlicesKernel.SetComputeSize(make_dim3(projDim.x, projDim.y, sliceNumber));
//    cropSlicesKernel(CTFbuffer_realRect,
//                     projDim,
//                     sliceNumber,
//                     config.CutLength,
//                     config.DimLength,
//                     corners,
//                     normVals);
//
//    // Batched FFT of forward projection: f_m -> F_m
//    cufftSafeCall(cufftExecR2C(FFThandleR2Call,
//                               (cufftReal*) CTFbuffer_realRect.GetDevicePtr(),
//                               (cufftComplex*) CTFbuffer_comp1.GetDevicePtr()));
//
//    // Compute sum([F_m * Q * CTF], defSlices)
//    if (config.CtfMode == Configuration::Config::CTFM_YES){
//        postFilterSum.SetComputeSize(fftDim.x, fftDim.y, sliceNumber);
//        postFilterSum(CTFbuffer_comp1,
//                      d_prefilter_fft,
//                      fft_d,
//                      fftDim,
//                      ctfHandler.GetGlobalConstants(),
//                      ctfHandler.GetImageConstants(stackIdx),
//                      ctfHandler.GetDefocusOffsets(stackIdx));
//    } else {
//        postFilter(CTFbuffer_comp1,
//                   d_prefilter_fft,
//                   fft_d,
//                   (proj.GetWidth() / 2 + 1), proj.GetHeight(),
//                   (float) proj.GetWidth() * (float) proj.GetHeight());
//    }
//
//    // Radial Average of signal power for SNR, no normalization for FFT as this
//    // already happened in postFilterSum
//    multiplicity_d.Memset(0);
//    signal_power_d.Memset(0);
//    radialSum(fft_d,
//              signal_power_d,
//              multiplicity_d,
//              fftDim,
//              projDimF,
//              asymCorrFac,
//              1.f);
//    nppSafeCall(nppsDiv_32f_I((Npp32f*)multiplicity_d.GetDevicePtr(),
//                              (Npp32f*)signal_power_d.GetDevicePtr(),
//                              snrShells));
//
//    // IFFT of the summed transform
//    cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*) fft_d.GetDevicePtr(), (cufftReal*) proj_dv_d.GetDevicePtr()));
//    copyToPitched(proj_dv_d, proj_d, projDim);
//}


void Reconstructor::DistanceChildren(Volume<float>* parentVol,
                                     std::vector<Volume<float>*>& childVols,
                                     DeviceVolume& maskDevVol,
                                     int stackIdx, bool volumeIsEmpty, int iter, bool noSync)
{
    float runtime;

    // Volume dimensions
    uint3 childDim;
    childDim.x = (uint) childVols[0]->GetSubVolumeDimension(0).x;
    childDim.y = (uint) childVols[0]->GetSubVolumeDimension(0).y;
    childDim.z = (uint) childVols[0]->GetSubVolumeDimension(0).z;

    uint3 parentDim;
    parentDim.x = (uint) parentVol->GetSubVolumeDimension(0).x;
    parentDim.y = (uint) parentVol->GetSubVolumeDimension(0).y;
    parentDim.z = (uint) parentVol->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Compute lookup table
    computePostFilter(childVols[0], stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);


    // Thickness map
    proj_dv_d.Memset(0);
    dist_children_d.Memset(0);
    prefilter3DX(maskDevVol.surface_card(), maskDevVol.surface_dual(), (int)childDim.x, (int)childDim.y, (int)childDim.z);
    prefilter3DY(maskDevVol.surface_dual(), maskDevVol.surface_dual(), (int)childDim.x, (int)childDim.y, (int)childDim.z);
    prefilter3DZ(maskDevVol.surface_dual(), maskDevVol.surface_dual(), (int)childDim.x, (int)childDim.y, (int)childDim.z);

    // Project all Masks
    for (auto child : childVols) {
        // System Matrix
        float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, child->VolumeMatrix());
        runtime = fpOrthoSSKernel(projDim,
                                childDim,
                                ctfHandler.GetImageConstants(stackIdx),
                                systemMatrix,
                                proj_dv_d,
                                maskDevVol.surface_dual());
    }

    // f_m = f .* mask
    cropKernel(proj_dv_d,
               projDim,
               config.CutLength,
               config.DimLength,
               corners,
               normVals);

    // f_m -> F_m
    cufftSafeCall(cufftExecR2C(handleR2C,
                               (cufftReal*)proj_dv_d.GetDevicePtr(),
                               (cufftComplex*)fft_d.GetDevicePtr()));

    // F' = [F_m * Q]
    float vs = childVols[0]->GetVoxelSize().x;
    postFilter(fft_d,
               d_prefilter_fft,
               fft_d,
               fftDim,
               (float) projSize * (1 / (vs * vs)));

    // IFFT --> F' -> f
    cufftSafeCall(cufftExecC2R(handleC2R,
                               (cufftComplex*) fft_d.GetDevicePtr(),
                               (cufftReal*)proj_dv_d.GetDevicePtr()));
    copyToPitched(proj_dv_d, dist_children_d, projDim);


    {
        auto img = new float[proj.GetWidth() * proj.GetHeight()];
        dist_children_d.CopyDeviceToHost(img);
        stringstream DP;
        DP << "distance_raw_" << stackIdx << "_" << iter << ".em";
        emwrite(DP.str(), img, proj.GetWidth(),
                proj.GetHeight());
        delete[] img;
    }

    CudaPitchedDeviceVariable mask1(proj.GetWidth() * sizeof(char), proj.GetHeight(), sizeof(char));

    float thresh = 10.f;

    nppSafeCall(nppiThreshold_LTValGTVal_32f_C1IR((Npp32f*)dist_children_d.GetDevicePtr(),
                                          (int)dist_children_d.GetPitch(),
                                          roiAll,
                                          0,
                                          0,
                                                  thresh,
                                                  thresh));

    {
        auto img = new float[proj.GetWidth() * proj.GetHeight()];
        dist_children_d.CopyDeviceToHost(img);
        stringstream DP;
        DP << "distance_thresh_" << stackIdx << "_" << iter << ".em";
        emwrite(DP.str(), img, proj.GetWidth(),
                proj.GetHeight());
        delete[] img;
    }

    nppSafeCall(nppiThreshold_LTVal_32f_C1IR((Npp32f*)dist_children_d.GetDevicePtr(),
                                                  (int)dist_children_d.GetPitch(),
                                                  roiAll,
                                             thresh,
                                                  0));

    nppSafeCall(nppiDivC_32f_C1IR(thresh,
                                  (Npp32f*)dist_children_d.GetDevicePtr(),
                                  (int)dist_children_d.GetPitch(),
                                  roiAll));

    nppSafeCall(nppiConvert_32f8u_C1R((Npp32f*)dist_children_d.GetDevicePtr(),
                                      (int)dist_children_d.GetPitch(),
                                      (Npp8u*)mask1.GetDevicePtr(),
                                      (int)mask1.GetPitch(),
                                      roiAll,
                                      NPP_RND_ZERO));

    {
        auto img = new unsigned char[proj.GetWidth() * proj.GetHeight()];
        auto ret = new float[proj.GetWidth() * proj.GetHeight()];

        mask1.CopyDeviceToHost(img);

        for (int i = 0; i < proj.GetPixelCount(); i++){
            ret[i] = img[i];
        }

        stringstream DP;
        DP << "distance_mask_" << stackIdx << "_" << iter << ".em";
        emwrite(DP.str(), ret, proj.GetWidth(),
                proj.GetHeight());
        delete[] img;
        delete[] ret;
    }

    size_t dist_buffer_size = 0;
    nppSafeCall(nppiDistanceTransformPBAGetBufferSize(roiAll, &dist_buffer_size));

    CudaDeviceVariable dist_buffer(dist_buffer_size);

    NppStreamContext ctx;
    nppSafeCall(nppGetStreamContext(&ctx));
    nppSafeCall(nppiDistanceTransformAbsPBA_8u32f_C1R_Ctx((Npp8u*)mask1.GetDevicePtr(),
                                                           (int)mask1.GetPitch(),
                                                           1,
                                                           10,
                                                    NULL,
                                                          0,
                                                           NULL,
                                                           0,
                                                           NULL,
                                                           0,
                                                           (Npp32f*)dist_children_d.GetDevicePtr(),
                                                           (int)dist_children_d.GetPitch(),
                                                           roiAll,
                                                           (Npp8u*) dist_buffer.GetDevicePtr(),
                                                           ctx));

    nppSafeCall(nppiThreshold_GTVal_32f_C1IR((Npp32f*)dist_children_d.GetDevicePtr(),
                                             (int) dist_children_d.GetPitch(),
                                             roiAll,
                                             20,
                                             20));

    nppSafeCall(nppiDivC_32f_C1IR(20,
                                  (Npp32f*)dist_children_d.GetDevicePtr(),
                                  (int)dist_children_d.GetPitch(),
                                  roiAll));

    copyFromPitched(dist_children_d, proj_dv_d, projDim);

    nppSafeCall(nppsSubCRev_32f_I(1, (Npp32f*) proj_dv_d.GetDevicePtr(), projSize));

    copyToPitched(proj_dv_d, dist_children_d, projDim);
}

void Reconstructor::DistanceParent(Volume<float>* parentVol,
                                   int stackIdx, bool volumeIsEmpty, int iter, bool noSync)
{
    float runtime;

    // Volume dimensions
    uint3 parentDim;
    parentDim.x = (uint) parentVol->GetSubVolumeDimension(0).x;
    parentDim.y = (uint) parentVol->GetSubVolumeDimension(0).y;
    parentDim.z = (uint) parentVol->GetSubVolumeDimension(0).z;

    // Number of ctf slices
    int sliceNumber = ctfHandler.GetImageConstants(stackIdx).sliceNumber;

    // Compute lookup table
    computePostFilter(parentVol, stackIdx, proj.GetWidth(), proj.GetHeight(), 3.f);

    // Crop Dims
    vector<float2> corners;
    vector<float> normVals;
    proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);

    // Thickness map
    proj_dv_d.Memset(0);

    // System Matrix
    float4x4 systemMatrix = proj.SystemMatrix4x4(stackIdx, parentVol->VolumeMatrix());

    // Distance with parent
    runtime = distOrthoKernel(projDim,
                              parentDim,
                              ctfHandler.GetImageConstants(stackIdx),
                              systemMatrix,
                              proj_dv_d);

    // f_m = f .* mask
    cropKernel(proj_dv_d,
               projDim,
               config.CutLength,
               config.DimLength,
               corners,
               normVals);

#ifdef WRITEDEBUG
    {
        auto img = new float[proj.GetWidth()*proj.GetHeight()];
        proj_dv_d.CopyDeviceToHost(img, proj.GetWidth()*proj.GetHeight()*sizeof(float));
        stringstream DP;
        DP << "after_distcrop_" << stackIdx << "_" << iter << ".em";
        emwrite(DP.str(), img, proj.GetWidth(),
                proj.GetHeight());
        delete[] img;
    }
#endif

    // f_m -> F_m
    cufftSafeCall(cufftExecR2C(handleR2C,
                               (cufftReal*)proj_dv_d.GetDevicePtr(),
                               (cufftComplex*)fft_d.GetDevicePtr()));

    // F' = [F_m * Q]
    postFilter(fft_d,
               d_prefilter_fft,
               fft_d,
               fftDim,
               (float) projSize);

    // IFFT --> F' -> f
    cufftSafeCall(cufftExecC2R(handleC2R,
                               (cufftComplex*) fft_d.GetDevicePtr(),
                               (cufftReal*) proj_dv_d.GetDevicePtr()));
    copyToPitched(proj_dv_d, dist_d, projDim);

    // Save distance for later use (normed by voxel size sqr)
    nppSafeCall(nppsMax_32f((Npp32f*)dist_d.GetDevicePtr(),
                            proj.GetWidth()*proj.GetHeight(),
                            (Npp32f*) meanval.GetDevicePtr(),
                            (Npp8u*) meanbuffer.GetDevicePtr()));

    float volumeTraversalLength = 0.f;
    meanval.CopyDeviceToHost(&volumeTraversalLength, sizeof(float));

    proj_distance[stackIdx] = volumeTraversalLength * parentVol->GetVoxelSize().x * parentVol->GetVoxelSize().x;
}

void Reconstructor::PrepareProjection(void *img_h, int proj_index, float &meanValue, float &StdValue, int &BadPixels)
{
    // Not node 0, skip
	if (mpi_part != 0)
	{
		return;
	}

    // If projection not used, skip
    if (!proj.IsGood(proj_index)){
        return;
    }

    // Copy projection to device
	if (projSource->GetDataType() == DT_SHORT)
	{
		cudaSafeCall(cuMemcpyHtoD(realprojUS_d.GetDevicePtr(),
                                  img_h,
                                  proj.GetWidth() * proj.GetHeight() * sizeof(short)));
		nppSafeCall(nppiConvert_16s32f_C1R((Npp16s*)realprojUS_d.GetDevicePtr(),
                                           proj.GetWidth() * sizeof(short),
                                           (Npp32f*)realproj_d.GetDevicePtr(),
                                           (int)realproj_d.GetPitch(),
                                           roiAll));
	}
	else if (projSource->GetDataType() == DT_USHORT)
	{
		cudaSafeCall(cuMemcpyHtoD(realprojUS_d.GetDevicePtr(),
                                  img_h,
                                  proj.GetWidth() * proj.GetHeight() * sizeof(short)));
		nppSafeCall(nppiConvert_16u32f_C1R((Npp16u*)realprojUS_d.GetDevicePtr(),
                                           proj.GetWidth() * sizeof(short),
                                           (Npp32f*)realproj_d.GetDevicePtr(),
                                           (int)realproj_d.GetPitch(),
                                           roiAll));
	}
	else if (projSource->GetDataType() == DT_INT)
	{
		realprojUS_d.CopyHostToDevice(img_h);
		nppSafeCall(nppiConvert_32s32f_C1R((Npp32s*)realprojUS_d.GetDevicePtr(),
                                           (int)realprojUS_d.GetPitch(),
                                           (Npp32f*)realproj_d.GetDevicePtr(),
                                           (int)realproj_d.GetPitch(),
                                           roiAll));
	}
	else if (projSource->GetDataType() == DT_UINT)
	{
		realprojUS_d.CopyHostToDevice(img_h);
		nppSafeCall(nppiConvert_32u32f_C1R((Npp32u*)realprojUS_d.GetDevicePtr(),
                                           (int)realprojUS_d.GetPitch(),
                                           (Npp32f*)realproj_d.GetDevicePtr(),
                                           (int)realproj_d.GetPitch(),
                                           roiAll));
	}
	else if (projSource->GetDataType() == DT_FLOAT)
	{
		realproj_d.CopyHostToDevice(img_h);
	}
	else
	{
		return;
	}


    nppSafeCall(nppiMean_32f_C1R((Npp32f*)realproj_d.GetDevicePtr(),
                                 projPitchFloat,
                                 roiAll,
                                 (Npp8u*)meanbuffer.GetDevicePtr(),
                                 (Npp64f*)meanval.GetDevicePtr()));

	double mean = 0;
	meanval.CopyDeviceToHost(&mean);
	meanValue = (float)mean;

	if (config.CorrectBadPixels)
	{
        nppSafeCall(nppiCompareC_32f_C1R((Npp32f*)realproj_d.GetDevicePtr(),
                                         (int)realproj_d.GetPitch(),
                                         config.BadPixelValue * meanValue,
                                         (Npp8u*)badPixelMask_d.GetDevicePtr(),
                                         (int)badPixelMask_d.GetPitch(),
                                         roiAll,
                                         NPP_CMP_GREATER));
	}
	else
	{
        nppSafeCall(nppiSet_8u_C1R(0,
                                   (Npp8u*)badPixelMask_d.GetDevicePtr(),
                                   (int)badPixelMask_d.GetPitch(),
                                   roiAll));
	}

	nppSafeCall(nppiSum_8u_C1R((Npp8u*)badPixelMask_d.GetDevicePtr(),
                               (int)badPixelMask_d.GetPitch(),
                               roiAll,
                               (Npp8u*)meanbuffer.GetDevicePtr(),
                               (Npp64f*)meanval.GetDevicePtr()));

	meanval.CopyDeviceToHost(&mean);
	BadPixels = (int)(mean / 255.0);

    nppSafeCall(nppiSet_32f_C1MR(meanValue,
                                 (Npp32f*)realproj_d.GetDevicePtr(),
                                 (int)realproj_d.GetPitch(),
                                 roiAll,
                                 (Npp8u*)badPixelMask_d.GetDevicePtr(),
                                 (int)badPixelMask_d.GetPitch()));

	//When doing WBP we compute mean and std on the RAW image before Fourier filter and WBP weighting
//	if (config.WBP_NoSART)
//	{
//        // Mean and STD
//        nppSafeCall(nppiMean_StdDev_32f_C1R((Npp32f*)realproj_d.GetDevicePtr(),
//                                            (int)realproj_d.GetPitch(),
//                                            roiAll,
//                                            (Npp8u*)meanbuffer.GetDevicePtr(),
//                                            (Npp64f*)meanval.GetDevicePtr(),
//                                            (Npp64f*)stdval.GetDevicePtr()));
//
//		mean = 0;
//		meanval.CopyDeviceToHost(&mean);
//		double std_h = 0;
//		stdval.CopyDeviceToHost(&std_h);
//		StdValue = (float)std_h;
//		float std_hf = StdValue;
//
//		meanValue = (float)(mean);
//		float mean_hf = meanValue;
//
//        // Correct norm
//		if (config.ProjectionNormalization == Configuration::Config::PNM_MEAN)
//		{
//			std_hf = meanValue;
//		}
//		if (config.ProjectionNormalization == Configuration::Config::PNM_NONE)
//		{
//			std_hf = 1;
//			mean_hf = 0;
//		}
//		if (config.DownWeightTiltsForWBP)
//		{
//			//we divide here because we divide later using nppiDivC: at the end we multiply!
//			std_hf /= (float)cos(abs(markers(MarkerFileItem_enum::MFI_TiltAngle, proj_index, 0)) / 180.0 * M_PI);
//		}
//
//        // Subtract mean
//        nppSafeCall(nppiSubC_32f_C1IR(mean_hf,
//                                      (Npp32f*)realproj_d.GetDevicePtr(),
//                                      (int)realproj_d.GetPitch(),
//                                      roiAll));
//
//        // STD 1, also invert
//		nppSafeCall(nppiDivC_32f_C1IR(-std_hf,
//                                      (Npp32f*)realproj_d.GetDevicePtr(),
//                                      (int)realproj_d.GetPitch(),
//                                      roiAll));
//
//        copyFromPitched(realproj_d, proj_dv_d, projDim);
//	}



	if (!skipFilter)
	{
        copyFromPitched(realproj_d, proj_dv_d, projDim);
		cufftSafeCall(cufftExecR2C(handleR2C,
                                   (cufftReal*)proj_dv_d.GetDevicePtr(),
                                   (cufftComplex*)fft_d.GetDevicePtr()));

//		if (!skipFilter)
//		{
        auto lp = (float)config.fourFilterLP;
        auto hp = (float)config.fourFilterHP;
        auto lps = (float)config.fourFilterLPS;
        auto hps = (float)config.fourFilterHPS;
        int size = proj.GetMaxDimension();

        fourFilterKernel(fft_d,
                         fftPitchComplex,
                         projDim,
                         asymCorrFac,
                         lp, hp, lps, hps);
//		}

//		if (config.DoseWeighting)
//		{
//            // TODO:
//			doseWeightingKernel(fft_d,
//                                fftPitchComplex,
//                                proj.GetMaxDimension(),
//                                config.AccumulatedDose[proj_index],
//                                proj.GetPixelSize() * 10.0f);
//		}
//
//		if (config.WBP_NoSART)
//		{
//            if (config.WBPFilter == FM_EXACT){
//                ExactFilter(proj_index);
//            } else {
//                // Projection Matrix
//                Matrix<double> Mproj = proj.ProjectionMatrix<double>(proj_index);
//                auto volumeHeight = (float)config.RecDimensions.z * config.VoxelSize.z;
//                float thickness = (float)(proj.GetMaxDimension()) / (volumeHeight * 2.0f);
//
//                wbp(fft_d,
//                    fftPitchComplex,
//                    projDim,
//                    asymCorrFac,
//                    config.WBPFilter,
//                    proj.GetGoodProjCount() - 1,
//                    thickness,
//                    Mproj,
//                    det_mats_d);
//            }
//		}

		cufftSafeCall(cufftExecC2R(handleC2R,
                                   (cufftComplex*)fft_d.GetDevicePtr(),
                                   (cufftReal*)proj_dv_d.GetDevicePtr()));

        copyToPitched(proj_dv_d, realproj_d, projDim);
//		float normVal = (float)(projSize);

        nppSafeCall(nppiDivC_32f_C1IR((float)projSize,
                                      (Npp32f*)realproj_d.GetDevicePtr(),
                                      (int)realproj_d.GetPitch(),
                                      roiAll));
	}

	//Normalize from FFT
//	nppSafeCall(nppiDivC_32f_C1R((Npp32f*)realproj_d.GetDevicePtr(),
//                                 realproj_d.GetPitch(),
//                                 normVal,
//                                 (Npp32f*)realprojUS_d.GetDevicePtr(),
//                                 (int)realprojUS_d.GetPitch(),
//                                 roiAll));




	// When doing SART we compute mean and std on the filtered image
//	if (!config.WBP_NoSART)
//	{
    // Compute Mean/SD
    nppSafeCall(nppiMean_StdDev_32f_C1R((Npp32f*)realproj_d.GetDevicePtr(),
                                        (int)realproj_d.GetPitch(),
                                        roiAll,
                                        (Npp8u*)meanbuffer.GetDevicePtr(),
                                        (Npp64f*)meanval.GetDevicePtr(),
                                        (Npp64f*)stdval.GetDevicePtr()));

    mean = 0;
    meanval.CopyDeviceToHost(&mean);
    double std_h = 0;
    stdval.CopyDeviceToHost(&std_h);
    StdValue = (float)std_h;
    float std_hf = StdValue;

    meanValue = (float)(mean);
    float mean_hf = meanValue;

    if (config.ProjectionNormalization == Configuration::Config::PNM_MEAN)
    {
        std_hf = meanValue;
    }
    if (config.ProjectionNormalization == Configuration::Config::PNM_NONE)
    {
        std_hf = 1;
        mean_hf = 0;
    }

    // Sub mean
//    nppSafeCall(nppiSubC_32f_C1R((Npp32f*)realprojUS_d.GetDevicePtr(),
//                                 (int)realprojUS_d.GetPitch(),
//                                 mean_hf,
//                                 (Npp32f*)realproj_d.GetDevicePtr(),
//                                 (int)realproj_d.GetPitch(),
//                                 roiAll));

    nppSafeCall(nppiSubC_32f_C1IR(mean_hf,
                                 (Npp32f*)realproj_d.GetDevicePtr(),
                                 (int)realproj_d.GetPitch(),
                                 roiAll));

    // STD = 1, also invert
    nppSafeCall(nppiDivC_32f_C1IR(-std_hf,
                                  (Npp32f*)realproj_d.GetDevicePtr(),
                                  (int)realproj_d.GetPitch(),
                                  roiAll));

    // When doing WBP we filter after the regular normalization
    printf("\nMode is %i\n", config.WBP_NoSART);
    if (config.WBP_NoSART)
    {
        copyFromPitched(realproj_d, proj_dv_d, projDim);

        cufftSafeCall(cufftExecR2C(handleR2C,
                                   (cufftReal*)proj_dv_d.GetDevicePtr(),
                                   (cufftComplex*)fft_d.GetDevicePtr()));

        if (config.WBPFilter == FM_EXACT){
            ExactFilter(proj_index);
        } else {
            // Projection Matrix
            Matrix<double> Mproj = proj.ProjectionMatrix<double>(proj_index);
            auto volumeHeight = (float)config.RecDimensions.z * config.VoxelSize.z;
            float thickness = (float)(proj.GetMaxDimension()) / (volumeHeight * 2.0f);

            wbp(fft_d,
                fftPitchComplex,
                projDim,
                asymCorrFac,
                config.WBPFilter,
                proj.GetGoodProjCount() - 1,
                thickness,
                Mproj,
                det_mats_d);
        }

        cufftSafeCall(cufftExecC2R(handleC2R,
                                   (cufftComplex*)fft_d.GetDevicePtr(),
                                   (cufftReal*)proj_dv_d.GetDevicePtr()));

        copyToPitched(proj_dv_d, realproj_d, projDim);

        // Norm for FFT
        nppSafeCall(nppiDivC_32f_C1IR((float)projSize,
                                      (Npp32f*)realproj_d.GetDevicePtr(),
                                      (int)realproj_d.GetPitch(),
                                      roiAll));
    }


    realproj_d.CopyDeviceToHost(img_h);
//	}
//	else
//	{
//		realprojUS_d.CopyDeviceToHost(img_h);
//	}
}

void Reconstructor::ExactFilter(int aIndex)
{
    // Assumes projection is in fft_d

    // Copy detector matrices of all good projections except this one
    // to device
    auto* Mdet = new float3x3[proj.GetGoodProjCount()-1];
    int c = 0;

    for (int i = 0; i < projSource->GetProjectionCount(); i++)
    {
        if (i == aIndex) continue;

        if (proj.IsGood(i)){
            Mdet[c] = proj.DetectorMatrix3x3(i);
            c++;
        }
    }
    det_mats_d.CopyHostToDevice(Mdet, (proj.GetGoodProjCount()-1) * sizeof(float3x3));
    delete[] Mdet;


    // Projection Matrix
    Matrix<double> Mproj = proj.ProjectionMatrix<double>(aIndex);

    // Parameters for exact weighting
    auto volumeHeight = (float)config.RecDimensions.z * config.VoxelSize.z;
    float thickness = (float)(proj.GetMaxDimension()) / (volumeHeight * 2.0f);


    wbp(fft_d,
        fftPitchComplex,
        projDim,
        asymCorrFac,
        FM_EXACT,
        proj.GetGoodProjCount() - 1,
        thickness,
        Mproj,
        det_mats_d);
}

template<class TVol>
void Reconstructor::Compare(Volume<TVol>* vol, char* originalImage, uint stackIdx, bool computeSNR, int iter, bool normForLength, float length)
{
	if (mpi_part == 0)
	{
        float volumeTraversalLength = 1.f;

        // If we just subtract we do not need to divide by length for WBP.
        if (normForLength) {
            nppSafeCall(nppsMax_32f((Npp32f *) dist_d.GetDevicePtr(),
                                    proj.GetWidth() * proj.GetHeight(),
                                    (Npp32f *) meanval.GetDevicePtr(),
                                    (Npp8u *) meanbuffer.GetDevicePtr()));


            meanval.CopyDeviceToHost(&volumeTraversalLength, sizeof(float));
        } else {
            volumeTraversalLength = length;
            nppSafeCall(nppiSet_32f_C1R(volumeTraversalLength,
                                        (Npp32f*)dist_d.GetDevicePtr(),
                                        dist_d.GetPitch(),
                                        roiAll));
        }

        if (computeSNR) {
            // Radial Average of signal power for SNR, no normalization for FFT as this
            // already happened in postFilterSum

            // FFT fwd proj parent
            copyFromPitched(proj_children_d, proj_dv_d, projDim);
            cufftSafeCall(cufftExecR2C(handleR2C,
                                       (cufftReal*)proj_dv_d.GetDevicePtr(),
                                       (cufftComplex*)fft_d.GetDevicePtr()));

            multiplicity_d.Memset(0);
            signal_power_d.Memset(0);
            radialSum(fft_d,
                      signal_power_d,
                      multiplicity_d,
                      fftDim,
                      projDimF,
                      asymCorrFac,
                      1.f);//projDimF.x * projDimF.y
            nppSafeCall(nppsDiv_32f_I((Npp32f*)multiplicity_d.GetDevicePtr(),
                                      (Npp32f*)signal_power_d.GetDevicePtr(),
                                      snrShells));
            //signal_power_d.CopyDeviceToHost(stackp);
            signal_power_arr_d.CopyFromDeviceToArray(signal_power_d);

            // To stack for later reconstruction
            float* stackp = sp_stack + stackIdx * snrShells;
            signal_power_d.CopyDeviceToHost(stackp);
        }

        // Crop Dims
        vector<float2> corners;
        vector<float> normVals;
        proj.ComputeHitPointsNew(*vol, stackIdx, corners, normVals);

        realproj_d.CopyHostToDevice(originalImage);
		float runtime = compKernel(realproj_d,
                                   proj_d,
                                   dist_d,
                                   projDim,
                                   config.CutLength,
                                   config.DimLength,
                                   volumeTraversalLength,
                                   corners,
                                   normVals,
                                   vol->GetVoxelSize().x);

        if (computeSNR){
            // FFT diff proj parent (i.e. noise)
            copyFromPitched(proj_d, proj_dv_d, projDim);
            cufftSafeCall(cufftExecR2C(handleR2C,
                                       (cufftReal*)proj_dv_d.GetDevicePtr(),
                                       (cufftComplex*)fft_d.GetDevicePtr()));

            // Radial average noise power, normalize for FFT
            multiplicity_d.Memset(0);
            noise_power_d.Memset(0);
            radialSum(fft_d,
                      noise_power_d,
                      multiplicity_d,
                      fftDim,
                      projDimF,
                      asymCorrFac,
                      volumeTraversalLength); //1.f/(projDimF.x * projDimF.y)* ///config.DeconvStrength);
            nppSafeCall(nppsDiv_32f_I((Npp32f*)multiplicity_d.GetDevicePtr(),
                                      (Npp32f*)noise_power_d.GetDevicePtr(),
                                      snrShells));

            // To array for texture
            noise_power_arr_d.CopyFromDeviceToArray(noise_power_d);

            // To stack for later reconstruction
            float* stackp = np_stack + stackIdx * snrShells;
            noise_power_d.CopyDeviceToHost(stackp);

            if (config.WriteDebug) {
                {
                    auto img = new float[snrShells];
                    noise_power_arr_d.CopyFromArrayToHost(img);
                    stringstream DP;
                    DP << "noise_power_" << stackIdx << "_" << iter << ".em";
                    emwrite(DP.str(), img, snrShells, 1);
                    delete[] img;
                }

                {
                    auto img = new float[snrShells];
                    signal_power_arr_d.CopyFromArrayToHost(img);
                    stringstream DP;
                    DP << "signal_power_" << stackIdx << "_" << iter << ".em";
                    emwrite(DP.str(), img, snrShells, 1);
                    delete[] img;
                }
            }
        }
	}
}
template void Reconstructor::Compare(Volume<unsigned short>* vol, char* originalImage, uint aIndex, bool computeSNR, int iter, bool normForLength, float length);
template void Reconstructor::Compare(Volume<float>* vol, char* originalImage, uint aIndex, bool computeSNR, int iter, bool normForLength, float length);

void Reconstructor::PrepareForWBP(Volume<float>* parentVol,
                                  vector<Volume<float>*>& childVols,
                                  char* originalImage,
                                  uint stackIdx,
                                  int iter)
{
    if (snr_loaded){
        // If SNR and distance are known, divide projection by distance and copy signal/noise power.
        proj_d.CopyHostToDevice(originalImage);
        proj_children_d.CopyDeviceToDevice(proj_d);

        nppSafeCall(nppiDivC_32f_C1IR(proj_distance[stackIdx]/(parentVol->GetVoxelSize().x * parentVol->GetVoxelSize().x),
                                      (Npp32f*)proj_d.GetDevicePtr(),
                                      (int)proj_d.GetPitch(),
                                      roiAll));

        // Noise estimates are for all particles in a projection. The actual power for a single particle
        // is NP * sqrt(N).
        float* stackp;
        stackp = np_stack + stackIdx * snrShells;
        noise_power_d.CopyHostToDevice(stackp);
        nppSafeCall(nppsMulC_32f_I(sqrtf((float)childVols.size()),
                                   (Npp32f*) noise_power_d.GetDevicePtr(),
                                   snrShells));
        noise_power_arr_d.CopyFromDeviceToArray(noise_power_d);


        stackp = sp_stack + stackIdx * snrShells;
        signal_power_d.CopyHostToDevice(stackp);
//        nppSafeCall(nppsDivC_32f_I(sqrtf((float)childVols.size()),
//                                   (Npp32f*) signal_power_d.GetDevicePtr(),
//                                   snrShells));
        signal_power_arr_d.CopyFromDeviceToArray(signal_power_d);
    } else {
        // If unknown, just backproject the image as is.
        proj_d.CopyHostToDevice(originalImage);
        proj_children_d.CopyDeviceToDevice(proj_d);
    }
}

void Reconstructor::CompareChildren(char* originalImage,
                                     Volume<float>* vol,
                                     std::vector<Volume<float>*>&  childVols,
                                     DeviceVolumeFFT& childDevVol,
                                     DeviceVolumeBuf& maskDevVol,
                                     uint stackIdx,
                                     bool computeSNR,
                                     int iter)
{
    if (mpi_part == 0)
    {
        float dimension = childVols[0]->GetDimension().x;
        float voxelsize = childVols[0]->GetVoxelSize().x;
        float volumeTraversalLength = dimension / (voxelsize*voxelsize);
        nppSafeCall(nppiSet_32f_C1R(volumeTraversalLength,
                                    (Npp32f*)dist_d.GetDevicePtr(),
                                    dist_d.GetPitch(),
                                    roiAll));

        // Crop Dims
        vector<float2> corners;
        vector<float> normVals;
        proj.ComputeHitPointsNew(*vol, stackIdx, corners, normVals);

        realproj_d.CopyHostToDevice(originalImage);
        float runtime = compKernel(realproj_d,
                                   proj_d,
                                   dist_d,
                                   projDim,
                                   config.CutLength,
                                   config.DimLength,
                                   volumeTraversalLength,
                                   corners,
                                   normVals,
                                   vol->GetVoxelSize().x);
    }
}

void Reconstructor::MultiplicityChildren(std::vector<Volume<float>*>&  childVols,
                                         DeviceVolumeBuf& multDevVol,
                                         uint stackIdx, int iter)
{
    // Compute Multiplicity
    multDevVol.reset();
    multiplicity3D.SetComputeSize(multDevVol.GetDim());
    for (auto child : childVols){
        // System Matrix
        float3x3 systemMatrix = proj.SystemMatrix3x3(stackIdx, child->VolumeMatrixNorm());
        multiplicity3D(multDevVol.device_var(), multDevVol.GetDim(), systemMatrix);
    }

    // Store actual value in Card array
    multDevVol.VarToCard();

    // Normalize by particle number
    multDevVol.DivC((float)childVols.size());

    // Store in Dual array
    multDevVol.VarToDual();
}

void Reconstructor::PowerChildren(std::vector<Volume<float>*>&  childVols,
                                  DeviceVolumeFFT& childDevVol,
                                  DeviceVolumeBuf& maskDevVol,
                                  uint stackIdx,
                                  int iter)
{
    float dimension = childVols[0]->GetDimension().x;
    float voxelsize = childVols[0]->GetVoxelSize().x;

    signal_power_d.Memset(0);
    childDevVol.SSNRmasked(signal_power_d,
                           maskDevVol,
                           proj.GetMaxShells(),
                           proj.GetMaxDimension()/2+1,
                           1.f,
                           voxelsize);

    // To stack for later reconstruction
    float* stackp = sp_stack + stackIdx * snrShells;
    signal_power_d.CopyDeviceToHost(stackp);

    // Reduce signal strength to match single particle
//    nppSafeCall(nppsDivC_32f_I(sqrtf((float)childVols.size()),
//                           (Npp32f*) signal_power_d.GetDevicePtr(),
//                           snrShells));

    // To array for interpolation
    signal_power_arr_d.CopyFromDeviceToArray(signal_power_d);

    if (config.WriteDebug) {
        {
            auto img = new float[snrShells];
            signal_power_arr_d.CopyFromArrayToHost(img);
            stringstream DP;
            DP << "signal_power_" << stackIdx << "_" << iter << ".em";
            emwrite(DP.str(), img, snrShells, 1);
            delete[] img;
        }
    }
}

void Reconstructor::PowerChildrenHS(std::vector<Volume<float>*>&  childVols,
                                    DeviceVolumeFFT& childDevVol,
                                    std::vector<DeviceVolumeFFT*>& childDevHalfs,
                                    DeviceVolumeBuf& maskDevVol,
                                    uint stackIdx,
                                    int iter)
{
    float dimension = childVols[0]->GetDimension().x;
    float voxelsize = childVols[0]->GetVoxelSize().x;

    // Power for half sets
    for (int half = 0; half < 2; half++) {
        signal_power_d.Memset(0);
        childDevHalfs[half]->SSNRmasked(signal_power_d,
                               maskDevVol,
                               proj.GetMaxShells(),
                               proj.GetMaxDimension() / 2 + 1,
                               1.f,
                               voxelsize);

        // To stack for later reconstruction
        float *stackp = sp_stack_HS[half] + stackIdx * snrShells;
        signal_power_d.CopyDeviceToHost(stackp);

        // Reduce signal strength to match single particle
//        nppSafeCall(nppsDivC_32f_I(sqrtf((float)childVols.size() * 0.5f),
//                                   (Npp32f*) signal_power_d.GetDevicePtr(),
//                                   snrShells));

        // To array for interpolation
        signal_power_arr_d_HS[half].CopyFromDeviceToArray(signal_power_d);

        if (config.WriteDebug) {
            {
                auto img = new float[snrShells];
                signal_power_arr_d_HS[half].CopyFromArrayToHost(img);
                stringstream DP;
                DP << "signal_power_parent_" << stackIdx << "_" << iter << "_" << half << ".em";
                emwrite(DP.str(), img, snrShells, 1);
                delete[] img;
            }
        }
    }

    // Power for full
    signal_power_d.Memset(0);
    multiplicity_d.Memset(0);
//    childDevHalfs[0]->FSCmasked(*childDevHalfs[1],
//                                maskDevVol,
//                                &signal_power_d,
//                                proj.GetMaxShells(),
//                                proj.GetMaxDimension()/2+1,
//                                1.f,
//                                voxelsize);
//
//    if (config.WriteDebug) {
//        {
//            auto img = new float[snrShells];
//            signal_power_d.CopyDeviceToHost(img);
//            stringstream DP;
//            DP << "fsc_" << stackIdx << "_" << iter << ".em";
//            emwrite(DP.str(), img, snrShells, 1);
//            delete[] img;
//        }
//    }
//
//    // FSC > 0.001
//    nppSafeCall(nppsThreshold_LTVal_32f_I((Npp32f*)signal_power_d.GetDevicePtr(),
//                                          snrShells,
//                                          0.001f,
//                                          0.001f));
//    // FSC < 0.999
//    nppSafeCall(nppsThreshold_GTVal_32f_I((Npp32f*)signal_power_d.GetDevicePtr(),
//                                          snrShells,
//                                          0.999f,
//                                          0.999f));
//
//    if (config.WriteDebug) {
//        {
//            auto img = new float[snrShells];
//            signal_power_d.CopyDeviceToHost(img);
//            stringstream DP;
//            DP << "fsc_clip_" << stackIdx << "_" << iter << ".em";
//            emwrite(DP.str(), img, snrShells, 1);
//            delete[] img;
//        }
//    }
//
//    // 1 - FSC
//    nppSafeCall(nppsSubCRev_32f((Npp32f*)signal_power_d.GetDevicePtr(),
//                                1.f,
//                                (Npp32f*)multiplicity_d.GetDevicePtr(),
//                                snrShells));
//
//    // FSC / (1 - FSC) --> SNR
//    nppSafeCall(nppsDiv_32f_I((Npp32f*)multiplicity_d.GetDevicePtr(),
//                              (Npp32f*)signal_power_d.GetDevicePtr(),
//                              snrShells));
//
//
//    if (config.WriteDebug) {
//        {
//            auto img = new float[snrShells];
//            signal_power_d.CopyDeviceToHost(img);
//            stringstream DP;
//            DP << "snr_" << stackIdx << "_" << iter << ".em";
//            emwrite(DP.str(), img, snrShells, 1);
//            delete[] img;
//        }
//    }


    childDevVol.SSNRmasked(signal_power_d,
                           maskDevVol,
                           proj.GetMaxShells(),
                           proj.GetMaxDimension()/2+1,
                           1.f,
                           voxelsize);

    // To stack for later reconstruction
    float* stackp = sp_stack + stackIdx * snrShells;
    signal_power_d.CopyDeviceToHost(stackp);

    // Reduce signal strength to match single particle
//    nppSafeCall(nppsDivC_32f_I(sqrtf((float)childVols.size()),
//                               (Npp32f*) signal_power_d.GetDevicePtr(),
//                               snrShells));

    // To array for interpolation
    signal_power_arr_d.CopyFromDeviceToArray(signal_power_d);

    if (config.WriteDebug) {
        {
            auto img = new float[snrShells];
            signal_power_arr_d.CopyFromArrayToHost(img);
            stringstream DP;
            DP << "signal_power_full_" << stackIdx << "_" << iter << ".em";
            emwrite(DP.str(), img, snrShells, 1);
            delete[] img;
        }
    }

}



void Reconstructor::PowerOrphans(std::vector<Volume<float>*>&  childVols,
                                 DeviceVolumeFFT& childDevVol,
                                 DeviceVolumeBuf& maskDevVol,
                                 uint stackIdx,
                                 int iter)
{
    float dimension = childVols[0]->GetDimension().x;
    float voxelsize = childVols[0]->GetVoxelSize().x;

    // Compute Noise power (radial average)
    noise_power_d.Memset(0);
    childDevVol.SSNRmasked(noise_power_d,
                           maskDevVol,
                           proj.GetMaxShells(),
                           proj.GetMaxDimension()/2+1,
                           1.f,
                           voxelsize);

    // To stack for later reconstruction
    float* stackp = np_stack + stackIdx * snrShells;
    noise_power_d.CopyDeviceToHost(stackp);

    // Scale up noise to match single particle
    nppSafeCall(nppsMulC_32f_I(sqrtf((float)childVols.size()),
                           (Npp32f*) signal_power_d.GetDevicePtr(),
                           snrShells));

    noise_power_arr_d.CopyFromDeviceToArray(noise_power_d);

    if (config.WriteDebug) {
        {
            auto img = new float[snrShells];
            noise_power_arr_d.CopyFromArrayToHost(img);
            stringstream DP;
            DP << "noise_power_" << stackIdx << "_" << iter << ".em";
            emwrite(DP.str(), img, snrShells, 1);
            delete[] img;
        }
    }
}


void Reconstructor::PowerOrphansHS(std::vector<Volume<float>*>&  childVols,
                                   DeviceVolumeFFT& childDevVol,
                                   std::vector<DeviceVolumeFFT*>& childDevHalfs,
                                   DeviceVolumeBuf& maskDevVol,
                                   uint stackIdx,
                                   int iter)
{
    float dimension = childVols[0]->GetDimension().x;
    float voxelsize = childVols[0]->GetVoxelSize().x;

    // Power for half sets
    for (int half = 0; half < 2; half++) {

        // Compute Noise power (radial average)
        noise_power_d.Memset(0);
        childDevHalfs[half]->SSNRmasked(noise_power_d,
                               maskDevVol,
                               proj.GetMaxShells(),
                               proj.GetMaxDimension() / 2 + 1,
                               1.f,
                               voxelsize);

        // To stack for later reconstruction
        float *stackp = np_stack_HS[half] + stackIdx * snrShells;
        noise_power_d.CopyDeviceToHost(stackp);

        // Scale up noise to match single particle
        nppSafeCall(nppsMulC_32f_I(sqrtf((float)childVols.size() * 0.5f),
                                   (Npp32f*) signal_power_d.GetDevicePtr(),
                                   snrShells));


        // To array for interpolation
        noise_power_arr_d_HS[half].CopyFromDeviceToArray(noise_power_d);

        if (config.WriteDebug) {
            {
                auto img = new float[snrShells];
                noise_power_arr_d_HS[half].CopyFromArrayToHost(img);
                stringstream DP;
                DP << "noise_power_" << stackIdx << "_" << iter << "_" << half << ".em";
                emwrite(DP.str(), img, snrShells, 1);
                delete[] img;
            }
        }
    }

    // Compute Noise power (radial average)
    noise_power_d.Memset(0);
    childDevVol.SSNRmasked(noise_power_d,
                           maskDevVol,
                           proj.GetMaxShells(),
                           proj.GetMaxDimension()/2+1,
                           1.f,
                           voxelsize);

    // To stack for later reconstruction
    float* stackp = np_stack + stackIdx * snrShells;
    noise_power_d.CopyDeviceToHost(stackp);

    // Scale up noise to match single particle
    nppSafeCall(nppsMulC_32f_I(sqrtf((float)childVols.size()),
                               (Npp32f*) signal_power_d.GetDevicePtr(),
                               snrShells));
//    nppSafeCall(nppsSet_32f(1.f,
//                               (Npp32f*) signal_power_d.GetDevicePtr(),
//                               snrShells));

    noise_power_arr_d.CopyFromDeviceToArray(noise_power_d);


    if (config.WriteDebug) {
        {
            auto img = new float[snrShells];
            noise_power_arr_d.CopyFromArrayToHost(img);
            stringstream DP;
            DP << "noise_power_full_ " << stackIdx << "_" << iter << ".em";
            emwrite(DP.str(), img, snrShells, 1);
            delete[] img;
        }
    }
}


//void Reconstructor::CompareSpecial(Volume<float>* parentVol, Volume<float>* childVol, char* originalImage, uint stackIdx, bool computeSNR, int iter)
//{
//    if (mpi_part == 0)
//    {
//        nppSafeCall(nppsMax_32f((Npp32f*)dist_d.GetDevicePtr(),
//                                proj.GetWidth()*proj.GetHeight(),
//                                (Npp32f*) meanval.GetDevicePtr(),
//                                (Npp8u*) meanbuffer.GetDevicePtr()));
//
//        float volumeTraversalLength = 0.f;
//        meanval.CopyDeviceToHost(&volumeTraversalLength, sizeof(float));
//
//        if (computeSNR) {
//            // Radial Average of signal power for SNR, no normalization for FFT as this
//            // already happened in postFilterSum
//            float* stackp = sp_stack + stackIdx * snrShells;
//
//            // FFT fwd proj parent
//            copyFromPitched(proj_d, proj_dv_d, projDim);
//            cufftSafeCall(cufftExecR2C(handleR2C,
//                                       (cufftReal*)proj_dv_d.GetDevicePtr(),
//                                       (cufftComplex*)fft_d.GetDevicePtr()));
//
//            multiplicity_d.Memset(0);
//            signal_power_d.Memset(0);
//            radialSum(fft_d,
//                      signal_power_d,
//                      multiplicity_d,
//                      fftDim,
//                      projDimF,
//                      asymCorrFac,
//                      1.f);//projDimF.x * projDimF.y
//            nppSafeCall(nppsDiv_32f_I((Npp32f*)multiplicity_d.GetDevicePtr(),
//                                      (Npp32f*)signal_power_d.GetDevicePtr(),
//                                      snrShells));
//            //signal_power_d.CopyDeviceToHost(stackp);
//            signal_power_arr_d.CopyFromDeviceToArray(signal_power_d);
//
//            // FFT fwd proj child
//            copyFromPitched(proj_children_d, proj_dv_d, projDim);
//            cufftSafeCall(cufftExecR2C(handleR2C,
//                                       (cufftReal*)proj_dv_d.GetDevicePtr(),
//                                       (cufftComplex*)fft_d.GetDevicePtr()));
//
//            multiplicity_d.Memset(0);
//            signal_power_d.Memset(0);
//            radialSum(fft_d,
//                      signal_power_d,
//                      multiplicity_d,
//                      fftDim,
//                      projDimF,
//                      asymCorrFac,
//                      1.f);//projDimF.x * projDimF.y
//            nppSafeCall(nppsDiv_32f_I((Npp32f*)multiplicity_d.GetDevicePtr(),
//                                      (Npp32f*)signal_power_d.GetDevicePtr(),
//                                      snrShells));
//            //signal_power_d.CopyDeviceToHost(stackp);
//            signal_power_child_arr_d.CopyFromDeviceToArray(signal_power_d);
//        }
//
//        // Crop Dims
//        vector<float2> corners;
//        vector<float> normVals;
//        proj.ComputeHitPointsNew(*parentVol, stackIdx, corners, normVals);
//
//        realproj_d.CopyHostToDevice(originalImage);
//        compSpecialKernel.SetComputeSize(projDim);
//        float runtime = compSpecialKernel(realproj_d,
//                                          proj_d,
//                                          proj_children_d,
//                                          dist_d,
//                                          dist_children_d,
//                                          projDim,
//                                          config.CutLength,
//                                          config.DimLength,
//                                          volumeTraversalLength,
//                                          childVol->GetDimension().x/childVol->GetVoxelSize().x,
//                                          corners,
//                                          normVals,
//                                          parentVol->GetVoxelSize().x,
//                                          childVol->GetVoxelSize().x);
//
//        {
//            auto img = new float[proj.GetWidth()*proj.GetHeight()];
//            proj_d.CopyDeviceToHost(img);
//            stringstream DP;
//            DP << "error_parent_" << stackIdx << "_" << iter <<".em";
//            emwrite(DP.str(), img, proj.GetWidth(),
//                    proj.GetHeight());
//            delete[] img;
//        }
//
//        {
//            auto img = new float[proj.GetWidth()*proj.GetHeight()];
//            proj_children_d.CopyDeviceToHost(img);
//            stringstream DP;
//            DP << "error_child_" << stackIdx << "_" << iter <<".em";
//            emwrite(DP.str(), img, proj.GetWidth(),
//                    proj.GetHeight());
//            delete[] img;
//        }
//
//        if (computeSNR){
//
//            // Expects signal power to reside in signal_power_d already.
//            float* stackp = np_stack + stackIdx * snrShells;
//            //noise_power_arr_d.CopyFromHostToArray(stackp);
//
//
//            // FFT diff proj parent (i.e. noise)
//            copyFromPitched(proj_d, proj_dv_d, projDim);
//            cufftSafeCall(cufftExecR2C(handleR2C,
//                                       (cufftReal*)proj_dv_d.GetDevicePtr(),
//                                       (cufftComplex*)fft_d.GetDevicePtr()));
//
//            // Radial average noise power, normalize for FFT
//            multiplicity_d.Memset(0);
//            noise_power_d.Memset(0);
//            radialSum(fft_d,
//                      noise_power_d,
//                      multiplicity_d,
//                      fftDim,
//                      projDimF,
//                      asymCorrFac,
//                      volumeTraversalLength); //1.f/(projDimF.x * projDimF.y)* ///config.DeconvStrength);
//            nppSafeCall(nppsDiv_32f_I((Npp32f*)multiplicity_d.GetDevicePtr(),
//                                      (Npp32f*)noise_power_d.GetDevicePtr(),
//                                      snrShells));
//            //noise_power_d.CopyDeviceToHost(stackp);
//            noise_power_parent_arr_d.CopyFromDeviceToArray(noise_power_d);
//
//
//            {
//                auto img = new float[snrShells];
//                noise_power_parent_arr_d.CopyFromArrayToHost(img);
//                stringstream DP;
//                DP << "noise_power_parent_" << stackIdx << "_" << iter <<".em";
//                emwrite(DP.str(), img, snrShells, 1);
//                delete[] img;
//            }
//
//            {
//                auto img = new float[snrShells];
//                signal_power_parent_arr_d.CopyFromArrayToHost(img);
//                stringstream DP;
//                DP << "signal_power_parent_" << stackIdx << "_" << iter <<".em";
//                emwrite(DP.str(), img, snrShells, 1);
//                delete[] img;
//            }
//
//            // FFT diff proj child (i.e. noise)
//            copyFromPitched(proj_children_d, proj_dv_d, projDim);
//            cufftSafeCall(cufftExecR2C(handleR2C,
//                                       (cufftReal*)proj_dv_d.GetDevicePtr(),
//                                       (cufftComplex*)fft_d.GetDevicePtr()));
//
//            // Radial average noise power, normalize for FFT
//            multiplicity_d.Memset(0);
//            noise_power_d.Memset(0);
//            radialSum(fft_d,
//                      noise_power_d,
//                      multiplicity_d,
//                      fftDim,
//                      projDimF,
//                      asymCorrFac,
//                      volumeTraversalLength); //1.f/(projDimF.x * projDimF.y)* ///config.DeconvStrength);
//            nppSafeCall(nppsDiv_32f_I((Npp32f*)multiplicity_d.GetDevicePtr(),
//                                      (Npp32f*)noise_power_d.GetDevicePtr(),
//                                      snrShells));
//            //noise_power_d.CopyDeviceToHost(stackp);
//            noise_power_child_arr_d.CopyFromDeviceToArray(noise_power_d);
//
//
//            {
//                auto img = new float[snrShells];
//                noise_power_child_arr_d.CopyFromArrayToHost(img);
//                stringstream DP;
//                DP << "noise_power_child_" << stackIdx << "_" << iter <<".em";
//                emwrite(DP.str(), img, snrShells, 1);
//                delete[] img;
//            }
//
//            {
//                auto img = new float[snrShells];
//                signal_power_child_arr_d.CopyFromArrayToHost(img);
//                stringstream DP;
//                DP << "signal_power_child_" << stackIdx << "_" << iter <<".em";
//                emwrite(DP.str(), img, snrShells, 1);
//                delete[] img;
//            }
//
//            // SNR now resides in snr_arr
//        }
//    }
//}

void Reconstructor::SubtractError(float* error)
{
    if (mpi_part == 0)
    {
        realproj_d.CopyHostToDevice(error);
        float runtime = subEKernel(proj_d, realproj_d, dist_d);
    }
}


template<class TVol>
void Reconstructor::PrintGeometry(Volume<TVol>* vol, int index)
{
	int x = proj.GetWidth();
	int y = proj.GetHeight();

	printf("\n\nProjection: %d\n", index);

	if (config.WBP_NoSART)
	{
		magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
		magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
	}

	SetConstantValues(slicerKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
	SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

	float t_in, t_out;
	GetDefocusDistances(t_in, t_out, index, vol);

	//Shoot ray from center of volume:
	float3 c_projNorm = proj.GetNormalVector(index);
	float3 c_detektor = proj.GetPosition(index);
	float3 MC_bBoxMin;
	float3 MC_bBoxMax;
	MC_bBoxMin = vol->GetVolumeBBoxMin();
	MC_bBoxMax = vol->GetVolumeBBoxMax();
	float3 volDim = vol->GetDimension();
	float3 hitPoint;
	float t;

	t = (c_projNorm.x * (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f)) +
		c_projNorm.y * (MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f)) +
		c_projNorm.z * (MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f)));
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = abs(t);

	printf("t: %f\n", t);

	hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f));
	hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f));
	hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f));

	float4x4 c_DetectorMatrix;

	proj.GetDetectorMatrix(index, (float*)&c_DetectorMatrix, 1);
	MatrixVector3Mul(c_DetectorMatrix, &hitPoint);

	//--> pixelBorders.x = x.min; pixelBorders.z = y.min;
	float hitX = round(hitPoint.x);
	float hitY = round(hitPoint.y);

	printf("HitXY: %d %d\n", (int)hitX, (int)hitY);

	//Shoot ray from hit point on projection towards volume to get the distance to entry and exit point
	float3 pos = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitY * proj.GetPixelVPitch(index);
	hitX = (float)proj.GetWidth() * 0.5f;
	hitY = (float)proj.GetHeight() * 0.5f;
	float3 pos2 = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitX * proj.GetPixelVPitch(index);

	printf("Center: %f %f %f\n", pos2.x, pos2.y, pos2.z);

	float3 nvec = proj.GetNormalVector(index);

	t_in = 2 * -DIST;
	t_out = 2 * DIST;

	for (int x = 0; x <= 1; x++)
		for (int y = 0; y <= 1; y++)
			for (int z = 0; z <= 1; z++)
			{
				t = (nvec.x * (MC_bBoxMin.x + x * (MC_bBoxMax.x - MC_bBoxMin.x))
					+ nvec.y * (MC_bBoxMin.y + y * (MC_bBoxMax.y - MC_bBoxMin.y))
					+ nvec.z * (MC_bBoxMin.z + z * (MC_bBoxMax.z - MC_bBoxMin.z)));
				t += (-nvec.x * pos.x - nvec.y * pos.y - nvec.z * pos.z);

				if (t < t_in) t_in = t;
				if (t > t_out) t_out = t;
			}

	for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize())
	{
		dist_d.Memset(0);

		float defocusAngle = defocus.GetAstigmatismAngle(index) + (float)(proj.GetImageRotationToCompensate((uint)index) / M_PI * 180.0);
		float defocusMin;
		float defocusMax;
		GetDefocusMinMax(ray + config.CTFSliceThickness / proj.GetPixelSize() * 0.5f, index, defocusMin, defocusMax);

		printf("Defocus: %-8d nm\n", (int)defocusMin);
		
	}
}
template void Reconstructor::PrintGeometry(Volume<unsigned short>* vol, int index);
template void Reconstructor::PrintGeometry(Volume<float>* vol, int index);


void Reconstructor::ResetProjectionsDevice()
{
	proj_d.Memset(0);
    proj_children_d.Memset(0);
	dist_d.Memset(0);
    dist_children_d.Memset(0);
    fft_d.Memset(0);
    fft_d2.Memset(0);

	CTFbuffer_realRect.Memset(0);
	CTFbuffer_comp1.Memset(0);
}

void Reconstructor::SaveSNR(string& aFile, string suffix)
{
    stringstream signal_out;
    signal_out << aFile << "signal_power_" << suffix << ".em";
    emwrite(signal_out.str(), sp_stack, snrShells, proj.GetProjCount());

    stringstream noise_out;
    noise_out << aFile << "noise_power_" << suffix << ".em";
    emwrite(noise_out.str(), np_stack, snrShells, proj.GetProjCount());

    stringstream dist_out;
    dist_out << aFile << "distance_" << suffix << ".em";
    emwrite(dist_out.str(), proj_distance, proj.GetProjCount(), 1);
}

void Reconstructor::LoadSNR(string& aFile, string suffix)
{
    stringstream signal_in;
    signal_in << aFile << "signal_power_" << suffix << ".em";
    EmFile sigIn(signal_in.str());
    sigIn.OpenAndRead();
    std::memcpy(sp_stack, sigIn.GetData(), snrShells * proj.GetProjCount() * sizeof(float));

    stringstream noise_in;
    noise_in << aFile << "noise_power_" << suffix << ".em";
    EmFile noiIn(noise_in.str());
    noiIn.OpenAndRead();
    std::memcpy(np_stack, noiIn.GetData(), snrShells * proj.GetProjCount() * sizeof(float));

    stringstream dist_in;
    dist_in << aFile << "distance_" << suffix << ".em";
    EmFile distIn(dist_in.str());
    distIn.OpenAndRead();
    std::memcpy(proj_distance, distIn.GetData(), proj.GetProjCount() * sizeof(float));

    snr_loaded = true;
}


void Reconstructor::ResetChildProj()
{
    proj_children_d.Memset(0);
}

void Reconstructor::CopyProjectionToHost(float * buffer)
{
	proj_d.CopyDeviceToHost(buffer);
}

void Reconstructor::CopyChildProjectionToHost(float * buffer)
{
    proj_children_d.CopyDeviceToHost(buffer);
}

void Reconstructor::CopyDistanceImageToHost(float * buffer)
{
	dist_d.CopyDeviceToHost(buffer);
}

void Reconstructor::CopyChildDistanceImageToHost(float * buffer)
{
    dist_children_d.CopyDeviceToHost(buffer);
}

void Reconstructor::CopyRealProjectionToHost(float * buffer)
{
	realproj_d.CopyDeviceToHost(buffer);
}

void Reconstructor::CopyProjectionToDevice(float * buffer)
{
	proj_d.CopyHostToDevice(buffer);
}

void Reconstructor::CopyDistanceImageToDevice(float * buffer)
{
	dist_d.CopyHostToDevice(buffer);
}

void Reconstructor::CopyRealProjectionToDevice(float * buffer)
{
	realproj_d.CopyHostToDevice(buffer);
}

void Reconstructor::MPIBroadcast(float ** buffers, int bufferCount)
{
#ifdef USE_MPI
	for (int i = 0; i < bufferCount; i++)
	{
		MPI_Bcast(buffers[i], proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
#endif
}

#ifdef REFINE_MODE
void Reconstructor::CopyProjectionToSubVolumeProjection()
{
	if (mpi_part == 0)
	{
		projSubVols_d.CopyDeviceToDevice(proj_d);
	}
}
#endif

void Reconstructor::ConvertVolumeFP16(float * slice, Cuda::CudaSurfaceObject3D& surf, int z)
{
	if (volTemp_d.GetWidth() != config.RecDimensions.x ||
		volTemp_d.GetHeight() != config.RecDimensions.y)
	{
		volTemp_d.Alloc(config.RecDimensions.x * sizeof(float), config.RecDimensions.y, sizeof(float));
	}
	convVolKernel(volTemp_d, surf, z);
	volTemp_d.CopyDeviceToHost(slice);
}

//template<class TVol>
//void Reconstructor::computePostFilter(Volume<TVol>* vol, int projIndex, int dim_x, int dim_y, float degree)
//{
//    // Compute the actual spline
//    int2 maxVal = make_int2(dim_x, dim_y);
//    int2 pixelcount = maxVal;
//    int2 pixelcount_half = make_int2(pixelcount.x/2, pixelcount.y/2);
//    float2 maxFreq = make_float2(0.5f, 0.5f);
//    float2 freqStepSize = make_float2(maxFreq.x/(float)pixelcount_half.x, maxFreq.y/(float)pixelcount_half.y);
//
//    // Projection system
//    float3 c_projNorm = proj.GetNormalVector(projIndex);
//    float3 c_detektor = proj.GetPosition(projIndex);
//    float4x4 c_DetectorMatrix;
//
//    // Box Spline directions
//    proj.GetDetectorMatrix(projIndex, (float *) &c_DetectorMatrix, 1);
//    //MatrixScalarMul(c_DetectorMatrix, config.VoxelSize.x);
//    Matrix<float> c_magAnisoM = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg,
//                                                      0.f, 0.f);
//    float3x3 c_magAniso = *(float3x3 *) c_magAnisoM.GetData();
//
//    Matrix<double> old_det(4, 4);
//    old_det(0, 0) = c_DetectorMatrix.m[0].x;
//    old_det(0, 1) = c_DetectorMatrix.m[0].y;
//    old_det(0, 2) = c_DetectorMatrix.m[0].z;
//    old_det(0, 3) = c_DetectorMatrix.m[0].w;
//    old_det(1, 0) = c_DetectorMatrix.m[1].x;
//    old_det(1, 1) = c_DetectorMatrix.m[1].y;
//    old_det(1, 2) = c_DetectorMatrix.m[1].z;
//    old_det(1, 3) = c_DetectorMatrix.m[1].w;
//    old_det(2, 0) = c_DetectorMatrix.m[2].x;
//    old_det(2, 1) = c_DetectorMatrix.m[2].y;
//    old_det(2, 2) = c_DetectorMatrix.m[2].z;
//    old_det(2, 3) = c_DetectorMatrix.m[2].w;
//    old_det(3, 0) = c_DetectorMatrix.m[3].x;
//    old_det(3, 1) = c_DetectorMatrix.m[3].y;
//    old_det(3, 2) = c_DetectorMatrix.m[3].z;
//    old_det(3, 3) = c_DetectorMatrix.m[3].w;
//
//    Matrix<double> M_det = proj.DetectorMatrix<double>(projIndex);
//    Matrix<double> M_vol = vol->VolumeMatrix();
//
//    Matrix<double> M_proj = proj.ProjectionMatrix<double>(projIndex);
//    Matrix<double> rot(4, 1);
//
//
//    Matrix<double> center_new(4, 1);
//    center_new(0, 0) = 0;
//    center_new(1, 0) = 0;
//    center_new(2, 0) = 0;
//    center_new(3, 0) = 1;
//
//    Matrix<double> center_old(4, 1);
//    center_old(0, 0) = -128;
//    center_old(1, 0) = -128;
//    center_old(2, 0) = -128;
//    center_old(3, 0) = 1;
//
//
//    Matrix<double> M_full(4,4);
//    M_full = M_det * M_vol;
//
//    center_new = M_full * center_new;
//    center_old = old_det * center_old;
//
//    rot = center_new;
//    rot(0, 0) = center_new(0, 0);
//    rot(1, 0) = center_new(1, 0);
//    rot(2, 0) = 0;
//    rot(3, 0) = 1;
//    //rot(3, 0) = 0;
//    rot = M_proj * rot;
//
//    printf("\nOld Detector\n");
//    printf("[ %f %f %f %f;\n", old_det(0, 0), old_det(0, 1), old_det(0, 2), old_det(0, 3));
//    printf("  %f %f %f %f;\n", old_det(1, 0), old_det(1, 1), old_det(1, 2), old_det(1, 3));
//    printf("  %f %f %f %f;\n", old_det(2, 0), old_det(2, 1), old_det(2, 2), old_det(2, 3));
//    printf("  %f %f %f %f]\n", old_det(3, 0), old_det(3, 1), old_det(3, 2), old_det(3, 3));
//
//    printf("\nNew Detector\n");
//    printf("[ %f %f %f %f;\n", M_full(0, 0), M_full(0, 1), M_full(0, 2), M_full(0, 3));
//    printf("  %f %f %f %f;\n", M_full(1, 0), M_full(1, 1), M_full(1, 2), M_full(1, 3));
//    printf("  %f %f %f %f;\n", M_full(2, 0), M_full(2, 1), M_full(2, 2), M_full(2, 3));
//    printf("  %f %f %f %f]\n", M_full(3, 0), M_full(3, 1), M_full(3, 2), M_full(3, 3));
//
//    printf("\nNew Detector2\n");
//    printf("[ %f %f %f %f;\n", M_det(0, 0), M_det(0, 1), M_det(0, 2), M_det(0, 3));
//    printf("  %f %f %f %f;\n", M_det(1, 0), M_det(1, 1), M_det(1, 2), M_det(1, 3));
//    printf("  %f %f %f %f;\n", M_det(2, 0), M_det(2, 1), M_det(2, 2), M_det(2, 3));
//    printf("  %f %f %f %f]\n", M_det(3, 0), M_det(3, 1), M_det(3, 2), M_det(3, 3));
//
//    printf("\nVol\n");
//    printf("[ %f %f %f %f;\n", M_vol(0, 0), M_vol(0, 1), M_vol(0, 2), M_vol(0, 3));
//    printf("  %f %f %f %f;\n", M_vol(1, 0), M_vol(1, 1), M_vol(1, 2), M_vol(1, 3));
//    printf("  %f %f %f %f;\n", M_vol(2, 0), M_vol(2, 1), M_vol(2, 2), M_vol(2, 3));
//    printf("  %f %f %f %f]\n", M_vol(3, 0), M_vol(3, 1), M_vol(3, 2), M_vol(3, 3));
//
//    printf("\nProj\n");
//    printf("[ %f %f %f %f;\n", M_proj(0, 0), M_proj(0, 1), M_proj(0, 2), M_proj(0, 3));
//    printf("  %f %f %f %f;\n", M_proj(1, 0), M_proj(1, 1), M_proj(1, 2), M_proj(1, 3));
//    printf("  %f %f %f %f;\n", M_proj(2, 0), M_proj(2, 1), M_proj(2, 2), M_proj(2, 3));
//    printf("  %f %f %f %f]\n", M_proj(3, 0), M_proj(3, 1), M_proj(3, 2), M_proj(3, 3));
//
//    printf("\n Project Old \n");
//    printf("\n %f, %f, %f, %f \n", center_old(0,0), center_old(1,0), center_old(2,0), center_old(3,0));
//
//    printf("\n Project New \n");
//    printf("\n %f, %f, %f, %f \n", center_new(0,0), center_new(1,0), center_new(2,0), center_new(3,0));
//
//    printf("\n point-proj New \n");
//    printf("\n %f, %f, %f, %f\n",  rot(0,0),  rot(1,0),  rot(2,0),  rot(3,0));
//
//    printf("\n c_projNorm \n");
//    printf("\n %f, %f, %f \n", c_projNorm.x, c_projNorm.y, c_projNorm.z);
//
//    printf("\n c_detektor \n");
//    printf("\n %f, %f, %f \n",  c_detektor.x, c_detektor.y, c_detektor.z);
//
//    float t;
//    float ts;
//    t = (c_projNorm.x * center_old(0, 0) + c_projNorm.y * center_old(1, 0) + c_projNorm.z * center_old(2, 0));
//    t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
//    t = abs(t) + DIST;
//    ts = t - (*entryPoints)[projIndex];
//
//    printf("\n t \n");
//    printf("\n %f \n",  t);
//
//    printf("\n ts \n");
//    printf("\n %f \n",  ts);
//
//    float2 Xi_x, Xi_y, Xi_z;
//    Xi_x = make_float2(c_DetectorMatrix.m[0].x, c_DetectorMatrix.m[1].x);
//    Xi_y = make_float2(c_DetectorMatrix.m[0].y, c_DetectorMatrix.m[1].y);
//    Xi_z = make_float2(c_DetectorMatrix.m[0].z, c_DetectorMatrix.m[1].z);
//
//    MatrixVector3Mul(c_magAniso, Xi_x);
//    MatrixVector3Mul(c_magAniso, Xi_y);
//    MatrixVector3Mul(c_magAniso, Xi_z);
//
//    float3 nu = make_float3(degree+1, degree+1, degree+1);
//
//    postFilterBox(pixelcount,
//                  freqStepSize,
//                  Xi_x, Xi_y, Xi_z, nu,
//                  d_prefilter_fft);
//}

void Reconstructor::computePostFilter(Volume<float>* aVol, int projIndex, int dim_x, int dim_y, float degree)
{
    // Compute the actual spline
    int2 maxVal = make_int2(dim_x, dim_y);
    int2 pixelcount = maxVal;
    int2 pixelcount_half = make_int2(pixelcount.x/2, pixelcount.y/2);
    float2 maxFreq = make_float2(0.5f, 0.5f);
    float2 freqStepSize = make_float2(maxFreq.x/(float)pixelcount_half.x, maxFreq.y/(float)pixelcount_half.y);

    // Projection system
//    float3 c_projNorm = proj.GetNormalVector(projIndex);
//    float4x4 c_DetectorMatrix;

    // Box Spline directions
    Matrix<double> M_det = proj.DetectorMatrix<double>(projIndex) * aVol->GetVoxelSize().x;
    float3x3 Mdetector = MatrixTo3x3(M_det);//proj.DetectorMatrix3x3(projIndex);
//    proj.GetDetectorMatrix(projIndex, (float *) &c_DetectorMatrix, 1);
//    MatrixScalarMul(c_DetectorMatrix, aVol->GetVoxelSize().x);
//    Matrix<float> c_magAnisoM = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg,
//                                                      0.f, 0.f);
//    float3x3 c_magAniso = *(float3x3 *) c_magAnisoM.GetData();

//    float2 Xi_x, Xi_y, Xi_z;
//    Xi_x = make_float2(c_DetectorMatrix.m[0].x, c_DetectorMatrix.m[1].x);
//    Xi_y = make_float2(c_DetectorMatrix.m[0].y, c_DetectorMatrix.m[1].y);
//    Xi_z = make_float2(c_DetectorMatrix.m[0].z, c_DetectorMatrix.m[1].z);

    float2 Xi_x, Xi_y, Xi_z;
    Xi_x = make_float2(Mdetector.m[0].x, Mdetector.m[1].x);
    Xi_y = make_float2(Mdetector.m[0].y, Mdetector.m[1].y);
    Xi_z = make_float2(Mdetector.m[0].z, Mdetector.m[1].z);

//    MatrixVector3Mul(c_magAniso, Xi_x);
//    MatrixVector3Mul(c_magAniso, Xi_y);
//    MatrixVector3Mul(c_magAniso, Xi_z);

    float3 nu = make_float3(degree+1, degree+1, degree+1);

    postFilterBox(pixelcount,
                  freqStepSize,
                  Xi_x, Xi_y, Xi_z, nu, 1.f,
                  d_prefilter_fft);

//    Matrix<double> old_det(4, 4);
//    old_det(0, 0) = c_DetectorMatrix.m[0].x;
//    old_det(0, 1) = c_DetectorMatrix.m[0].y;
//    old_det(0, 2) = c_DetectorMatrix.m[0].z;
//    old_det(0, 3) = c_DetectorMatrix.m[0].w;
//    old_det(1, 0) = c_DetectorMatrix.m[1].x;
//    old_det(1, 1) = c_DetectorMatrix.m[1].y;
//    old_det(1, 2) = c_DetectorMatrix.m[1].z;
//    old_det(1, 3) = c_DetectorMatrix.m[1].w;
//    old_det(2, 0) = c_DetectorMatrix.m[2].x;
//    old_det(2, 1) = c_DetectorMatrix.m[2].y;
//    old_det(2, 2) = c_DetectorMatrix.m[2].z;
//    old_det(2, 3) = c_DetectorMatrix.m[2].w;
//    old_det(3, 0) = c_DetectorMatrix.m[3].x;
//    old_det(3, 1) = c_DetectorMatrix.m[3].y;
//    old_det(3, 2) = c_DetectorMatrix.m[3].z;
//    old_det(3, 3) = c_DetectorMatrix.m[3].w;
//
//    printf("\nOld Detector\n");
//    printf("[ %f %f %f %f;\n", old_det(0, 0), old_det(0, 1), old_det(0, 2), old_det(0, 3));
//    printf("  %f %f %f %f;\n", old_det(1, 0), old_det(1, 1), old_det(1, 2), old_det(1, 3));
//    printf("  %f %f %f %f;\n", old_det(2, 0), old_det(2, 1), old_det(2, 2), old_det(2, 3));
//    printf("  %f %f %f %f]\n", old_det(3, 0), old_det(3, 1), old_det(3, 2), old_det(3, 3));
//
//    Matrix<double> M_det = proj.DetectorMatrix<double>(projIndex);
//    printf("\nNew Detector\n");
//    printf("[ %f %f %f %f;\n", M_det(0, 0), M_det(0, 1), M_det(0, 2), M_det(0, 3));
//    printf("  %f %f %f %f;\n", M_det(1, 0), M_det(1, 1), M_det(1, 2), M_det(1, 3));
//    printf("  %f %f %f %f;\n", M_det(2, 0), M_det(2, 1), M_det(2, 2), M_det(2, 3));
//    printf("  %f %f %f %f]\n", M_det(3, 0), M_det(3, 1), M_det(3, 2), M_det(3, 3));
}


void Reconstructor::computePreFilter(int projIndex, int dim_x, int dim_y, float degree)
{
    // Compute the actual spline
    int2 maxVal = make_int2(dim_x, dim_y); //maxSupportLUT.y - maxSupportLUT.x;
    int2 pixelcount = maxVal;//config.LUTStep;
    int2 pixelcount_half = make_int2(pixelcount.x/2, pixelcount.y/2);
    float2 maxFreq = make_float2(0.5f, 0.5f);
    float2 freqStepSize = make_float2(maxFreq.x/(float)pixelcount_half.x, maxFreq.y/(float)pixelcount_half.y);

    // Projection system
    float3 c_projNorm = proj.GetNormalVector(projIndex);
    float4x4 c_DetectorMatrix;

    // Box Spline directions
    proj.GetDetectorMatrix(projIndex, (float *) &c_DetectorMatrix, 1);
    MatrixScalarMul(c_DetectorMatrix, config.VoxelSize.x);
    Matrix<float> c_magAnisoM = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg,
                                                      0.f, 0.f);
    float3x3 c_magAniso = *(float3x3 *) c_magAnisoM.GetData();

    float2 Xi_x, Xi_y, Xi_z;
    Xi_x = make_float2(c_DetectorMatrix.m[0].x, c_DetectorMatrix.m[1].x);
    Xi_y = make_float2(c_DetectorMatrix.m[0].y, c_DetectorMatrix.m[1].y);
    Xi_z = make_float2(c_DetectorMatrix.m[0].z, c_DetectorMatrix.m[1].z);

    MatrixVector3Mul(c_magAniso, Xi_x);
    MatrixVector3Mul(c_magAniso, Xi_y);
    MatrixVector3Mul(c_magAniso, Xi_z);

    float3 nu = make_float3(degree+1, degree+1, degree+1);

    postFilterBox(pixelcount,
                  freqStepSize,
                  Xi_x, Xi_y, Xi_z, nu, 1.f,
                  d_prefilter_fft);

//    preFilterBox(pixelcount,
//                 freqStepSize,
//                 Xi_x, Xi_y, Xi_z, nu,
//                 d_prefilter_fft);
}

//#define WRITEDEBUG 1
#ifdef REFINE_MODE
float2 Reconstructor::GetDisplacement(bool MultiPeakDetection, float* CCValue)
{
	float2 shift;
	shift.x = 0;
	shift.y = 0;
	
	if (mpi_part == 0)
	{
#ifdef WRITEDEBUG
		float* test = new float[proj.GetMaxDimension() * proj.GetMaxDimension()];
#endif
		/*float* test = new float[proj.GetWidth() * proj.GetHeight()];
		proj_d.CopyDeviceToHost(test);
		
		double summe = 0;
		for (size_t i = 0; i < proj.GetWidth() * proj.GetHeight(); i++)
		{
			summe += test[i];
		}
		emwrite("testCTF2.em", test, proj.GetWidth(), proj.GetHeight());
		delete[] test;*/

		// proj_d contains the original Projection minus the proj(reconstructionWithoutSubVols)
		// make square
		cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
#ifdef WRITEDEBUG
		projSquare_d.CopyDeviceToHost(test);
		emwrite("projection3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif
        // Make mean free
		nppSafeCall(nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));
		double MeanA = 0;
		meanval.CopyDeviceToHost(&MeanA, sizeof(double));
		nppSafeCall(nppiSubC_32f_C1IR((float)(MeanA), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

		// Square, and compute the sum of the squared projection
		nppSafeCall(nppiSqr_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), (Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

		nppSafeCall(nppiSum_32f_C1R((Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));

		double SumA = 0;
		meanval.CopyDeviceToHost(&SumA, sizeof(double)); // this now contains the square counter-intuitively

        // Real-to-Complex FFT of background subtracted REAL projection
		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));
		//fourFilterKernel(fft_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension(), config.fourFilterLP, 12, config.fourFilterLPS, 4);

		//missuse ctf_d as second fft variable
		//projSubVols_d contains the projection of the model
		// Make square
		cts(projSubVols_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);

		// Make mean free
		nppSafeCall(nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));
		double MeanB = 0;
		meanval.CopyDeviceToHost(&MeanB, sizeof(double));
		nppSafeCall(nppiSubC_32f_C1IR((float)(MeanB), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

		// Square, and compute the sum of the squared projection
		nppSafeCall(nppiSqr_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), (Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

		nppSafeCall(nppiSum_32f_C1R((Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));

		double SumB = 0;
		meanval.CopyDeviceToHost(&SumB, sizeof(double));

#ifdef WRITEDEBUG
		projSquare_d.CopyDeviceToHost(test);
		emwrite("realprojection3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif
        // Real-to-Complex FFT of FAKE projection
		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)ctf_d.GetDevicePtr()));
		//fourFilterKernel(ctf_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension(), 150, 2, 20, 1);

		// Cross-correlation
		conjKernel(fft_d, ctf_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension());

		// Get CC map
		cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));
#ifdef WRITEDEBUG
		projSquare_d.CopyDeviceToHost(test);
		emwrite("cc3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif
		
		int maxShift = 10;
#ifdef REFINE_MODE
		maxShift = config.MaxShift;
#endif
		// Normalize cross correlation result
		nppSafeCall(nppiDivC_32f_C1IR((float)(proj.GetMaxDimension() * proj.GetMaxDimension() * sqrt(SumA * SumB)), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

		//printf("Divs: %f %f\n", (float)SumA, (float)SumB);


		NppiSize ccSize;
		ccSize.width = roiCC1.width;
		ccSize.height = roiCC1.height;
		nppSafeCall(nppiCopy_32f_C1R(
			(float*)((char*)projSquare_d.GetDevicePtr() + roiCC1.y * proj.GetMaxDimension() * sizeof(float) + roiCC1.x * sizeof(float)),
			proj.GetMaxDimension() * sizeof(float),
			(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC1.y * ccMap_d.GetPitch() + roiDestCC1.x * sizeof(float)),
			(int)ccMap_d.GetPitch(), ccSize));

		nppSafeCall(nppiCopy_32f_C1R(
			(float*)((char*)projSquare_d.GetDevicePtr() + roiCC2.y * proj.GetMaxDimension() * sizeof(float) + roiCC2.x * sizeof(float)),
			proj.GetMaxDimension() * sizeof(float),
			(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC2.y * ccMap_d.GetPitch() + roiDestCC2.x * sizeof(float)),
			(int)ccMap_d.GetPitch(), ccSize));

		nppSafeCall(nppiCopy_32f_C1R(
			(float*)((char*)projSquare_d.GetDevicePtr() + roiCC3.y * proj.GetMaxDimension() * sizeof(float) + roiCC3.x * sizeof(float)),
			proj.GetMaxDimension() * sizeof(float),
			(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC3.y * ccMap_d.GetPitch() + roiDestCC3.x * sizeof(float)),
			(int)ccMap_d.GetPitch(), ccSize));

		nppSafeCall(nppiCopy_32f_C1R(
			(float*)((char*)projSquare_d.GetDevicePtr() + roiCC4.y * proj.GetMaxDimension() * sizeof(float) + roiCC4.x * sizeof(float)),
			proj.GetMaxDimension() * sizeof(float),
			(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC4.y * ccMap_d.GetPitch() + roiDestCC4.x * sizeof(float)),
			(int)ccMap_d.GetPitch(), ccSize));

		ccMap_d.CopyDeviceToHost(ccMap);


		maxShiftKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension(), maxShift);

		nppSafeCall(nppiMaxIndx_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, 
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp32f*)meanval.GetDevicePtr(), (int*)stdval.GetDevicePtr(), 
			(int*)(stdval.GetDevicePtr() + sizeof(int))));


#ifdef WRITEDEBUG
		projSquare_d.CopyDeviceToHost(test);
		emwrite("shiftTest3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif

		int maxPixels[2];
		stdval.CopyDeviceToHost(maxPixels, 2 * sizeof(int));

		float maxVal;
		meanval.CopyDeviceToHost(&maxVal, sizeof(float));
		//printf("\nMaxVal: %f", maxVal);
		if (CCValue != NULL)
		{
			*CCValue = maxVal;
		}

		if (MultiPeakDetection)
		{
			//multiPeak
			nppSafeCall(nppiSet_8u_C1R(255, (Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch(), roiSquare));

			findPeakKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), badPixelMask_d, proj.GetMaxDimension(), maxVal * 0.9f);

			nppiSet_32f_C1R(1.0f, (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare);
			nppiSet_32f_C1MR(0.0f, (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, (Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch());

			maxShiftWeightedKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension(), maxShift);


			nppSafeCall(nppiMaxIndx_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
				(Npp8u*)meanbuffer.GetDevicePtr(), (Npp32f*)meanval.GetDevicePtr(), (int*)stdval.GetDevicePtr(),
				(int*)(stdval.GetDevicePtr() + sizeof(int))));

			stdval.CopyDeviceToHost(maxPixels, 2 * sizeof(int));


			//NppiSize ccSize;
			ccSize.width = roiCC1.width;
			ccSize.height = roiCC1.height;
			nppSafeCall(nppiCopy_32f_C1R(
				(float*)((char*)projSquare_d.GetDevicePtr() + roiCC1.y * proj.GetMaxDimension() * sizeof(float) + roiCC1.x * sizeof(float)),
				proj.GetMaxDimension() * sizeof(float),
				(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC1.y * ccMap_d.GetPitch() + roiDestCC1.x * sizeof(float)),
				(int)ccMap_d.GetPitch(), ccSize));

			nppSafeCall(nppiCopy_32f_C1R(
				(float*)((char*)projSquare_d.GetDevicePtr() + roiCC2.y * proj.GetMaxDimension() * sizeof(float) + roiCC2.x * sizeof(float)),
				proj.GetMaxDimension() * sizeof(float),
				(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC2.y * ccMap_d.GetPitch() + roiDestCC2.x * sizeof(float)),
				(int)ccMap_d.GetPitch(), ccSize));

			nppSafeCall(nppiCopy_32f_C1R(
				(float*)((char*)projSquare_d.GetDevicePtr() + roiCC3.y * proj.GetMaxDimension() * sizeof(float) + roiCC3.x * sizeof(float)),
				proj.GetMaxDimension() * sizeof(float),
				(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC3.y * ccMap_d.GetPitch() + roiDestCC3.x * sizeof(float)),
				(int)ccMap_d.GetPitch(), ccSize));

			nppSafeCall(nppiCopy_32f_C1R(
				(float*)((char*)projSquare_d.GetDevicePtr() + roiCC4.y * proj.GetMaxDimension() * sizeof(float) + roiCC4.x * sizeof(float)),
				proj.GetMaxDimension() * sizeof(float),
				(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC4.y * ccMap_d.GetPitch() + roiDestCC4.x * sizeof(float)),
				(int)ccMap_d.GetPitch(), ccSize));

			ccMap_d.CopyDeviceToHost(ccMapMulti);
		}

		//Get shift:
		shift.x = (float)maxPixels[0];
		shift.y = (float)maxPixels[1];

		if (shift.x > proj.GetMaxDimension() / 2)
		{
			shift.x -= proj.GetMaxDimension();
		}
		
		if (shift.y > proj.GetMaxDimension() / 2)
		{
			shift.y -= proj.GetMaxDimension();
		}

		if (maxVal <= 0)
		{
			//something went wrong, no shift found
			shift.x = -1000;
			shift.y = -1000;
		}
	}
	return shift;
}

//TODO: The output correlation values are not normalized (not in range 0 < v < 1), but this isn't strictly necessary here, so it would add useless computation. Maybe fix this later
float2 Reconstructor::GetDisplacementPC(bool MultiPeakDetection, float* CCValue)
{
    float2 shift;
    shift.x = 0;
    shift.y = 0;

    if (mpi_part == 0)
    {
#ifdef WRITEDEBUG
        float* test = new float[proj.GetMaxDimension() * proj.GetMaxDimension()];
#endif

        // proj_d contains the original Projection minus the proj(reconstructionWithoutSubVols)
        // make square
        cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
#ifdef WRITEDEBUG
        projSquare_d.CopyDeviceToHost(test);
		emwrite("projection3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif
        // Make mean free
        nppSafeCall(nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
                                     (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));
        double MeanA = 0;
        meanval.CopyDeviceToHost(&MeanA, sizeof(double));
        nppSafeCall(nppiSubC_32f_C1IR((float)(MeanA), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

        // Square, and compute the sum of the squared projection
        //nppSafeCall(nppiSqr_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), (Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

        //nppSafeCall(nppiSum_32f_C1R((Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));

        //double SumA = 0;
        //meanval.CopyDeviceToHost(&SumA, sizeof(double)); // this now contains the square counter-intuitively

        // Real-to-Complex FFT of background subtracted REAL projection
        cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

        // missuse ctf_d as second fft variable
        // projSubVols_d contains the projection of the model
        // Make square
        cts(projSubVols_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);

        // Make mean free
        nppSafeCall(nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
                                     (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));
        double MeanB = 0;
        meanval.CopyDeviceToHost(&MeanB, sizeof(double));
        nppSafeCall(nppiSubC_32f_C1IR((float)(MeanB), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

        // Square, and compute the sum of the squared projection
        // nppSafeCall(nppiSqr_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), (Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

        // nppSafeCall(nppiSum_32f_C1R((Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));

        //double SumB = 0;
        //meanval.CopyDeviceToHost(&SumB, sizeof(double));

#ifdef WRITEDEBUG
        projSquare_d.CopyDeviceToHost(test);
		emwrite("realprojection3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif
        // Real-to-Complex FFT of FAKE projection
        cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)ctf_d.GetDevicePtr()));
        //fourFilterKernel(ctf_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension(), 150, 2, 20, 1);

        // Phase-correlation
        pcKernel(fft_d, ctf_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension());

        //Cuda::CudaDeviceVariable& img, size_t stride, int pixelcount, float lp, float hp, float lps, float hps
        fourFilterKernel(fft_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension(), config.PhaseCorrSigma, 0, config.PhaseCorrSigma, 0);

        // Get CC map (transform back)
        cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));
#ifdef WRITEDEBUG
        projSquare_d.CopyDeviceToHost(test);
		emwrite("cc3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif

        int maxShift = 10;
#ifdef REFINE_MODE
        maxShift = config.MaxShift;
#endif
        // Normalize cross correlation result
        //nppSafeCall(nppiDivC_32f_C1IR((float)(proj.GetMaxDimension() * proj.GetMaxDimension() * sqrt(SumA * SumB)), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

        //printf("Divs: %f %f\n", (float)SumA, (float)SumB);

        // FFT-shift using NPPI
        NppiSize ccSize;
        ccSize.width = roiCC1.width;
        ccSize.height = roiCC1.height;
        nppSafeCall(nppiCopy_32f_C1R(
                (float*)((char*)projSquare_d.GetDevicePtr() + roiCC1.y * proj.GetMaxDimension() * sizeof(float) + roiCC1.x * sizeof(float)),
                proj.GetMaxDimension() * sizeof(float),
                (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC1.y * ccMap_d.GetPitch() + roiDestCC1.x * sizeof(float)),
                (int)ccMap_d.GetPitch(), ccSize));

        nppSafeCall(nppiCopy_32f_C1R(
                (float*)((char*)projSquare_d.GetDevicePtr() + roiCC2.y * proj.GetMaxDimension() * sizeof(float) + roiCC2.x * sizeof(float)),
                proj.GetMaxDimension() * sizeof(float),
                (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC2.y * ccMap_d.GetPitch() + roiDestCC2.x * sizeof(float)),
                (int)ccMap_d.GetPitch(), ccSize));

        nppSafeCall(nppiCopy_32f_C1R(
                (float*)((char*)projSquare_d.GetDevicePtr() + roiCC3.y * proj.GetMaxDimension() * sizeof(float) + roiCC3.x * sizeof(float)),
                proj.GetMaxDimension() * sizeof(float),
                (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC3.y * ccMap_d.GetPitch() + roiDestCC3.x * sizeof(float)),
                (int)ccMap_d.GetPitch(), ccSize));

        nppSafeCall(nppiCopy_32f_C1R(
                (float*)((char*)projSquare_d.GetDevicePtr() + roiCC4.y * proj.GetMaxDimension() * sizeof(float) + roiCC4.x * sizeof(float)),
                proj.GetMaxDimension() * sizeof(float),
                (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC4.y * ccMap_d.GetPitch() + roiDestCC4.x * sizeof(float)),
                (int)ccMap_d.GetPitch(), ccSize));

        ccMap_d.CopyDeviceToHost(ccMap);


        maxShiftKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension(), maxShift);

        nppSafeCall(nppiMaxIndx_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
                                        (Npp8u*)meanbuffer.GetDevicePtr(), (Npp32f*)meanval.GetDevicePtr(), (int*)stdval.GetDevicePtr(),
                                        (int*)(stdval.GetDevicePtr() + sizeof(int))));


#ifdef WRITEDEBUG
        projSquare_d.CopyDeviceToHost(test);
		emwrite("shiftTest3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif

        int maxPixels[2];
        stdval.CopyDeviceToHost(maxPixels, 2 * sizeof(int));

        float maxVal;
        meanval.CopyDeviceToHost(&maxVal, sizeof(float));
        //printf("\nMaxVal: %f", maxVal);
        if (CCValue != NULL)
        {
            *CCValue = maxVal;
        }

        if (MultiPeakDetection)
        {
            //multiPeak
            nppSafeCall(nppiSet_8u_C1R(255, (Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch(), roiSquare));

            findPeakKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), badPixelMask_d, proj.GetMaxDimension(), maxVal * 0.9f);

            nppiSet_32f_C1R(1.0f, (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare);
            nppiSet_32f_C1MR(0.0f, (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, (Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch());

            maxShiftWeightedKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension(), maxShift);


            nppSafeCall(nppiMaxIndx_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
                                            (Npp8u*)meanbuffer.GetDevicePtr(), (Npp32f*)meanval.GetDevicePtr(), (int*)stdval.GetDevicePtr(),
                                            (int*)(stdval.GetDevicePtr() + sizeof(int))));

            stdval.CopyDeviceToHost(maxPixels, 2 * sizeof(int));


            //NppiSize ccSize;
            ccSize.width = roiCC1.width;
            ccSize.height = roiCC1.height;
            nppSafeCall(nppiCopy_32f_C1R(
                    (float*)((char*)projSquare_d.GetDevicePtr() + roiCC1.y * proj.GetMaxDimension() * sizeof(float) + roiCC1.x * sizeof(float)),
                    proj.GetMaxDimension() * sizeof(float),
                    (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC1.y * ccMap_d.GetPitch() + roiDestCC1.x * sizeof(float)),
                    (int)ccMap_d.GetPitch(), ccSize));

            nppSafeCall(nppiCopy_32f_C1R(
                    (float*)((char*)projSquare_d.GetDevicePtr() + roiCC2.y * proj.GetMaxDimension() * sizeof(float) + roiCC2.x * sizeof(float)),
                    proj.GetMaxDimension() * sizeof(float),
                    (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC2.y * ccMap_d.GetPitch() + roiDestCC2.x * sizeof(float)),
                    (int)ccMap_d.GetPitch(), ccSize));

            nppSafeCall(nppiCopy_32f_C1R(
                    (float*)((char*)projSquare_d.GetDevicePtr() + roiCC3.y * proj.GetMaxDimension() * sizeof(float) + roiCC3.x * sizeof(float)),
                    proj.GetMaxDimension() * sizeof(float),
                    (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC3.y * ccMap_d.GetPitch() + roiDestCC3.x * sizeof(float)),
                    (int)ccMap_d.GetPitch(), ccSize));

            nppSafeCall(nppiCopy_32f_C1R(
                    (float*)((char*)projSquare_d.GetDevicePtr() + roiCC4.y * proj.GetMaxDimension() * sizeof(float) + roiCC4.x * sizeof(float)),
                    proj.GetMaxDimension() * sizeof(float),
                    (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC4.y * ccMap_d.GetPitch() + roiDestCC4.x * sizeof(float)),
                    (int)ccMap_d.GetPitch(), ccSize));

            ccMap_d.CopyDeviceToHost(ccMapMulti);
        }

        //Get shift:
        shift.x = (float)maxPixels[0];
        shift.y = (float)maxPixels[1];

        if (shift.x > proj.GetMaxDimension() / 2)
        {
            shift.x -= proj.GetMaxDimension();
        }

        if (shift.y > proj.GetMaxDimension() / 2)
        {
            shift.y -= proj.GetMaxDimension();
        }

        if (maxVal <= 0)
        {
            //something went wrong, no shift found
            shift.x = -1000;
            shift.y = -1000;
        }
    }
    return shift;
}

void Reconstructor::rotVol(Cuda::CudaDeviceVariable & vol, float phi, float psi, float theta)
{
	rotKernel(vol, phi, psi, theta);
}

void Reconstructor::setRotVolData(float * data)
{
	rotKernel.SetData(data);
}
float * Reconstructor::GetCCMap()
{
	return ccMap;
}
float * Reconstructor::GetCCMapMulti()
{
	return ccMapMulti;
}

void Reconstructor::GetCroppedProjection(float *outImage, int2 roiMin, int2 roiMax) {

    int outW = roiMax.x-roiMin.x + 1;
    int outH = roiMax.y-roiMin.y + 1;
    //printf("outW: %i outH: %i \n", outW, outH);
    memset(outImage, 0, outW*outH*sizeof(float));

    auto buffer = new float[proj.GetHeight()*proj.GetWidth()];
    proj_d.CopyDeviceToHost(buffer);

    //stringstream ss;
    //ss << "projjjjjj.em";
    //emwrite(ss.str(), buffer, proj.GetWidth(), proj.GetHeight());

    for (int x = roiMin.x; x < roiMax.x+1; x++){
        for (int y = roiMin.y; y < roiMax.y+1; y++){
            if(x > proj.GetWidth()-1) continue;
            if(y > proj.GetHeight()-1) continue;

            if(x < 0) continue;
            if(y < 0) continue;

            int xx = x-roiMin.x;
            int yy = y-roiMin.y;

            outImage[xx+outW*yy] = buffer[x+proj.GetWidth()*y];
            //printf("%s", typeid(buffer).name());
            //printf("xx: %i yy: %i x: %i y: %i buffer: %f out: %f\n", xx, yy, x, y, buffer[y+proj.GetHeight()*x], outImage[yy+outH*xx]);
        }
    }

    delete[] buffer;
}

void Reconstructor::GetCroppedProjection(float *outImage, float *inImage, int2 roiMin, int2 roiMax) {

    int outW = roiMax.x-roiMin.x + 1;
    int outH = roiMax.y-roiMin.y + 1;
    //printf("outW: %i outH: %i \n", outW, outH);
    memset(outImage, 0, outW*outH*sizeof(float));

    //auto buffer = new float[proj.GetHeight()*proj.GetWidth()];
    //proj_d.CopyDeviceToHost(buffer);

    //stringstream ss;
    //ss << "projjjjjj.em";
    //emwrite(ss.str(), buffer, proj.GetWidth(), proj.GetHeight());

    for (int x = roiMin.x; x < roiMax.x+1; x++){
        for (int y = roiMin.y; y < roiMax.y+1; y++){
            if(x > proj.GetWidth()-1) continue;
            if(y > proj.GetHeight()-1) continue;

            if(x < 0) continue;
            if(y < 0) continue;

            int xx = x-roiMin.x;
            int yy = y-roiMin.y;

            outImage[xx+outW*yy] = inImage[x+proj.GetWidth()*y];
            //printf("%s", typeid(buffer).name());
            //printf("xx: %i yy: %i x: %i y: %i buffer: %f out: %f\n", xx, yy, x, y, buffer[y+proj.GetHeight()*x], outImage[yy+outH*xx]);
        }
    }
}

#endif

