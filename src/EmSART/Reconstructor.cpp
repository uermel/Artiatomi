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
#include "CudaKernelBinarys.h"
#include <typeinfo>

using namespace std;
using namespace Cuda;

KernelModuls::KernelModuls(Cuda::CudaContext* aCuCtx)
	:compilerOutput(false),
	infoOutput(false)
{
	modFP = aCuCtx->LoadModulePTX(KernelForwardProjectionRayMarcher_TL, 0, infoOutput, compilerOutput);
	modSlicer = aCuCtx->LoadModulePTX(KernelForwardProjectionSlicer, 0, infoOutput, compilerOutput);
	modVolTravLen = modSlicer;
	modComp = aCuCtx->LoadModulePTX(KernelCompare, 0, infoOutput, compilerOutput);
	modWBP = aCuCtx->LoadModulePTX(KernelWbpWeighting, 0, infoOutput, compilerOutput);
	modBP = aCuCtx->LoadModulePTX(KernelBackProjectionOS, 0, infoOutput, compilerOutput);
	modCTF = aCuCtx->LoadModulePTX(Kernelctf, 0, infoOutput, compilerOutput);
	modCTS = aCuCtx->LoadModulePTX(KernelCopyToSquare, 0, infoOutput, compilerOutput);
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
    //erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
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
//	printf("PosInVol2: %f, %f, %f\n", (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f + vol->GetVoxelSize().x * 0.5f)),
//		(MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f + vol->GetVoxelSize().x * 0.5f)),
//		(MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f + vol->GetVoxelSize().x * 0.5f)));

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

	//printf("t_in: %f; t_out: %f\n", t_in, t_out);
	//t_in = 2*-DIST;
	//t_out = 2*DIST;

	//for (int x = 0; x <= 1; x++)
	//	for (int y = 0; y <= 1; y++)
	//		for (int z = 0; z <= 1; z++)
	//		{
	//			//float t;

	//			t = (nvec.x * (MC_bBoxMin.x + x * (MC_bBoxMax.x - MC_bBoxMin.x))
	//				+ nvec.y * (MC_bBoxMin.y + y * (MC_bBoxMax.y - MC_bBoxMin.y))
	//				+ nvec.z * (MC_bBoxMin.z + z * (MC_bBoxMax.z - MC_bBoxMin.z)));
	//			t += (-nvec.x * pos2.x - nvec.y * pos2.y - nvec.z * pos2.z);

	//			if (t < t_in) t_in = t;
	//			if (t > t_out) t_out = t;
	//		}
	////printf("t_in: %f; t_out: %f\n", t_in, t_out);



	//{
	//	float xAniso = 2366.25f;
	//	float yAniso = 4527.75f;

	//	float3 c_source = c_detektor;
	//	float3 c_uPitch = proj.GetPixelUPitch(index);
	//	float3 c_vPitch = proj.GetPixelVPitch(index);
	//	c_source = c_source + (xAniso)* c_uPitch;
	//	c_source = c_source + (yAniso)* c_vPitch;

	//	//////////// BOX INTERSECTION (partial Volume) /////////////////
	//	float3 tEntry;
	//	tEntry.x = (MC_bBoxMin.x - c_source.x) / (c_projNorm.x);
	//	tEntry.y = (MC_bBoxMin.y - c_source.y) / (c_projNorm.y);
	//	tEntry.z = (MC_bBoxMin.z - c_source.z) / (c_projNorm.z);

	//	float3 tExit;
	//	tExit.x = (MC_bBoxMax.x - c_source.x) / (c_projNorm.x);
	//	tExit.y = (MC_bBoxMax.y - c_source.y) / (c_projNorm.y);
	//	tExit.z = (MC_bBoxMax.z - c_source.z) / (c_projNorm.z);


	//	float3 tmin = fminf(tEntry, tExit);
	//	float3 tmax = fmaxf(tEntry, tExit);

	//	t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	//	t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
	//	printf("t_in: %f; t_out: %f\n", t_in, t_out);
	//}
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
	Projection & aProj, ProjectionSource* aProjectionSource,
	MarkerFile& aMarkers, CtfFile& aDefocus, KernelModuls& modules, int aMpi_part, int aMpi_size)
	: 
	proj(aProj), 
	projSource(aProjectionSource), 
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
	bpKernel(modules.modBP, aConfig.FP16Volume),
	convVolKernel(modules.modBP),
	convVol3DKernel(modules.modBP),
	ctf(modules.modCTF),
	cts(modules.modCTS),
	dimBordersKernel(modules.modComp),
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
	//Set kernel work dimensions for 2D images:
	fpKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);
	slicerKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);
	volTravLenKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);
	compKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);
	subEKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);
	cropKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);

	wbp.SetComputeSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
	ctf.SetComputeSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
	fourFilterKernel.SetComputeSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
	doseWeightingKernel.SetComputeSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
	conjKernel.SetComputeSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
    pcKernel.SetComputeSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
	cts.SetComputeSize(proj.GetMaxDimension(), proj.GetMaxDimension(), 1);
	maxShiftKernel.SetComputeSize(proj.GetMaxDimension(), proj.GetMaxDimension(), 1);
	convVolKernel.SetComputeSize(config.RecDimensions.x, config.RecDimensions.y, 1);
	dimBordersKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);

	//Alloc device variables
	realprojUS_d.Alloc(proj.GetWidth() * sizeof(int), proj.GetHeight(), sizeof(int));
	proj_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));
	realproj_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));
	dist_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));
	filterImage_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));

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

	//Bind back projection image to texref in BP Kernel
	if (aConfig.CtfMode == Configuration::Config::CTFM_YES)
	{
		texImage.Bind(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &dist_d, CU_AD_FORMAT_FLOAT, 1);
		//CudaTextureLinearPitched2D::Bind(&bpKernel, "tex", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
		//	CU_TR_FILTER_MODE_LINEAR, 0, &dist_d, CU_AD_FORMAT_FLOAT, 1);
	}
	else
	{
		texImage.Bind(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &proj_d, CU_AD_FORMAT_FLOAT, 1);
		//CudaTextureLinearPitched2D::Bind(&bpKernel, "tex", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
		//	CU_TR_FILTER_MODE_POINT, 0, &proj_d, CU_AD_FORMAT_FLOAT, 1);
	}

	ctf_d.Alloc((proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension(), sizeof(cuComplex));
	fft_d.Alloc((proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex) * proj.GetMaxDimension());
	projSquare_d.Alloc(proj.GetMaxDimension() * sizeof(float) * proj.GetMaxDimension());
	badPixelMask_d.Alloc(proj.GetMaxDimension() * sizeof(char), proj.GetMaxDimension(), 4 * sizeof(char));

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

	size_t free = 0;
	size_t total = 0;
	cudaMemGetInfo(&free, &total);
    printf("before crash: free: %zu total: %zu", free, total);

	cufftSafeCall(cufftPlan2d(&handleR2C, proj.GetMaxDimension(), proj.GetMaxDimension(), CUFFT_R2C));
	cufftSafeCall(cufftPlan2d(&handleC2R, proj.GetMaxDimension(), proj.GetMaxDimension(), CUFFT_C2R));

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

	cufftSafeCall(cufftDestroy(handleR2C));
	cufftSafeCall(cufftDestroy(handleC2R));
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

template<class TVol>
void Reconstructor::ForwardProjectionNoCTF(Volume<TVol>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync)
{	
	//proj_d.Memset(0);
	//dist_d.Memset(0);
	float runtime;

	if (!volumeIsEmpty)
	{
		SetConstantValues(fpKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
		runtime = fpKernel(proj.GetWidth(), proj.GetHeight(), proj_d, dist_d, texVol);

#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)proj_d.GetDevicePtr(), (int)proj_d.GetPitch(), roiAll));
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
#endif

				// To avoid aliasing artifacts low pass filter to Nyquist of tomogram or Projection fourier filter, whichever is lower.
				if ((config.VoxelSize.x > 1) && (config.LimitToNyquist)) // assume cubic voxel sizes
				{
					int2 pA, pB, pC, pD;
					pA.x = 0;
					pA.y = proj.GetHeight() - 1;
					pB.x = proj.GetWidth() - 1;
					pB.y = proj.GetHeight() - 1;
					pC.x = 0;
					pC.y = 0;
					pD.x = proj.GetWidth() - 1;
					pD.y = 0;

					cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);

					cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);

					fft_d.Memset(0);
					cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

					// Set the low pass filter to nyquist of the Tomogram
					float lp = (float)proj.GetMaxDimension() / config.VoxelSize.x - 20.f;
					float lps = 20.f;
					// Use the low pass from the config if it's lower and filter is not skipped
					if (!config.SkipFilter) {
                        if (lp > (float) config.fourFilterLP) {
                            lp = (float) config.fourFilterLP;
                            lps = (float) config.fourFilterLPS;
                        }
                    }

					int size = proj.GetMaxDimension();

					fourFilterKernel(fft_d, roiFFT.width * sizeof(Npp32fc), size, lp, 0, lps, 0);


					cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));

					nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), (float)(proj.GetMaxDimension() * proj.GetMaxDimension()),
						(Npp32f*)proj_d.GetDevicePtr(), (int)proj_d.GetPitch(), roiAll));


					cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
				}
#ifdef USE_MPI
			}
			else
			{
				proj_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif
	}
	else
	{
		SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

		runtime = volTravLenKernel(proj.GetWidth(), proj.GetHeight(), dist_d);
#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
			}
			else
			{
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif
	}	
}
template void Reconstructor::ForwardProjectionNoCTF(Volume<unsigned short>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);
template void Reconstructor::ForwardProjectionNoCTF(Volume<float>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);


template<class TVol>
void Reconstructor::ForwardProjectionCTF(Volume<TVol>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync)
{
	//proj_d.Memset(0);
	//dist_d.Memset(0);
	float runtime;
	int x = proj.GetWidth();
	int y = proj.GetHeight();

	//if (mpi_part == 0)
	//	printf("\n");

	if (!volumeIsEmpty)
	{
		//Forward projection is not done in WBP --> no need to adapt magAnisotropy
		SetConstantValues(slicerKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
		SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

		float t_in, t_out;
		GetDefocusDistances(t_in, t_out, index, vol);

		for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize())
		{
			dist_d.Memset(0);

			float defocusAngle = defocus.GetAstigmatismAngle(index) + (float)(proj.GetImageRotationToCompensate((uint)index) / M_PI * 180.0);
			float defocusMin;
			float defocusMax;
			GetDefocusMinMax(ray, index, defocusMin, defocusMax);

			if (mpi_part == 0)
			{
				printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b FP Defocus: %-8d nm", (int)defocusMin);
				fflush(stdout);
			}
			//printf("\n");
			runtime = slicerKernel(x, y, dist_d, ray, ray + config.CTFSliceThickness / proj.GetPixelSize(), texVol);

#ifdef USE_MPI
			if (!noSync)
			{
				if (mpi_part == 0)
				{
					for (int mpi = 1; mpi < mpi_size; mpi++)
					{
						MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						realprojUS_d.CopyHostToDevice(MPIBuffer);
						nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
					}
				}
				else
				{
					dist_d.CopyDeviceToHost(MPIBuffer);
					MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
				}
			}
#endif
			//CTF filtering is only done on GPU 0!
			if (mpi_part == 0)
			{
				/*dist_d.CopyDeviceToHost(MPIBuffer);
				writeBMP(string("VorCTF.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/

				int2 pA, pB, pC, pD;
				pA.x = 0;
				pA.y = proj.GetHeight() - 1;
				pB.x = proj.GetWidth() - 1;
				pB.y = proj.GetHeight() - 1;
				pC.x = 0;
				pC.y = 0;
				pD.x = proj.GetWidth() - 1;
				pD.y = 0;

				cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);

				cts(dist_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);

				fft_d.Memset(0);
				cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

				ctf(fft_d, defocusMin, defocusMax, defocusAngle, true, config.PhaseFlipOnly, config.WienerFilterNoiseLevel, (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), config.CTFBetaFac);

                // To avoid aliasing artifacts low pass filter to Nyquist of Tomogram or Projection fourier filter, whichever is lower.
				if ((config.VoxelSize.x > 1) && (config.LimitToNyquist)) // assume cubic voxel sizes
				{
                    // Set the low pass filter to nyquist of the Tomogram
                    float lp = (float)proj.GetMaxDimension() / config.VoxelSize.x - 20.f;
                    float lps = 20.f;
                    // Use the low pass from the config if it's lower
                    if (!config.SkipFilter) {
                        if (lp > (float) config.fourFilterLP) {
                            lp = (float) config.fourFilterLP;
                            lps = (float) config.fourFilterLPS;
                        }
                    }

					int size = proj.GetMaxDimension();

					fourFilterKernel(fft_d, roiFFT.width * sizeof(Npp32fc), size, lp, 0, lps, 0);
				}

				cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));

				nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), (float)(proj.GetMaxDimension() * proj.GetMaxDimension()),
					(Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));


				cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
				nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), (Npp32f*)proj_d.GetDevicePtr(), (int)proj_d.GetPitch(), roiAll));
			}
		}
		/*proj_d.CopyDeviceToHost(MPIBuffer);
		writeBMP(string("Proj.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
		//Get Volume traversal lengths
		dist_d.Memset(0);
		runtime = volTravLenKernel(x, y, dist_d);

#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
				/*proj_d.CopyDeviceToHost(MPIBuffer);
				printf("\n");
				writeBMP(string("proj3.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());
				printf("\n");
				dist_d.CopyDeviceToHost(MPIBuffer);
				writeBMP(string("dist.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
			}
			else
			{
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif

	}
	else
	{
		//Forward projection is not done in WBP --> no need to adapt magAnisotropy
		SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

		runtime = volTravLenKernel(proj.GetWidth(), proj.GetHeight(), dist_d);
#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
			}
			else
			{
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif
	}
}
template void Reconstructor::ForwardProjectionCTF(Volume<unsigned short>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);
template void Reconstructor::ForwardProjectionCTF(Volume<float>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);

template<class TVol>
void Reconstructor::ForwardProjectionNoCTFROI(Volume<TVol>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync)
{	
	//proj_d.Memset(0);
	//dist_d.Memset(0);
	float runtime;

	if (!volumeIsEmpty)
	{
		//Forward projection is not done in WBP --> no need to adapt magAnisotropy
		SetConstantValues(fpKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
		runtime = fpKernel(proj.GetWidth(), proj.GetHeight(), proj_d, dist_d, texVol, roiMin, roiMax);

#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)proj_d.GetDevicePtr(), (int)proj_d.GetPitch(), roiAll));
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}

#endif
                // To avoid aliasing artifacts low pass filter to Nyquist of Tomogram or Projection fourier filter, whichever is lower.
				if ((config.VoxelSize.x > 1) && (config.LimitToNyquist)) // assume cubic voxel sizes
				{
					int2 pA, pB, pC, pD;
					pA.x = 0;
					pA.y = proj.GetHeight() - 1;
					pB.x = proj.GetWidth() - 1;
					pB.y = proj.GetHeight() - 1;
					pC.x = 0;
					pC.y = 0;
					pD.x = proj.GetWidth() - 1;
					pD.y = 0;

					cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);

					cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);

					fft_d.Memset(0);
					cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

                    // Set the low pass filter to nyquist of the Tomogram
                    float lp = (float)proj.GetMaxDimension() / config.VoxelSize.x - 20.f;
                    float lps = 20.f;
                    // Use the low pass from the config if it's lower
                    if (!config.SkipFilter) {
                        if (lp > (float) config.fourFilterLP) {
                            lp = (float) config.fourFilterLP;
                            lps = (float) config.fourFilterLPS;
                        }
                    }

					int size = proj.GetMaxDimension();

					fourFilterKernel(fft_d, roiFFT.width * sizeof(Npp32fc), size, lp, 0, lps, 0);


					cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));

					nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), (float)(proj.GetMaxDimension() * proj.GetMaxDimension()),
						(Npp32f*)proj_d.GetDevicePtr(), (int)proj_d.GetPitch(), roiAll));


					cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
				}
#ifdef USE_MPI
			}
			else
			{
				proj_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif
	}
	else
	{
		//Forward projection is not done in WBP --> no need to adapt magAnisotropy
		SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

		runtime = volTravLenKernel(proj.GetWidth(), proj.GetHeight(), dist_d, roiMin, roiMax);
#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
			}
			else
			{
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif
	}	
}
template void Reconstructor::ForwardProjectionNoCTFROI(Volume<unsigned short>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
template void Reconstructor::ForwardProjectionNoCTFROI(Volume<float>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);


template<class TVol>
void Reconstructor::ForwardProjectionCTFROI(Volume<TVol>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync)
{
	//proj_d.Memset(0);
	//dist_d.Memset(0);
	float runtime;
	int x = proj.GetWidth();
	int y = proj.GetHeight();

	//if (mpi_part == 0)
	//	printf("\n");

	if (!volumeIsEmpty)
	{
		//Forward projection is not done in WBP --> no need to adapt magAnisotropy
		SetConstantValues(slicerKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
		SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

		float t_in, t_out;
		GetDefocusDistances(t_in, t_out, index, vol);
		/*t_in -= 2*config.CTFSliceThickness / proj.GetPixelSize();
		t_out += 2*config.CTFSliceThickness / proj.GetPixelSize();*/

		for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize())
		{
			dist_d.Memset(0);

			float defocusAngle = defocus.GetAstigmatismAngle(index) + (float)(proj.GetImageRotationToCompensate((uint)index) / M_PI * 180.0);
			float defocusMin;
			float defocusMax;
			GetDefocusMinMax(ray, index, defocusMin, defocusMax);

			if (mpi_part == 0)
			{
				printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b FP Defocus: %-8d nm", (int)defocusMin);
				fflush(stdout);
			}
			//printf("\n");
			runtime = slicerKernel(x, y, dist_d, ray, ray + config.CTFSliceThickness / proj.GetPixelSize(), texVol, roiMin, roiMax);

#ifdef USE_MPI
			if (!noSync)
			{
				if (mpi_part == 0)
				{
					for (int mpi = 1; mpi < mpi_size; mpi++)
					{
						MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						realprojUS_d.CopyHostToDevice(MPIBuffer);
						nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
					}
				}
				else
				{
					dist_d.CopyDeviceToHost(MPIBuffer);
					MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
				}
			}
#endif
			//CTF filtering is only done on GPU 0!
			if (mpi_part == 0)
			{
				/*dist_d.CopyDeviceToHost(MPIBuffer);
				writeBMP(string("VorCTF.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/

				int2 pA, pB, pC, pD;
				pA.x = 0;
				pA.y = proj.GetHeight() - 1;
				pB.x = proj.GetWidth() - 1;
				pB.y = proj.GetHeight() - 1;
				pC.x = 0;
				pC.y = 0;
				pD.x = proj.GetWidth() - 1;
				pD.y = 0;

				cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);

				cts(dist_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);

				fft_d.Memset(0);
				cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

				ctf(fft_d, defocusMin, defocusMax, defocusAngle, true, config.PhaseFlipOnly, config.WienerFilterNoiseLevel, (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), config.CTFBetaFac);

                // To avoid aliasing artifacts low pass filter to Nyquist of Tomogram or Projection fourier filter, whichever is lower.
				if ((config.VoxelSize.x > 1) && (config.LimitToNyquist)) // assume cubic voxel sizes
				{
                    // Set the low pass filter to nyquist of the Tomogram
                    float lp = (float)proj.GetMaxDimension() / config.VoxelSize.x - 20.f;
                    float lps = 20.f;
                    // Use the low pass from the config if it's lower
                    if (!config.SkipFilter) {
                        if (lp > (float) config.fourFilterLP) {
                            lp = (float) config.fourFilterLP;
                            lps = (float) config.fourFilterLPS;
                        }
                    }

					int size = proj.GetMaxDimension();

					fourFilterKernel(fft_d, roiFFT.width * sizeof(Npp32fc), size, lp, 0, lps, 0);
				}

				cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));

				nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), (float)(proj.GetMaxDimension() * proj.GetMaxDimension()),
					(Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));


				cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
				nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), (Npp32f*)proj_d.GetDevicePtr(), (int)proj_d.GetPitch(), roiAll));
			}
		}
		/*proj_d.CopyDeviceToHost(MPIBuffer);
		writeBMP(string("Proj.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
		//Get Volume traversal lengths
		dist_d.Memset(0);
		runtime = volTravLenKernel(x, y, dist_d, roiMin, roiMax);

#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
				/*proj_d.CopyDeviceToHost(MPIBuffer);
				printf("\n");
				writeBMP(string("proj3.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());
				printf("\n");
				dist_d.CopyDeviceToHost(MPIBuffer);
				writeBMP(string("dist.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
			}
			else
			{
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif

	}
	else
	{
		//Forward projection is not done in WBP --> no need to adapt magAnisotropy
		SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

		runtime = volTravLenKernel(proj.GetWidth(), proj.GetHeight(), dist_d, roiMin, roiMax);
#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
			}
			else
			{
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif
	}
}
template void Reconstructor::ForwardProjectionCTFROI(Volume<unsigned short>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
template void Reconstructor::ForwardProjectionCTFROI(Volume<float>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);

template<class TVol>
void Reconstructor::BackProjectionNoCTF(Volume<TVol>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount)
{
	float3 volDim = vol->GetSubVolumeDimension(mpi_part);
	bpKernel.SetComputeSize((int)volDim.x, (int)volDim.y, (int)volDim.z);

	if (config.WBP_NoSART)
	{
		magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
		magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
	}

    // Find area shaded by volume, cut and dim borders
    int2 pA, pB, pC, pD;
    float2 hitA, hitB, hitC, hitD;
    proj.ComputeHitPoints(*vol, proj_index, pA, pB, pC, pD);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pA.x, (float)pA.y, hitA.x, hitA.y);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pB.x, (float)pB.y, hitB.x, hitB.y);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pC.x, (float)pC.y, hitC.x, hitC.y);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pD.x, (float)pD.y, hitD.x, hitD.y);
    pA.x = (int)hitA.x; pA.y = (int)hitA.y;
    pB.x = (int)hitB.x; pB.y = (int)hitB.y;
    pC.x = (int)hitC.x; pC.y = (int)hitC.y;
    pD.x = (int)hitD.x; pD.y = (int)hitD.y;
	cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
	
	// Prepare and execute Backprojection
	SetConstantValues(bpKernel, *vol, proj, proj_index, mpi_part, magAnisotropy, magAnisotropyInv);

	float runtime = bpKernel(proj.GetWidth(), proj.GetHeight(), config.Lambda / SIRTCount, 
		config.OverSampling, 1.0f / (float)(config.OverSampling), texImage, surface, 0, 9999999999999.0f);

}
template void Reconstructor::BackProjectionNoCTF(Volume<unsigned short>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);
template void Reconstructor::BackProjectionNoCTF(Volume<float>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);


#ifdef SUBVOLREC_MODE
template<class TVol>
void Reconstructor::BackProjectionNoCTF(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index)
{
	size_t batchSize = subVolumes.size();

	if (config.WBP_NoSART)
	{
		magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
		magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
	}

	for (size_t batch = 0; batch < batchSize; batch++)
	{
		//bind surfref to correct array:
		//cudaSafeCall(cuSurfRefSetArray(surfref, vecArrays[batch]->GetCUarray(), 0));
		CudaSurfaceObject3D surface(vecArrays[batch]);

		//set additional shifts:
		proj.SetExtraShift(proj_index, vecExtraShifts[batch]);

		float3 volDim = subVolumes[batch]->GetSubVolumeDimension(0);
		bpKernel.SetComputeSize((int)volDim.x, (int)volDim.y, (int)volDim.z);
		
		SetConstantValues(bpKernel, *(subVolumes[batch]), proj, proj_index, 0, magAnisotropy, magAnisotropyInv);

		float runtime = bpKernel(proj.GetWidth(), proj.GetHeight(), 1.0f,
			config.OverSampling, 1.0f / (float)(config.OverSampling), texImage, surface, 0, 9999999999999.0f);
	}
}
template void Reconstructor::BackProjectionNoCTF(Volume<unsigned short>* vol, vector<Volume<unsigned short>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index);
template void Reconstructor::BackProjectionNoCTF(Volume<float>* vol, vector<Volume<float>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index);
#endif

template<class TVol>
void Reconstructor::BackProjectionCTF(Volume<TVol>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTcount)
{
	float3 volDim = vol->GetSubVolumeDimension(mpi_part);
	bpKernel.SetComputeSize((int)volDim.x, (int)volDim.y, (int)volDim.z);
	float runtime;
	int x = proj.GetWidth();
	int y = proj.GetHeight();

	if (config.WBP_NoSART)
	{
		magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
		magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
	}

	if (mpi_part == 0)
		printf("\n");

	float t_in, t_out;
	GetDefocusDistances(t_in, t_out, proj_index, vol);

	// Find area shaded by volume, cut and dim borders
    int2 pA, pB, pC, pD;
    float2 hitA, hitB, hitC, hitD;
    proj.ComputeHitPoints(*vol, proj_index, pA, pB, pC, pD);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pA.x, (float)pA.y, hitA.x, hitA.y);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pB.x, (float)pB.y, hitB.x, hitB.y);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pC.x, (float)pC.y, hitC.x, hitC.y);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pD.x, (float)pD.y, hitD.x, hitD.y);
    pA.x = (int)hitA.x; pA.y = (int)hitA.y;
    pB.x = (int)hitB.x; pB.y = (int)hitB.y;
    pC.x = (int)hitC.x; pC.y = (int)hitC.y;
    pD.x = (int)hitD.x; pD.y = (int)hitD.y;
    cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);


	for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize())
	{
		SetConstantValues(bpKernel, *vol, proj, proj_index, mpi_part, magAnisotropy, magAnisotropyInv);


		float defocusAngle = defocus.GetAstigmatismAngle(proj_index) + (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0);
		float defocusMin;
		float defocusMax;
		GetDefocusMinMax(ray, proj_index, defocusMin, defocusMax);
		float tiltAngle = (markers(MFI_TiltAngle, proj_index, 0) + config.AddTiltAngle) / 180.0f * (float)M_PI;
		
		if (mpi_part == 0)
		{
			printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b BP Defocus: %-8d nm", (int)defocusMin);
			fflush(stdout);
		}

		cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);
		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));
		
		ctf(fft_d, defocusMin, defocusMax, defocusAngle, false, config.PhaseFlipOnly, config.WienerFilterNoiseLevel, (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), config.CTFBetaFac);

		cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));
		nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), (float)(proj.GetMaxDimension() * proj.GetMaxDimension()),
			(Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));

		cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);

		runtime = bpKernel(x, y, config.Lambda / SIRTcount, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage, surface, ray, ray + config.CTFSliceThickness / proj.GetPixelSize());
	}
}
template void Reconstructor::BackProjectionCTF(Volume<unsigned short>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);
template void Reconstructor::BackProjectionCTF(Volume<float>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);

#ifdef SUBVOLREC_MODE
template<class TVol>
void Reconstructor::BackProjectionCTF(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index)
{
	size_t batchSize = subVolumes.size();
	//TODO
	//float3 volDim = vol->GetSubVolumeDimension(mpi_part);
	bpKernel.SetComputeSize(config.SizeSubVol, config.SizeSubVol, config.SizeSubVol);
	float runtime;
	int x = proj.GetWidth();
	int y = proj.GetHeight();

	if (config.WBP_NoSART)
	{
		magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
		magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
	}

	if (mpi_part == 0)
		printf("\n");

	float t_in, t_out;
	GetDefocusDistances(t_in, t_out, proj_index, vol);

	for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize())
	{
		float defocusAngle = defocus.GetAstigmatismAngle(proj_index) + (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0);
		float defocusMin;
		float defocusMax;
		GetDefocusMinMax(ray, proj_index, defocusMin, defocusMax);
		float tiltAngle = (markers(MFI_TiltAngle, proj_index, 0) + config.AddTiltAngle) / 180.0f * (float)M_PI;

		if (mpi_part == 0)
		{
			printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b BP Defocus: %-8d nm", (int)defocusMin);
			fflush(stdout);
		}

		//Do CTF correction:
		cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);
		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

		ctf(fft_d, defocusMin, defocusMax, defocusAngle, false, config.PhaseFlipOnly, config.WienerFilterNoiseLevel, (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), config.CTFBetaFac);

		cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));
		nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), (float)(proj.GetMaxDimension() * proj.GetMaxDimension()),
			(Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));


		for (size_t batch = 0; batch < batchSize; batch++)
		{
			//bind surfref to correct array:
			//cudaSafeCall(cuSurfRefSetArray(surfref, vecArrays[batch]->GetCUarray(), 0));
			CudaSurfaceObject3D surface(vecArrays[batch]);

			//set additional shifts:
			proj.SetExtraShift(proj_index, vecExtraShifts[batch]);

			float3 volDim = subVolumes[batch]->GetSubVolumeDimension(0);
			bpKernel.SetComputeSize((int)volDim.x, (int)volDim.y, (int)volDim.z);

			SetConstantValues(bpKernel, *(subVolumes[batch]), proj, proj_index, 0, magAnisotropy, magAnisotropyInv);

			//Most of the time, no volume should get hit...
			runtime = bpKernel(x, y, 1.0f, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage, surface, ray, ray + config.CTFSliceThickness / proj.GetPixelSize());
		}
	}
}
template void Reconstructor::BackProjectionCTF(Volume<unsigned short>* vol, vector<Volume<unsigned short>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index);
template void Reconstructor::BackProjectionCTF(Volume<float>* vol, vector<Volume<float>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index);
#endif

bool Reconstructor::ComputeFourFilter()
{
	//if (skipFilter)
	//{
	//	return false;
	//}

	//if (mpi_part != 0)
	//{
	//	return false;
	//}

	//float lp = config.fourFilterLP, hp = config.fourFilterHP, lps = config.fourFilterLPS, hps = config.fourFilterHPS;
	//int size = proj.GetMaxDimension();
	//float2* filter = new float2[size * size];
	//float2* fourFilter = new float2[(proj.GetMaxDimension() / 2 + 1) * proj.GetMaxDimension()];

	//if ((lp > size || lp < 0 || hp > size || hp < 0 || hps > size || hps < 0) && !skipFilter)
	//{
	//	//Filter parameters are not good!
	//	skipFilter = true;
	//	return false;
	//}

	//lp = lp - lps;
	//hp = hp + hps;


	//for (int y = 0; y < size; y++)
	//{
	//	for (int x = 0; x < size; x++)
	//	{
	//		float _x = -size / 2 + y;
	//		float _y = -size / 2 + x;

	//		float dist = (float)sqrtf(_x * _x + _y * _y);
	//		float fil = 0;
	//		//Low pass
	//		if (lp > 0)
	//		{
	//			if (dist <= lp) fil = 1;
	//		}
	//		else
	//		{
	//			if (dist <= size / 2 - 1) fil = 1;
	//		}

	//		//Gauss
	//		if (lps > 0)
	//		{
	//			float fil2;
	//			if (dist < lp) fil2 = 1;
	//			else fil2 = 0;

	//			fil2 = (-fil + 1.0f) * (float)expf(-((dist - lp) * (dist - lp) / (2 * lps * lps)));
	//			if (fil2 > 0.001f)
	//				fil = fil2;
	//		}

	//		if (lps > 0 && lp == 0 && hp == 0 && hps == 0)
	//			fil = (float)expf(-((dist - lp) * (dist - lp) / (2 * lps * lps)));

	//		if (hp > lp) return -1;

	//		if (hp > 0)
	//		{
	//			float fil2 = 0;
	//			if (dist >= hp) fil2 = 1;

	//			fil *= fil2;

	//			if (hps > 0)
	//			{
	//				float fil3 = 0;
	//				if (dist < hp) fil3 = 1;
	//				fil3 = (-fil2 + 1) * (float)expf(-((dist - hp) * (dist - hp) / (2 * hps * hps)));
	//				if (fil3 > 0.001f)
	//					fil = fil3;

	//			}
	//		}
	//		float2 filcplx;
	//		filcplx.x = fil;
	//		filcplx.y = 0;
	//		filter[y * size + x] = filcplx;
	//	}
	//}
	////writeBMP("test.bmp", test, size, size);

	//cuFloatComplex* filterTemp = new cuFloatComplex[size * (size / 2 + 1)];

	////Do FFT Shift in X direction (delete double coeffs)
	//for (int y = 0; y < size; y++)
	//{
	//	for (int x = size / 2; x <= size; x++)
	//	{
	//		int oldX = x;
	//		if (oldX == size) oldX = 0;
	//		int newX = x - size / 2;
	//		filterTemp[y * (size / 2 + 1) + newX] = filter[y * size + oldX];
	//	}
	//}
	////Do FFT Shift in Y direction
	//for (int y = 0; y < size; y++)
	//{
	//	for (int x = 0; x < size / 2 + 1; x++)
	//	{
	//		int oldY = y + size / 2;
	//		if (oldY >= size) oldY -= size;
	//		fourFilter[y * (size / 2 + 1) + x] = filterTemp[oldY * (size / 2 + 1) + x];
	//	}
	//}
	//
	//ctf_d.CopyHostToDevice(fourFilter);
	//delete[] filterTemp;
	//delete[] filter;
	//delete[] fourFilter;
	return true;
}

void Reconstructor::PrepareProjection(void * img_h, int proj_index, float & meanValue, float & StdValue, int & BadPixels)
{
	if (mpi_part != 0)
	{
		return;
	}

	if (projSource->GetDataType() == DT_SHORT)
	{
		//printf("SIGNED SHORT\n");
		cudaSafeCall(cuMemcpyHtoD(realprojUS_d.GetDevicePtr(), img_h, proj.GetWidth() * proj.GetHeight() * sizeof(short)));
		nppSafeCall(nppiConvert_16s32f_C1R((Npp16s*)realprojUS_d.GetDevicePtr(), proj.GetWidth() * sizeof(short), (Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
	}
	else if (projSource->GetDataType() == DT_USHORT)
	{
		//printf("UNSIGNED SHORT\n");
		cudaSafeCall(cuMemcpyHtoD(realprojUS_d.GetDevicePtr(), img_h, proj.GetWidth() * proj.GetHeight() * sizeof(short)));
		nppSafeCall(nppiConvert_16u32f_C1R((Npp16u*)realprojUS_d.GetDevicePtr(), proj.GetWidth() * sizeof(short), (Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
	}
	else if (projSource->GetDataType() == DT_INT)
	{
		realprojUS_d.CopyHostToDevice(img_h);
		nppSafeCall(nppiConvert_32s32f_C1R((Npp32s*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
	}
	else if (projSource->GetDataType() == DT_UINT)
	{
		realprojUS_d.CopyHostToDevice(img_h);
		nppSafeCall(nppiConvert_32u32f_C1R((Npp32u*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
	}
	else if (projSource->GetDataType() == DT_FLOAT)
	{
		realproj_d.CopyHostToDevice(img_h);
	}
	else
	{
		return;
	}

	projSquare_d.Memset(0);
	if (config.GetFileReadMode() == Configuration::Config::FRM_DM4)
	{
		float t = cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, true, false);
	}
	else
	{
		float t = cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
	}

	nppSafeCall(nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));

	double mean = 0;
	meanval.CopyDeviceToHost(&mean);
	meanValue = (float)mean;

	if (config.CorrectBadPixels)
	{
		nppSafeCall(nppiCompareC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), config.BadPixelValue * meanValue, 
			(Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch(), roiSquare, NPP_CMP_GREATER));
	}
	else
	{
		nppSafeCall(nppiSet_8u_C1R(0, (Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch(), roiSquare));
	}

	nppSafeCall(nppiSum_8u_C1R((Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch(), roiSquare,
		(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));

	meanval.CopyDeviceToHost(&mean);
	BadPixels = (int)(mean / 255.0);

	nppSafeCall(nppiSet_32f_C1MR(meanValue, (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, 
		(Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch()));

	float normVal = 1;

	//When doing WBP we compute mean and std on the RAW image before Fourier filter and WBP weighting
	if (config.WBP_NoSART)
	{
		nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), 1.0f, 
			(Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), roiAll));

		nppSafeCall(nppiMean_StdDev_32f_C1R((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), roiAll,
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr(), (Npp64f*)stdval.GetDevicePtr()));

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
		if (config.DownWeightTiltsForWBP)
		{
			//we devide here because we devide later using nppiDivC: add the end we multiply!
			std_hf /= (float)cos(abs(markers(MarkerFileItem_enum::MFI_TiltAngle, proj_index, 0)) / 180.0 * M_PI);
		}

		nppSafeCall(nppiSubC_32f_C1R((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), mean_hf,
			(Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
		nppSafeCall(nppiDivC_32f_C1IR(-std_hf, (Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));


		//rotate image so that tilt axis lies parallel to image axis to avoid smearing if WBP-filter wedge:

		Matrix<double> shiftToCenter(3, 3);
		Matrix<double> rotate(3, 3);
		Matrix<double> shiftBack(3, 3);
		double psiAngle = proj.GetImageRotationToCompensate((uint)proj_index);

		shiftToCenter(0, 0) = 1; shiftToCenter(0, 1) = 0; shiftToCenter(0, 2) = -proj.GetWidth() / 2.0 + 0.5;
		shiftToCenter(1, 0) = 0; shiftToCenter(1, 1) = 1; shiftToCenter(1, 2) = -proj.GetHeight() / 2.0 + 0.5;
		shiftToCenter(2, 0) = 0; shiftToCenter(2, 1) = 0; shiftToCenter(2, 2) = 1;

		rotate(0, 0) = cos(psiAngle); rotate(0, 1) = -sin(psiAngle); rotate(0, 2) = 0;
		rotate(1, 0) = sin(psiAngle); rotate(1, 1) =  cos(psiAngle); rotate(1, 2) = 0;
		rotate(2, 0) = 0;             rotate(2, 1) = 0;              rotate(2, 2) = 1;

		shiftBack(0, 0) = 1; shiftBack(0, 1) = 0; shiftBack(0, 2) = proj.GetWidth() / 2.0 - 0.5;
		shiftBack(1, 0) = 0; shiftBack(1, 1) = 1; shiftBack(1, 2) = proj.GetHeight() / 2.0 - 0.5;
		shiftBack(2, 0) = 0; shiftBack(2, 1) = 0; shiftBack(2, 2) = 1;

		Matrix<double> rotationMatrix = shiftBack * (rotate * shiftToCenter);

		double affineMatrix[2][3];
		affineMatrix[0][0] = rotationMatrix(0, 0); affineMatrix[0][1] = rotationMatrix(0, 1); affineMatrix[0][2] = rotationMatrix(0, 2);
		affineMatrix[1][0] = rotationMatrix(1, 0); affineMatrix[1][1] = rotationMatrix(1, 1); affineMatrix[1][2] = rotationMatrix(1, 2);

		NppiSize imageSize;
		NppiRect roi;
		imageSize.width = proj.GetWidth();
		imageSize.height = proj.GetHeight();
		roi.x = 0;
		roi.y = 0;
		roi.width = proj.GetWidth();
		roi.height = proj.GetHeight();

		dimBordersKernel(realproj_d, config.Crop, config.CropDim);
		realprojUS_d.Memset(0);

		nppSafeCall(nppiWarpAffine_32f_C1R((Npp32f*)realproj_d.GetDevicePtr(), imageSize, (int)realproj_d.GetPitch(), roi,
			(Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), roi, affineMatrix, NppiInterpolationMode::NPPI_INTER_CUBIC));

		float t = cts(realprojUS_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);

//        float* test = new float[proj.GetMaxDimension()*proj.GetMaxDimension()];
//        projSquare_d.CopyDeviceToHost(test);
//
//        stringstream ss;
//        ss << "test_" << proj_index << ".em";
//        emwrite(ss.str(), (float*)test, proj.GetMaxDimension(), proj.GetMaxDimension());
//
//        delete[] test;
	}

	// good until here

	if (!skipFilter || config.WBP_NoSART)
	{
		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

		if (!skipFilter)
		{
			float lp = (float)config.fourFilterLP, hp = (float)config.fourFilterHP, lps = (float)config.fourFilterLPS, hps = (float)config.fourFilterHPS;
			int size = proj.GetMaxDimension();
			
			fourFilterKernel(fft_d, roiFFT.width * sizeof(Npp32fc), size, lp, hp, lps, hps);
			/*nppSafeCall(nppiMul_32fc_C1IR((Npp32fc*)ctf_d.GetDevicePtr(), ctf_d.GetPitch(),
				(Npp32fc*)fft_d.GetDevicePtr(), roiFFT.width * sizeof(Npp32fc), roiFFT));*/
		}
		if (config.DoseWeighting)
		{
			//cout << "Dose: " << config.AccumulatedDose[proj_index] << "; Pixelsize: " << proj.GetPixelSize() * 10.0f << endl;
			/*Npp32fc* temp = new Npp32fc[roiFFT.width * roiFFT.height];
			float* temp2 = new float[roiFFT.width * roiFFT.height];
			fft_d.CopyDeviceToHost(temp);
			for (size_t i = 0; i < roiFFT.width * roiFFT.height; i++)
			{
				temp2[i] = sqrtf(temp[i].re * temp[i].re + temp[i].im * temp[i].im);
			}
			emwrite("before.em", temp2, roiFFT.width, roiFFT.height);*/
			doseWeightingKernel(fft_d, roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), config.AccumulatedDose[proj_index], proj.GetPixelSize() * 10.0f);
			/*fft_d.CopyDeviceToHost(temp);
			for (size_t i = 0; i < roiFFT.width * roiFFT.height; i++)
			{
				temp2[i] = temp[i].re;
			}
			emwrite("after.em", temp2, roiFFT.width, roiFFT.height);*/
		}
		if (config.WBP_NoSART)
		{
			if (config.WBPFilter == FM_EXACT)
			{
				float* tiltAngles = new float[projSource->GetProjectionCount()];
				for (int i = 0; i < projSource->GetProjectionCount(); i++)
				{
					tiltAngles[i] = markers(MFI_TiltAngle, i, 0) * M_PI / 180.0f;
					if (!markers.CheckIfProjIndexIsGood(i))
					{
						tiltAngles[i] = -999.0f;
					}
				}

				meanbuffer.CopyHostToDevice(tiltAngles, projSource->GetProjectionCount() * sizeof(float));
				delete[] tiltAngles;
			}

			float volumeHeight = config.RecDimensions.z;
			float voxelSize = config.VoxelSize.z;
			volumeHeight *= voxelSize;
#ifdef SUBVOLREC_MODE
			/*volumeHeight = config.SizeSubVol;
			voxelSize = config.VoxelSizeSubVol;*/
#endif
			float D = (proj.GetMaxDimension() / 2) / volumeHeight * 2.0f;


			//Do WBP weighting
            double psiAngle = markers(MFI_RotationPsi, (uint)proj_index, 0) / 180.0 * (double)M_PI;
            if (Configuration::Config::GetConfig().UseFixPsiAngle)
                psiAngle = Configuration::Config::GetConfig().PsiAngle / 180.0 * (double)M_PI;

            // Fix for WBP of rectangular images (otherwise stuff is rotated out too far)
            float flipAngle;
            if (abs(abs(psiAngle) - ((double)M_PI/2.)) < ((double)M_PI/4.)){
                flipAngle = 90.;
            } else {
                flipAngle = 0.;
            }

            printf("PSI ANGLE: %f \n", flipAngle);

            wbp(fft_d, roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), flipAngle, config.WBPFilter, proj_index, projSource->GetProjectionCount(), D, meanbuffer);


			/*float2* test = new float2[fft_d.GetSize() / 4 / 2];
			float* test2 = new float[fft_d.GetSize() / 4 / 2];
			fft_d.CopyDeviceToHost(test);

			for (size_t i = 0; i < fft_d.GetSize() / 4 / 2; i++)
			{
				test2[i] = test[i].x;
			}

			stringstream ss;
			ss << "projFilter_" << proj_index << ".em";
			emwrite(ss.str(), test2, proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension());
			delete[] test;
			delete[] test2;*/

		}

		cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));

		normVal = (float)(proj.GetMaxDimension() * proj.GetMaxDimension());

//        float* test = new float[proj.GetMaxDimension()*proj.GetMaxDimension()];
//        projSquare_d.CopyDeviceToHost(test);
//
//        stringstream ss;
//        ss << "test_" << proj_index << ".em";
//        emwrite(ss.str(), (float*)test, proj.GetMaxDimension(), proj.GetMaxDimension());
//
//        delete[] test;
	}

	//Normalize from FFT
	nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), 
		normVal, (Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), roiAll));


	//When doing SART we compute mean and std on the filtered image
	if (!config.WBP_NoSART)
	{
		NppiSize roiForMean;
		Npp32f* ptr = (Npp32f*)realprojUS_d.GetDevicePtr();
		//When doing SART the projection must be mean free, so compute mean on center of image only for IMOD aligned stacks...
		if (config.ProjectionNormalization == Configuration::Config::PNM_NONE) //IMOD aligned stack
		{
			roiForMean.height = roiAll.height / 2;
			roiForMean.width = roiAll.width / 2;

			//Move start pointer:
			char* ptrChar = (char*)ptr;
			ptrChar += realprojUS_d.GetPitch() * (roiAll.height / 4); //Y
			ptr = (float*)ptrChar;
			ptr += roiAll.width / 4; //X

		}
		else
		{
			roiForMean.height = roiAll.height;
			roiForMean.width = roiAll.width;
		}

		nppSafeCall(nppiMean_StdDev_32f_C1R(ptr, (int)realprojUS_d.GetPitch(), roiForMean,
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr(), (Npp64f*)stdval.GetDevicePtr()));

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
			//meanValue = 0;
			//printf("I DID NAAAHHT.\n");
		}
        //mean_hf = 0.f;

		nppSafeCall(nppiSubC_32f_C1R((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), mean_hf,
			(Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
		nppSafeCall(nppiDivC_32f_C1IR(-std_hf, (Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
		realproj_d.CopyDeviceToHost(img_h);


	}
	else
	{
		realprojUS_d.CopyDeviceToHost(img_h);

//		stringstream ss;
//		ss << "test_" << proj_index << ".em";
//		emwrite(ss.str(), (float*)img_h, proj.GetWidth(), proj.GetHeight());
	}
}

template<class TVol>
void Reconstructor::Compare(Volume<TVol>* vol, char* originalImage, int index)
{
	if (mpi_part == 0)
	{
		float z_Direction = proj.GetNormalVector(index).z;
		float z_VolMinZ = vol->GetVolumeBBoxMin().z;
		float z_VolMaxZ = vol->GetVolumeBBoxMax().z;
		float volumeTraversalLength = fabs((DIST - z_VolMinZ) / z_Direction - (DIST - z_VolMaxZ) / z_Direction);

        realproj_d.CopyHostToDevice(originalImage);

		//nppiSet_32f_C1R(1.0f, (float*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);

		float runtime = compKernel(realproj_d, proj_d, dist_d, volumeTraversalLength, config.Crop, config.CropDim, config.ProjectionScaleFactor);
		/*proj_d.CopyDeviceToHost(MPIBuffer);
		writeBMP(string("Comp.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
	}
}
template void Reconstructor::Compare(Volume<unsigned short>* vol, char* originalImage, int index);
template void Reconstructor::Compare(Volume<float>* vol, char* originalImage, int index);

void Reconstructor::SubtractError(float* error)
{
    if (mpi_part == 0)
    {
        realproj_d.CopyHostToDevice(error);
        float runtime = subEKernel(proj_d, realproj_d, dist_d);
    }
}


template<class TVol>
void Reconstructor::ForwardProjection(Volume<TVol>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync)
{
	if (config.CtfMode == Configuration::Config::CTFM_YES)
	{
		ForwardProjectionCTF(vol, texVol, index, volumeIsEmpty, noSync);
	}
	else
	{
		ForwardProjectionNoCTF(vol, texVol, index, volumeIsEmpty, noSync);
	}
}
template void Reconstructor::ForwardProjection(Volume<unsigned short>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync);
template void Reconstructor::ForwardProjection(Volume<float>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync);


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


template<class TVol>
void Reconstructor::ForwardProjectionROI(Volume<TVol>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync)
{
	if (config.CtfMode == Configuration::Config::CTFM_YES)
	{
		ForwardProjectionCTFROI(vol, texVol, index, volumeIsEmpty, roiMin, roiMax, noSync);
	}
	else
	{
		ForwardProjectionNoCTFROI(vol, texVol, index, volumeIsEmpty, roiMin, roiMax, noSync);
	}
}
template void Reconstructor::ForwardProjectionROI(Volume<unsigned short>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
template void Reconstructor::ForwardProjectionROI(Volume<float>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);

template<typename TVol>
void Reconstructor::ForwardProjectionDistanceOnly(Volume<TVol>* vol, int index)
{
	SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

	float runtime = volTravLenKernel(proj.GetWidth(), proj.GetHeight(), dist_d);
#ifdef USE_MPI
	if (mpi_part == 0)
	{
		for (int mpi = 1; mpi < mpi_size; mpi++)
		{
			MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			realprojUS_d.CopyHostToDevice(MPIBuffer);
			nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
		}
	}
	else
	{
		dist_d.CopyDeviceToHost(MPIBuffer);
		MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	}
#endif
}
template void Reconstructor::ForwardProjectionDistanceOnly(Volume<unsigned short>* vol, int index);
template void Reconstructor::ForwardProjectionDistanceOnly(Volume<float>* vol, int index);



template<class TVol>
void Reconstructor::BackProjection(Volume<TVol>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount)
{
	if (config.CtfMode == Configuration::Config::CTFM_YES)
	{
		BackProjectionCTF(vol, surface, proj_index, SIRTCount);
	}
	else
	{
		BackProjectionNoCTF(vol, surface, proj_index, SIRTCount);
	}
}

template void Reconstructor::BackProjection(Volume<unsigned short>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);
template void Reconstructor::BackProjection(Volume<float>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);


#ifdef SUBVOLREC_MODE
template<class TVol>
void Reconstructor::BackProjection(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index)
{
	if (config.CtfMode == Configuration::Config::CTFM_YES)
	{
		BackProjectionCTF(vol, subVolumes, vecExtraShifts, vecArrays, proj_index);
	}
	else
	{
		BackProjectionNoCTF(vol, subVolumes, vecExtraShifts, vecArrays, proj_index);
	}
}

template void Reconstructor::BackProjection(Volume<unsigned short>* vol, vector<Volume<unsigned short>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index);
template void Reconstructor::BackProjection(Volume<float>* vol, vector<Volume<float>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index);
#endif

//template<class TVol>
//void Reconstructor::OneSARTStep(Volume<TVol>* vol, Cuda::CudaTextureObject3D & texVol, Cuda::CudaSurfaceObject3D& surface, int index, bool volumeIsEmpty, char * originalImage, float SIRTCount, float* MPIBuffer)
//{
//	ForwardProjection(vol, texVol, index, volumeIsEmpty);
//	if (mpi_part == 0)
//	{
//		Compare(vol, originalImage, index);
//		CopyProjectionToHost(MPIBuffer);
//	}
//
//	//spread the content to all other nodes:
//	MPIBroadcast(&MPIBuffer, 1);
//	CopyProjectionToDevice(MPIBuffer);
//	BackProjection(vol, surface, index, SIRTCount);
//}
//template void Reconstructor::OneSARTStep(Volume<unsigned short>* vol, Cuda::CudaTextureObject3D & texVol, Cuda::CudaSurfaceObject3D& surface, int index, bool volumeIsEmpty, char * originalImage, float SIRTCount, float* MPIBuffer);
//template void Reconstructor::OneSARTStep(Volume<float>* vol, Cuda::CudaTextureObject3D & texVol, Cuda::CudaSurfaceObject3D& surface, int index, bool volumeIsEmpty, char * originalImage, float SIRTCount, float* MPIBuffer);

//template<class TVol>
//void Reconstructor::BackProjectionWithPriorWBPFilter(Volume<TVol>* vol, int proj_index, char* originalImage, float* MPIBuffer)
//{
//	if (mpi_part == 0)
//	{
//		realproj_d.CopyHostToDevice(originalImage);
//		projSquare_d.Memset(0);
//		cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
//
//		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));
//
//		//Do WBP weighting
//		wbp(fft_d, roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), markers(MFI_RotationPsi, proj_index, 0), FilterMethod::FM_RAMP);
//		cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));
//
//		float normVal = proj.GetMaxDimension() * proj.GetMaxDimension();
//
//		//Normalize from FFT
//		nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float),
//			normVal, (Npp32f*)proj_d.GetDevicePtr(), proj_d.GetPitch(), roiAll));
//
//		CopyProjectionToHost(MPIBuffer);
//	}
//	//spread the content to all other nodes:
//	MPIBroadcast(&MPIBuffer, 1);
//	CopyProjectionToDevice(MPIBuffer);
//	BackProjection(vol, proj_index, 1);
//}
//template void Reconstructor::BackProjectionWithPriorWBPFilter(Volume<unsigned short>* vol, int proj_index, char* originalImage, float* MPIBuffer);
//template void Reconstructor::BackProjectionWithPriorWBPFilter(Volume<float>* vol, int proj_index, char* originalImage, float* MPIBuffer);

//template<class TVol>
//void Reconstructor::RemoveProjectionFromVol(Volume<TVol>* vol, int proj_index, char* originalImage, float* MPIBuffer)
//{
//	if (mpi_part == 0)
//	{
//		realproj_d.CopyHostToDevice(originalImage);
//		projSquare_d.Memset(0);
//		cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
//
//		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));
//
//		//Do WBP weighting
//		wbp(fft_d, roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), markers(MFI_RotationPsi, proj_index, 0), FilterMethod::FM_RAMP);
//		cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));
//
//		float normVal = proj.GetMaxDimension() * proj.GetMaxDimension();
//		//negate to remove from volume:
//		normVal *= -1;
//
//		//Normalize from FFT
//		nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float),
//			normVal, (Npp32f*)proj_d.GetDevicePtr(), proj_d.GetPitch(), roiAll));
//
//		CopyProjectionToHost(MPIBuffer);
//	}
//	//spread the content to all other nodes:
//	MPIBroadcast(&MPIBuffer, 1);
//	CopyProjectionToDevice(MPIBuffer);
//	BackProjection(vol, proj_index, 1);
//}
//template void Reconstructor::RemoveProjectionFromVol(Volume<unsigned short>* vol, int proj_index, char* originalImage, float* MPIBuffer);
//template void Reconstructor::RemoveProjectionFromVol(Volume<float>* vol, int proj_index, char* originalImage, float* MPIBuffer);

void Reconstructor::ResetProjectionsDevice()
{
	proj_d.Memset(0);
	dist_d.Memset(0);
}

void Reconstructor::CopyProjectionToHost(float * buffer)
{
	proj_d.CopyDeviceToHost(buffer);
}

void Reconstructor::CopyDistanceImageToHost(float * buffer)
{
	dist_d.CopyDeviceToHost(buffer);
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

