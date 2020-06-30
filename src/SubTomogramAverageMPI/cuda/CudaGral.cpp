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


// CudaGral.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//
//
//#include "stdafx.h"
//#include "Array.h"
//#include "CudaContext.h"
//#include "CudaVariables.h"
//#include "CudaKernel.h"
//#include "CudaArrays.h"
//#include "CudaTextures.h"
//#include "CudaDeviceProperties.h"
//#include "Config.h"
//#include "ConfigExceptions.h"
//#include "Topographie.h"
//#include "GralGebiet.h"
//#include "GrammGebiet.h"
//#include "KartesischesGitter.h"
//#include "WindfeldGramm.h"
//#include "Eingabedaten.h"
//#include "Attraktor.h"
//#include "Quellen.h"
//#include "Punktquellen.h"
//#include "Linienquellen.h"
//#include "Flaechenquellen.h"
//#include "Katasterquellen.h"
//#include "Wetter.h"
//
//#include "FileReadException.h"
//
//#include "writeBMP.h"
//
//#include <builtin_types.h> 
//
//#include <time.h>
//
//
//using namespace std;
//using namespace GRAL;
//using namespace Configuration;
//
//void WaitForInput(int exitCode)
//{
//	char c;
//	cout << ("\nPress <Enter> to exit...");  
//	c = cin.get();// getchar(); 
//	exit(exitCode);
//}
//
//namespace GRAL
//{
//	//! Main application class
//	/*!
//			\warning Der portierte C++-Code ist nur eine "quick-n-dirty"-Portierung des original FORTRAN Codes.
//					 Zweck der Portierung war nur ein besseres Verständis von GRAL zu erlangen und weiterhin den
//					 CUDA-Teil mit den üblichen C/C++-Entwicklungstools bequemer entwickeln zu können. Die C/C++-
//					 Abschnitte erheben weder Anspruch auf Vollständigkeit, noch auf Korrektheit.
//	*/
//	// Main application class
//	class MainApp
//	{
//	private:
//		Config mConfig;
//	public:
//		MainApp(Config& aConfig) : mConfig(aConfig)
//		{
//		
//		}
//
//
//		//! Executes the main application
//		/*!
//			\return Returns the CudaContext* used and to be destroyed manually after successful execution.
//		*/
//		// Executes the main application
//		Cuda::CudaContext* run()
//		{
//			Gralgebiet gralGebiet("GRAL.geb");
//
//			Grammgebiet grammGebiet("GRAMM.geb", &gralGebiet);
//
//			KartesischesGitter kartgitter(&gralGebiet, &grammGebiet);
//
//			WindfeldGramm windfeld(&gralGebiet, &grammGebiet, &kartgitter);
//		
//			Eingabedaten eingabedaten("ein.dat", &gralGebiet, &grammGebiet, &windfeld);
//
//			Attraktor attraktor;
//
//			windfeld.ReadLanduseDat(eingabedaten.Z0);
//
//
//			Quellen quellen(&gralGebiet, &grammGebiet, &eingabedaten, &kartgitter, &windfeld);
//
//			grammGebiet.ResetGebietsgrenzen();
//
//			Wetter wetter(&gralGebiet, &grammGebiet, &eingabedaten, &windfeld, &quellen.FlaechenQuellen);
//				
//			windfeld.ReadWindfeld(wetter.AktuelleWetterlage+1);
//			bool res = wetter.ReadNextWetterlage();
//
//
//			//TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
//			//  Kartesisches Windfeld erzeugen
//			//TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
//
//			if (grammGebiet.Topo->TopoExists && !kartgitter.ControlDatExists)
//				windfeld.InterpolateToKartGrid();
//
//			//Vorerst ist es egal ob Gebäude existieren oder nicht...
//			if (!grammGebiet.Topo->TopoExists)
//				wetter.ComputeKartWindField(&kartgitter, windfeld.MoninObLengthOB(0, 0));
//
//			//Die Überprüfung wie und ob Gebäude berücksichtigt werden sollen wird in AccountBuilings(...) erledigt
//			kartgitter.AccountBuildings(eingabedaten.ComputationLevel, windfeld.ComputePotTemperature(), (float)windfeld.Z0GRAMM(0,0));
//
//			kartgitter.ComputeMassConservativeField();
//		
//
//			//TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
//			//  Quellenverteilung
//			//TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
//
//			quellen.PunktQuellen.Quellhoehe(windfeld.MoninObLengthOB(0,0), &wetter, &kartgitter, &windfeld);
//			quellen.CalcAnfangskoordinaten();
//			//TODO: Punktquellen, Tunnelquellen
//		
//			//TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
//			//   Ab hier CUDA
//			//TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
//		
//			cout << "Now starting CUDA..." << endl;
//
//			//Init CUDA
//			Cuda::CudaContext* cuCtx = Cuda::CudaContext::CreateInstance(mConfig.CudaDeviceID, CU_CTX_MAP_HOST);
//
//			//Lade Kernel für Linien und Katasterquellen
//			//CUmodule module = cuCtx->LoadModulePTX(mConfig.CudaNormalParticleKernel.c_str(), 0, NULL, NULL);
//			CUmodule moduleNQ = cuCtx->LoadModule(mConfig.CudaNormalParticleKernel.c_str());
//			Cuda::CudaKernel kernelNQ(mConfig.CudaNormalParticleKernelName.c_str(), moduleNQ, cuCtx);
//
//			CUmodule modulePQ = cuCtx->LoadModule(mConfig.CudaPointSourceKernel.c_str());
//			Cuda::CudaKernel kernelPQ(mConfig.CudaPointSourceKernelName.c_str(), modulePQ, cuCtx);
//		
//			cout << "Kernel image normal sources: " << mConfig.CudaNormalParticleKernel << endl;
//			cout << "Kernel image point sources:  " << mConfig.CudaPointSourceKernel << endl;
//
//			//Verknüpfe U und V Komponente des Windfeldes
//			float2* windUV = new float2[kartgitter.WindComponentUK.GetSize()];
//			for (int i = 0; i < kartgitter.WindComponentUK.GetSize(); i++)
//			{
//				windUV[i].x = kartgitter.WindComponentUK.GetData()[i];
//				windUV[i].y = kartgitter.WindComponentVK.GetData()[i];
//			}
//
//			//Gib Info über Speicherkonfiguration aus
//			size_t totSize = cuCtx->GetMemorySize() / 1024 / 1024;
//			size_t freeCuda = cuCtx->GetFreeMemorySize() / 1024 / 1024;
//
//			cout << "Frei Speicher vor alloc: " << freeCuda << " MB" << endl;
//
//			//Wenn ZeroCopy verwendet werden soll
//			Cuda::CudaPageLockedHostVariable* hpl_windUV;
//			Cuda::CudaPageLockedHostVariable* hpl_windW;
//			CUdeviceptr d_uv;
//			CUdeviceptr d_w;
//
//			//Wenn keine Texturen verwendet werden sollen
//			Cuda::CudaDeviceVariable* dv_uv;
//			Cuda::CudaDeviceVariable* dv_w;
//
//			if (mConfig.UseZeroCopy)
//			{
//				hpl_windUV = new Cuda::CudaPageLockedHostVariable(kartgitter.WindComponentUK.GetSizeInBytes() * 2, CU_MEMHOSTALLOC_DEVICEMAP);
//				hpl_windW = new Cuda::CudaPageLockedHostVariable(kartgitter.WindComponentWK.GetSizeInBytes(), CU_MEMHOSTALLOC_DEVICEMAP);
//				memcpy(hpl_windUV->GetHostPtr(), windUV, kartgitter.WindComponentUK.GetSizeInBytes() * 2);
//				memcpy(hpl_windW->GetHostPtr() , kartgitter.WindComponentWK.GetData(), kartgitter.WindComponentVK.GetSizeInBytes());
//			
//				CUresult res1 = cuMemHostGetDevicePointer(&d_uv, hpl_windUV->GetHostPtr(), 0);
//				CUresult res2 = cuMemHostGetDevicePointer(&d_w , hpl_windW->GetHostPtr() , 0);
//			}
//		
//			if (!mConfig.UseCudaTextures)
//			{			
//				dv_uv = new Cuda::CudaDeviceVariable(kartgitter.WindComponentUK.GetSizeInBytes() * 2);
//				dv_w = new Cuda::CudaDeviceVariable(kartgitter.WindComponentWK.GetSizeInBytes());
//
//				dv_uv->CopyHostToDevice(windUV);
//				dv_w->CopyHostToDevice(kartgitter.WindComponentWK.GetData());
//			
//				d_uv = dv_uv->GetDevicePtr();
//				d_w  = dv_w->GetDevicePtr();
//			}
//
//
//			//Cuda textures for Kernel "Normal particle"
//			Cuda::CudaTextureArray3D texWindfeldUV(&kernelNQ, "texWindfeldUV", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, 
//												   CU_TR_FILTER_MODE_LINEAR, 0, CU_AD_FORMAT_FLOAT, 
//												   kartgitter.DimMeshNII + 1, kartgitter.DimMeshNJJ + 1, kartgitter.DimMeshNKK, 2);
//		
//			Cuda::CudaTextureArray3D texWindfeldW(&kernelNQ, "texWindfeldW", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, 
//												   CU_TR_FILTER_MODE_LINEAR, 0, CU_AD_FORMAT_FLOAT, 
//												   kartgitter.DimMeshNII + 1, kartgitter.DimMeshNJJ + 1, kartgitter.DimMeshNKK, 1);/**/
//
//			Cuda::CudaTextureArray2D texWindGeschw10m(&kernelNQ, "texWindGeschw10m", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
//													  CU_TR_FILTER_MODE_LINEAR, 0, CU_AD_FORMAT_FLOAT,
//													  kartgitter.DimMeshNII, kartgitter.DimMeshNJJ, 1);
//
//			Cuda::CudaTextureArray2D texCombined(&kernelNQ, "texCombined", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
//													  CU_TR_FILTER_MODE_POINT, 0, CU_AD_FORMAT_FLOAT,
//													  kartgitter.DimMeshNII, kartgitter.DimMeshNJJ, 4);
//
//			Cuda::CudaTextureArray2D texCutk(&kernelNQ, "texCutk", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
//													  CU_TR_FILTER_MODE_POINT, 0, CU_AD_FORMAT_FLOAT,
//													  kartgitter.DimMeshNII, kartgitter.DimMeshNJJ, 1);
//
//			Cuda::CudaTextureArray2D texKkart(&kernelNQ, "texKkart", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
//													  CU_TR_FILTER_MODE_POINT, 0, CU_AD_FORMAT_UNSIGNED_INT32,
//													  kartgitter.DimMeshNII, kartgitter.DimMeshNJJ, 1);
//
//		
//			//Cuda textures for Kernel "Point source"
//			Cuda::CudaTextureArray3D texWindfeldUV_p(&kernelPQ, "texWindfeldUV", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, 
//												   CU_TR_FILTER_MODE_LINEAR, 0, texWindfeldUV.GetArray());
//		
//			Cuda::CudaTextureArray3D texWindfeldW_p(&kernelPQ, "texWindfeldW", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, 
//												   CU_TR_FILTER_MODE_LINEAR, 0, texWindfeldW.GetArray());
//		
//			Cuda::CudaTextureArray2D texWindGeschw10m_p(&kernelPQ, "texWindGeschw10m", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
//												   CU_TR_FILTER_MODE_LINEAR, 0, texWindGeschw10m.GetArray());
//		
//			Cuda::CudaTextureArray2D texCombined_p(&kernelPQ, "texCombined", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
//												   CU_TR_FILTER_MODE_POINT, 0, texCombined.GetArray());
//		
//			Cuda::CudaTextureArray2D texCutk_p(&kernelPQ, "texCutk", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
//												   CU_TR_FILTER_MODE_POINT, 0, texCutk.GetArray());
//		
//			Cuda::CudaTextureArray2D texKkart_p(&kernelPQ, "texKkart", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
//												   CU_TR_FILTER_MODE_POINT, 0, texKkart.GetArray());
//
//
//
//
//
//		
//			Cuda::CudaDeviceVariable d_coordInit((quellen.KatasterQIndex.size() + quellen.LinienQIndex.size()) * sizeof(float3));
//			Cuda::CudaDeviceVariable d_velInit((quellen.KatasterQIndex.size() + quellen.LinienQIndex.size()) * sizeof(float3));
//
//			Cuda::CudaDeviceVariable d_iquellKQ((quellen.KatasterQIndex.size() + quellen.LinienQIndex.size()) * sizeof(int));
//			Cuda::CudaDeviceVariable d_iquellPQ(quellen.PunktQIndex.size() * sizeof(int));
//
//			Cuda::CudaDeviceVariable d_coordInitP(quellen.PunktQIndex.size() * sizeof(float3));
//			Cuda::CudaDeviceVariable d_velInitP(quellen.PunktQIndex.size() * sizeof(float3));
//			Cuda::CudaDeviceVariable d_zmax(quellen.PunktQIndex.size() * sizeof(float));
//			Cuda::CudaDeviceVariable d_FRHp(quellen.PunktQIndex.size() * sizeof(float));
//			Cuda::CudaDeviceVariable d_Fhurly(quellen.PunktQIndex.size() * sizeof(float));
//			Cuda::CudaDeviceVariable d_Ghurly(quellen.PunktQIndex.size() * sizeof(float));
//			Cuda::CudaDeviceVariable d_Mhurly(quellen.PunktQIndex.size() * sizeof(float));
//			Cuda::CudaDeviceVariable d_Rhurly(quellen.PunktQIndex.size() * sizeof(float));
//			Cuda::CudaDeviceVariable d_up(quellen.PunktQIndex.size() * sizeof(float));
//			Cuda::CudaDeviceVariable d_wp(quellen.PunktQIndex.size() * sizeof(float));
//
//			Array4D<int> conz3d(gralGebiet.DimCountArrayNXL, gralGebiet.DimCountArrayNYL, gralGebiet.CountHorizontalCutsNS, gralGebiet.CountSourceGroupes, 0);
//			Cuda::CudaDeviceVariable d_conz3d(conz3d.GetSizeInBytes());
//			cuCtx->ClearMemory(d_conz3d.GetDevicePtr(), 0, conz3d.GetSizeInBytes());
//		
//			freeCuda = cuCtx->GetFreeMemorySize() / 1024 / 1024;
//
//			cout << "Free nachher: " << freeCuda << " MB" << endl;
//			//copy Texture data to device
//			texWindfeldUV.GetArray()->CopyFromHostToArray(windUV);
//			texWindfeldW.GetArray()->CopyFromHostToArray(kartgitter.WindComponentWK.GetData());
//			texWindGeschw10m.GetArray()->CopyFromHostToArray(kartgitter.ComputeWindspeedMap10m()->GetData());
//				
//			Array2D<float>* usternA = windfeld.ComputeUsternMap();
//			Array2D<float>* obA = windfeld.ComputeObMap();
//
//			float* ustern = usternA->GetData();
//			float* ob = obA->GetData();
//			float* z0 = windfeld.ComputeZ0Map()->GetData();
//
//			float4* combined = new float4[kartgitter.DimMeshNII * kartgitter.DimMeshNJJ];
//		
//			for (int i = 0; i < kartgitter.DimMeshNII * kartgitter.DimMeshNJJ; i++)
//			{
//				combined[i].x = kartgitter.TerrainElevationAHK.GetData()[i];
//				combined[i].y = ob[i];
//				combined[i].z = z0[i];
//				combined[i].w = ustern[i];
//
//			}
//
//			texCombined.GetArray()->CopyFromHostToArray(combined);
//			
//			texCutk.GetArray()->CopyFromHostToArray(kartgitter.HeightBuildingCUTK.GetData());
//			texKkart.GetArray()->CopyFromHostToArray(kartgitter.HeightIndexTerrainKKART.GetData());
//		
//			//Konstanten setzen
//
//			kernelNQ.SetConstantValue("c_NII", &(kartgitter.DimMeshNII));
//			kernelPQ.SetConstantValue("c_NII", &(kartgitter.DimMeshNII));
//
//			kernelNQ.SetConstantValue("c_NJJ", &(kartgitter.DimMeshNJJ));
//			kernelPQ.SetConstantValue("c_NJJ", &(kartgitter.DimMeshNJJ));
//		
//			float NKK = (float)kartgitter.DimMeshNKK;
//			kernelNQ.SetConstantValue("c_NKK", &(NKK));
//			kernelPQ.SetConstantValue("c_NKK", &(NKK));
//
//			kernelNQ.SetConstantValue("c_AHMin", &(kartgitter.AHMin));
//			kernelPQ.SetConstantValue("c_AHMin", &(kartgitter.AHMin));
//
//			kernelNQ.SetConstantValue("c_AHMax", &(kartgitter.AHMax));
//			kernelPQ.SetConstantValue("c_AHMax", &(kartgitter.AHMax));
//		
//			kernelNQ.SetConstantValue("c_DXK", &(gralGebiet.CellSizeDXK));
//			kernelPQ.SetConstantValue("c_DXK", &(gralGebiet.CellSizeDXK));
//		
//			kernelNQ.SetConstantValue("c_DYK", &(gralGebiet.CellSizeDYK));
//			kernelPQ.SetConstantValue("c_DYK", &(gralGebiet.CellSizeDYK));
//		
//			kernelNQ.SetConstantValue("c_DZK0", &(gralGebiet.pCellSizeDZK[0]));
//			kernelPQ.SetConstantValue("c_DZK0", &(gralGebiet.pCellSizeDZK[0]));
//		
//			kernelNQ.SetConstantValue("c_strech", &(gralGebiet.StretchFactor));
//			kernelPQ.SetConstantValue("c_strech", &(gralGebiet.StretchFactor));
//		
//			float logStretch = logf(gralGebiet.StretchFactor);
//			kernelNQ.SetConstantValue("c_LogStrech", &(logStretch));
//			kernelPQ.SetConstantValue("c_LogStrech", &(logStretch));
//		
//			kernelNQ.SetConstantValue("c_NXL", &(gralGebiet.DimCountArrayNXL));
//			kernelPQ.SetConstantValue("c_NXL", &(gralGebiet.DimCountArrayNXL));
//		
//			kernelNQ.SetConstantValue("c_NYL", &(gralGebiet.DimCountArrayNYL));
//			kernelPQ.SetConstantValue("c_NYL", &(gralGebiet.DimCountArrayNYL));
//
//			float blh = 0;
//			if (eingabedaten.Statistik != INPUT_ZR)
//			{
//				double blhd = 0;
//				if (windfeld.MoninObLengthOB(0, 0) >= 0 && windfeld.MoninObLengthOB(0, 0) < 100)
//					blhd = min(0.4 * sqrt(windfeld.FrictionVelocityUSTERN(0,0) * windfeld.MoninObLengthOB(0,0)
//						/ eingabedaten.CoriolisParameterFCOR), 2000.0);
//				else if(windfeld.MoninObLengthOB(0, 0) < 0 && windfeld.MoninObLengthOB(0, 0) > -100)
//					blhd = 800.0;
//				else
//					blhd = min(0.2 * windfeld.FrictionVelocityUSTERN(0, 0) / eingabedaten.CoriolisParameterFCOR, 2000.0);
//				blh = float(blhd);
//			}
//		
//			kernelNQ.SetConstantValue("c_blh", &(blh));
//			kernelPQ.SetConstantValue("c_blh", &(blh));
//
//			kernelNQ.SetConstantValue("c_standv", &(eingabedaten.standv));
//			kernelPQ.SetConstantValue("c_standv", &(eingabedaten.standv));
//		
//			kernelNQ.SetConstantValue("c_NS", &(gralGebiet.CountHorizontalCutsNS));
//			kernelPQ.SetConstantValue("c_NS", &(gralGebiet.CountHorizontalCutsNS));
//		
//			kernelNQ.SetConstantValue("c_dx", &(gralGebiet.SizeCountArrayDX));
//			kernelPQ.SetConstantValue("c_dx", &(gralGebiet.SizeCountArrayDX));
//		
//			kernelNQ.SetConstantValue("c_dy", &(gralGebiet.SizeCountArrayDY));
//			kernelPQ.SetConstantValue("c_dy", &(gralGebiet.SizeCountArrayDY));
//		
//			kernelNQ.SetConstantValue("c_dz", &(gralGebiet.SizeCountArrayDZ));;
//			kernelPQ.SetConstantValue("c_dz", &(gralGebiet.SizeCountArrayDZ));
//		
//			float MASSE = (float)quellen.SchadstoffMasse;
//			kernelNQ.SetConstantValue("c_MASSE", &(MASSE));
//			kernelPQ.SetConstantValue("c_MASSE", &(MASSE));
//
//			kernelNQ.SetConstantValue("c_dV", &(gralGebiet.SizeCountArrayDV));
//			kernelPQ.SetConstantValue("c_dV", &(gralGebiet.SizeCountArrayDV));
//
//			float taus = (float)(eingabedaten.DispersionTimeTAUS);
//			kernelNQ.SetConstantValue("c_Taus", &taus);
//			kernelPQ.SetConstantValue("c_Taus", &taus);
//
//			kernelNQ.SetConstantValue("c_hor", gralGebiet.pHeightHorizontalCutsHOR);
//			kernelPQ.SetConstantValue("c_hor", gralGebiet.pHeightHorizontalCutsHOR);
//
// 
//			float3* coordInit = new float3[(quellen.KatasterQIndex.size() + quellen.LinienQIndex.size())];
//			float3* velInit = new float3[(quellen.KatasterQIndex.size() + quellen.LinienQIndex.size())];
//			int*    iquellKQ = new int[(quellen.KatasterQIndex.size() + quellen.LinienQIndex.size())];
//
//			for (size_t i = 0; i < quellen.KatasterQIndex.size(); i++)
//			{
//				int index = quellen.KatasterQIndex[i];
//				coordInit[i].x = (float)quellen.pXcoord[index];
//				coordInit[i].y = (float)quellen.pYcoord[index];
//				coordInit[i].z = quellen.pZcoord[index];
//
//				int indexI = int(coordInit[i].x / gralGebiet.CellSizeDXK);
//				int indexJ = int(coordInit[i].y / gralGebiet.CellSizeDYK);
//
//				float differenz = coordInit[i].z - kartgitter.TerrainElevationAHK(indexI, indexJ);
//				float u0int = (*usternA)(indexI, indexJ) * powf(2.0f + 4.0f * (differenz / abs((*obA)(indexI, indexJ) + 0.001f)), 0.6f);
//				u0int = max(u0int, 0.3f) / eingabedaten.standv;
//				velInit[i].x = (float)RandGauss() * u0int;
//				velInit[i].y = (float)RandGauss() * u0int;
//				velInit[i].z = 0;
//
//				iquellKQ[i] = quellen.pIquell[index];
//			}
//			for (size_t i = quellen.KatasterQIndex.size(); i < quellen.KatasterQIndex.size() + quellen.LinienQIndex.size(); i++)
//			{
//				int index = quellen.LinienQIndex[i - quellen.KatasterQIndex.size()];
//				coordInit[i].x = (float)quellen.pXcoord[index];
//				coordInit[i].y = (float)quellen.pYcoord[index];
//				coordInit[i].z = quellen.pZcoord[index];
//
//				int indexI = int(coordInit[i].x / gralGebiet.CellSizeDXK);
//				int indexJ = int(coordInit[i].y / gralGebiet.CellSizeDYK);
//
//				float differenz = coordInit[i].z - kartgitter.TerrainElevationAHK(indexI, indexJ);
//				float u0int = (*usternA)(indexI, indexJ) * powf(2.0f + 4.0f * (differenz / abs((*obA)(indexI, indexJ) + 0.001f)), 0.6f);
//				u0int = max(u0int, 0.3f) / eingabedaten.standv;
//				velInit[i].x = (float)RandGauss() * u0int;
//				velInit[i].y = (float)RandGauss() * u0int;
//				velInit[i].z = 0;
//
//				iquellKQ[i] = quellen.pIquell[index];
//			}
//
//			cout << "Anzahl Partikel: " << quellen.KatasterQIndex.size() + quellen.LinienQIndex.size() << endl;
//		
//			d_coordInit.CopyHostToDevice(coordInit);
//			d_velInit.CopyHostToDevice(velInit);
//			d_iquellKQ.CopyHostToDevice(iquellKQ);
//
//			kernelNQ.SetDevicePtrParameter(d_conz3d.GetDevicePtr());
//			kernelNQ.SetDevicePtrParameter(d_coordInit.GetDevicePtr());
//			kernelNQ.SetDevicePtrParameter(d_velInit.GetDevicePtr());
//			kernelNQ.SetDevicePtrParameter(d_iquellKQ.GetDevicePtr());
//			if (!mConfig.UseCudaTextures)
//			{
//				kernelNQ.SetDevicePtrParameter(d_uv);//.GetDevicePtr()
//				kernelNQ.SetDevicePtrParameter(d_w);//.GetDevicePtr()
//			}
//
//			kernelNQ.SetDynamicSharedMemory(sizeof(float3) * mConfig.CudaNormalParticleBlockWidth * 2);//
//			kernelNQ.SetBlockDimensions(mConfig.CudaNormalParticleBlockWidth, 1, 1);
//			kernelNQ.SetGridDimensions(uint(quellen.KatasterQIndex.size() + quellen.LinienQIndex.size()) / mConfig.CudaNormalParticleBlockWidth, 1);
//
//			cout << "Running kernel for normal particles..." << endl;
//
//			float rt = 0;
//			rt = kernelNQ();
//
//			cout << "Runtime: " << rt / 1000.0f << " sec." << endl << endl;
//
//
//
//			//kernel ausführen Punktquellen:
//		
//			float3* coordInitP = new float3[quellen.PunktQIndex.size()];
//			float3* velInitP = new float3[quellen.PunktQIndex.size()];
//			float* zmax = new float[quellen.PunktQIndex.size()];
//			float* FRHp = new float[quellen.PunktQIndex.size()];
//			float* Fhurly = new float[quellen.PunktQIndex.size()];
//			float* Ghurly = new float[quellen.PunktQIndex.size()];
//			float* Mhurly = new float[quellen.PunktQIndex.size()];
//			float* Rhurly = new float[quellen.PunktQIndex.size()];
//			float* up = new float[quellen.PunktQIndex.size()];
//			float* wp = new float[quellen.PunktQIndex.size()];
//			int*   iquellPQ = new int[quellen.PunktQIndex.size()];
//		
//			for (size_t i = 0; i < quellen.PunktQIndex.size(); i++)
//			{
//				int index = quellen.PunktQIndex[i];
//				int quelleID = quellen.pKenn[index] - quellen.FlaechenQuellen.Count;
//
//				coordInitP[i].x = (float)quellen.pXcoord[index];
//				coordInitP[i].y = (float)quellen.pYcoord[index];
//				coordInitP[i].z = quellen.pZcoord[index];
//
//				int indexI = int(coordInitP[i].x / gralGebiet.CellSizeDXK);
//				int indexJ = int(coordInitP[i].y / gralGebiet.CellSizeDYK);
//
//				float differenz = coordInitP[i].z - kartgitter.TerrainElevationAHK(indexI, indexJ);
//				float u0int = (*usternA)(indexI, indexJ) * powf(2.0f + 4.0f * (differenz / abs((*obA)(indexI, indexJ) + 0.001f)), 0.6f);
//				u0int = max(u0int, 0.3f) / eingabedaten.standv;
//				velInitP[i].x = (float)RandGauss() * u0int;
//				velInitP[i].y = (float)RandGauss() * u0int;
//				velInitP[i].z = 0;
//						
//				zmax[i] = quellen.PunktQuellen.pZmax[quelleID];
//				FRHp[i] = quellen.PunktQuellen.FRHp[quelleID];
//				Fhurly[i] = quellen.PunktQuellen.pFhurly[quelleID];
//				Ghurly[i] = quellen.PunktQuellen.pGhurly[quelleID];
//				Mhurly[i] = quellen.PunktQuellen.pMhurly[quelleID];
//				Rhurly[i] = quellen.PunktQuellen.pRhurly[quelleID];
//				up[i] = quellen.PunktQuellen.pUp[quelleID];
//				wp[i] = quellen.PunktQuellen.pWp[quelleID];
//				
//				iquellPQ[i] = quellen.pIquell[index];
//			}
//
//			cout << "Anzahl Partikel: " << quellen.PunktQIndex.size() << endl;
//		
//
//			d_coordInitP.CopyHostToDevice(coordInitP);
//			d_velInitP.CopyHostToDevice(velInitP);
//			d_zmax.CopyHostToDevice(zmax);
//			d_FRHp.CopyHostToDevice(FRHp);
//			d_Fhurly.CopyHostToDevice(Fhurly);
//			d_Ghurly.CopyHostToDevice(Ghurly);
//			d_Mhurly.CopyHostToDevice(Mhurly);
//			d_Rhurly.CopyHostToDevice(Rhurly);
//			d_up.CopyHostToDevice(up);
//			d_wp.CopyHostToDevice(wp);
//			d_iquellPQ.CopyHostToDevice(iquellPQ);
//
//			kernelPQ.SetDevicePtrParameter(d_conz3d.GetDevicePtr());
//			kernelPQ.SetDevicePtrParameter(d_coordInitP.GetDevicePtr());
//			kernelPQ.SetDevicePtrParameter(d_velInitP.GetDevicePtr());
//
//			kernelPQ.SetDevicePtrParameter(d_zmax.GetDevicePtr());
//			kernelPQ.SetDevicePtrParameter(d_FRHp.GetDevicePtr());
//			kernelPQ.SetDevicePtrParameter(d_Fhurly.GetDevicePtr());
//			kernelPQ.SetDevicePtrParameter(d_Ghurly.GetDevicePtr());
//			kernelPQ.SetDevicePtrParameter(d_Mhurly.GetDevicePtr());
//			kernelPQ.SetDevicePtrParameter(d_Rhurly.GetDevicePtr());
//			kernelPQ.SetDevicePtrParameter(d_up.GetDevicePtr());
//			kernelPQ.SetDevicePtrParameter(d_wp.GetDevicePtr());
//			kernelPQ.SetDevicePtrParameter(d_iquellPQ.GetDevicePtr());
//		
//			kernelPQ.SetDynamicSharedMemory(sizeof(float3) * mConfig.CudaPointSourceBlockWidth * 2);//
//			kernelPQ.SetBlockDimensions(mConfig.CudaPointSourceBlockWidth, 1, 1);
//			kernelPQ.SetGridDimensions(uint(quellen.PunktQIndex.size()) / mConfig.CudaPointSourceBlockWidth, 1);
//
//			cout << "Running kernel for point sources..." << endl;
//
//			rt = kernelPQ();
//
//			cout << "Runtime: " << rt / 1000.0f << " sec." << endl << endl;
//
//
//			//Ergebnis zu host kopieren
//			d_conz3d.CopyDeviceToHost(conz3d.GetData());
//		
//			//Ergebnis in Datei schreiben
//			ofstream result("firstResult.bin", ios::binary);
//
//			result.write((char*)(conz3d.GetData()), conz3d.GetSizeInBytes());
//
//			result.close();
//
//			//TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
//			//  Clean up
//			//TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
//
//			if (mConfig.UseZeroCopy)
//			{
//				delete hpl_windUV;
//				delete hpl_windW;
//			
//				hpl_windUV = NULL;
//				hpl_windW = NULL;
//			}
//		
//			if (!mConfig.UseCudaTextures)
//			{			
//				delete dv_uv;
//				delete dv_w;
//
//				dv_uv = NULL;
//				dv_w = NULL;
//			}
//		
//			return cuCtx;
//		}
//	};
//}

//! Haupteinstiegspunkt
//Haupteinstiegspunkt
//int main(int argc, char* argv[])
//{
//	try
//	{
//		cout << endl;
//		
//		cout << "                       :::::::::::::::::::::::::::::::::" << endl;
//		cout << "                       :      GRAL 8.10 (for CUDA)     :" << endl;      
//		cout << "                       :::::::::::::::::::::::::::::::::" << endl;
//
//		cout << endl;
//
//		Config config("gral.cfg");
//		
//		
//		MainApp mainapp(config);
//
//		Cuda::CudaContext* cuCtx = mainapp.run();
//
//		Cuda::CudaContext::DestroyContext(cuCtx);
//
//
//	}
//	catch (ConfigException& e)
//	{
//		cout << "\n\nERROR:\n";
//		cout << e.GetMessage() << endl << endl;
//		WaitForInput(-1);
//	}
//	catch (FileReadException& e)
//	{
//		cout << "\n\nERROR:\n";
//		cout << e.GetMessage() << endl << endl;
//		WaitForInput(-1);
//	}
//	catch (Cuda::CudaException& e)
//	{
//		cout << "\n\nERROR:\n";
//		cout << e.GetMessage() << endl << endl;
//		WaitForInput(-1);
//	}
//	catch (GralInputException& e)
//	{
//		cout << "\n\nERROR:\n";
//		cout << e.GetMessage() << endl << endl;
//		WaitForInput(-1);
//	}
//	cout << "\n\nAll tests passed!\n\n";
//	WaitForInput(0);
//}
//
