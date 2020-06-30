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


// FileIOTestApp.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include "../FileIO/FileIO.h"
#include "../FileIO/SingleFrame.h"
#include "../FileIO/TiltSeries.h"
#include "../FileIO/MovieStack.h"
#include "../FileIO/SERFile.h"
#include "../FileIO/EmFile.h"
#include "../FileIO/TIFFFile.h"
#include "../FileIO/MarkerFile.h"
#include "../FileIO/ImodFiducialFile.h"
#include <future>
#include "../Threading/ThreadPool.h"
#include "../Threading/SpecificBackgroundThread.h"
#include "../FileIO/ImodFiducialFile.h"
#include "../FileIO/MDocFile.h"

//#include "../../Logging/spdlog.h"


void test(FileReader::FileReaderStatus val)
{
	cout << val.bytesRead * 100 / val.bytesToRead << endl;
	cout << "Thread ID: " << this_thread::get_id() << endl;
}

int doSomeStuff(int val1, float val2)
{
	cout << val1 << "; " << val2 << endl;
	cout << "Thread ID: " << this_thread::get_id() << endl;
	return 4;
}

static void status(int percent)
{
	cout << percent << "%" << endl;
}


int _tmain(int argc, _TCHAR* argv[])
{
	//SingleFrame sf("Z:/microscope/Anna/14fef17_/(16).dm4");
	//float testps = sf.GetPixelSize();
	MarkerFile* testf = MarkerFile::ImportFromIMOD("Y:\\Documents\\Misha\\tomo32.fid");
	testf->Save("Y:\\Documents\\Misha\\tomo32.em");

	ImodFiducialFile imod("Y:\\Documents\\Misha\\tomo32.fid");
	bool okres = imod.OpenAndRead();
	int countMarker = imod.GetMarkerCount();
	int countProj = imod.GetProjectionCount();

	TiltSeries ts("C:\\Users\\Michael Kunz\\Desktop\\tomo32.mrc.st");




	MDocFile mdoc("Y:\\microscope\\kunz\\TMV\\06mar17\\Tomo1.st.mdoc");
	vector<MDocEntry> vec = mdoc.GetEntries();

	/**/
	float x[] = { -221, -3457, 1028, 537, -2851 };
	float y[] = { -202, 2402, 3724, -2982, -2651 };
	float z[] = { -10, 60, 396, -185, -357 };

	int projCount = 45;
	int markerCount = 5;
	float* thetas = new float[projCount];
	float* psis = new float[projCount];
	float* mags = new float[projCount];
	float* xshift = new float[projCount];
	float* yshift = new float[projCount];
	float** xpos = new float*[projCount];
	float** ypos = new float*[projCount];

	for (size_t i = 0; i < projCount; i++)
	{
		thetas[i] = (float)i * 3 - 66;
		psis[i] = -5 + 0*thetas[i] /100.0f;
		mags[i] = 1;
		xshift[i] = 0;
		yshift[i] = 0;
		xpos[i] = new float[markerCount];
		ypos[i] = new float[markerCount];
	}

	MarkerFile::projectTestBeads(3712 * 2, markerCount, x, y, z, projCount, thetas, psis, 0.5f, mags, xshift, yshift, 1.0f, xpos, ypos);


	MarkerFile m2(projCount, thetas);
	for (size_t i = 1; i < markerCount; i++)
	{
		m2.AddMarker();
	}
	for (size_t proj = 0; proj < projCount; proj++)
	{
		for (size_t marker = 0; marker < markerCount; marker++)
		{
			m2(MarkerFileItem_enum::MFI_Magnifiaction, proj, marker) = mags[proj];
			m2(MarkerFileItem_enum::MFI_RotationPsi, proj, marker) = psis[proj];
			m2(MarkerFileItem_enum::MFI_X_Shift, proj, marker) = xshift[proj];
			m2(MarkerFileItem_enum::MFI_Y_Shift, proj, marker) = yshift[proj];
			m2(MarkerFileItem_enum::MFI_X_Coordinate, proj, marker) = xpos[proj][marker];
			m2(MarkerFileItem_enum::MFI_Y_Coordinate, proj, marker) = ypos[proj][marker];
		}
	}

	m2.Save("Y:\\microscope\\kunz\\calibration\\Preradiated30sec\\simuMarker_Phi0.5_Rot0.em");

	//ImodFiducialFile imod("Z:\\microscope\\Diana\\29mar17\\29mar17_8_test_alignments\\29mar17_8.st_Alig.fid");
	//ImodFiducialFile imod("Z:/hodirnau/data/PhD/miller-frog/Cryo-EM/Cryo-MillerTrees/02.06-test5/02.06-test5.st_Alig.fid");
	//bool te = imod.OpenAndRead();
	
	MarkerFile* mf = MarkerFile::ImportFromIMOD("Z:\\microscope\\Diana\\29mar17\\29mar17_8_test_alignments\\29mar17_8.st_Alig.fid");
	mf->Save("Z:\\microscope\\Diana\\29mar17\\29mar17_8_test_alignments\\markerImod2.em");

	bool test = MarkerFile::CanReadAsMarkerfile("Z:\\microscope\\kunz\\calibration\\Preradiated30sec\\marker50.em");
	MarkerFile m("Z:\\microscope\\kunz\\calibration\\Preradiated30sec\\marker50.em");

	/*float error = 0;
	float phi = -0.12;
	float* perProj = new float[m.GetProjectionCount()];
	float* perMarker = new float[m.GetMarkerCount()];
	float* x = new float[m.GetMarkerCount()];
	float* y = new float[m.GetMarkerCount()];
	float* z = new float[m.GetMarkerCount()];

	m.SetMagAnisotropy(1.016f, 43.0f, 0, 0);
//	m.Align3D(0, 3712 * 2, 3838 * 2, error, phi, true, true, false, true, true, true, true, 5, 10, 0, perProj, perMarker, x, y, z, &status);

	cout << "Error: " << error << endl;
	
	m.Save("Z:\\microscope\\kunz\\calibration\\Preradiated30sec\\marker50Alig.em");

*/
	return 0;

	/*TiltSeries ts("C:\\Users\\Michael Kunz\\Desktop\\tomo_7.mrc.st"); 
	float ps = ts.GetPixelSize();

	MRCFile::InitHeaders("C:\\Users\\Michael Kunz\\Desktop\\tomo_7LE.mrc", (int)ts.GetWidth(), (int)ts.GetHeight(), ts.GetPixelSize(), DT_FLOAT, true);
	for (size_t i = 0; i < ts.GetImageCount(); i++)
	{
		MRCFile::AddPlaneToMRCFile("C:\\Users\\Michael Kunz\\Desktop\\tomo_7LE.mrc", DT_FLOAT, ts.GetData(i), ts.GetTiltAngle(i));
	}*/

	//SingleFrame sf("Z:\\microscope\\Anna\\F30\\correlative\\Frame.dm3");
	//float pst = sf.GetPixelSize();

	////TIFFFile tif("Z:\\microscope\\geiss\\TestMeasurementsCTF\\TMV+Lacey\\17_DF3s.tif");
	//TIFFFile tif("Z:\\microscope\\Anna\\F30\\16.02.05_2.mouse liver neg\\tif\\Liver extract2_1.tif");
	//tif.OpenAndRead();

	//uchar* data = (uchar*)tif.GetData();

	//bool t = TIFFFile::WriteTIFF("C:\\Users\\Michael Kunz\\Desktop\\1_Copy.tif", tif.GetWidth(), tif.GetHeight()*2, 1, DT_UCHAR, data);
	//

	//SERFile ser("C:\\Users\\Michael Kunz\\Desktop\\1_1.ser");
	//ser.OpenAndRead();

	////float pixSize = sf.GetPixelSize();

	//rand();
	//int t1 = sizeof(TiffFileHeader);
	//int t2 = sizeof(TiffTag);

	//TIFFFile tiff("C:\\Users\\Michael Kunz\\Desktop\\1.tif");
	//tiff.OpenAndRead();

	//ushort* d = (ushort*)tiff.GetData();
	///*float* f = new float[4096*4096];

	//for (size-t i = 0; i < 4096*4096; i++)
	//{
	//	f[i] = d[i];
	//}*/
	//MRCFile::InitHeaders("C:\\Users\\Michael Kunz\\Desktop\\Test2.mrc", 4096, 4096, 1, DT_USHORT);
	//MRCFile::AddPlaneToMRCFile("C:\\Users\\Michael Kunz\\Desktop\\Test2.mrc", DT_USHORT, d, 0);
	//spdlog::basic_logger_mt("logger", false);
	/*spdlog::get("logger")->info("Hallo logger!");
	spdlog::set_level(spdlog::level::info);*/

	// Create basic file logger (not rotated)
	//auto my_logger = spd::basic_logger_mt("basic_logger", "logs/basic.txt");



	/*auto erg2 = RunInThread(doSomeStuff, 10, 15.0f);
	erg2.get();*/
	////ThreadPool tt(1);
	//ThreadPool* a = SingletonThread::Get(12);
	//auto erg = a->enqueue(doSomeStuff, 14, 15.0f);
	//erg.get();

	//erg = SingletonThread::Get(12)->enqueue(doSomeStuff, 15, 16.0f);
	//erg.get();

	//erg = SingletonThread::Get(13)->enqueue(doSomeStuff, 16, 17.0f);
	//erg.get();

	///*future<TiltSeries*> fut = async(TiltSeries::CreateInstance, "Z:\\microscope\\kunz\\driff_test\\driff_test.st", test);
	//cout << "Loading file..." << endl;
	//cout << "Thread ID: " << this_thread::get_id() << endl;
	//TiltSeries* sf = fut.get();

	//delete sf;*/

	//EmFile em("C:\\Users\\Michael Kunz\\Desktop\\markerfile02.em");
	//em.OpenAndRead();

	//bool erg3 = SingleFrame::CanReadFile("C:\\Users\\Michael Kunz\\Desktop\\markerfile02.em");
	//if (erg3)
	//SingleFrame s("C:\\Users\\Michael Kunz\\Desktop\\markerfile02.em");

/*	int* dataI = (int*)s.GetData();
	ushort* dataF = (ushort*)s.GetData();

	ushort mini = 40000000, maxi = 0;

	for (size_t i = 0; i < 4096 * 4096; i++)
	{
		dataF[i] = (ushort)dataI[i];

		if (dataF[i] < mini) mini = dataF[i];
		if (dataF[i] > maxi) maxi = dataF[i];

	}

	MRCFile::InitHeaders("C:\\Users\\Michael Kunz\\Desktop\\Test2.mrc", s.GetWidth(), s.GetHeight(), 1, DT_USHORT);
	MRCFile::AddPlaneToMRCFile("C:\\Users\\Michael Kunz\\Desktop\\Test2.mrc", DT_USHORT, s.GetData(), 0);
*/

	//MovieStack ms("Z:\\microscope\\Cramer2D\\20100124\\impol2_0625.mrc");
	
	/*MRCFile::InitHeaders("F:\\TestData\\testWrite.mrc", ms.GetWidth(), ms.GetHeight(), 1.1f, DT_FLOAT);
	for (size_t i = 0; i < ms.GetImageCount(); i++)
	{
		MRCFile::AddPlaneToMRCFile("F:\\TestData\\testWrite.mrc", DT_FLOAT, ms.GetData(i), 0);
	}*/
	return 0;
}

