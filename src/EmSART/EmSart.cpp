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


//#define USE_MPI
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "EmSartDefault.h"
#include "Projection.h"
#include "Volume.h"
#include <CudaContext.h>
#include "utils/Config.h"
#include "io/FileSource.h"
#ifdef USE_MPI
#include "io/MPISource.h"
#endif
#include <MarkerFile.h>
#include "io/writeBMP.h"
#include <CtfFile.h>
#include <time.h>
#include <cufft.h>
#include <npp.h>
#include <algorithm>
#include <iomanip>
#include "utils/SimpleLogger.h"
#include "Reconstructor.h"
#include "kernels/kernels.h"
#include "ncurses.h"

using namespace std;
using namespace Cuda;

#ifdef WIN32
#define round(x) ((x)>=0)?(int)((x)+0.5):(int)((x)-0.5)
//#define CUDACONFFILE "cuda.cfg"
#define CONFFILE "emsart.cfg"
#else
//#define CUDACONFFILE "/home/Group/Software/tomography/kunzFunctions/EmSART/cuda.cfg"
#define CONFFILE "emsart.cfg"
#include <unistd.h>
#include <limits.h>
#endif


void WaitForInput(int exitCode)
{
    char c;
    cout << ("\nPress <Enter> to exit...");
    c = cin.get();
    exit(exitCode);
}

int main(int argc, char* argv[])
{
    int mpi_part = 0;

    int mpi_size = 1;
    const int mpi_max_name_size = 256;
    char mpi_name[mpi_max_name_size];
    int mpi_sizename = mpi_max_name_size;
    int mpi_host_id = 0;
    int mpi_host_rank = 0;
    int mpi_offset = 0;

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_part);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Get_processor_name(mpi_name, &mpi_sizename);


	vector<string> hostnames;
	vector<string> singlehostnames;
    //printf("MPI process %d of %d on PC %s\n", mpi_part, mpi_size, mpi_name);

	if (mpi_part == 0)
	{
		hostnames.push_back(string(mpi_name));
		for (int i = 1; i < mpi_size; i++)
		{
			char tempname[mpi_max_name_size];
			MPI_Recv(tempname, mpi_max_name_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			hostnames.push_back(string(tempname));
		}

		//printf("Found %d hostnames\n", hostnames.size());

		for (int i = 0; i < mpi_size; i++)
		{
			bool exists = false;
			for (int h = 0; h < singlehostnames.size(); h++)
			{
				if (hostnames[i] == singlehostnames[h])
					exists = true;
			}
			if (!exists)
				singlehostnames.push_back(hostnames[i]);
		}

		//sort host names alphabetically to obtain deterministic host IDs
		sort(singlehostnames.begin(), singlehostnames.end());

		for (int i = 1; i < mpi_size; i++)
		{
			int host_id;
			int host_rank = 0;
			int offset = 0;

			string hostname = hostnames[i];

			for (int h = 0; h < singlehostnames.size(); h++)
			{
				if (singlehostnames[h] == hostname)
				{
					host_id = h;
					break;
				}
			}

			for (int h = 0; h < i; h++)
			{
				if (hostnames[h] == hostname)
				{
					host_rank++;
				}
			}

			for (int h = 0; h < host_id; h++)
			{
				for (int n = 0; n < hostnames.size(); n++)
				{
					if (hostnames[n] == singlehostnames[h])
					{
						offset++;
					}
				}
			}

			MPI_Send(&host_id, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&host_rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}

		for (int h = 0; h < singlehostnames.size(); h++)
		{
			if (singlehostnames[h] == string(mpi_name))
			{
				mpi_host_id = h;
				break;
			}
		}


		for (int h = 0; h < mpi_host_id; h++)
		{
			for (int n = 0; n < hostnames.size(); n++)
			{
				if (hostnames[n] == singlehostnames[h])
				{
					mpi_offset++;
				}
			}
		}
		mpi_host_rank = 0;

	}
	else
	{
		MPI_Send(mpi_name, mpi_max_name_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

		MPI_Recv(&mpi_host_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&mpi_host_rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&mpi_offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	printf("Host ID: %d; host rank: %d; offset: %d; global rank: %d; name: %s\n", mpi_host_id, mpi_host_rank, mpi_offset, mpi_part, mpi_name);fflush(stdout);

	MPI_Barrier(MPI_COMM_WORLD);
#endif

    clock_t start, stop;
    double runtime = 0.0;
    CudaContext* cuCtx;

    string logfile;
    bool doLog = false;
    if (mpi_part == 0)
    {
        for (int arg = 0; arg < argc - 1; arg++)
        {
            if (string(argv[arg]) == "-log")
            {
                logfile = string(argv[arg+1]);
                doLog = true;
            }
        }
    }

    SimpleLogger log(logfile, SimpleLogger::LOG_ERROR, !doLog);

    try
    {
        if (mpi_part == 0) printf("\n\n                                EmSART 2.0\n\n\n");
        if (mpi_part == 0) printf("Read configuration file ");
        //Load configuration files
        Configuration::Config aConfig = Configuration::Config::GetConfig(CONFFILE, argc, argv, mpi_part, NULL);
        if (mpi_part == 0) printf("Done\n");fflush(stdout);

        if (mpi_part == 0) printf("Projection source: %s\n", aConfig.ProjectionFile.c_str());
        if (mpi_part == 0) printf("Marker source: %s\n", aConfig.MarkerFile.c_str());
        if (mpi_part == 0) printf("Volume shifts: %f, %f, %f\n", aConfig.VolumeShift.x, aConfig.VolumeShift.y, aConfig.VolumeShift.z);
        if (mpi_part == 0) printf("Volume file name: %s\n", aConfig.OutVolumeFile.c_str());
        if (mpi_part == 0) printf("Lambda: %f\n", aConfig.Lambda);
        if (mpi_part == 0) printf("Iterations: %i\n\n", aConfig.Iterations);

#ifdef USE_MPI
        log << "Running on " << mpi_size << " GPUs in " << (int)singlehostnames.size() << " Hosts:" << endl;
		for (int i = 0; i < singlehostnames.size(); i++)
		{
			log << "Host " << i << ": " << singlehostnames[i] << endl;
		}
#else
        log << "Running in single GPU (no MPI) mode" << endl;
#endif

        log << "Configuration file: " << aConfig.GetConfigFileName() << endl;
        log << "Projection source: " <<aConfig.ProjectionFile << endl;
        log << "Marker source: " << aConfig.MarkerFile << endl;
        log << "Volume file name: " << aConfig.OutVolumeFile << endl;
        log << "Volume shifts: " << aConfig.VolumeShift << endl;
        log << "Lambda: " << aConfig.Lambda << endl;
        log << "Iterations: " << aConfig.Iterations << endl;
        log << "Performing CTF correction: " << (aConfig.CtfMode != Configuration::Config::CTFM_NO ? "TRUE" : "FALSE") << endl;
        if (aConfig.CtfMode != Configuration::Config::CTFM_NO)
        {
            log << "Ignore volume Z-shift for CTF correction: " << (aConfig.IgnoreZShiftForCTF ? "TRUE" : "FALSE") << endl;
            log << "Slice thickness for CTF correction in nm: " << aConfig.CTFSliceThickness << endl;
        }

        CtfFile* defocus = NULL;

        if (aConfig.CtfMode == Configuration::Config::CTFM_YES)
        {
            defocus = new CtfFile(aConfig.CtfFile);
        }


        //Check volume dimensions:
        bool recDimOK = true;
        if (aConfig.RecDimensions.x % 4 != 0)
        {
            printf("Error: RecDimensions.x (%d) is not a multiple of 4\n", aConfig.RecDimensions.x);
            recDimOK = false;

            log << SimpleLogger::LOG_ERROR;
            log << "RecDimensions.x (" << aConfig.RecDimensions.x << ") is not a multiple of 4" << endl;
        }
        if (aConfig.RecDimensions.y % 2 != 0)
        {
            printf("Error: RecDimensions.y (%d) is not even\n", aConfig.RecDimensions.y);
            recDimOK = false;

            log << SimpleLogger::LOG_ERROR;
            log << "RecDimensions.y (" << aConfig.RecDimensions.y << ") is not even" << endl;
        }

        if (!recDimOK) WaitForInput(-1);

        printf("Create CUDA context on device %d ... \n", aConfig.CudaDeviceIDs[mpi_offset + mpi_host_rank]);fflush(stdout);
        //Create CUDA context
        cuCtx = Cuda::CudaContext::CreateInstance(aConfig.CudaDeviceIDs[mpi_offset + mpi_host_rank]);

        printf("Using CUDA device %s\n", cuCtx->GetDeviceProperties()->GetDeviceName().c_str());fflush(stdout);

        printf("Compute Capability: %f\n", cuCtx->GetDeviceProperties()->GetComputeCapability());fflush(stdout);

        printf("Available Memory on device: %llu MB\n", cuCtx->GetFreeMemorySize() / 1024 / 1024);fflush(stdout);

        ProjectionSource* projSource;
        //Load projection data file
        if (mpi_part == 0)
        {
            if (aConfig.GetFileReadMode() == Configuration::Config::FRM_DM4 ||
                aConfig.GetFileReadMode() == Configuration::Config::FRM_MRC)
            {
                printf("\nLoading projections...\n");
                projSource = new FileSource(aConfig.ProjectionFile);


                printf("\nLoaded %d projections.\n\n", projSource->GetProjectionCount());
            }
            else
            {
                printf("Error: Projection file format not supported. Supported formats are: DM4 file series, MRC stacks, ST stacks.");
                log << SimpleLogger::LOG_ERROR;
                log << "Projection file format not supported. Supported formats are: DM4 file series, MRC stacks, ST stacks." << endl;
                WaitForInput(-1);
            }

#ifdef USE_MPI
            float pixelsize = projSource->GetPixelSize();
			int dims[4];
			dims[0] = projSource->GetWidth();
			dims[1] = projSource->GetHeight();
			dims[2] = projSource->GetProjectionCount();
			dims[3] = *((int*)&pixelsize);
			MPI_Bcast(dims, 4, MPI_INT, 0, MPI_COMM_WORLD);
#endif
        }
#ifdef USE_MPI
        else
		{
			int dims[4];
			MPI_Bcast(dims, 4, MPI_INT, 0, MPI_COMM_WORLD);
			projSource = new MPISource(dims[0], dims[1], dims[2], *((float*)&(dims[3])));
		}
#endif

        //Load marker/alignment file
        MarkerFile markers(aConfig.MarkerFile, aConfig.ReferenceMarker);

        //Create projection object to handle projection data
        //Projection proj(projSource, &markers, aConfig.WBP_NoSART);
        Projection proj(projSource, &markers, false);

        //Create volume dataset (host)
        Volume<float> *vol = NULL;
#ifdef USE_MPI
        if (aConfig.FP16Volume)
			volFP16 = new Volume<unsigned short>(aConfig.RecDimensions, mpi_size, mpi_part);
		else
			vol = new Volume<float>(aConfig.RecDimensions, mpi_size, mpi_part);
#else
        vol = new Volume<float>(aConfig.RecDimensions);
#endif

        vol->PositionInSpace(aConfig.VoxelSize, aConfig.VolumeShift, proj.GetMinimumTiltShift(), 0, 0, 0);
        log << "Using FP32 internal storage format for volume";

        if (aConfig.FP16Volume && !aConfig.WriteVolumeAsFP16)
            log << "; Convert to FP32 when saving to file";
        log << endl;

        float3 subVolDim;
        subVolDim = vol->GetSubVolumeDimension(mpi_part);

        size_t sizeDataType;
        sizeDataType = sizeof(float);

        if (mpi_part == 0) printf("Memory space required by volume data: %llu MB\n", (size_t)aConfig.RecDimensions.x * (size_t)aConfig.RecDimensions.y * (size_t)aConfig.RecDimensions.z * sizeDataType / 1024 / 1024);
        if (mpi_part == 0) printf("Memory space required by partial volume: %llu MB\n", (size_t)aConfig.RecDimensions.x * (size_t)aConfig.RecDimensions.y * (size_t)subVolDim.z * sizeDataType / 1024 / 1024);

        //Load Kernels
        KernelModules modules(cuCtx);

        //Alloc device variables
        float3 volSize;
        CUarray_format arrayFormat;

        volSize = vol->GetSubVolumeDimension(mpi_part);
        arrayFormat = CU_AD_FORMAT_FLOAT;

        uint3 volDimU = make_uint3((uint)volSize.x, (uint)volSize.y, (uint)volSize.z);
        DeviceVolume deviceVolume(volDimU, modules);


        if (mpi_part == 0) printf("Copy volume to device ... ");

        bool volumeIsEmpty = false;


        log << "Volume dimensions: " << vol->GetDimension() << endl;
        log << "Sub-Volume dimensions: " << endl;
        for (int sv = 0; sv < vol->GetSubVolumeCount(); sv++)
            log << "Sub-Volume " << sv << ": " << vol->GetSubVolumeDimension(sv) << endl;


        if (mpi_part == 0) printf("Done\n");fflush(stdout);


        int* indexList;
        int projCount;
        proj.CreateProjectionIndexList(PLT_RANDOM_START_ZERO_TILT, &projCount, &indexList);

        if (mpi_part == 0)
        {
            printf("Projection index list:\n");
            log << "Projection index list:" << endl;
            for (int i = 0; i < projCount; i++)
            {
                printf("%3d,", indexList[i]);
                log << indexList[i];
                if (i < projCount - 1)
                    log << ", ";
            }
            log << endl;
            printf("\b \n\n");

        }

        Reconstructor reconstructor(aConfig, proj, projSource, markers, *defocus, modules, mpi_part, mpi_size);
        reconstructor.PlanCTFCorrection(vol, markers.GetProjectionCount(), projSource->GetProjectionCount(), indexList);

        if (mpi_part == 0) printf("Free Memory on device after allocations: %llu MB\n", cuCtx->GetFreeMemorySize() / 1024 / 1024);

/////////////////////////////////////
/// Filter Projections
/////////////////////////////////////
        if (mpi_part == 0)
        {

            float lp = aConfig.fourFilterLP, hp = aConfig.fourFilterHP, lps = aConfig.fourFilterLPS, hps = aConfig.fourFilterHPS;
            bool skipFilter = aConfig.SkipFilter;

            log << "Bandpass filter for projections applied: " << (skipFilter ? "false" : "true") << endl;
            log << "Bandpass filter values (lp, lps, hp, hps): " << lp << ", " << lps << ", " << hp << ", " << hps << endl;

            log << "Projection datatype: " << projSource->GetDataType() << endl;

            if (aConfig.ProjectionNormalization == Configuration::Config::PNM_STANDARD_DEV)
                log << "Normalizing projections by standard deviation [im = (im - mean) / std]" << endl;
            else
                log << "Normalizing projections by mean [im = (im - mean) / mean]" << endl;

            log << "Scaling projection values by: " << aConfig.ProjectionScaleFactor << endl;
            log << "Pixel size is: " << proj.GetPixelSize() << " nm" << endl;

            log << "Projection statistics:" << endl;

            printf("\r\n");
            for (int i = 0; i < projSource->GetProjectionCount(); i++)
            {
                if (!markers.CheckIfProjIndexIsGood(i))
                {
                    continue;
                }

                //printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
                printf("\r");
                printf("Filtering projection: %i", i);
                log << "Projection " << i;
                fflush(stdout);

                //projSource->GetProjection(i) always points to an array with an element size of 4 bytes,
                //Even if original data is stored in shorts! We can therefore cast data and keep the same pointer.
                char* imgUS = projSource->GetProjection(i);

                //Check if data format is supported
                if (projSource->GetDataType() != DT_SHORT &&
                    projSource->GetDataType() != DT_USHORT &&
                    projSource->GetDataType() != DT_INT &&
                    projSource->GetDataType() != DT_UINT &&
                    projSource->GetDataType() != DT_FLOAT)
                {
                    cerr << "Projections have wrong data type: supported types are: short, ushort, int, uint and float.";
                    log << SimpleLogger::LOG_ERROR;
                    log << "Projections have wrong data type: supported types are: short, ushort, int, uint and float." << endl;
                    WaitForInput(-1);
                }

                float meanValue, stdValue;
                int badPixels;
                reconstructor.PrepareProjection(imgUS, i, meanValue, stdValue, badPixels);

                printf(" Bad Pixels: %d Mean: %f Std: %f", badPixels, meanValue, stdValue);
                log << ": Bad Pixels: " << badPixels << " Mean: " << meanValue << " Std. dev.: " << stdValue << endl;
            }
        }

/////////////////////////////////////
/// End Filter Projections
/////////////////////////////////////


        if (mpi_part == 0)printf("\nPixel size is: %f nm, Cs: %.2f mm, Voltage: %.2f kV\n", proj.GetPixelSize(), aConfig.Cs, aConfig.Voltage);

        int SIRTcount = aConfig.SIRTCount;
        if (aConfig.WBP_NoSART)
            SIRTcount = 1;
        if (aConfig.WBP_NoSART)
            aConfig.Iterations = 1;


        float** SIRTBuffer = new float*[SIRTcount];
        for (int i = 0; i < SIRTcount; i++)
        {
            uint size = proj.GetWidth() * proj.GetHeight();
            SIRTBuffer[i] = new float[size];
            memset(SIRTBuffer[i], 0, size * 4);
        }

        if (mpi_part == 0)printf("\n\nStart reconstruction ...\n\n");
        fflush(stdout);
        start = clock();

        auto dist = new float[proj.GetProjCount() * proj.GetWidth() * proj.GetHeight()];

        // Whether or not to use
        bool initialize_adhoc = true;

        if (aConfig.WBP_NoSART){
            if (!aConfig.SNRFile.empty()){
                string type = "volume";
                reconstructor.LoadSNR(aConfig.SNRFile, type);
                initialize_adhoc = false;
            }
        }

/////////////////////////////////////
/// Begin Reconstruction
/////////////////////////////////////

        stringstream output;
        output << std::fixed;
        output << std::setprecision(2);
        float total_steps = (float)aConfig.Iterations * (float)proj.GetGoodProjCount();

        for (int iter = 0; iter < aConfig.Iterations; iter++) {//2; iter++){//aConfig.Iterations; iter++) {
            for (uint projIdx = 0; projIdx < projCount; projIdx++) {//3;projIdx++){//projCount; projIdx++) {
                // Index in stack
                int stackIdx = indexList[projIdx];

                // On first iteration or in case of WBP without prior knowledge we need to initialize the volume using
                // an ad-hoc wiener factor.
                bool useSNR = true;
                if (iter < 1 && initialize_adhoc) useSNR = false;

                // Some terminal output
                float progress = (float)(iter * proj.GetGoodProjCount() + projIdx) / total_steps * 100;
                output.str(std::string());
                output << "Progress: " << progress <<"%"
                       << " | Iteration: " << iter + 1
                       << " | Step: " << projIdx + 1 << " | ";


                reconstructor.ResetProjectionsDevice();

                // Forward projection only when doing SART
                if (!aConfig.WBP_NoSART) {
                    // Forward projection
                    reconstructor.ForwardProjectionParent(vol,
                                                          deviceVolume,
                                                          stackIdx, false, iter, output);


                    // Distance projection
                    cout << "\r\e[K" << flush;
                    cout << output.str() << "Ds " << stackIdx << flush;

                    float *distp = dist + stackIdx * proj.GetHeight() * proj.GetWidth();
                    if (iter < 1) {
                        reconstructor.DistanceParent(vol, stackIdx, false, iter, false);
                        reconstructor.CopyDistanceImageToHost(distp);
                    } else {
                        reconstructor.CopyDistanceImageToDevice(distp);
                    }

                    if (aConfig.WriteDebug) {
                        {
                            auto img = new float[proj.GetWidth() * proj.GetHeight()];
                            reconstructor.CopyProjectionToHost(img);
                            stringstream DP;
                            DP << "forward_" << stackIdx << "_" << iter << ".em";
                            emwrite(DP.str(), img, proj.GetWidth(),
                                    proj.GetHeight());
                            delete[] img;
                        }

                        {
                            auto img = new float[proj.GetWidth() * proj.GetHeight()];
                            reconstructor.CopyDistanceImageToHost(img);
                            stringstream DP;
                            DP << "distance_" << stackIdx << "_" << iter << ".em";
                            emwrite(DP.str(), img, proj.GetWidth(),
                                    proj.GetHeight());
                            delete[] img;
                        }
                    }

                }

                if (aConfig.WriteDebug) {
                    {
                        stringstream DP;
                        DP << "real_" << stackIdx << "_" << iter << ".em";
                        emwrite(DP.str(), (float *) projSource->GetProjection(stackIdx), proj.GetWidth(),
                                proj.GetHeight());
                    }
                }

                // Compare
                cout << "\r\e[K" << flush;
                cout << output.str() << "Df " << stackIdx << flush;

                // Compare only when doing SART
                if (!aConfig.WBP_NoSART) {
                    reconstructor.Compare(vol, projSource->GetProjection(stackIdx), stackIdx, true, iter);

                    if (aConfig.WriteDebug) {
                        auto img = new float[proj.GetWidth()*proj.GetHeight()];
                        reconstructor.CopyProjectionToHost(img);
                        stringstream DP;
                        DP << "comp_" << stackIdx << "_" << iter <<".em";
                        emwrite(DP.str(), img, proj.GetWidth(),
                                proj.GetHeight());
                        delete[] img;
                    }

                } else {
                    reconstructor.PrepareForWBP(vol, projSource->GetProjection(stackIdx), stackIdx, iter);

                    if (aConfig.WriteDebug) {
                        {
                            auto img = new float[proj.GetWidth() * proj.GetHeight()];
                            reconstructor.CopyProjectionToHost(img);
                            stringstream DP;
                            DP << "wbp_prepped_" << stackIdx << "_" << iter << ".em";
                            emwrite(DP.str(), img, proj.GetWidth(),
                                    proj.GetHeight());
                            delete[] img;
                        }
                    }
                }

                // Back projection
                reconstructor.BackProjectionParent(vol,
                                                   deviceVolume,
                                                   stackIdx, 1, iter, output, useSNR);
            }


            if (aConfig.WriteDebug) {
                auto img = new float[vol->GetSubVolumeSizeInVoxels(0)];
                deviceVolume.array_card().CopyFromArrayToHost(img);
                for (int i=0; i<vol->GetSubVolumeSizeInVoxels(0); i++){
                    img[i] = img[i] * -1.f;
                }
                stringstream DP;
                DP << "volume_"  << iter << ".em";
                emwrite(DP.str(), img, vol->GetDimension().x,
                        vol->GetDimension().y, vol->GetDimension().z);
                delete[] img;
            }
        }

/////////////////////////////////////
/// End Reconstruction
/////////////////////////////////////

/////////////////////////////////////
/// Begin Saving
/////////////////////////////////////

        if (!aConfig.SNRFile.empty() && !aConfig.WBP_NoSART){
            string type = "volume";
            reconstructor.SaveSNR(aConfig.SNRFile, type);
        }

		if (!(aConfig.FP16Volume && !aConfig.WriteVolumeAsFP16))
		{
			if (mpi_part == 0) printf("\n\nCopying Data back to host ... ");fflush(stdout);

            deviceVolume.CardToHost(vol->GetPtrToSubVolume(mpi_part));

			if (mpi_part == 0) printf("Done\n");
		}

        stop = clock();
        runtime = (double) (stop-start)/CLOCKS_PER_SEC;

        if (mpi_part == 0) printf("\n\nTotal time for reconstruction: %.2i:%.2i min.\n\n", (int)floor(runtime / 60.0), (int)floor(((runtime / 60.0) - floor(runtime / 60.0))*60.0));

		if (mpi_part == 0)
		{
			std::ofstream* mVol = new std::ofstream();
			mVol->open(aConfig.OutVolumeFile.c_str(), ios_base::out | ios_base::binary);
			if (!(mVol->is_open() && mVol->good()))
				printf("Cannot open File!\n");
			else
			{
				printf("Write Volume to disk (part 0) ... ");fflush(stdout);
				Configuration::Config::FILE_SAVE_MODE fsm = aConfig.GetFileSaveMode();
				float3 dims;

                dims = vol->GetDimension();
                vol->Invert();


				size_t sizeDataType = sizeof(float);
				if (fsm == Configuration::Config::FSM_MRC)
				{
					sizeDataType = sizeof(float);
					MrcHeader header;
					memset(&header, 0, sizeof(MrcHeader));
					header.NX = (int)dims.x;
					header.NY = (int)dims.y;
					header.NZ = (int)dims.z;
					header.MODE = MRCMODE_F;
					if (aConfig.FP16Volume && aConfig.WriteVolumeAsFP16)
						header.MODE = MRCMODE_HALF;

					header.NXSTART = 0;
					header.NYSTART = 0;
					header.NZSTART = 0;
					header.MX = (int)dims.x;
					header.MY = (int)dims.y;
					header.MZ = (int)dims.z;

					header.Xlen = proj.GetPixelSize() * dims.x * aConfig.VoxelSize.x * 10.0f;
					header.Ylen = proj.GetPixelSize() * dims.y * aConfig.VoxelSize.y * 10.0f;
					header.Zlen = proj.GetPixelSize() * dims.z * aConfig.VoxelSize.z * 10.0f;

					header.MAPC = MRCAXIS_X;
					header.MAPR = MRCAXIS_Y;
					header.MAPS = MRCAXIS_Z;
					mVol->write((char*)&header, sizeof(MrcHeader));
				}
				else if (fsm == Configuration::Config::FSM_EM)
				{
					EmHeader header;
					memset(&header, 0, sizeof(EmHeader));

					header.MachineCoding = EMMACHINE_PC;
					header.DataType = EMDATATYPE_FLOAT;
					if (aConfig.FP16Volume && aConfig.WriteVolumeAsFP16)
						header.DataType = EMDATATYPE_HALF;

					header.DimX = (int)dims.x;
					header.DimY = (int)dims.y;
					header.DimZ = (int)dims.z;
					header.ObjectPixelSize = (int)(proj.GetPixelSize() * aConfig.VoxelSize.x * 1000);

					mVol->write((char*)&header, sizeof(EmHeader));
					sizeDataType = sizeof(float);
				}
				else
				{
					int temp = (int)dims.x;
					mVol->write((char*)&temp, 4);
					temp = (int)dims.y;
					mVol->write((char*)&temp, 4);
					temp = (int)dims.z;
					mVol->write((char*)&temp, 4);

					float maxVal = -999999.0f;
					if (!aConfig.FP16Volume)
						maxVal = vol->GetMax();
					mVol->write((char*)&maxVal, 4);
				}



                //close file. It will be reopened in WriteToFile-method
                mVol->flush();
                mVol->close();

                vol->WriteToFile(aConfig.OutVolumeFile, mpi_part);

                printf("Done\n");fflush(stdout);

#ifdef USE_MPI
				int ack = 0;
				if (mpi_size > 1)
					MPI_Send(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
#endif
			}
		}
#ifdef USE_MPI
		else
		{
			int ack = 0;
			MPI_Recv(&ack, 1, MPI_INT, mpi_part-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			
			printf("\nWrite Volume to disk (part %d) ... ", mpi_part);
			if (aConfig.FP16Volume && !aConfig.WriteVolumeAsFP16)
			{				
				std::ofstream* mVol = new std::ofstream();
				mVol->open(aConfig.OutVolumeFile.c_str(), ios_base::out | ios_base::binary | ios_base::app);
				if (!(mVol->is_open() && mVol->good()))
				{
					printf("Cannot open File!\n");fflush(stdout);
				}
				else
				{
					size_t sizeDataType = sizeof(float);
					printf("   ");
					float3 dim = volFP16->GetSubVolumeDimension(mpi_part);
					size_t dimI = (size_t)dim.x * (size_t)dim.y * sizeDataType;
					int alreadyDone = 0;
					for (int subVol = 0; subVol < mpi_part; subVol++)
					{
						alreadyDone += (int)volFP16->GetSubVolumeDimension(subVol).z;
					}
					float* volTemp_h = new float[aConfig.RecDimensions.x * aConfig.RecDimensions.y];
					for (int zSlice = 0; zSlice < dim.z; zSlice++)
					{
						reconstructor.ConvertVolumeFP16(volTemp_h, surfObj, zSlice);
						mVol->write((char*)volTemp_h, dimI);
						mVol->flush();
					
						printf("\b\b\b\b% 3d%%", (alreadyDone + zSlice + 1) * 100 / (int)volFP16->GetDimension().z);fflush(stdout);
					}
					mVol->close();fflush(stdout);
					delete[] volTemp_h;
				}
			}
			else
			{
				if (aConfig.FP16Volume)
				{
					volFP16->WriteToFile(aConfig.OutVolumeFile, mpi_part);
				}
				else
				{
					vol->Invert();
					vol->WriteToFile(aConfig.OutVolumeFile, mpi_part);
				}
				printf("Done\n");fflush(stdout);
			}


			int ack2 = 0;
			if (mpi_part + 1 < mpi_size)
				MPI_Send(&ack2, 1, MPI_INT, mpi_part + 1, 0, MPI_COMM_WORLD);
			
		}
#endif
	}

/////////////////////////////////////
/// End Saving
/////////////////////////////////////

	catch (exception& e)
	{
		log << SimpleLogger::LOG_ERROR;
		log << "An error occured: " << string(e.what()) << endl;
		cout << "\n\nERROR:\n";
		cout << e.what() << endl << endl;
		WaitForInput(-1);
	}
	if (mpi_part == mpi_size - 1)
		cout << endl;
	
	CudaContext::DestroyContext(cuCtx);
#ifdef USE_MPI
	MPI_Finalize();
#endif
}
