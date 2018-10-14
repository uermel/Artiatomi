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
#include "default.h"
#include "Projection.h"
#include "Volume.h"
//#include "Kernels.h"
#include "cuda/CudaArrays.h"
#include "cuda/CudaContext.h"
#include "cuda/CudaTextures.h"
#include "cuda/CudaKernel.h"
#include "cuda/CudaDeviceProperties.h"
#include "utils/Config.h"
//#include "utils/CudaConfig.h"
#include "utils/Matrix.h"
#include "io/Dm4FileStack.h"
#include "io/MRCFile.h"
#ifdef USE_MPI
#include "io/MPISource.h"
#endif
#include "io/MarkerFile.h"
#include "io/writeBMP.h"
#include "io/mrcHeader.h"
#include "io/emHeader.h"
#include "io/CtfFile.h"
#include <time.h>
#include <cufft.h>
#include <npp.h>
//#include "CudaKernelBinarys.h"
#include <algorithm>
#include "utils/SimpleLogger.h"
#include "Reconstructor.h"

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
	    if (mpi_part == 0) printf("\n\n                          EmSART 2.0 Geometry testing\n\n\n");
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

	    printf("Create CUDA context on device %i ... \n", aConfig.CudaDeviceIDs[mpi_offset + mpi_host_rank]);fflush(stdout);
		//Create CUDA context
		cuCtx = Cuda::CudaContext::CreateInstance(aConfig.CudaDeviceIDs[mpi_offset + mpi_host_rank]);
        
        printf("Using CUDA device %s\n", cuCtx->GetDeviceProperties()->GetDeviceName().c_str());fflush(stdout);

        printf("Available Memory on device: %i MB\n", cuCtx->GetFreeMemorySize() / 1024 / 1024);fflush(stdout);

        ProjectionSource* projSource;
		//Load projection data file
		if (mpi_part == 0)
		{
			if (aConfig.GetFileReadMode() == Configuration::Config::FRM_DM4)
			{
				projSource = new Dm4FileStack(aConfig.ProjectionFile);
				printf("\nLoading projections...\n");
				if (!projSource->OpenAndRead())
				{
					printf("Error: cannot read projections from %s.\n", aConfig.ProjectionFile.c_str());
					WaitForInput(-1);
				}
				projSource->ReadHeaderInfo();

				printf("Loaded %d dm4 projections.\n\n", projSource->DimZ);
			}
			else if(aConfig.GetFileReadMode() == Configuration::Config::FRM_MRC)
			{
				//Load projection data file
				projSource = new MRCFile(aConfig.ProjectionFile);
				((MRCFile*)projSource)->OpenAndReadHeader();
			}
			else
			{
				printf("Error: Projection file format not supported. Supported formats are: DM4 file series, MRC stacks, ST stacks.");
				log << SimpleLogger::LOG_ERROR;
				log << "Projection file format not supported. Supported formats are: DM4 file series, MRC stacks, ST stacks." << endl;
				WaitForInput(-1);
			}

#ifdef USE_MPI
			float pixelsize = projSource->PixelSize[0];
			int dims[4];
			dims[0] = projSource->DimX;
			dims[1] = projSource->DimY;
			dims[2] = projSource->DimZ;
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
		Projection proj(projSource, &markers);

		//Create volume dataset (host)
		Volume<unsigned short> *volFP16 = NULL;
		Volume<float> *vol = NULL;
#ifdef USE_MPI
		if (aConfig.FP16Volume)
			volFP16 = new Volume<unsigned short>(aConfig.RecDimensions, mpi_size, mpi_part);
		else
			vol = new Volume<float>(aConfig.RecDimensions, mpi_size, mpi_part);
#else
		if (aConfig.FP16Volume)
			volFP16 = new Volume<unsigned short>(aConfig.RecDimensions);
		else
			vol = new Volume<float>(aConfig.RecDimensions);
#endif
		



		if (aConfig.FP16Volume)
		{
			volFP16->PositionInSpace(aConfig.VoxelSize, aConfig.VolumeShift, proj.GetMinimumTiltShift());
			log << "Using FP16 internal storage format for volume";
		}
		else
		{
			vol->PositionInSpace(aConfig.VoxelSize, aConfig.VolumeShift, proj.GetMinimumTiltShift());
			log << "Using FP32 internal storage format for volume";
		}
		if (aConfig.FP16Volume && !aConfig.WriteVolumeAsFP16)
			log << "; Convert to FP32 when saving to file";
		log << endl;

		float3 subVolDim;
		if (aConfig.FP16Volume)
			subVolDim = volFP16->GetSubVolumeDimension(mpi_part);
		else
			subVolDim = vol->GetSubVolumeDimension(mpi_part);
		
        size_t sizeDataType;
		if (aConfig.FP16Volume)
		{
			sizeDataType = sizeof(unsigned short);
		}
		else
		{
			sizeDataType = sizeof(float);
		}
        if (mpi_part == 0) printf("Memory space required by volume data: %i MB\n", aConfig.RecDimensions.x * aConfig.RecDimensions.y * aConfig.RecDimensions.z * sizeDataType / 1024 / 1024);
        if (mpi_part == 0) printf("Memory space required by partial volume: %i MB\n", aConfig.RecDimensions.x * aConfig.RecDimensions.y * (size_t)subVolDim.z * sizeDataType / 1024 / 1024);

		//Load Kernels
		KernelModuls modules(cuCtx);

		//Alloc device variables
		float3 volSize;
		CUarray_format arrayFormat;
		if (aConfig.FP16Volume)
		{
			volSize = volFP16->GetSubVolumeDimension(mpi_part);
			arrayFormat = CU_AD_FORMAT_HALF;
		}
		else
		{
			volSize = vol->GetSubVolumeDimension(mpi_part);
			arrayFormat = CU_AD_FORMAT_FLOAT;
		}

		


		int* indexList;
		int projCount;
		proj.CreateProjectionIndexList(PLT_NORMAL, &projCount, &indexList);
		//proj.CreateProjectionIndexList(PLT_RANDOM, &projCount, &indexList);
		//proj.CreateProjectionIndexList(PLT_NORMAL, &projCount, &indexList);
		
		
		if (mpi_part == 0)
		{
			printf("Projection index list:\n");
			log << "Projection index list:" << endl;
			for (uint i = 0; i < projCount; i++)
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


        if (mpi_part == 0) printf("Free Memory on device after allocations: %i MB\n", cuCtx->GetFreeMemorySize() / 1024 / 1024);


        if (mpi_part == 0)printf("\nPixel size is: %f nm, Cs: %.2f mm, Voltage: %.2f kV\n", proj.GetPixelSize(0), aConfig.Cs, aConfig.Voltage);

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
		
        for (int iter = 0; iter < aConfig.Iterations; iter++)
        {			
			for (uint SIRTstep = 0; SIRTstep < (uint)(projCount + SIRTcount - 1) / (uint)SIRTcount; SIRTstep++)
			{
				for (uint i = 0; i < (uint)SIRTcount; i++)
				{
					if (SIRTstep * (uint)SIRTcount + i >= projCount) continue;
					int index = indexList[SIRTstep * (uint)SIRTcount + i];

					if (mpi_part == 0)
					{
						//printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
						//printf("Iter. %3i on proj. %3i/%3i, index %3i. FP  ", iter + 1, SIRTstep * (uint)SIRTcount + i + 1, projCount, index);
						//fflush(stdout);
					}
					reconstructor.PrintGeometry(volFP16, index);

				}
				                
            }
        }

		

        stop = clock();
        runtime = (double) (stop-start)/CLOCKS_PER_SEC;

        if (mpi_part == 0) printf("\n\nTotal time for reconstruction: %.2i:%.2i min.\n\n", (int)floor(runtime / 60.0), (int)floor(((runtime / 60.0) - floor(runtime / 60.0))*60.0));

		
		
	}
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
