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
#include "io/MotiveListe.h"
#include "io/ShiftFile.h"
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

typedef struct {
	float3 m[3];
} float3x3;

typedef struct {
	int particleNr;
	int particleNrInTomo;
	int tomoNr;
	int obtainedShiftFromID;
	float CCValue;
} groupRelations;

void MatrixVector3Mul(float3x3& M, float xIn, float yIn, float& xOut, float& yOut)
{
	xOut = M.m[0].x * xIn + M.m[0].y * yIn + M.m[0].z * 1.f;
	yOut = M.m[1].x * xIn + M.m[1].y * yIn + M.m[1].z * 1.f;
	//erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
}

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

	printf("Host ID: %d; host rank: %d; offset: %d; global rank: %d; name: %s\n", mpi_host_id, mpi_host_rank, mpi_offset, mpi_part, mpi_name); fflush(stdout);

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
				logfile = string(argv[arg + 1]);
				doLog = true;
			}
		}
	}

	SimpleLogger log(logfile, SimpleLogger::LOG_ERROR, !doLog);

	try
	{
		if (mpi_part == 0) printf("\n\n                          EmSART for local refinement\n\n\n");
		if (mpi_part == 0) printf("Read configuration file ");
		//Load configuration files
		Configuration::Config aConfig = Configuration::Config::GetConfig(CONFFILE, argc, argv, mpi_part, NULL);
		if (mpi_part == 0) printf("Done\n"); fflush(stdout);

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
		log << "Projection source: " << aConfig.ProjectionFile << endl;
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

		printf("Create CUDA context on device %i ... \n", aConfig.CudaDeviceIDs[mpi_offset + mpi_host_rank]); fflush(stdout);
		//Create CUDA context
		cuCtx = Cuda::CudaContext::CreateInstance(aConfig.CudaDeviceIDs[mpi_offset + mpi_host_rank]);

		printf("Using CUDA device %s\n", cuCtx->GetDeviceProperties()->GetDeviceName().c_str()); fflush(stdout);

		printf("Available Memory on device: %i MB\n", cuCtx->GetFreeMemorySize() / 1024 / 1024); fflush(stdout);
		
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
			else if (aConfig.GetFileReadMode() == Configuration::Config::FRM_MRC)
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

		MotiveList ml(aConfig.MotiveList, aConfig.ScaleMotivelistPosition, aConfig.ScaleMotivelistShift);
		vector<MotiveList> supportMotiveLists;
		for (size_t i = 0; i < aConfig.SupportingMotiveLists.size(); i++)
		{
			supportMotiveLists.push_back(MotiveList(aConfig.SupportingMotiveLists[i], aConfig.ScaleMotivelistPosition, aConfig.ScaleMotivelistShift));
		}

		bool* processedParticle = new bool[ml.DimY];
		memset(processedParticle, 0, ml.DimY * sizeof(bool));
		float* minDistOfProcessedParticles = new float[ml.DimY];
		groupRelations* groupRelationList = new groupRelations[ml.DimY];
		memset(groupRelationList, 0, ml.DimY * sizeof(groupRelations));
		
		
		EMFile reconstructedVol(aConfig.OutVolumeFile);
		reconstructedVol.OpenAndReadHeader();
		reconstructedVol.ReadHeaderInfo();
		dim3 volDims = make_dim3(reconstructedVol.DimX, reconstructedVol.DimY, reconstructedVol.DimZ);

		//Create volume dataset (host)
		Volume<unsigned short> *volFP16 = NULL;
		Volume<float> *volSubVol = NULL;
		Volume<float> *volSubVolRot = NULL;

		//Additional volumes for support references
		vector<Volume<float> *> volsSupport;
		

		//Volume<float> *volReconstructed = new Volume<float>(volDims, mpi_size, mpi_part);
		Volume<float> *volWithoutSubVols = new Volume<float>(volDims, mpi_size, mpi_part);
		//Needed to avoid slow loading from disk:
		Volume<float> *volWithSubVols = new Volume<float>(volDims, mpi_size, mpi_part);
		//Volume<float> *volOnlySubVols = new Volume<float>(volDims, mpi_size, mpi_part);
		//volReconstructed->PositionInSpace(aConfig.VoxelSize, make_float3(0, 0, 0));
		volWithoutSubVols->PositionInSpace(aConfig.VoxelSize, aConfig.VolumeShift);
		//volOnlySubVols->PositionInSpace(aConfig.VoxelSize, make_float3(0, 0, 0));
#ifdef USE_MPI
		////if (mpi_part == 0)
		//{
		//	volSubVol = new Volume<float>(make_dim3(aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol));
		//	volSubVolRot = new Volume<float>(make_dim3(aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol));
		//}
		//this is now independent of MPI!
#else
		
		
#endif
		volSubVol = new Volume<float>(make_dim3(aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol));
		volSubVolRot = new Volume<float>(make_dim3(aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol));

		if (aConfig.SupportingReferences.size() > 0)
		{
			for (size_t i = 0; i < aConfig.SupportingReferences.size(); i++)
			{
				EMFile ref(aConfig.SupportingReferences[i]);
				ref.OpenAndReadHeader();
				ref.ReadHeaderInfo();
				Volume<float>* newvol = new Volume<float>(make_dim3(ref.DimX, ref.DimY, ref.DimZ));
				newvol->LoadFromFile(aConfig.SupportingReferences[i], 0);
				volsSupport.push_back(newvol);
			}
		}


		CudaDeviceVariable volRot(aConfig.SizeSubVol * aConfig.SizeSubVol * aConfig.SizeSubVol * sizeof(float));


		if (aConfig.FP16Volume && !aConfig.WriteVolumeAsFP16)
			log << "; Convert to FP32 when saving to file";
		log << endl;

		float3 subVolDim;
		if (aConfig.FP16Volume)
			subVolDim = volFP16->GetSubVolumeDimension(mpi_part);
		else
			subVolDim = volSubVol->GetSubVolumeDimension(0);

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
			volSize = volWithoutSubVols->GetSubVolumeDimension(mpi_part);
			arrayFormat = CU_AD_FORMAT_FLOAT;
		}

		CudaArray3D vol_Array(arrayFormat, volSize.x, volSize.y, volSize.z, 1, 2);
		CudaTextureObject3D texObj(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &vol_Array);

		CudaArray3D vol_ArraySubVol(arrayFormat, aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol, 1, 2);
		CudaTextureObject3D texObjSubVol(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &vol_ArraySubVol);

		vector<RotKernel*> supportRotators;
		vector<CudaDeviceVariable*> supportvolRots;

		vector<CudaArray3D*> supportvol_ArraySubVols;
		vector<CudaTextureObject3D*> supporttexObjSubVols;

		if (aConfig.SupportingReferences.size() > 0)
		{
			for (size_t i = 0; i < volsSupport.size(); i++)
			{
				RotKernel* rotator = new RotKernel(modules.modWBP, (int)volsSupport[i]->GetDimension().x);
				rotator->SetData(volsSupport[i]->GetPtrToSubVolume(0));
				supportRotators.push_back(rotator);
				supportvolRots.push_back(new CudaDeviceVariable((int)volsSupport[i]->GetDimension().x * (int)volsSupport[i]->GetDimension().x * (int)volsSupport[i]->GetDimension().x * sizeof(float)));

				CudaArray3D* arr = new CudaArray3D(arrayFormat, (int)volsSupport[i]->GetDimension().x, (int)volsSupport[i]->GetDimension().x, (int)volsSupport[i]->GetDimension().x, 1, 2);
				supportvol_ArraySubVols.push_back(arr);
				supporttexObjSubVols.push_back(new CudaTextureObject3D(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, arr));
			}
		}



		/*CUsurfref surfref;
		cudaSafeCall(cuModuleGetSurfRef(&surfref, modules.modBP, "surfref"));
		cudaSafeCall(cuSurfRefSetArray(surfref, vol_ArraySubVol.GetCUarray(), 0));*/

		if (mpi_part == 0) printf("Copy volume to device ... ");

		bool volumeIsEmpty = true;

		if (aConfig.FP16Volume)
		{
			vol_Array.CopyFromHostToArray(volFP16->GetPtrToSubVolume(mpi_part));
			log << "Volume dimensions: " << volFP16->GetDimension() << endl;
			log << "Sub-Volume dimensions: " << endl;
			for (int sv = 0; sv < volFP16->GetSubVolumeCount(); sv++)
				log << "Sub-Volume " << sv << ": " << volFP16->GetSubVolumeDimension(sv) << endl;
		}
		else
		{
			//vol_Array.CopyFromHostToArray(volReconstructed->GetPtrToSubVolume(mpi_part));
			log << "Volume dimensions: " << volWithoutSubVols->GetDimension() << endl;
			log << "Sub-Volume dimensions: " << endl;
			for (int sv = 0; sv < volWithoutSubVols->GetSubVolumeCount(); sv++)
				log << "Sub-Volume " << sv << ": " << volWithoutSubVols->GetSubVolumeDimension(sv) << endl;
		}

		if (mpi_part == 0) printf("Done\n"); fflush(stdout);


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
		/////////////////////////////////////
		/// Filter Projections
		/////////////////////////////////////
		if (mpi_part == 0)
		{
			float lp = aConfig.fourFilterLP, hp = aConfig.fourFilterHP, lps = aConfig.fourFilterLPS, hps = aConfig.fourFilterHPS;
			bool skipFilter = aConfig.SkipFilter;

			if (!reconstructor.ComputeFourFilter() && !skipFilter)
			{
				log << SimpleLogger::LOG_ERROR;
				log << "Invalid filter parameters: Skiping filter." << endl;
				printf("Invalid filter parameters. Skiping filter...\n");
				log << SimpleLogger::LOG_INFO;
				skipFilter = true;
			}

			log << "Bandpass filter for projections applied: " << (skipFilter ? "false" : "true") << endl;
			log << "Bandpass filter values (lp, lps, hp, hps): " << lp << ", " << lps << ", " << hp << ", " << hps << endl;


			log << "Projection datatype: " << projSource->GetDataType() << endl;

			if (aConfig.ProjectionNormalization == Configuration::Config::PNM_STANDARD_DEV)
				log << "Normalizing projections by standard deviation [im = (im - mean) / std]" << endl;
			else
				log << "Normalizing projections by mean [im = (im - mean) / mean]" << endl;

			log << "Scaling projection values by: " << aConfig.ProjectionScaleFactor << endl;
			log << "Pixel size is: " << proj.GetPixelSize(0) << " nm" << endl;

			log << "Projection statistics:" << endl;

			printf("\r\n");
			for (int i = 0; i < projSource->DimZ; i++)
			{
				if (!markers.CheckIfProjIndexIsGood(i))
				{
					continue;
				}

				printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
				printf("Filtering projection: %i", i);
				log << "Projection " << i;
				fflush(stdout);

				//projSource->GetProjection(i) always points to an array with an element size of 4 bytes,
				//Even if original data is stored in shorts! We can therfor cast data and keep the same pointer.
				char* imgUS = projSource->GetProjection(i);
				float tilt = projSource->TiltAlpha[i];
				float weight = 1.0f / cos(tilt / 180.0f * M_PI);

				//Check if data format is supported
				if (projSource->GetDataType() != FDT_SHORT &&
					projSource->GetDataType() != FDT_USHORT &&
					projSource->GetDataType() != FDT_INT &&
					projSource->GetDataType() != FDT_UINT &&
					projSource->GetDataType() != FDT_FLOAT)
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

		
		/*float2* extraShiftsOld = new float2[projSource->DimZ];
		int minTiltIdx = proj.GetMinimumTiltIndex();
		int minTilt = -1;
		for (size_t i = 0; i < projCount; i++)
		{
			if (indexList[i] == minTiltIdx)
			{
				minTilt = i;
				break;
			}
		}*/
		/*if (minTilt < 0 && mpi_part == 0)
		{
			printf("0 degree tilt is not part of the tilt series. Aborting...\n");
			exit(-1);
		}*/



		/*float2* extraShifts = new float2[projSource->DimZ];
		memset(extraShifts, 0, projSource->DimZ * sizeof(float2));*/
		//printf("Opening Shift file:\n"); fflush(stdout);

		ShiftFile sf(aConfig.ShiftOutputFile, projSource->DimZ, ml.DimY);
		//printf("No crash here :)\n"); fflush(stdout);

		//Load reconstruction:
		volWithSubVols->LoadFromFile(aConfig.OutVolumeFile, mpi_part);

		//int testGroupCount = ml.GetGroupCount(aConfig);

		if (mpi_part == 0)
		{
			volSubVol->LoadFromFile(aConfig.Reference, 0);
			reconstructor.setRotVolData(volSubVol->GetPtrToSubVolume(0));
		}
				
		for (int group = 0; group < ml.GetGroupCount(aConfig); group++)
		{
			Matrix<float> magAnisotropyInv(reconstructor.GetMagAnistropyMatrix(1.0f / aConfig.MagAnisotropyAmount, aConfig.MagAnisotropyAngleInDeg, proj.GetWidth(), proj.GetHeight()));

			//reset Volume:
			{
				volWithoutSubVols->LoadFromVolume(*volWithSubVols, mpi_part);
			}

			
			volumeIsEmpty = false;
			
			//Get the motive list entries for this group:
			vector<motive> motives = ml.GetNeighbours(group, aConfig);

			if (aConfig.GroupMode == Configuration::Config::GM_MAXCOUNT ||
				aConfig.GroupMode == Configuration::Config::GM_MAXDIST)
			{
				if (processedParticle[ml.GetGlobalIdx(motives[0])])
				{
					continue;
				}
			}

			vector<supportMotive> support = ml.GetNeighbours(ml.GetGlobalIdx(motives[0]), aConfig.MaxDistanceSupport, supportMotiveLists);
			if (mpi_part == 0)
			{
				printf("Added %d support particles\n", (int)support.size());
				log << "Added " << (int)support.size() <<" support particles" << endl;
			}

#ifdef USE_MPI
			MPI_Barrier(MPI_COMM_WORLD);
#endif
			//Remove the subVolumes from the reconstruction and fill the space with zeros:
			for (int motlIdx = 0; motlIdx < motives.size(); motlIdx++)
			{
				motive m = motives[motlIdx];

				float3 posSubVol = make_float3(m.x_Coord, m.y_Coord, m.z_Coord);
				float3 shift = make_float3(m.x_Shift, m.y_Shift, m.z_Shift);

				float binningAdjust = aConfig.VoxelSize.x / aConfig.VoxelSizeSubVol;

				//adjust subVolsize to radius --> / 2!
				volWithoutSubVols->RemoveSubVolume(posSubVol.x + shift.x, posSubVol.y + shift.y, posSubVol.z + shift.z, aConfig.SizeSubVol / binningAdjust / 2, 0, mpi_part);
			}

			//Remove the supporting subVolumes from the reconstruction and fill the space with zeros:
			for (int motlIdx = 0; motlIdx < support.size(); motlIdx++)
			{
				supportMotive m = support[motlIdx];

				float3 posSubVol = make_float3(m.m.x_Coord, m.m.y_Coord, m.m.z_Coord);
				float3 shift = make_float3(m.m.x_Shift, m.m.y_Shift, m.m.z_Shift);

				float binningAdjust = aConfig.VoxelSize.x / aConfig.VoxelSizeSubVol;

				//adjust subVolsize to radius --> / 2!
				volWithoutSubVols->RemoveSubVolume(posSubVol.x + shift.x, posSubVol.y + shift.y, posSubVol.z + shift.z, volsSupport[m.index]->GetDimension().x / binningAdjust / 2, 0, mpi_part);
			}
			//Copy the holey volume to GPU
			vol_Array.CopyFromHostToArray(volWithoutSubVols->GetPtrToSubVolume(mpi_part));

			if (mpi_part == 0)
				switch (aConfig.GroupMode)
				{
				case Configuration::Config::GM_BYGROUP:
					printf("Group nr: %d/%d, group size: %d\n", group, (int)ml.GetGroupCount(aConfig), (int)motives.size());
					log << "Group nr: " << group << "/" << (int)ml.GetGroupCount(aConfig) << ", group size: " << (int)motives.size() << endl;
					break;
				case Configuration::Config::GM_MAXDIST:
					printf("Group nr: %d/%d, group size: %d\n", group, (int)ml.GetGroupCount(aConfig), (int)motives.size());
					log << "Group nr: " << group << "/" << (int)ml.GetGroupCount(aConfig) << ", group size: " << (int)motives.size() << endl;
					break;
				case Configuration::Config::GM_MAXCOUNT:
					{
						float dist = ml.GetDistance(motives[0], motives[motives.size() - 1]);
						printf("Group nr: %d/%d, max distance : %f [voxel]\n", group, (int)ml.GetGroupCount(aConfig), dist);
						log << "Group nr: " << group << "/" << (int)ml.GetGroupCount(aConfig) << ", max distance: " << dist << " [voxel]" << endl;
					}
					break;
				}



			//Project the single sub-Volumes:
			for (int i = 0; i < projCount; i++)
			{
				if (mpi_part == 0)
				{
					printf("Projection nr: %d/%d\n", i+1, projCount);
				}

				//Reset the minDist Values for each projection.
				for (size_t i = 0; i < ml.DimY; i++)
				{
					minDistOfProcessedParticles[i] = 100000000.0f; //some large value...
				}

				int index = indexList[i];
				//index = 20;
				reconstructor.ResetProjectionsDevice();
				int2 roiMin = make_int2(proj.GetWidth(), proj.GetHeight());
				int2 roiMax = make_int2(0, 0);

				if (mpi_part == 0)
				{
					for (int motlIdx = 0; motlIdx < motives.size(); motlIdx++)
					{
						motive m = motives[motlIdx];
						
						reconstructor.rotVol(volRot, m.phi, m.psi, m.theta);
						//volRot.CopyDeviceToHost(volSubVolRot->GetPtrToSubVolume(0));
						vol_ArraySubVol.CopyFromDeviceToArray(volRot);


						float3 posSubVol = make_float3(m.x_Coord, m.y_Coord, m.z_Coord);
						float3 shift = make_float3(m.x_Shift, m.y_Shift, m.z_Shift);
						//printf("posSubVol: (%f, %f, %f) * %f\n", posSubVol.x, posSubVol.y, posSubVol.z, aConfig.ScaleMotivelistPosition);
						//printf("shift: (%f, %f, %f)\n", shift.x, shift.y, shift.z);

						int2 hitPoint;
						float3 bbMin = volWithoutSubVols->GetVolumeBBoxMin();
						proj.ComputeHitPoint(bbMin.x + (posSubVol.x + shift.x - 1) * aConfig.VoxelSize.x + 0.5f * aConfig.VoxelSize.x,
							bbMin.y + (posSubVol.y + shift.y - 1) * aConfig.VoxelSize.y + 0.5f * aConfig.VoxelSize.y,
							bbMin.z + (posSubVol.z + shift.z - 1) * aConfig.VoxelSize.z + 0.5f * aConfig.VoxelSize.z,
							index, hitPoint);
						//printf("HitPoint: (%d, %d)\n", hitPoint.x, hitPoint.y);

						float hitX, hitY;
						MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), hitPoint.x, hitPoint.y, hitX, hitY);

						int safeDist = 100 + aConfig.VoxelSizeSubVol * aConfig.SizeSubVol * 2 + aConfig.MaxShift * 2;
						int hitXMin = floor(hitX) - safeDist;
						int hitXMax = ceil(hitX) + safeDist;
						int hitYMin = floor(hitY) - safeDist;
						int hitYMax = ceil(hitY) + safeDist;

						if (hitXMin < 0) hitXMin = 0;
						if (hitYMin < 0) hitYMin = 0;

						if (hitXMin < roiMin.x)
							roiMin.x = hitXMin;
						if (hitXMax > roiMax.x)
							roiMax.x = hitXMax;

						if (hitYMin < roiMin.y)
							roiMin.y = hitYMin;
						if (hitYMax > roiMax.y)
							roiMax.y = hitYMax;

						volSubVolRot->PositionInSpace(aConfig.VoxelSize, aConfig.VoxelSizeSubVol, *volWithoutSubVols, posSubVol, shift);

						int2 roiMinLocal = make_int2(hitXMin, hitYMin);
						int2 roiMaxLocal = make_int2(hitXMax, hitYMax);
						//printf("ROILocal: xMin: %d, yMin: %d, xMax: %d, yMax: %d\n", roiMinLocal.x, roiMinLocal.y, roiMaxLocal.x, roiMaxLocal.y);

						//Project actual index:						
						/*printf("Do projection...\n");
						fflush(stdout);*/
						reconstructor.ForwardProjectionROI(volSubVolRot, texObjSubVol, index, volumeIsEmpty, roiMinLocal, roiMaxLocal, true);
						//reconstructor.ForwardProjection(volSubVolRot, texObjSubVol, index, volumeIsEmpty, true);
					}
					//project support particles
					for (int motlIdx = 0; motlIdx < support.size(); motlIdx++)
					{
						supportMotive m = support[motlIdx];
						(*supportRotators[m.index])(*supportvolRots[m.index], m.m.phi, m.m.psi, m.m.theta);
						supportvol_ArraySubVols[m.index]->CopyFromDeviceToArray(*supportvolRots[m.index]);


						float3 posSubVol = make_float3(m.m.x_Coord, m.m.y_Coord, m.m.z_Coord);
						float3 shift = make_float3(m.m.x_Shift, m.m.y_Shift, m.m.z_Shift);
						//printf("posSubVol: (%f, %f, %f) * %f\n", posSubVol.x, posSubVol.y, posSubVol.z, aConfig.ScaleMotivelistPosition);
						//printf("shift: (%f, %f, %f)\n", shift.x, shift.y, shift.z);

						int2 hitPoint;
						float3 bbMin = volWithoutSubVols->GetVolumeBBoxMin();
						proj.ComputeHitPoint(bbMin.x + (posSubVol.x + shift.x - 1) * aConfig.VoxelSize.x + 0.5f * aConfig.VoxelSize.x,
							bbMin.y + (posSubVol.y + shift.y - 1) * aConfig.VoxelSize.y + 0.5f * aConfig.VoxelSize.y,
							bbMin.z + (posSubVol.z + shift.z - 1) * aConfig.VoxelSize.z + 0.5f * aConfig.VoxelSize.z,
							index, hitPoint);
						//printf("HitPoint: (%d, %d)\n", hitPoint.x, hitPoint.y);

						float hitX, hitY;
						MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), hitPoint.x, hitPoint.y, hitX, hitY);

						int safeDist = 100 + aConfig.VoxelSizeSubVol * aConfig.SizeSubVol * 2 + aConfig.MaxShift * 2;
						int hitXMin = floor(hitX) - safeDist;
						int hitXMax = ceil(hitX) + safeDist;
						int hitYMin = floor(hitY) - safeDist;
						int hitYMax = ceil(hitY) + safeDist;

						if (hitXMin < 0) hitXMin = 0;
						if (hitYMin < 0) hitYMin = 0;

						if (hitXMin < roiMin.x)
							roiMin.x = hitXMin;
						if (hitXMax > roiMax.x)
							roiMax.x = hitXMax;

						if (hitYMin < roiMin.y)
							roiMin.y = hitYMin;
						if (hitYMax > roiMax.y)
							roiMax.y = hitYMax;

						volsSupport[m.index]->PositionInSpace(aConfig.VoxelSize, aConfig.VoxelSizeSubVol, *volWithoutSubVols, posSubVol, shift);
						
						int2 roiMinLocal = make_int2(hitXMin, hitYMin);
						int2 roiMaxLocal = make_int2(hitXMax, hitYMax);
						//printf("ROILocal: xMin: %d, yMin: %d, xMax: %d, yMax: %d\n", roiMinLocal.x, roiMinLocal.y, roiMaxLocal.x, roiMaxLocal.y);

						//Project actual index:						
						/*printf("Do projection...\n");
						fflush(stdout);*/
						reconstructor.ForwardProjectionROI(volsSupport[m.index], *supporttexObjSubVols[m.index], index, volumeIsEmpty, roiMinLocal, roiMaxLocal, true);
						//reconstructor.ForwardProjection(volSubVolRot, texObjSubVol, index, volumeIsEmpty, true);
					}
				}

				//This is the final projection of the model:
				reconstructor.CopyProjectionToSubVolumeProjection();
				if (mpi_part == 0)
				if (aConfig.DebugImages)//for debugging save images
				{
					printf("Save image...\n");
					fflush(stdout);
					reconstructor.CopyProjectionToHost(SIRTBuffer[0]);

					stringstream ss;
					ss << "projSubVols_" << group << "_" << index << ".em";
					emwrite(ss.str(), SIRTBuffer[0], proj.GetWidth(), proj.GetHeight());
				}

#ifdef USE_MPI
				//MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(&roiMin, 2, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Bcast(&roiMax, 2, MPI_INT, 0, MPI_COMM_WORLD);
#endif

				//printf("ROIAfter: xMin: %d, yMin: %d, xMax: %d, yMax: %d\n", roiMin.x, roiMin.y, roiMax.x, roiMax.y);

				reconstructor.ResetProjectionsDevice();
				//project Reconstruction without sub-Volumes:
				{
					reconstructor.ForwardProjectionROI(volWithoutSubVols, texObj, index, false, roiMin, roiMax);
					//reconstructor.ForwardProjection(volWithoutSubVols, texObj, index, false);
					
					/*reconstructor.CopyProjectionToHost(SIRTBuffer[0]);

					stringstream ss;
					ss << "projVorComp_" << index << ".em";
					emwrite(ss.str(), SIRTBuffer[0], proj.GetWidth(), proj.GetHeight());

					reconstructor.CopyDistanceImageToHost(SIRTBuffer[0]);
					stringstream ss2;
					ss2 << "distVorComp_" << index << ".em";
					emwrite(ss2.str(), SIRTBuffer[0], proj.GetWidth(), proj.GetHeight());*/

					reconstructor.Compare(volWithoutSubVols, projSource->GetProjection(index), index);
					//We now have in proj_d the distance weighted 'noise free' approximation of the original projection.
				}
				if (mpi_part == 0)
				if (aConfig.DebugImages)
				{
					printf("Save image...\n");
					fflush(stdout);
					reconstructor.CopyProjectionToHost(SIRTBuffer[0]);

					stringstream ss;
					ss << "projComp_" << group << "_" << index << ".em";
					emwrite(ss.str(), SIRTBuffer[0], proj.GetWidth(), proj.GetHeight());
				}

				float2 shift;
				float ccValue;
				float* ccMap;
				float* ccMapMulti;
				if (mpi_part == 0)
				{
					shift = reconstructor.GetDisplacement(aConfig.MultiPeakDetection, &ccValue);
					ccMap = reconstructor.GetCCMap();
					if (aConfig.MultiPeakDetection)
					{
						ccMapMulti = reconstructor.GetCCMapMulti();
					}
				}
				bool dumpCCMap = aConfig.CCMapFileName.length() > 3;
				if (mpi_part == 0)
				if (dumpCCMap)
				{
					stringstream ss;
					ss << aConfig.CCMapFileName << index << ".em";
					if (group == 0) //create a new file and overwrite the old one
					{
						emwrite(ss.str(), ccMap, aConfig.MaxShift * 4, aConfig.MaxShift * 4);
					}
					else //append to existing file
					{
						EMFile::AddSlice(ss.str(), ccMap, aConfig.MaxShift * 4, aConfig.MaxShift * 4);
					}
					if (aConfig.MultiPeakDetection)
					{
						stringstream ss2;
						ss2 << aConfig.CCMapFileName << "Multi_" << index << ".em";
						if (group == 0) //create a new file and overwrite the old one
						{
							emwrite(ss2.str(), ccMapMulti, aConfig.MaxShift * 4, aConfig.MaxShift * 4);
						}
						else //append to existing file
						{
							EMFile::AddSlice(ss2.str(), ccMapMulti, aConfig.MaxShift * 4, aConfig.MaxShift * 4);
						}
					}
				}
				

				if (mpi_part == 0)
					printf("\nShift is: %f, %f\n", shift.x, shift.y);

				//extraShifts[index] = shift;
				
#ifdef USE_MPI
				MPI_Barrier(MPI_COMM_WORLD);
#endif
				//Do this on all nodes to stay in sync!
				//if (mpi_part == 0)
				{
					//save shift for every entry in the group
					if (aConfig.GroupMode == Configuration::Config::GM_BYGROUP)
					{
						for (int motlIdx = 0; motlIdx < motives.size(); motlIdx++)
						{
							motive m = motives[motlIdx];

							int totalIdx = ml.GetGlobalIdx(m);
							//printf("Set values at: %d, %d\n", index, motlIdx); fflush(stdout);
							sf.SetValue(index, totalIdx, shift);
							groupRelationList[totalIdx].particleNr = m.partNr;
							groupRelationList[totalIdx].particleNrInTomo = m.partNrInTomo;
							groupRelationList[totalIdx].tomoNr = m.tomoNr;
							groupRelationList[totalIdx].obtainedShiftFromID = group;
							groupRelationList[totalIdx].CCValue = ccValue;
						}
					}
					else //save shift only for the first entry in the group
					{
						motive m = motives[0];
						int count = 0;
						vector<pair<int, float> > closeIndx;
						int totalIdx = ml.GetGlobalIdx(m);
						for (int motlIdx = 1; motlIdx < motives.size(); motlIdx++)
						{
							motive m2 = motives[motlIdx];
							float d = ml.GetDistance(m, m2);
							int totalIdx2 = ml.GetGlobalIdx(m2);
							if (d <= aConfig.SpeedUpDistance && d < minDistOfProcessedParticles[totalIdx2])
							{
								sf.SetValue(index, totalIdx2, shift);
								processedParticle[totalIdx2] = true;
								minDistOfProcessedParticles[totalIdx2] = d;
								closeIndx.push_back(pair<int, float>(totalIdx2, d));
								count++;

								groupRelationList[totalIdx2].particleNr = m2.partNr;
								groupRelationList[totalIdx2].particleNrInTomo = m2.partNrInTomo;
								groupRelationList[totalIdx2].tomoNr = m2.tomoNr;
								groupRelationList[totalIdx2].obtainedShiftFromID = totalIdx;
								groupRelationList[totalIdx2].CCValue = ccValue;
							}
						}

						if (mpi_part == 0 && i == 0)
						{
							printf("Found %d close particles\n", count);
							log << "Found " << count << " close particles:" << endl;
							log << "Particle " << totalIdx << " --> " ;
							for (size_t p = 0; p < count; p++)
							{
								log << closeIndx[p].first << " (" << closeIndx[p].second << ")  ";
							}
							log << endl;
						}

						//printf("Set values at: %d, %d\n", index, motlIdx); fflush(stdout);
						sf.SetValue(index, totalIdx, shift);
						processedParticle[totalIdx] = true;
					}
				}

				//Save measured local shifts to an EM-file after every projection in case we crash...
				if (mpi_part == 0)
				{
					sf.OpenAndWrite();
				}
			}

			if (mpi_part == 0)
			{
				printf("\n");
			}
		}

		//Save measured local shifts to an EM-file:
		if (mpi_part == 0)
		{
			sf.OpenAndWrite();

			ofstream fs;
			fs.open(aConfig.ShiftOutputFile + ".relations");
			for (size_t i = 0; i < ml.DimY; i++)
			{
				fs << groupRelationList[i].particleNr << "; " << groupRelationList[i].particleNrInTomo << "; " << groupRelationList[i].particleNrInTomo << "; " << groupRelationList[i].obtainedShiftFromID << "; " << groupRelationList[i].CCValue << std::endl;
			}
			fs.close();
		}

		//emwrite(aConfig.ShiftOutputFile.c_str(), (float*)extraShifts, 2, projSource->DimZ);


		stop = clock();
		runtime = (double)(stop - start) / CLOCKS_PER_SEC;

		if (mpi_part == 0) printf("\n\nTotal time for local shift measurements: %.2i:%.2i min.\n\n", (int)floor(runtime / 60.0), (int)floor(((runtime / 60.0) - floor(runtime / 60.0))*60.0));

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
