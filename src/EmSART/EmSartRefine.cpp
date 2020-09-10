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
//#include "Kernels.h"
#include <CudaContext.h>
#include "utils/Config.h"
//#include "utils/CudaConfig.h"
//#include "utils/Matrix.h"
//#include "io/Dm4FileStack.h"
//#include "io/MRCFile.h"
#include "io/FileSource.h"
#ifdef USE_MPI
#include "io/MPISource.h"
#endif
#include <MarkerFile.h>
#include "io/writeBMP.h"
//#include "io/mrcHeader.h"
//#include "io/emHeader.h"
#include <CtfFile.h>
#include <MotiveListe.h>
#include <ShiftFile.h>
#include <time.h>
#include <cufft.h>
#include <npp.h>
//#include "CudaKernelBinarys.h"
#include <algorithm>
#include "utils/SimpleLogger.h"
#include "Reconstructor.h"
#include "cuda_profiler_api.h"
#include <npp.h>

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

//typedef struct {
//	float3 m[3];
//} float3x3;

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

        //Create CUDA context
		printf("Create CUDA context on device %i ... \n", aConfig.CudaDeviceIDs[mpi_offset + mpi_host_rank]); fflush(stdout);
		cuCtx = Cuda::CudaContext::CreateInstance(aConfig.CudaDeviceIDs[mpi_offset + mpi_host_rank]);
		printf("Using CUDA device %s\n", cuCtx->GetDeviceProperties()->GetDeviceName().c_str()); fflush(stdout);
		printf("Available Memory on device: %i MB\n", cuCtx->GetFreeMemorySize() / 1024 / 1024); fflush(stdout);

        //Load projection data file
		ProjectionSource* projSource;
		if (mpi_part == 0)
		{
			if (aConfig.GetFileReadMode() == Configuration::Config::FRM_DM4 ||
				aConfig.GetFileReadMode() == Configuration::Config::FRM_MRC)
			{
				printf("\nLoading projections...\n");
				projSource = new FileSource(aConfig.ProjectionFile);


				printf("Loaded %d projections.\n\n", projSource->GetProjectionCount());
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

        /////////////////////////////////////
        /// Prepare tomogram  START
        /////////////////////////////////////

		//Load marker/alignment file
		MarkerFile markers(aConfig.MarkerFile, aConfig.ReferenceMarker);

		//Create projection object to handle projection data
		Projection proj(projSource, &markers, false);

        // Get header of reconstructed volume
        EmFile reconstructedVol(aConfig.OutVolumeFile);
        reconstructedVol.OpenAndReadHeader();
        dim3 volDims = make_dim3(reconstructedVol.GetFileHeader().DimX, reconstructedVol.GetFileHeader().DimY, reconstructedVol.GetFileHeader().DimZ);

        // Init holey volume
        auto volWithoutSubVols = new Volume<float>(volDims, mpi_size, mpi_part);
        volWithoutSubVols->PositionInSpace(aConfig.VoxelSize, aConfig.VolumeShift);

        //Needed to avoid slow loading from disk:
        auto volWithSubVols = new Volume<float>(volDims, mpi_size, mpi_part);

        // Dummy
        Volume<unsigned short> *volFP16 = NULL;

        if (aConfig.FP16Volume && !aConfig.WriteVolumeAsFP16)
            log << "; Convert to FP32 when saving to file";
        log << endl;

        // For printing figure out how large the tomogram is in GPU memory
        float3 subVolDim;
        if (aConfig.FP16Volume)
            subVolDim = volFP16->GetSubVolumeDimension(0);
        else
            subVolDim = volWithSubVols->GetSubVolumeDimension(0);

        size_t sizeDataType;
        if (aConfig.FP16Volume)
        {
            sizeDataType = sizeof(unsigned short);
        }
        else
        {
            sizeDataType = sizeof(float);
        }

        printf("subVolDim.z: %f\n", subVolDim.z);
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

        size_t free = 0;
        size_t total = 0;
        cudaMemGetInfo(&free, &total);
        printf("before array: free: %zu total: %zu", free, total);

        // Tomogram's array on the device (needed for texture interpolation)
        CudaArray3D vol_Array(arrayFormat, volSize.x, volSize.y, volSize.z, 1, 2);
        // Tomogram's texture object for SART
        CudaTextureObject3D texObj(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &vol_Array);
        CudaSurfaceObject3D surfObj(&vol_Array);

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


        int* projIndexList;
        int projCount;
        proj.CreateProjectionIndexList(PLT_NORMAL, &projCount, &projIndexList);
        //proj.CreateProjectionprojIndexList(PLT_RANDOM, &projCount, &projIndexList);
        //proj.CreateProjectionprojIndexList(PLT_NORMAL, &projCount, &projIndexList);


        if (mpi_part == 0)
        {
            printf("Projection index list:\n");
            log << "Projection index list:" << endl;
            for (uint i = 0; i < projCount; i++)
            {
                printf("%3d,", projIndexList[i]);
                log << projIndexList[i];
                if (i < projCount - 1)
                    log << ", ";
            }
            log << endl;
            printf("\b \n\n");
        }

        free = 0;
        total = 0;
        cudaMemGetInfo(&free, &total);
        printf("before reconstructor: free: %zu total: %zu", free, total);

        Reconstructor reconstructor(aConfig, proj, projSource, markers, *defocus, modules, mpi_part, mpi_size);
        /////////////////////////////////////
        /// Prepare tomogram  END
        /////////////////////////////////////

        /////////////////////////////////////
        /// Preprocess particles START
        /////////////////////////////////////
        if (mpi_part == 0) printf("Start preprocessing particles and masks\n");
		// Get Motivelist and select for tomogram
		MotiveList ml(aConfig.MotiveList, aConfig.ScaleMotivelistPosition, aConfig.ScaleMotivelistShift);
		ml.selectTomo(aConfig.TomogramIndex);

		// Get number and IDs of unique references
		vector<int> unique_ref_ids; // Unique reference IDs
		int unique_ref_count = 0; // Number of unique IDs
		auto ref_ids = new int[ml.GetParticleCount()]; // For each particle, the corresponding index within unique_ref_ids
		ml.getRefIndeces(unique_ref_ids, ref_ids, unique_ref_count);

		// Init processed list
        //auto processedParticle = new bool[ml.GetParticleCount()];
		//memset(processedParticle, 0, ml.GetParticleCount() * sizeof(bool));

		// Init distance list
		auto minDistOfProcessedParticles = new float[ml.GetParticleCount()];
		//groupRelations* groupRelationList = new groupRelations[ml.GetParticleCount()];
		//memset(groupRelationList, 0, ml.GetParticleCount() * sizeof(groupRelations));

        // Create working memory (host)
        vector<Volume<unsigned short> *> v_volFP16; // TODO:What the hell does this do?

        // Storage of original particle, mask for particle, mask for volume (size: unique_ref_count)
        vector<Volume<float> *> vh_ori_particle;
        vector<Volume<float> *> vh_ori_refmask;
        vector<Volume<float> *> vh_ori_volmask;

        // Storage of rotated particles, masks (size: ml.GetParticleCount());
        vector<Volume<float> *> vh_rot_particle;
        vector<Volume<float> *> vh_rot_refmask;
        vector<Volume<float> *> vh_rot_volmask;

        // Storage of particle, mask dimensions (dimensions of refmask same as particle)
        vector<uint3> vh_particle_dims;
        vector<uint3> vh_volmask_dims;

        // Init the original reference volumes on host, read from file, get dimensions
        vector<float3> vh_subVolDim;
        vector<float3> vh_volMaskDim;
        for (size_t i = 0; i < unique_ref_count; i++)
        {
            // Build names
            stringstream ref_name;
            stringstream refmask_name;
            stringstream volmask_name;
            ref_name << aConfig.Reference << unique_ref_ids[i] << ".em";
            refmask_name << aConfig.ReferenceMask << unique_ref_ids[i] << ".em";
            volmask_name << aConfig.VolumeMask << unique_ref_ids[i] << ".em";

            //--> REFERENCE
            // Read header
            EmFile ref(ref_name.str());
            ref.OpenAndReadHeader();
            // Create volume object
            uint3 tempdim = make_dim3(ref.GetFileHeader().DimX, ref.GetFileHeader().DimY, ref.GetFileHeader().DimZ);
            auto temp_ref = new Volume<float>(tempdim);
            // Read and append volume and dimensions
            temp_ref->LoadFromFile(ref_name.str(), 0);
            vh_ori_particle.push_back(temp_ref);
            vh_particle_dims.push_back(tempdim);

            //--> REFERENCE MASK
            // Read header
            EmFile refmask(refmask_name.str());
            refmask.OpenAndReadHeader();
            // Create volume object
            tempdim = make_dim3(refmask.GetFileHeader().DimX, refmask.GetFileHeader().DimY, refmask.GetFileHeader().DimZ);
            auto temp_refmask = new Volume<float>(tempdim);
            // Read and append volume (dimensions same as reference)
            temp_refmask->LoadFromFile(refmask_name.str(), 0);
            vh_ori_refmask.push_back(temp_refmask);

            //--> VOLUME MASK
            // Read header
            EmFile volmask(volmask_name.str());
            volmask.OpenAndReadHeader();
            // Create volume object
            tempdim = make_dim3(volmask.GetFileHeader().DimX, volmask.GetFileHeader().DimY, volmask.GetFileHeader().DimZ);
            auto temp_volmask = new Volume<float>(tempdim);
            // Read and append volume and dimensions
            temp_volmask->LoadFromFile(volmask_name.str(), 0);
            vh_ori_volmask.push_back(temp_volmask);
            vh_volmask_dims.push_back(tempdim);

            // Get dimension of particle
            if (aConfig.FP16Volume)
                vh_subVolDim.push_back(v_volFP16[i]->GetSubVolumeDimension(mpi_part));
            else
                vh_subVolDim.push_back(vh_ori_particle[i]->GetSubVolumeDimension(0));
        }

        // Device array
        //CudaDeviceVariable d_rot_particle(aConfig.SizeSubVol * aConfig.SizeSubVol * aConfig.SizeSubVol * sizeof(float));

        // Rotation kernels for the reference volumes/masks of (potentially) different sizes (size: unique_ref_count)
        vector<RotKernel*> vd_ref_rotators;
        vector<RotKernel*> vd_refmask_rotators;
        vector<RotKernel*> vd_volmask_rotators;
        // Output device arrays for the reference volumes/masks of (potentially) different sizes (size: unique_ref_count)
        vector<CudaDeviceVariable*> vd_rot_particle;
        vector<CudaDeviceVariable*> vd_rot_refmask;
        vector<CudaDeviceVariable*> vd_rot_volmask;
        // Mask kernels for masking volumes on the GPU
        vector<ApplyMaskKernel*> vd_apply_mask_kernels;
        vector<RestoreVolumeKernel*> vd_restore_vol_kernels;

        // Initialize the kernels and device variables
        for (size_t i = 0; i < unique_ref_count; i++)
        {
            //--> REFERENCE
            // Create the kernel objects
            auto ref_rotator = new RotKernel(modules.modWBP, (int)vh_ori_particle[i]->GetDimension().x);
            // Set the unrotated reference as default data
            ref_rotator->SetData(vh_ori_particle[i]->GetPtrToSubVolume(0));
            vd_ref_rotators.push_back(ref_rotator);
            // Create the output arrays
            vd_rot_particle.push_back(new CudaDeviceVariable((int)vh_ori_particle[i]->GetDimension().x * (int)vh_ori_particle[i]->GetDimension().x * (int)vh_ori_particle[i]->GetDimension().x * sizeof(float)));

            //--> REFERENCE MASK
            // Create the kernel objects
            auto refmask_rotator = new RotKernel(modules.modWBP, (int)vh_ori_refmask[i]->GetDimension().x);
            // Set the unrotated mask as default data
            refmask_rotator->SetData(vh_ori_refmask[i]->GetPtrToSubVolume(0));
            vd_refmask_rotators.push_back(refmask_rotator);
            // Create the output arrays
            vd_rot_refmask.push_back(new CudaDeviceVariable((int)vh_ori_refmask[i]->GetDimension().x * (int)vh_ori_refmask[i]->GetDimension().x * (int)vh_ori_refmask[i]->GetDimension().x * sizeof(float)));

            //--> VOLUME MASK
            // Create the kernel objects
            auto volmask_rotator = new RotKernel(modules.modWBP, (int)vh_ori_volmask[i]->GetDimension().x);
            // Set the unrotated mask as default data
            volmask_rotator->SetData(vh_ori_volmask[i]->GetPtrToSubVolume(0));
            vd_volmask_rotators.push_back(volmask_rotator);
            // Create the output arrays
            vd_rot_volmask.push_back(new CudaDeviceVariable((int)vh_ori_volmask[i]->GetDimension().x * (int)vh_ori_volmask[i]->GetDimension().x * (int)vh_ori_volmask[i]->GetDimension().x * sizeof(float)));
            // Create mask application kernels
            auto volmask_apply = new ApplyMaskKernel(modules.modWBP, (int)vh_ori_volmask[i]->GetDimension().x);
            vd_apply_mask_kernels.push_back(volmask_apply);
            // Create volume restoration kernels
            auto volmask_restore = new RestoreVolumeKernel(modules.modWBP, (int)vh_ori_volmask[i]->GetDimension().x);
            vd_restore_vol_kernels.push_back(volmask_restore);
        }

        // Now rotate all particles/masks, apply rotated masks to rotated particles and store them on host
        CudaDeviceVariable d_refsum(sizeof(float));
        CudaDeviceVariable d_masksum(sizeof(float));
        CudaDeviceVariable d_mean(sizeof(float));
        float h_refsum = 0;
        float h_masksum = 0;
        float h_mean = 0;
        for (size_t motlIdx = 0; motlIdx < ml.GetParticleCount(); motlIdx++)
        {
            // Get current particle's data
            motive m = ml.GetAt(motlIdx);

            // Rotate refs and masks
            int ref_id_idx = ref_ids[motlIdx];
            (*vd_ref_rotators[ref_id_idx])(*vd_rot_particle[ref_id_idx], m.phi, m.psi, m.theta);
            (*vd_refmask_rotators[ref_id_idx])(*vd_rot_refmask[ref_id_idx], m.phi, m.psi, m.theta);
            (*vd_volmask_rotators[ref_id_idx])(*vd_rot_volmask[ref_id_idx], m.phi, m.psi, m.theta);

            //--> Apply mask to reference correctly (zero mean inside mask).
            // Set old vars 0
            h_refsum = 0;
            h_masksum = 0;
            h_mean = 0;
            d_refsum.Memset(0);
            d_masksum.Memset(0);

            //--> Compute ref .* mask (result in place in ref)
            auto npp_ref = (Npp32f*)vd_rot_particle[ref_id_idx]->GetDevicePtr();
            auto npp_refmask = (Npp32f*)vd_rot_refmask[ref_id_idx]->GetDevicePtr();
            auto npp_volmask = (Npp32f*)vd_rot_volmask[ref_id_idx]->GetDevicePtr();

            int refdim = (int)vh_ori_particle[ref_id_idx]->GetDimension().x * (int)vh_ori_particle[ref_id_idx]->GetDimension().x * (int)vh_ori_particle[ref_id_idx]->GetDimension().x;
            int volmaskdim = (int)vh_ori_volmask[ref_id_idx]->GetDimension().x * (int)vh_ori_volmask[ref_id_idx]->GetDimension().x * (int)vh_ori_volmask[ref_id_idx]->GetDimension().x;
            nppSafeCall(nppsMulC_32f_I((Npp32f) -1.f, npp_refmask, refdim)); // Volumes are loaded inverted, so fix that (WTF?!?!?!!?)
            nppSafeCall(nppsMulC_32f_I((Npp32f) -1.f, npp_volmask, volmaskdim)); // Volumes are loaded inverted, so fix that (WTF?!?!?!!?)
            //TODO: Overload volume loading function to load non-inverted masks.
            nppSafeCall(nppsMul_32f_I(npp_refmask, npp_ref, refdim));

            printf("Finished Mult\n");

            //--> Compute sum(ref(:))
            d_refsum.CopyDeviceToHost(&h_refsum, sizeof(float));
            printf("Copy successful before\n");
            auto npp_refsum = (Npp32f *)d_refsum.GetDevicePtr();
            // Compute/Alloc scratch buffer size
            int nBufferSize;
            CUdeviceptr d_scratch;
            nppSafeCall(nppsSumGetBufferSize_32f(refdim, &nBufferSize));
            printf("Buffersize computed\n");
            cudaSafeCall(cuMemAlloc(&d_scratch, nBufferSize));
            printf("Buffer allocated\n");
            // Compute sum (is now in d_refum)
            nppSafeCall(nppsSum_32f(npp_ref, refdim, npp_refsum, (Npp8u*)d_scratch));
            printf("Sum executed\n");
            //printf("npp:%p var:%p", (void *)&npp_refsum, (void *)d_refsum.GetDevicePtr());
            // To host
            d_refsum.CopyDeviceToHost(&h_refsum, sizeof(float));
            printf("Copy to host\n");

            //--> Compute sum(mask(:))
            auto npp_masksum = (Npp32f *)d_masksum.GetDevicePtr();
            // Compute/Alloc scratch buffer size
            nppSafeCall(nppsSumGetBufferSize_32f(refdim, &nBufferSize));
            cudaSafeCall(cuMemAlloc(&d_scratch, nBufferSize));
            // Compute sum (is now in d_refum)
            nppSafeCall(nppsSum_32f(npp_refmask, refdim, npp_masksum, (Npp8u*)d_scratch));
            // To host
            d_masksum.CopyDeviceToHost(&h_masksum, sizeof(float));

            printf("Computed sum mask\n");

            //--> Compute mean in mask
            h_mean = h_refsum/h_masksum;
            printf("mean: %f\n", h_mean);

            //--> Multiply mask with mean
            CudaDeviceVariable d_scaled_mask(sizeof(float)*refdim);
            auto npp_scaled_mask = (Npp32f*)d_scaled_mask.GetDevicePtr();
            nppSafeCall(nppsMulC_32f(npp_refmask, (Npp32f)h_mean, npp_scaled_mask, refdim));
            //nppSafeCall(nppsMulC_32f(npp_mask, (Npp32f)-1.f, npp_scaled_mask, refdim));
            printf("Scaled mask\n");

            //--> Subtract scaled mask from rotated ref (in place).
            nppSafeCall(nppsSub_32f_I(npp_scaled_mask, npp_ref, refdim));
            //--> Final reference is now at the original device pointer

            printf("Finished shit\n");

            // Create host storage (using remembered dimensions from above) and copy back to host
            //--> REFERENCE
            auto tempref = new Volume<float>(vh_particle_dims[ref_id_idx]);
            vd_rot_particle[ref_id_idx]->CopyDeviceToHost(tempref->GetPtrToSubVolume(0));
            vh_rot_particle.push_back(tempref);

            //--> REFERENCE MASK
            auto temprefmask = new Volume<float>(vh_particle_dims[ref_id_idx]);
            vd_rot_refmask[ref_id_idx]->CopyDeviceToHost(temprefmask->GetPtrToSubVolume(0));
            vh_rot_refmask.push_back(temprefmask);

            //--> VOLUME MASK
            auto tempvolmask = new Volume<float>(vh_volmask_dims[ref_id_idx]);
            vd_rot_volmask[ref_id_idx]->CopyDeviceToHost(tempvolmask->GetPtrToSubVolume(0));
            vh_rot_volmask.push_back(tempvolmask);
        }

        // vh_rot_particle now contains all rotated, porperly masked particles in the order of the motivelist.
        // vh_rot_refmask now contains all rotated refrence masks.
        // vh_rot_volmask now contains all rotated volume masks for forward projection.

        // Free memory allocated for rotation (rot kernel + device output array)
        vd_ref_rotators.clear();
        vd_refmask_rotators.clear();
        vd_volmask_rotators.clear();
        vd_rot_particle.clear();
        vd_rot_refmask.clear();
        vd_rot_volmask.clear();

        // Test particle
        //emwrite("/home/uermel/Programs/artia-build/refinement/testparticle.em", vh_rot_particle[0]->GetPtrToSubVolume(0), (int)vh_rot_particle[0]->GetSubVolumeDimension(0).x, (int)vh_rot_particle[0]->GetSubVolumeDimension(0).x, (int)vh_rot_particle[0]->GetSubVolumeDimension(0).x);

        if (mpi_part == 0) printf("End preprocessing particles and masks\n");
        /////////////////////////////////////
        /// Preprocess particles END
        /////////////////////////////////////
		


		if (mpi_part == 0) printf("Free Memory on device after allocations: %i MB\n", cuCtx->GetFreeMemorySize() / 1024 / 1024);
		/////////////////////////////////////
		/// Filter Projections START
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
			log << "Pixel size is: " << proj.GetPixelSize() << " nm" << endl;

			log << "Projection statistics:" << endl;

			printf("\r\n");
			for (int i = 0; i < projSource->GetProjectionCount(); i++)
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
				//float tilt = projSource->TiltAlpha[i];
				//float weight = 1.0f / cos(tilt / 180.0f * M_PI);

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
				//TODO:reactivate this !!!!!!!!!!!!!!!!
				reconstructor.PrepareProjection(imgUS, i, meanValue, stdValue, badPixels);

				printf(" Bad Pixels: %d Mean: %f Std: %f", badPixels, meanValue, stdValue);
				log << ": Bad Pixels: " << badPixels << " Mean: " << meanValue << " Std. dev.: " << stdValue << endl;
			}
		}
		/////////////////////////////////////
		/// Filter Projections END
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

		ShiftFile sf(aConfig.ShiftOutputFile, projSource->GetProjectionCount(), ml.GetParticleCount());
		//printf("No crash here :)\n"); fflush(stdout);

		// Load reconstruction:
		volWithSubVols->LoadFromFile(aConfig.OutVolumeFile, mpi_part);

		// Send to GPU
		vol_Array.CopyFromHostToArray(volWithSubVols->GetPtrToSubVolume(mpi_part));

		//int testGroupCount = ml.GetGroupCount(aConfig);

		//if (mpi_part == 0)
		//{
		//	volSubVol->LoadFromFile(aConfig.Reference, 0);
		//	reconstructor.setRotVolData(volSubVol->GetPtrToSubVolume(0));
		//}

        // Get min and max z of subvolume on this mpi-part (this is necessary for masking in 3D later
        int zMin = 0;
        int zMax = (int)volWithSubVols->GetSubVolumeDimension(0).z;
        for (int i = 1; i <= mpi_part; i++)
        {
            zMin += (int)volWithSubVols->GetSubVolumeDimension(i).z;
            zMax += (int)volWithSubVols->GetSubVolumeDimension(i).z;
        }
        int xMax = (int)volWithSubVols->GetSubVolumeDimension(mpi_part).x;
        int yMax = (int)volWithSubVols->GetSubVolumeDimension(mpi_part).y;

        int3 volmin = make_int3(0, 0, zMin);
        int3 volmax = make_int3(xMax, yMax, zMax);

//        auto errorStack = new float*[projCount];
//        if(aConfig.SubtractError){
//            //project Reconstruction without sub-Volumes:
//            if (mpi_part == 0) {
//                for (int i = 0; i < projCount; i++) {
//                    errorStack[i] = new float[proj.GetWidth() * proj.GetHeight()];
//                    int projIndex = projIndexList[i];
//
//                    reconstructor.ResetProjectionsDevice();
//                    // Project full volume, result in proj_d
//                    reconstructor.ForwardProjection(volWithoutSubVols, texObj, projIndex, false);
//                    // Distance weighted error, result in proj_d
//                    reconstructor.Compare(volWithoutSubVols, projSource->GetProjection(projIndex), projIndex);
//
//                    reconstructor.CopyProjectionToHost(errorStack[i]);
//                    stringstream ss2;
//                    ss2 << "error_" << projIndex << ".em";
//                    emwrite(ss2.str(), errorStack[i], proj.GetWidth(), proj.GetHeight());
//
//                    // Subtract error from image, result in realproj_d
//                    //reconstructor.SubtractError();
//                    // Get realproj_d from device and save now.
//                    //reconstructor.CopyRealProjectionToHost(SIRTBuffer[0]);
//
//                    //stringstream ss;
//                    //ss << "projWithoutError_" << projIndex << ".em";
//                    //emwrite(ss.str(), SIRTBuffer[0], proj.GetWidth(), proj.GetHeight());
//                }
//            }
//        }


        /////////////////////////////////////
        /// Main refine loop START
        /////////////////////////////////////

        //int projNum = markers.GetProjectionCount();
        cudaProfilerStart();
        ///  START Loop over projections START///
        for(int projInd = 0; projInd < projCount; projInd++)
        //for(int projInd = 0; projInd < 1; projInd++)
        {
            // Index of projection in stack
            int stackIndex = projIndexList[projInd];

            // Display progress
            if (mpi_part == 0) {
                printf("Projection nr: %d/%d\n", stackIndex + 1, projCount);
                printf("\tUpdating volume ...\n");
            }

            if (mpi_part == 0)
                if (aConfig.DebugImages)//for debugging save images
                {
                    stringstream ss;
                    ss << "realProj_" << stackIndex << ".em";
                    emwrite(ss.str(), (float *) projSource->GetProjection(stackIndex), proj.GetWidth(),
                            proj.GetHeight());
                }

            /// Update volume to fit current projection ///
            reconstructor.ResetProjectionsDevice();
            // Project full volume, result in proj_d
            reconstructor.ForwardProjection(volWithoutSubVols, texObj, stackIndex, false);
            // Distance weighted error, result in proj_d
            reconstructor.Compare(volWithoutSubVols, projSource->GetProjection(stackIndex), stackIndex);
            // Correct volume for this projection
            reconstructor.BackProjection(volWithoutSubVols, surfObj, stackIndex, (float)SIRTcount);
            // The volume is now corrected for the reprojection error of this projection.

            auto updatedVolume = new float[volWithoutSubVols->GetSubVolumeSizeInVoxels(0)];
            vol_Array.CopyFromArrayToHost(updatedVolume);

//            stringstream ss;
//            ss << "updatedVolume1.em";
//            emwrite(ss.str(), updatedVolume, volWithoutSubVols->GetDimension().x, volWithoutSubVols->GetDimension().y, volWithoutSubVols->GetDimension().z);
//            delete[] updatedVolume;

            //Reset the minDist Values for each projection.
            for (size_t i = 0; i < ml.GetParticleCount(); i++)
            {
                minDistOfProcessedParticles[i] = 100000000.0f; //some large value...
            }

            // Init processed list
            auto processedParticle = new bool[ml.GetParticleCount()];
            memset(processedParticle, 0, ml.GetParticleCount() * sizeof(bool));

            // Display progress
            if (mpi_part == 0) {
                printf("\tRefining ...\n");
            }

            ///  START Loop over groups of particles and project each group together START///
            for(int group = 0; group < ml.GetGroupCount(aConfig.GroupMode); group++)
            //for(int group = 0; group < 1; group++)
            {

                // Display progress
                if (mpi_part == 0) {
                    printf("\t\tGroup: %i/%i\n", group+1, ml.GetGroupCount(aConfig.GroupMode));
                }

                //Get the motive list entries for this group:
                vector<motive> motives = ml.GetNeighbours(group, aConfig.GroupMode, aConfig.MaxDistance, aConfig.GroupSize);

                // Skip if particle shifts were assigned using SpeedUpDistance
                if (aConfig.GroupMode == MotiveList::GroupMode_enum::GM_MAXCOUNT ||
                    aConfig.GroupMode == MotiveList::GroupMode_enum::GM_MAXDIST)
                {
                    if (processedParticle[ml.GetGlobalIdx(motives[0])])
                    {
                        // Display progress
                        if (mpi_part == 0) {
                            printf("\t\t\tAlready assigned, skipping.\n");
                        }
                        continue;
                    }
                }

                // Compute mag anisotropy
                Matrix<float> magAnisotropyInv(reconstructor.GetMagAnistropyMatrix(1.0f / aConfig.MagAnisotropyAmount, aConfig.MagAnisotropyAngleInDeg, proj.GetWidth(), proj.GetHeight()));

                // Volume is full
                volumeIsEmpty = false;

                // Vector for restoration of masked regions
                vector<CudaDeviceVariable*> vd_temp_store;

                // Rotated particle's cuda array on the device (needed for texture interpolation), (size: unique_ref_count)
                vector<CudaArray3D*> vd_arr_rot_particle;
                // Texture object of the rotated particle for SART, (size: unique_ref_count)
                vector<CudaTextureObject3D*> vd_tex_rot_particle;

                // Initialize reconstructor and ROI for the group
                reconstructor.ResetProjectionsDevice();
                int2 roiMin = make_int2(proj.GetWidth(), proj.GetHeight());
                int2 roiMax = make_int2(0, 0);

                // Display progress
                if (mpi_part == 0) {
                    printf("\t\t\tRemoving particles and projecting references ...\n");
                }

                int2 roiMinCrop, roiMaxCrop;

                /// START Loop over each particle in one group START ///
                for (int groupIdx = 0; groupIdx < motives.size(); groupIdx++)
                {
                    // Get particle and associated ref_id
                    motive m = motives[groupIdx];
                    int globalIdx = ml.GetGlobalIdx(m);
                    int ref_id_idx = ref_ids[globalIdx];

                    /// 1. Remove particle from reconstructed Volume
                    // Mask dimensions and positions
                    dim3 maskDims = vh_volmask_dims[ref_id_idx];

                    int3 dimMask = make_int3(maskDims.x, maskDims.y, maskDims.z);
                    //printf("dimMask: x: %i y:%i z: %i\n", dimMask.x, dimMask.y, dimMask.z);
                    int3 radiusMask = make_int3(maskDims.x/2, maskDims.y/2, maskDims.z/2);
                    //printf("radiusMask: x: %i y:%i z: %i\n", radiusMask.x, radiusMask.y, radiusMask.z);
                    int3 centerInVol = make_int3(roundf(m.x_Coord + m.x_Shift), roundf(m.y_Coord + m.y_Shift), roundf(m.z_Coord + m.z_Shift));
                    //printf("centerInVol: x: %i y:%i z: %i\n", centerInVol.x, centerInVol.y, centerInVol.z);

                    //printf("volmin: x: %i y:%i z: %i\n", volmin.x, volmin.y, volmin.z);
                    //printf("volmax: x: %i y:%i z: %i\n", volmax.x, volmax.y, volmax.z);

                    // Get rotated mask and create temporary storage
                    CudaDeviceVariable d_volmask(maskDims.x * maskDims.x * maskDims.x * sizeof(float));
                    d_volmask.CopyHostToDevice(vh_rot_volmask[globalIdx]->GetPtrToSubVolume(0));
                    auto temp = new CudaDeviceVariable(maskDims.x * maskDims.x * maskDims.x * sizeof(float));
                    temp->Memset(0);
                    vd_temp_store.push_back(temp);

                    // Apply the volume mask for this particle
                    (*vd_apply_mask_kernels[ref_id_idx])(surfObj, d_volmask, *vd_temp_store[groupIdx], volmin, volmax, dimMask, radiusMask, centerInVol);

//                    auto temppart = new float[dimMask.x*dimMask.y*dimMask.z];
//                    vd_temp_store[groupIdx]->CopyDeviceToHost(temppart, dimMask.x*dimMask.y*dimMask.z* sizeof(float));
//
//                    stringstream ss;
//                    ss << "cutout_" << stackIndex << "_" << m.tomoNr << "_" << m.partNrInTomo << ".em";
//                    emwrite(ss.str(), temppart, dimMask.x, dimMask.y, dimMask.z);
//                    delete[] temppart;


                    /// 2. Project the reference volume
                    // Create the cuda array for the rotated particle (source data for interpolation during SART-FP)
                    auto arr = new CudaArray3D(arrayFormat, (int)vh_ori_particle[ref_id_idx]->GetDimension().x, (int)vh_ori_particle[ref_id_idx]->GetDimension().x, (int)vh_ori_particle[ref_id_idx]->GetDimension().x, 1, 2);
                    // Populate the array with the rotated particle
                    arr->CopyFromHostToArray(vh_rot_particle[globalIdx]->GetPtrToSubVolume(0));
                    // Add to group vector
                    vd_arr_rot_particle.push_back(arr); // pirate vector :)
                    // Create the texture object for the rotated particle (for interpolation during FP)
                    vd_tex_rot_particle.push_back(new CudaTextureObject3D(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, arr));

                    // Position info from motive list
                    float3 posSubVol = make_float3(m.x_Coord, m.y_Coord, m.z_Coord);
                    float3 shift = make_float3(m.x_Shift, m.y_Shift, m.z_Shift);

                    // Figure out the 2D-ROI
                    int2 hitPoint;
                    float3 bbMin = volWithoutSubVols->GetVolumeBBoxMin();
                    proj.ComputeHitPoint(bbMin.x + (posSubVol.x + shift.x - 1) * aConfig.VoxelSize.x +
                                         0.5f * aConfig.VoxelSize.x,
                                         bbMin.y + (posSubVol.y + shift.y - 1) * aConfig.VoxelSize.y +
                                         0.5f * aConfig.VoxelSize.y,
                                         bbMin.z + (posSubVol.z + shift.z - 1) * aConfig.VoxelSize.z +
                                         0.5f * aConfig.VoxelSize.z,
                                         stackIndex, hitPoint);

                    float hitX, hitY;
                    MatrixVector3Mul(*(float3x3 *) magAnisotropyInv.GetData(), hitPoint.x, hitPoint.y, hitX,
                                     hitY);

                    int safeDist =
                            100 + aConfig.VoxelSizeSubVol * aConfig.SizeSubVol * 2 + aConfig.MaxShift * 2;
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

                    int2 roiMinLocal = make_int2(hitXMin, hitYMin);
                    int2 roiMaxLocal = make_int2(hitXMax, hitYMax);

                    roiMinCrop = make_int2(floor(hitX) - 128, floor(hitY) - 128);
                    roiMaxCrop = make_int2(floor(hitX) + 127, floor(hitY) + 127);

                    // Set the 3D position
                    vh_rot_particle[globalIdx]->PositionInSpace(aConfig.VoxelSize, aConfig.VoxelSizeSubVol,
                                                                *volWithoutSubVols, posSubVol, shift);



                    //Project the particle:
                    reconstructor.ForwardProjectionROI(vh_rot_particle[globalIdx],
                                                       *vd_tex_rot_particle[groupIdx], stackIndex, volumeIsEmpty,
                                                       roiMinLocal, roiMaxLocal, true);
                }
                /// END Loop over each particle in one group END///

//                auto updatedVolume2 = new float[volWithoutSubVols->GetSubVolumeSizeInVoxels(0)];
//                vol_Array.CopyFromArrayToHost(updatedVolume2);
//
//                stringstream ss2;
//                ss2 << "updatedVolume2.em";
//                emwrite(ss2.str(), updatedVolume2, volWithoutSubVols->GetDimension().x, volWithoutSubVols->GetDimension().y, volWithoutSubVols->GetDimension().z);
//                delete[] updatedVolume2;

                /// This is the final projection of the reference volumes of this group:
                reconstructor.CopyProjectionToSubVolumeProjection();

                if (mpi_part == 0)
                    if (aConfig.DebugImages)//for debugging save images
                    {
                        reconstructor.CopyProjectionToHost(SIRTBuffer[0]);

                        stringstream ss;
                        ss << "projSubVols_" << group << "_" << stackIndex << ".em";
                        emwrite(ss.str(), SIRTBuffer[0], proj.GetWidth(), proj.GetHeight());
                    }

                /// Now, project the holey volume and subtract the FP from the real projection
                // Display progress
                if (mpi_part == 0) {
                    printf("\t\t\tProjecting holey volume ...\n");
                }
                reconstructor.ResetProjectionsDevice();
                reconstructor.ForwardProjectionROI(volWithoutSubVols, texObj, stackIndex, false, roiMin, roiMax);

                if (mpi_part == 0)
                    if (aConfig.DebugImages)//for debugging save images
                    {
                        reconstructor.CopyProjectionToHost(SIRTBuffer[0]);
                        stringstream ss;
                        ss << "projVorComp_" << stackIndex << ".em";
                        emwrite(ss.str(), SIRTBuffer[0], proj.GetWidth(), proj.GetHeight());

                        reconstructor.CopyDistanceImageToHost(SIRTBuffer[0]);
                        stringstream ss2;
                        ss2 << "distVorComp_" << stackIndex << ".em";
                        emwrite(ss2.str(), SIRTBuffer[0], proj.GetWidth(), proj.GetHeight());
                    }

                // Display progress
                if (mpi_part == 0) {
                    printf("\t\t\tFinding displacements ...\n");
                }

                reconstructor.Compare(volWithoutSubVols, projSource->GetProjection(stackIndex), stackIndex);
                /// We now have in proj_d the distance weighted 'background noise free' approximation of the original projection.

                // FOR CROPPING:
//                auto cropped = new float[256*256];
//                printf("roiMinCrop: %i, %i\n", roiMinCrop.x, roiMinCrop.y);
//                printf("roiMaxCrop: %i, %i\n", roiMaxCrop.x, roiMaxCrop.y);
//                // real crop version
//                //reconstructor.GetCroppedProjection(cropped, (float*)projSource->GetProjection(stackIndex), roiMinCrop, roiMaxCrop);
//                // old/new method
//                reconstructor.GetCroppedProjection(cropped, roiMinCrop, roiMaxCrop);
//
//                stringstream ss;
//                ss << "part_" << group << "_" << stackIndex << ".em";
//                emwrite(ss.str(), cropped, 256, 256);
//                delete[] cropped;

                if (mpi_part == 0)
                    if (aConfig.DebugImages)
                    {
                        reconstructor.CopyProjectionToHost(SIRTBuffer[0]);

                        stringstream ss;
                        ss << "projComp_" << group << "_" << stackIndex << ".em";
                        emwrite(ss.str(), SIRTBuffer[0], proj.GetWidth(), proj.GetHeight());
                    }

                /// Now compute the displacement
                float2 shift;
                float ccValue;
                float* ccMap;
                float* ccMapMulti;
                if (mpi_part == 0)
                {
                    shift = reconstructor.GetDisplacement(aConfig.MultiPeakDetection, &ccValue);
                    printf("\t\t\tX: %f, Y: %f, CC: %f\n", shift.x, shift.y, ccValue);
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
                        ss << aConfig.CCMapFileName << stackIndex << ".em";
                        if (group == 0) //create a new file and overwrite the old one
                        {
                            emwrite(ss.str(), ccMap, aConfig.MaxShift * 4, aConfig.MaxShift * 4);
                        }
                        else //append to existing file
                        {
                            EmFile::AddSlice(ss.str(), ccMap, aConfig.MaxShift * 4, aConfig.MaxShift * 4);
                        }
                        if (aConfig.MultiPeakDetection)
                        {
                            stringstream ss2;
                            ss2 << aConfig.CCMapFileName << "Multi_" << stackIndex << ".em";
                            if (group == 0) //create a new file and overwrite the old one
                            {
                                emwrite(ss2.str(), ccMapMulti, aConfig.MaxShift * 4, aConfig.MaxShift * 4);
                            }
                            else //append to existing file
                            {
                                EmFile::AddSlice(ss2.str(), ccMapMulti, aConfig.MaxShift * 4, aConfig.MaxShift * 4);
                            }
                        }
                    }
                /// Got displacement

                // Save shift for every entry in the group
                if (aConfig.GroupMode == MotiveList::GroupMode_enum::GM_BYGROUP)
                {
                    for (int motlIdx = 0; motlIdx < motives.size(); motlIdx++)
                    {
                        motive m = motives[motlIdx];

                        int totalIdx = ml.GetGlobalIdx(m);
                        //printf("Set values at: %d, %d\n", index, motlIdx); fflush(stdout);

                        // Convert to my_float to avoid FileIO dependency on Cuda
                        sf.SetValue(stackIndex, totalIdx, my_float2(shift.x, shift.y));
                        //groupRelationList[totalIdx].particleNr = m.partNr;
                        //groupRelationList[totalIdx].particleNrInTomo = m.partNrInTomo;
                        //groupRelationList[totalIdx].tomoNr = m.tomoNr;
                        //groupRelationList[totalIdx].obtainedShiftFromID = group;
                        //groupRelationList[totalIdx].CCValue = ccValue;
                    }
                }
                else //save shift only for the first entry in the group
                {
                    // Get first entry
                    motive m = motives[0];
                    int count = 0; // Number of particles within group that is below speedupdistance
                    vector<pair<int, float> > closeIndx;
                    int totalIdx = ml.GetGlobalIdx(m); // Global motivelist index

                    // Save the shift of the first entry
                    sf.SetValue(stackIndex, totalIdx, my_float2(shift.x, shift.y));
                    processedParticle[totalIdx] = true;

                    // For all entries in group except the first check if they are below speedup distance
                    for (int groupIdx = 1; groupIdx < motives.size(); groupIdx++) {
                        // Get entry
                        motive m2 = motives[groupIdx];

                        // Get distance/global index
                        float d = ml.GetDistance(m, m2);
                        int totalIdx2 = ml.GetGlobalIdx(m2);

                        // If distance <= SpeedUpDistance, save the shift and mark as processed
                        if (d <= aConfig.SpeedUpDistance && d < minDistOfProcessedParticles[totalIdx2]) {
                            // Convert to my_float to avoid FileIO dependency on Cuda
                            sf.SetValue(stackIndex, totalIdx2, my_float2(shift.x, shift.y));
                            processedParticle[totalIdx2] = true;
                            minDistOfProcessedParticles[totalIdx2] = d;
                            closeIndx.push_back(pair<int, float>(totalIdx2, d));
                            count++;

                            //groupRelationList[totalIdx2].particleNr = m2.partNr;
                            //groupRelationList[totalIdx2].particleNrInTomo = m2.partNrInTomo;
                            //groupRelationList[totalIdx2].tomoNr = m2.tomoNr;
                            //groupRelationList[totalIdx2].obtainedShiftFromID = totalIdx;
                            //groupRelationList[totalIdx2].CCValue = ccValue;
                        }
                    }
                }

                // Display progress
                if (mpi_part == 0) {
                    printf("\t\t\tResetting volume ...\n");
                }

                /// START Loop over each particle in one group START ///
                // Need to do this in reverse order because some particles already contain marked regions.
                for (int groupIdx = motives.size()-1; groupIdx > -1; groupIdx--)
                {
                    // Get particle and associated ref_id
                    motive m = motives[groupIdx];
                    int globalIdx = ml.GetGlobalIdx(m);
                    int ref_id_idx = ref_ids[globalIdx];

                    // Mask dimensions and positions
                    dim3 maskDims = vh_volmask_dims[ref_id_idx];
                    int3 dimMask = make_int3(maskDims.x, maskDims.y, maskDims.z);
                    int3 radiusMask = make_int3(maskDims.x/2, maskDims.y/2, maskDims.z/2);
                    int3 centerInVol = make_int3(roundf(m.x_Coord + m.x_Shift), roundf(m.y_Coord + m.y_Shift), roundf(m.z_Coord + m.z_Shift));

                    // Restore the volume
                    (*vd_restore_vol_kernels[ref_id_idx])(surfObj, *vd_temp_store[groupIdx], volmin, volmax, dimMask, radiusMask, centerInVol);
                }
                /// END Loop over each particle in one group END ///

//                auto updatedVolume = new float[volWithoutSubVols->GetSubVolumeSizeInVoxels(0)];
//                vol_Array.CopyFromArrayToHost(updatedVolume);
//
//                stringstream ss;
//                ss << "updatedVolume3.em";
//                emwrite(ss.str(), updatedVolume, volWithoutSubVols->GetDimension().x, volWithoutSubVols->GetDimension().y, volWithoutSubVols->GetDimension().z);
//                delete[] updatedVolume;

                // Clear memory
                for(int i = 0; i < vd_tex_rot_particle.size(); i++) {delete vd_tex_rot_particle[i];}
                for(int i = 0; i < vd_arr_rot_particle.size(); i++) {delete vd_arr_rot_particle[i];}
                for(int i = 0; i < vd_temp_store.size(); i++) {delete vd_temp_store[i];}
                vd_tex_rot_particle.clear();
                vd_arr_rot_particle.clear();
                vd_temp_store.clear();

                if (mpi_part == 0)
                {
                    printf("\n");
                }
            }
            /// END Loop over groups of particles and project each group together END ///
        }
        /// END Loop over projections END ///

		cudaProfilerStop();
        /////////////////////////////////////
        /// Main refine loop END
        /////////////////////////////////////

		//Save measured local shifts to an EM-file:
		if (mpi_part == 0)
		{
			sf.OpenAndWrite();

//			ofstream fs;
//			fs.open(aConfig.ShiftOutputFile + ".relations");
//			for (size_t i = 0; i < ml.GetParticleCount(); i++)
//			{
//				fs << groupRelationList[i].particleNr << "; " << groupRelationList[i].particleNrInTomo << "; " << groupRelationList[i].particleNrInTomo << "; " << groupRelationList[i].obtainedShiftFromID << "; " << groupRelationList[i].CCValue << std::endl;
//			}
//			fs.close();
		}

		//emwrite(aConfig.ShiftOutputFile.c_str(), (float*)extraShifts, 2, projSource->DimZ);


		stop = clock();
		runtime = (double)(stop - start) / CLOCKS_PER_SEC;

		if (mpi_part == 0) printf("\n\nTotal time for local shift measurements: %.2i:%.2i min.\n\n", (int)floor(runtime / 60.0), (int)floor(((runtime / 60.0) - floor(runtime / 60.0))*60.0));

        //for (int projIndex = 0; projIndex < projCount; projIndex++){
        //    delete[] errorStack[projIndex];
       // }

        //delete[] errorStack;

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
