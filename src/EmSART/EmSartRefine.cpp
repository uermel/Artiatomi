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
		Projection proj(projSource, &markers);

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

        // Tomogram's array on the device (needed for texture interpolation)
        CudaArray3D vol_Array(arrayFormat, volSize.x, volSize.y, volSize.z, 1, 2);
        // Tomogram's texture object for SART
        CudaTextureObject3D texObj(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &vol_Array);



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

        Reconstructor reconstructor(aConfig, proj, projSource, markers, *defocus, modules, mpi_part, mpi_size);

        /////////////////////////////////////
        /// Prepare tomogram  END
        /////////////////////////////////////

        /////////////////////////////////////
        /// Preprocess particles START
        /////////////////////////////////////
        if (mpi_part == 0) printf("Start preprocessing particles\n");
		// Get Motivelist and select for tomogram
		MotiveList ml(aConfig.MotiveList, aConfig.ScaleMotivelistPosition, aConfig.ScaleMotivelistShift);
		ml.selectTomo(aConfig.TomogramIndex);

		// Get number and IDs of unique references
		vector<int> unique_ref_ids; // Unique reference IDs
		int unique_ref_count = 0; // Number of unique IDs
		auto ref_ids = new int[ml.GetParticleCount()]; // For each particle, the corresponding index within unique_ref_ids
		ml.getRefIndeces(unique_ref_ids, ref_ids, unique_ref_count);

		// Init processed list
        auto processedParticle = new bool[ml.GetParticleCount()];
		memset(processedParticle, 0, ml.GetParticleCount() * sizeof(bool));

		// Init distance list
		auto minDistOfProcessedParticles = new float[ml.GetParticleCount()];
		//groupRelations* groupRelationList = new groupRelations[ml.GetParticleCount()];
		//memset(groupRelationList, 0, ml.GetParticleCount() * sizeof(groupRelations));

        // Create working memory (host)
        vector<Volume<unsigned short> *> v_volFP16; // TODO:What the hell does this do?

        // Storage of original particle (size: unique_ref_count)
        vector<Volume<float> *> vh_ori_particle;

        // Storage of rotated particles (size: ml.GetParticleCount());
        vector<Volume<float> *> vh_rot_particle;

        // Storage of particle dimensions
        vector<uint3> vh_particle_dims;

        // Init the original reference volumes on host, read from file, get dimensions
        vector<float3> vh_subVolDim;
        for (size_t i = 0; i < unique_ref_count; i++)
        {
            // Build name
            stringstream name;
            name << aConfig.Reference << unique_ref_ids[i] << ".em";
            // Get dims
            EmFile ref(name.str());
            ref.OpenAndReadHeader();
            // Create volume object
            uint3 tempdim = make_dim3(ref.GetFileHeader().DimX, ref.GetFileHeader().DimY, ref.GetFileHeader().DimZ);
            auto tempvol = new Volume<float>(tempdim);
            tempvol->LoadFromFile(name.str(), 0);
            vh_ori_particle.push_back(tempvol);
            vh_particle_dims.push_back(tempdim);

            // Get dimension of particle
            if (aConfig.FP16Volume)
                vh_subVolDim.push_back(v_volFP16[i]->GetSubVolumeDimension(mpi_part));
            else
                vh_subVolDim.push_back(vh_ori_particle[i]->GetSubVolumeDimension(0));
        }

        // Device array
        //CudaDeviceVariable d_rot_particle(aConfig.SizeSubVol * aConfig.SizeSubVol * aConfig.SizeSubVol * sizeof(float));

        // Rotation kernels for the reference volumes of (potentially) different sizes (size: unique_ref_count)
        vector<RotKernel*> vd_Rotators;
        // Output device arrays for the reference volumes of (potentially) different sizes (size: unique_ref_count)
        vector<CudaDeviceVariable*> vd_rot_particle;

        // Rotated particle's array on the device (needed for texture interpolation), (size: unique_ref_count)
        vector<CudaArray3D*> vd_arr_rot_particle;
        // Texture object of the rotated particle for SART, (size: unique_ref_count)
        vector<CudaTextureObject3D*> vd_tex_rot_particle;

        // Initialize the kernels and device variables
        for (size_t i = 0; i < unique_ref_count; i++)
        {
            // Create the kernel object
            auto rotator = new RotKernel(modules.modWBP, (int)vh_ori_particle[i]->GetDimension().x);
            // Set the unrotated reference as default data
            rotator->SetData(vh_ori_particle[i]->GetPtrToSubVolume(0));
            vd_Rotators.push_back(rotator);
            // Create the output arrays
            vd_rot_particle.push_back(new CudaDeviceVariable((int)vh_ori_particle[i]->GetDimension().x * (int)vh_ori_particle[i]->GetDimension().x * (int)vh_ori_particle[i]->GetDimension().x * sizeof(float)));

            // Create the cuda array for the rotated particle (source data for interpolation during SART-FP)
            auto arr = new CudaArray3D(arrayFormat, (int)vh_ori_particle[i]->GetDimension().x, (int)vh_ori_particle[i]->GetDimension().x, (int)vh_ori_particle[i]->GetDimension().x, 1, 2);
            vd_arr_rot_particle.push_back(arr); // pirate vector
            // Create the texture object for the rotated particle (for interpolation during SART-FP)
            vd_tex_rot_particle.push_back(new CudaTextureObject3D(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, arr));
        }

        // Now rotate all particles and store them on host
        for (size_t motlIdx = 0; motlIdx < ml.GetParticleCount(); motlIdx++)
        {
            // Get current particle's data
            motive m = ml.GetAt(motlIdx);

            // Rotate
            int ref_id_idx = ref_ids[motlIdx];
            (*vd_Rotators[ref_id_idx])(*vd_rot_particle[ref_id_idx], m.phi, m.psi, m.theta);

            // Create host storage (using remembered dimensions from above) and copy back to host
            auto tempvol = new Volume<float>(vh_particle_dims[ref_id_idx]);
            vd_rot_particle[ref_id_idx]->CopyDeviceToHost(tempvol->GetPtrToSubVolume(0));
            vh_rot_particle.push_back(tempvol);
        }

        // vh_rot_particle now contains all rotated particles in the order of the motivelist.

        // Free memory allocated for rotation (rot kernel + device output array)
        vd_Rotators.clear();
        vd_rot_particle.clear();
        if (mpi_part == 0) printf("End preprocessing particles\n");
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

		//Load reconstruction:
		volWithSubVols->LoadFromFile(aConfig.OutVolumeFile, mpi_part);

		//int testGroupCount = ml.GetGroupCount(aConfig);

		//if (mpi_part == 0)
		//{
		//	volSubVol->LoadFromFile(aConfig.Reference, 0);
		//	reconstructor.setRotVolData(volSubVol->GetPtrToSubVolume(0));
		//}

        /////////////////////////////////////
        /// Main refine loop START
        /////////////////////////////////////
		for (int group = 0; group < ml.GetGroupCount(aConfig.GroupMode); group++)
		{
			Matrix<float> magAnisotropyInv(reconstructor.GetMagAnistropyMatrix(1.0f / aConfig.MagAnisotropyAmount, aConfig.MagAnisotropyAngleInDeg, proj.GetWidth(), proj.GetHeight()));

			//reset Volume:
			{
				volWithoutSubVols->LoadFromVolume(*volWithSubVols, mpi_part);
			}

			volumeIsEmpty = false;
			
			//Get the motive list entries for this group:
			vector<motive> motives = ml.GetNeighbours(group, aConfig.GroupMode, aConfig.MaxDistance, aConfig.GroupSize);

			if (aConfig.GroupMode == MotiveList::GroupMode_enum::GM_MAXCOUNT ||
				aConfig.GroupMode == MotiveList::GroupMode_enum::GM_MAXDIST)
			{
				if (processedParticle[ml.GetGlobalIdx(motives[0])])
				{
					continue;
				}
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

			//Copy the holey volume to GPU
			vol_Array.CopyFromHostToArray(volWithoutSubVols->GetPtrToSubVolume(mpi_part));

			if (mpi_part == 0)
				switch (aConfig.GroupMode)
				{
				case MotiveList::GroupMode_enum::GM_BYGROUP:
					printf("Group nr: %d/%d, group size: %d\n", group, (int)ml.GetGroupCount(aConfig.GroupMode), (int)motives.size());
					log << "Group nr: " << group << "/" << (int)ml.GetGroupCount(aConfig.GroupMode) << ", group size: " << (int)motives.size() << endl;
					break;
				case MotiveList::GroupMode_enum::GM_MAXDIST:
					printf("Group nr: %d/%d, group size: %d\n", group, (int)ml.GetGroupCount(aConfig.GroupMode), (int)motives.size());
					log << "Group nr: " << group << "/" << (int)ml.GetGroupCount(aConfig.GroupMode) << ", group size: " << (int)motives.size() << endl;
					break;
				case MotiveList::GroupMode_enum::GM_MAXCOUNT:
					{
						float dist = ml.GetDistance(motives[0], motives[motives.size() - 1]);
						printf("Group nr: %d/%d, max distance : %f [voxel]\n", group, (int)ml.GetGroupCount(aConfig.GroupMode), dist);
						log << "Group nr: " << group << "/" << (int)ml.GetGroupCount(aConfig.GroupMode) << ", max distance: " << dist << " [voxel]" << endl;
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
				for (size_t i = 0; i < ml.GetParticleCount(); i++)
				{
					minDistOfProcessedParticles[i] = 100000000.0f; //some large value...
				}

				int projIndex = projIndexList[i];
				//projIndex = 20;
				reconstructor.ResetProjectionsDevice();
				int2 roiMin = make_int2(proj.GetWidth(), proj.GetHeight());
				int2 roiMax = make_int2(0, 0);

				if (mpi_part == 0)
				{
					for (int motlIdx = 0; motlIdx < motives.size(); motlIdx++)
					{
						motive m = motives[motlIdx];
						int globalIdx = ml.GetGlobalIdx(m);
						int ref_id_idx = ref_ids[globalIdx];
						
						//reconstructor.rotVol(volRot, m.phi, m.psi, m.theta);
						//volRot.CopyDeviceToHost(volSubVolRot->GetPtrToSubVolume(0));

						//vol_ArraySubVol.CopyFromDeviceToArray(volRot);
                        vd_arr_rot_particle[ref_id_idx]->CopyFromHostToArray(vh_rot_particle[globalIdx]->GetPtrToSubVolume(0));

						float3 posSubVol = make_float3(m.x_Coord, m.y_Coord, m.z_Coord);
						float3 shift = make_float3(m.x_Shift, m.y_Shift, m.z_Shift);
						//printf("posSubVol: (%f, %f, %f) * %f\n", posSubVol.x, posSubVol.y, posSubVol.z, aConfig.ScaleMotivelistPosition);
						//printf("shift: (%f, %f, %f)\n", shift.x, shift.y, shift.z);

						int2 hitPoint;
						float3 bbMin = volWithoutSubVols->GetVolumeBBoxMin();
						proj.ComputeHitPoint(bbMin.x + (posSubVol.x + shift.x - 1) * aConfig.VoxelSize.x + 0.5f * aConfig.VoxelSize.x,
							bbMin.y + (posSubVol.y + shift.y - 1) * aConfig.VoxelSize.y + 0.5f * aConfig.VoxelSize.y,
							bbMin.z + (posSubVol.z + shift.z - 1) * aConfig.VoxelSize.z + 0.5f * aConfig.VoxelSize.z,
							projIndex, hitPoint);
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

                        vh_rot_particle[globalIdx]->PositionInSpace(aConfig.VoxelSize, aConfig.VoxelSizeSubVol, *volWithoutSubVols, posSubVol, shift);

						int2 roiMinLocal = make_int2(hitXMin, hitYMin);
						int2 roiMaxLocal = make_int2(hitXMax, hitYMax);
						//printf("ROILocal: xMin: %d, yMin: %d, xMax: %d, yMax: %d\n", roiMinLocal.x, roiMinLocal.y, roiMaxLocal.x, roiMaxLocal.y);

						//Project actual index:						
						/*printf("Do projection...\n");
						fflush(stdout);*/
						reconstructor.ForwardProjectionROI(vh_rot_particle[globalIdx], *vd_tex_rot_particle[ref_id_idx], projIndex, volumeIsEmpty, roiMinLocal, roiMaxLocal, true);
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
					ss << "projSubVols_" << group << "_" << projIndex << ".em";
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
					reconstructor.ForwardProjectionROI(volWithoutSubVols, texObj, projIndex, false, roiMin, roiMax);
					//reconstructor.ForwardProjection(volWithoutSubVols, texObj, index, false);
					
					/*reconstructor.CopyProjectionToHost(SIRTBuffer[0]);

					stringstream ss;
					ss << "projVorComp_" << index << ".em";
					emwrite(ss.str(), SIRTBuffer[0], proj.GetWidth(), proj.GetHeight());

					reconstructor.CopyDistanceImageToHost(SIRTBuffer[0]);
					stringstream ss2;
					ss2 << "distVorComp_" << index << ".em";
					emwrite(ss2.str(), SIRTBuffer[0], proj.GetWidth(), proj.GetHeight());*/

					reconstructor.Compare(volWithoutSubVols, projSource->GetProjection(projIndex), projIndex);
					//We now have in proj_d the distance weighted 'noise free' approximation of the original projection.
				}
				if (mpi_part == 0)
				if (aConfig.DebugImages)
				{
					printf("Save image...\n");
					fflush(stdout);
					reconstructor.CopyProjectionToHost(SIRTBuffer[0]);

					stringstream ss;
					ss << "projComp_" << group << "_" << projIndex << ".em";
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
					ss << aConfig.CCMapFileName << projIndex << ".em";
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
						ss2 << aConfig.CCMapFileName << "Multi_" << projIndex << ".em";
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
					if (aConfig.GroupMode == MotiveList::GroupMode_enum::GM_BYGROUP)
					{
						for (int motlIdx = 0; motlIdx < motives.size(); motlIdx++)
						{
							motive m = motives[motlIdx];

							int totalIdx = ml.GetGlobalIdx(m);
							//printf("Set values at: %d, %d\n", index, motlIdx); fflush(stdout);

							// Convert to my_float to avoid FileIO dependency on Cuda
							sf.SetValue(projIndex, totalIdx, my_float2(shift.x, shift.y));
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
                        sf.SetValue(projIndex, totalIdx, my_float2(shift.x, shift.y));
                        processedParticle[totalIdx] = true;
						
						// For all entries in group except the first check if they are below speedup distance
						for (int groupIdx = 1; groupIdx < motives.size(); groupIdx++)
						{
						    // Get entry
							motive m2 = motives[groupIdx];
							
							// Get distance/global index
							float d = ml.GetDistance(m, m2);
							int totalIdx2 = ml.GetGlobalIdx(m2);
							
							// If distance <= SpeedUpDistance, save the shift and mark as processed
							if (d <= aConfig.SpeedUpDistance && d < minDistOfProcessedParticles[totalIdx2])
							{
								// Convert to my_float to avoid FileIO dependency on Cuda
								sf.SetValue(projIndex, totalIdx2, my_float2(shift.x, shift.y));
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
