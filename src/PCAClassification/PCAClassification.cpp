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

// System
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>

// Cuda
#include <cusolverDn.h>

// EmTools
#include <FileIO.h>
#include <EmFile.h>
#include <MarkerFile.h>
#include <CudaContext.h>
#include <Correlator3D.h>
#include <Rotator.h>
#include <SubDeviceKernel.h>
#include <Kernels/SubDeviceKernel.cu.h>
#include <ThreadPool.h>
#include <SpecificBackgroundThread.h>

// Self
#include "Kernels/kernels.cu.h"
#include "PCAKernels.h"
#include "utils/Config.h"
#include "KMeans.h"
#include <cudaProfiler.h>
#include <chrono>

using namespace std;
using namespace Cuda;

#define LOG(a, ...) (MKLog::Get()->Log(LL_INFO, a, __VA_ARGS__))

#define round(x) (x >= 0 ? (int)(x + 0.5) : (int)(x - 0.5))

#define runBatch SingletonThread::Get(batch)->enqueue

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

	CudaContext* ctx = NULL;

	try
	{
		const int batchSize = 4;
		Configuration::Config aConfig = Configuration::Config::GetConfig("pca.cfg", argc, argv, mpi_part, NULL);
		ctx = CudaContext::GetPrimaryContext(aConfig.CudaDeviceIDs[mpi_part]);
		ctx->SetCurrent();
		CUdevice dev;
		cudaSafeCall(cuDeviceGet(&dev, 0));
		cudaSafeCall(cuDevicePrimaryCtxSetFlags(dev, CU_CTX_SCHED_SPIN));

		vector<CUstream> streams(batchSize);
		vector<NppStreamContext> streamCtx(batchSize);
		for (size_t batch = 0; batch < batchSize; batch++)
		{
			cudaSafeCall(cuStreamCreate(&streams[batch], CU_STREAM_DEFAULT));
			nppSafeCall(nppSetStream(streams[batch]));
			nppSafeCall(nppGetStreamContext(&streamCtx[batch]));
		}
		nppSafeCall(nppSetStream(0));


		printf("\nLoad motivelist..."); fflush(stdout);
		MotiveList motl(aConfig.MotiveList, 1.0f, aConfig.ScaleMotivelistShift);
		printf("Done\n"); fflush(stdout);
	
		int totalCount = motl.GetParticleCount();
		int partCount = motl.GetParticleCount() / mpi_size;
		int lastPartCount = totalCount - (partCount * (mpi_size - 1));
		int startParticle = mpi_part * partCount;
	
		int totalCountPCA = motl.GetParticleCount();
		if (aConfig.LimitNumberOfParticlesInPCA)
		{
			totalCountPCA = aConfig.NumberOfParticlesInPCA;
		}

		int partCountPCA = totalCountPCA / mpi_size;
		int lastPartCountPCA = totalCountPCA - (partCountPCA * (mpi_size - 1));
		int startParticlePCA = mpi_part * partCountPCA;

		//adjust last part to fit really all particles (rounding errors...)
		if (mpi_part == mpi_size - 1)
		{
			partCount = lastPartCount;
			partCountPCA = lastPartCountPCA;
		}

		int endParticle = startParticle + partCount;
		int endParticlePCA = startParticlePCA + partCountPCA;


		vector<int> unique_wedge_ids; // Unique wedge IDs
		int unique_wedge_count = 0; // Number of unique wedge IDs
		int* wedge_ids = new int[motl.GetParticleCount()]; // For each particle, the corresponding wedge index within unique_ref_ids
		motl.getWedgeIndeces(unique_wedge_ids, wedge_ids, unique_wedge_count);

		map<int, EmFile*> wedges;
		if (aConfig.SingleWedge)
		{
			wedges.insert(pair<int, EmFile*>(0, new EmFile(aConfig.WedgeFile)));
			wedges[0]->OpenAndRead();
			unique_wedge_ids[0] = 0;
		}
		else
		{
			for (size_t i = 0; i < unique_wedge_count; i++)
			{
				stringstream sswedge;
				sswedge << aConfig.WedgeFile << unique_wedge_ids[i] << ".em";
				wedges.insert(pair<int, EmFile*>(unique_wedge_ids[i], new EmFile(sswedge.str())));
				wedges[unique_wedge_ids[i]]->OpenAndRead();
			}
		}

		EmFile mask(aConfig.Mask);
		mask.OpenAndRead();

		EmFile filter(aConfig.FilterFileName);
		if (aConfig.UseFilterVolume)
		{
			filter.OpenAndRead();
		}

		////////////////////////////////////
		/// Step 1: check input and init ///
		////////////////////////////////////

		if (mpi_part == 0)
		{
			printf("Step 1: Check input and init\n"); fflush(stdout);
		}

		int volSize = mask.GetFileHeader().DimX;
		int maskedVoxels = 0;
		vector<int> maskIndices;

		CUmodule mod = ctx->LoadModulePTX(PCAKernel, 0, false, false);
		CUmodule mod2 = ctx->LoadModulePTX(SubTomogramSubDeviceKernel, 0, false, false);
		ComputeEigenImagesKernel kernelEigenImages(mod);
		SubDivDeviceKernel kernelSubDiv(mod2);

		vector<CudaDeviceVariable> d_particle;
		vector<CudaDeviceVariable> d_particleMasked;
		vector<CudaDeviceVariable> d_particleRef;
		vector<CudaDeviceVariable> d_mask;
		vector<CudaDeviceVariable> d_filter;
		vector<CudaDeviceVariable> d_meanParticle;

		vector<CudaDeviceVariable> d_wedge1;
		vector<CudaDeviceVariable> d_wedge2;
		vector<CudaDeviceVariable> d_wedgeMerge;
		vector<CudaDeviceVariable> d_CC;
		vector<CudaPageLockedHostVariable> h_pagelockedBuffer;

		for (size_t batch = 0; batch < batchSize; batch++)
		{
			d_particle.push_back(std::move(CudaDeviceVariable((size_t)volSize * volSize * volSize * sizeof(float))));
			d_particleMasked.push_back(std::move(CudaDeviceVariable((size_t)volSize * volSize * volSize * sizeof(float))));
			d_particleRef.push_back(std::move(CudaDeviceVariable((size_t)volSize * volSize * volSize * sizeof(float))));
			d_mask.push_back(std::move(CudaDeviceVariable((size_t)volSize * volSize * volSize * sizeof(float))));
			d_filter.push_back(std::move(CudaDeviceVariable((size_t)volSize * volSize * volSize * sizeof(float))));
			d_meanParticle.push_back(std::move(CudaDeviceVariable((size_t)volSize * volSize * volSize * sizeof(float))));

			d_wedge1.push_back(std::move(CudaDeviceVariable((size_t)volSize * volSize * volSize * sizeof(float))));
			d_wedge2.push_back(std::move(CudaDeviceVariable((size_t)volSize * volSize * volSize * sizeof(float))));
			d_wedgeMerge.push_back(std::move(CudaDeviceVariable((size_t)volSize * volSize * volSize * sizeof(float))));
			d_CC.push_back(std::move(CudaDeviceVariable((size_t)volSize * volSize * volSize * sizeof(float))));

			h_pagelockedBuffer.push_back(std::move(CudaPageLockedHostVariable((size_t)volSize * volSize * volSize * sizeof(float), 0)));
		}

		cusolverDnHandle_t cusolverH = NULL;
		cusolverDnParams_t params = NULL;
		cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
		cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
		cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;

		CudaDeviceVariable d_eigenImages((size_t)aConfig.NumberOfEigenVectors * volSize * volSize * volSize * sizeof(float));
		CudaDeviceVariable d_covVarMat((size_t)totalCountPCA * totalCountPCA * sizeof(float));
		CudaDeviceVariable d_eigenVecs((size_t)totalCountPCA * totalCountPCA * sizeof(float)); //check size
		CudaDeviceVariable d_eigenVals(totalCountPCA * sizeof(float)); //check size
		CudaDeviceVariable d_info(sizeof(int));

		int64_t h_meig = 0;
		size_t workspaceInBytesOnDevice = 0;
		size_t workspaceInBytesOnHost = 0;
		float something = 0;

		if (mpi_part == 0)
		{
			cusolverSafeCall(cusolverDnCreate(&cusolverH));
			cusolverSafeCall(cusolverDnCreateParams(&params));
			cusolverSafeCall(cusolverDnXsyevdx_bufferSize(
				cusolverH,
				params,
				jobz,
				range,
				uplo,
				totalCountPCA,
				CUDA_R_32F,
				(float*)d_covVarMat.GetDevicePtr(),
				totalCountPCA,
				&something,
				&something,
				totalCountPCA - aConfig.NumberOfEigenVectors + 1,
				totalCountPCA,
				&h_meig,
				CUDA_R_32F,
				(float*)d_eigenVals.GetDevicePtr(),
				CUDA_R_32F,
				&workspaceInBytesOnDevice,
				&workspaceInBytesOnHost));
		}

		CudaDeviceVariable solverBuffer(workspaceInBytesOnDevice + 1); //can be 0...
		unsigned char* hostBuffer = new unsigned char[workspaceInBytesOnHost + 1]; //can be 0...

		int bufferSize = 0;
		nppSafeCall(nppsSumGetBufferSize_32f(volSize* volSize* volSize, &bufferSize));
		vector<CudaDeviceVariable> d_buffer;
		vector<CudaDeviceVariable> d_sum;

		for (size_t batch = 0; batch < batchSize; batch++)
		{
			d_buffer.push_back(std::move(CudaDeviceVariable((size_t)bufferSize)));
			d_sum.push_back(std::move(CudaDeviceVariable(sizeof(float))));
		}

		float* sumOfParticles = new float[(size_t)volSize * volSize * volSize];
		float* MPIBuffer = new float[(size_t)volSize * volSize * volSize];

		float* CCMatrix = new float[(size_t)totalCountPCA * totalCountPCA];
		memset(CCMatrix, 0, (size_t)totalCountPCA* totalCountPCA * sizeof(float));
		float* CCMatrixMPI = new float[(size_t)totalCountPCA * totalCountPCA];
		memset(CCMatrixMPI, 0, (size_t)totalCountPCA* totalCountPCA * sizeof(float));

		float* eigenValues = new float[aConfig.NumberOfEigenVectors];
		float* eigenValuesSorted = new float[aConfig.NumberOfEigenVectors];

		float* eigenImages = new float[aConfig.NumberOfEigenVectors * volSize * volSize * volSize];
		memset(eigenImages, 0, (size_t)aConfig.NumberOfEigenVectors * volSize * volSize * volSize * sizeof(float));
		float* eigenImagesMPI = new float[aConfig.NumberOfEigenVectors * volSize * volSize * volSize];
		memset(eigenImagesMPI, 0, (size_t)aConfig.NumberOfEigenVectors * volSize * volSize * volSize * sizeof(float));

		float* weightMatrix = new float[aConfig.NumberOfEigenVectors * totalCount];
		memset(weightMatrix, 0, (size_t)aConfig.NumberOfEigenVectors* totalCount * sizeof(float));
		float* weightMatrixMPI = new float[aConfig.NumberOfEigenVectors * totalCount];
		memset(weightMatrixMPI, 0, (size_t)aConfig.NumberOfEigenVectors* totalCount * sizeof(float));

		int* classes = new int[totalCount];

		if (wedges[unique_wedge_ids[0]]->GetFileHeader().DimX != volSize)
		{
			throw std::invalid_argument("Wedge volume dimension does not fit mask volume dimensions.");
		}

		if (aConfig.UseFilterVolume)
		{
			if (filter.GetFileHeader().DimX != volSize)
			{
				throw std::invalid_argument("Filter volume dimension does not fit mask volume dimensions.");
			}
			for (size_t batch = 0; batch < batchSize; batch++)
			{
				d_filter[batch].CopyHostToDevice(filter.GetData());
			}
			
		}

		{
			//check particle size
			stringstream ss;
			ss << aConfig.Particles;
			ss << motl.GetAt(0).GetIndexCoding(aConfig.NamingConv) << ".em";

			EmFile part(ss.str());
			part.OpenAndRead();
			if (part.GetFileHeader().DimX != volSize)
			{
				throw std::invalid_argument("Particle volume dimension does not fit mask volume dimensions.");
			}
		}
	
		if (mask.GetFileHeader().DataType != EMDATATYPE_FLOAT)
		{
			throw std::invalid_argument("Mask is not given in float datatype.");
		}

		//compute indices in mask and masked voxel count
		for (int i = 0; i < volSize * volSize * volSize; i++)
		{
			float value = ((float*)mask.GetData())[i];

			if (value != 0 && value != 1.0f)
			{
				throw std::invalid_argument("Mask is not binary!");
			}

			if (value == 1.0f)
			{
				maskedVoxels++;
				maskIndices.push_back(i);
			}
		}

		vector<Rotator*> rotators;
		vector<Correlator3D*> correlators;
		for (size_t batch = 0; batch < batchSize; batch++)
		{
			rotators.push_back(new Rotator(ctx, streams[batch], volSize));
			correlators.push_back(new Correlator3D(ctx, volSize, &d_filter[batch], aConfig.HighPass, aConfig.LowPass, aConfig.Sigma, aConfig.UseFilterVolume, streams[batch]));
		}
		

		//prepare motive list subset of NumberOfParticlesInPCA particles:
		std::default_random_engine rand(1234); 
		std::vector<int> motiveListIndeces(totalCount);

		if (mpi_part == 0)
		{
			std::iota(motiveListIndeces.begin(), motiveListIndeces.end(), 0);
			std::shuffle(motiveListIndeces.begin(), motiveListIndeces.end(), rand);
		}

#ifdef USE_MPI	
		//share random indeces, just to be sure they are the same on all nodes
		MPI_Bcast(&motiveListIndeces[0], totalCount, MPI_INT, 0, MPI_COMM_WORLD);
#endif

		for (size_t i = 0; i < batchSize; i++)
		{
			SingletonThread::Get(i)->enqueue([ctx] {ctx->SetCurrent(); });
		}


		////////////////////////////////////
		/// Step 2: sum up all particles ///
		////////////////////////////////////
	
		if (mpi_part == 0)
		{
			printf("Step 2: Sum up all particles\n"); fflush(stdout);
		}

		for (size_t batch = 0; batch < batchSize; batch++)
		{
			d_mask[batch].CopyHostToDevice(mask.GetData());
			d_meanParticle[batch].Memset(0);
		}

		ctx->Synchronize();
		if (aConfig.ComputeAverageParticle)
		{
			for (size_t i = startParticlePCA; i < endParticlePCA; i+=batchSize)
			{
				for (size_t batch = 0; batch < batchSize; batch++)
				{
					size_t particle = i + batch;
					if (particle >= endParticlePCA)
					{
						continue;
					}

					motive m = motl.GetAt(motiveListIndeces[particle]);
					stringstream ss;
					ss << aConfig.Particles;
					ss << m.GetIndexCoding(aConfig.NamingConv) << ".em";
					EmFile* part = new EmFile(ss.str());

					float3 shift = make_float3(-m.x_Shift, -m.y_Shift, -m.z_Shift);
					float phi = -m.psi;
					float psi = -m.phi;
					float theta = -m.theta;

					runBatch([&, batch, part, shift, phi, psi, theta] {
						part->OpenAndRead();
						memcpy(h_pagelockedBuffer[batch].GetHostPtr(), part->GetData(), part->GetDataSize());

						d_particle[batch].CopyHostToDeviceAsync(streams[batch], h_pagelockedBuffer[batch].GetHostPtr());
						correlators[batch]->FourierFilter(d_particle[batch]);
						//rotate and shift particle
						rotators[batch]->ShiftRotateTwoStep(shift, phi, psi, theta, d_particle[batch]);
						//apply mask
						nppSafeCall(nppsMul_32f_Ctx((float*)d_mask[batch].GetDevicePtr(), (float*)d_particle[batch].GetDevicePtr(), (float*)d_particleMasked[batch].GetDevicePtr(), volSize* volSize* volSize, streamCtx[batch]));
						//mean free (in masked area)
						nppSafeCall(nppsSum_32f_Ctx((float*)d_particleMasked[batch].GetDevicePtr(), volSize * volSize * volSize, (float*)d_sum[batch].GetDevicePtr(), (unsigned char*)d_buffer[batch].GetDevicePtr(), streamCtx[batch]));
						kernelSubDiv(streams[batch], d_sum[batch], d_particle[batch], volSize * volSize * volSize, (float)maskedVoxels);
						//sum up
						nppSafeCall(nppsAdd_32f_I_Ctx((float*)d_particle[batch].GetDevicePtr(), (float*)d_meanParticle[batch].GetDevicePtr(), volSize * volSize * volSize, streamCtx[batch]));

						//delete particle after queuing some work on GPU...
						delete part;
						});
				}
			}

			//Wait for command queue to finish
			for (size_t batch = 0; batch < batchSize; batch++)
			{
				SingletonThread::Get(batch)->SyncTasks();
			}
			//wait for GPU
			ctx->Synchronize();
			//merge batched results
			for (size_t batch = 1; batch < batchSize; batch++)
			{
				nppSafeCall(nppsAdd_32f_I((float*)d_meanParticle[batch].GetDevicePtr(), (float*)d_meanParticle[0].GetDevicePtr(), volSize * volSize * volSize));
			}
			ctx->Synchronize();

	#ifndef USE_MPI
			//If not in MPI mode, scale the sum directly here, otherwise later after gathering all partial results from nodes
			//Scale to number of particles
			nppSafeCall(nppsDivC_32f_I(totalCountPCA, (float*)d_meanParticle[0].GetDevicePtr(), volSize * volSize * volSize));
	#endif
			d_meanParticle[0].CopyDeviceToHost(sumOfParticles);


	#ifdef USE_MPI	
			//accumulate partial sums over MPI nodes
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, volSize * volSize * volSize, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					d_particle[0].CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppsAdd_32f_I((float*)d_particle[0].GetDevicePtr(), (float*)d_meanParticle[0].GetDevicePtr(), volSize * volSize * volSize));
				}
			}
			else
			{
				MPI_Send(sumOfParticles, volSize * volSize * volSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}

			//share entire sum with nodes
			if (mpi_part == 0)
			{
				//In MPI mode, scale the sum now
				//Scale to number of particles
				nppSafeCall(nppsDivC_32f_I(totalCountPCA, (float*)d_meanParticle[0].GetDevicePtr(), volSize * volSize * volSize));

				d_meanParticle[0].CopyDeviceToHost(sumOfParticles);
				for (size_t batch = 1; batch < batchSize; batch++)
				{
					d_meanParticle[batch].CopyDeviceToDevice(d_meanParticle[0]);
				}
			}
			MPI_Bcast(sumOfParticles, volSize * volSize * volSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

			if (mpi_part != 0)
			{
				for (size_t batch = 0; batch < batchSize; batch++)
				{
					d_meanParticle[batch].CopyHostToDevice(sumOfParticles);
				}
			}
	#endif

			if (aConfig.AverageParticleFile.size() > 1 && mpi_part == 0)
			{
				emwrite(aConfig.AverageParticleFile, sumOfParticles, volSize, volSize, volSize);
			}
		}//if compute particle sum
		else
		{
			if (mpi_part == 0)
			{
				printf("  Restore average particle from file: %s\n", aConfig.AverageParticleFile.c_str()); fflush(stdout);
			}
			EmFile partSum(aConfig.AverageParticleFile);
			partSum.OpenAndRead();
			for (size_t batch = 0; batch < batchSize; batch++)
			{
				d_meanParticle[batch].CopyHostToDevice(partSum.GetData());
			}
		}

		///////////////////////////////////////////
		/// Step 3: compute CC of all particles ///
		///////////////////////////////////////////

		if (mpi_part == 0)
		{
			printf("Step 3: Compute CC of all particles\n"); fflush(stdout);
		}


		float** particleBlock1 = new float* [aConfig.BlockSize];
		float** particleBlock2 = new float* [aConfig.BlockSize];
		for (size_t i = 0; i < aConfig.BlockSize; i++)
		{
			particleBlock1[i] = new float[(size_t)volSize * volSize * volSize];
			particleBlock2[i] = new float[(size_t)volSize * volSize * volSize];
		}

		float* cc_vol = new float[(size_t)volSize * volSize * volSize];
		
		for (size_t batch = 0; batch < batchSize; batch++)
		{
			correlators[batch]->PrepareMask(d_mask[batch], false); //mask must be binary anyway
		}
		
		ctx->Synchronize();
		if (aConfig.ComputeCovVarMatrix)
		{
			//distribute the CC values to compute evenly on compute nodes
			//the matrix is split this way:
			//n = number of particles
			//k = number of rows chosen for first node
			//x = number of compute nodes
			//number of rows (k) for the first node is defined by the ratio:
			//(n*k - (k*(k+1))/2) * ((x-1)/x) = ((n-k)*(n-k-1))/2 * (1/x)
			//which gives:
			//k = (+/- sqrt((2n-1)^2 * x - 4n(n-1)) + (2n - 1)*sqrt(x)) / (2 * sqrt(x))
			//scan iteratively through all nodes:

			vector<int> startEntries(mpi_size, 0);
			vector<int> endEntries(mpi_size, 0);
			vector<int> ccToCompute(mpi_size, 0);

			int totalRows = totalCountPCA;

			startEntries[0] = 0;
			int rowsDone = 0;

			for (int node = 0; node < mpi_size - 1; node++)
			{
				double n = totalRows;
				double x = mpi_size - node;
				double k = (-sqrt((2.0 * n + 1) * (2.0 * n + 1) * x - 4 * n * (n + 1.0)) + (2.0 * n + 1) * sqrt(x)) / (2.0 * sqrt(x));

				int kint = floor(k);

				endEntries[node] = kint + rowsDone;
				startEntries[node + 1] = kint + rowsDone;
				rowsDone += kint;

				ccToCompute[node] = n * kint - (kint * (kint + 1)) / 2;

				totalRows -= kint;
			}

			ccToCompute[mpi_size - 1] = totalRows * totalRows - (totalRows * (totalRows + 1)) / 2;

			endEntries[mpi_size - 1] = totalCountPCA;

			if (mpi_size == 1)
			{
				startEntries[0] = 0;
				endEntries[0] = totalCountPCA;
			}

			if (mpi_part == 0)
			{
				printf("  Distribution on nodes:\n");

				for (int i = 0; i < mpi_size; i++)
				{
					printf("  Node %d (rows %d -> %d): %d CCs\n", i, startEntries[i], endEntries[i], ccToCompute[i]);
				}
			}

			//loop over all particles on MPI node
			for (int pBlock1 = startEntries[mpi_part]; pBlock1 < endEntries[mpi_part]; pBlock1 += aConfig.BlockSize)
			{
				if (mpi_part == 0)
				{
					printf("  processing row %d of %d in matrix\n", pBlock1, endEntries[mpi_part]); fflush(stdout);
				}
				//load particles for block 1 from disk and make them mean free
				for (int i = pBlock1; i < pBlock1 + aConfig.BlockSize && i < endEntries[mpi_part]; i+=batchSize)
				{
					for (int batch = 0; batch < batchSize; batch++)
					{
						int p1 = i + batch;
						if (p1 >= pBlock1 + aConfig.BlockSize || p1 >= endEntries[mpi_part])
						{
							continue;
						}

						motive m = motl.GetAt(motiveListIndeces[p1]);
						stringstream ss;
						ss << aConfig.Particles;
						ss << m.GetIndexCoding(aConfig.NamingConv) << ".em";
						EmFile* part = new EmFile(ss.str());
						float3 shift = make_float3(-m.x_Shift, -m.y_Shift, -m.z_Shift);
						float phi = -m.psi;
						float psi = -m.phi;
						float theta = -m.theta;

						runBatch([&, batch, part, shift, phi, psi, theta, p1, pBlock1] {
							part->OpenAndRead();
							memcpy(h_pagelockedBuffer[batch].GetHostPtr(), part->GetData(), part->GetDataSize());
							d_particle[batch].CopyHostToDeviceAsync(streams[batch], h_pagelockedBuffer[batch].GetHostPtr());
							correlators[batch]->FourierFilter(d_particle[batch]);
							rotators[batch]->ShiftRotateTwoStep(shift, phi, psi, theta, d_particle[batch]);

							//remove mean particle
							nppSafeCall(nppsSub_32f_I_Ctx((float*)d_meanParticle[batch].GetDevicePtr(), (float*)d_particle[batch].GetDevicePtr(), volSize* volSize* volSize, streamCtx[batch]));

							//apply mask
							nppSafeCall(nppsMul_32f_Ctx((float*)d_mask[batch].GetDevicePtr(), (float*)d_particle[batch].GetDevicePtr(), (float*)d_particleMasked[batch].GetDevicePtr(), volSize * volSize * volSize, streamCtx[batch]));

							//mean free (in masked area)
							nppSafeCall(nppsSum_32f_Ctx((float*)d_particleMasked[batch].GetDevicePtr(), volSize * volSize * volSize, (float*)d_sum[batch].GetDevicePtr(), (unsigned char*)d_buffer[batch].GetDevicePtr(), streamCtx[batch]));
							kernelSubDiv(streams[batch], d_sum[batch], d_particle[batch], volSize * volSize * volSize, (float)maskedVoxels);

							delete part;
							d_particle[batch].CopyDeviceToHostAsync(streams[batch], h_pagelockedBuffer[batch].GetHostPtr());
							cudaSafeCall(cuStreamSynchronize(streams[batch]));

							memcpy(particleBlock1[p1 - pBlock1], h_pagelockedBuffer[batch].GetHostPtr(), d_particle[batch].GetSize());

							});
					}
				}
				//Wait for command queue to finish
				for (size_t batch = 0; batch < batchSize; batch++)
				{
					SingletonThread::Get(batch)->SyncTasks();
				}
				//wait for GPU
				ctx->Synchronize();

				//we have to scan the entire line, i.e. all colums = all particles
				for (int pBlock2 = pBlock1; pBlock2 < totalCountPCA; pBlock2 += aConfig.BlockSize)
				{
					//load particles for block 2 from disk and make them mean free
					for (int i = pBlock2; i < pBlock2 + aConfig.BlockSize && i < totalCountPCA; i += batchSize)
					{
						for (int batch = 0; batch < batchSize; batch++)
						{
							int p2 = i + batch;
							if (p2 >= pBlock2 + aConfig.BlockSize || p2 >= totalCountPCA)
							{
								continue;
							}

							motive m = motl.GetAt(motiveListIndeces[p2]);
							stringstream ss;
							ss << aConfig.Particles;
							ss << m.GetIndexCoding(aConfig.NamingConv) << ".em";
							EmFile* part = new EmFile(ss.str());
							float3 shift = make_float3(-m.x_Shift, -m.y_Shift, -m.z_Shift);
							float phi = -m.psi;
							float psi = -m.phi;
							float theta = -m.theta;

							runBatch([&, batch, part, shift, phi, psi, theta, p2, pBlock2] {
								part->OpenAndRead();
								memcpy(h_pagelockedBuffer[batch].GetHostPtr(), part->GetData(), part->GetDataSize());
								d_particle[batch].CopyHostToDeviceAsync(streams[batch], h_pagelockedBuffer[batch].GetHostPtr());
								correlators[batch]->FourierFilter(d_particle[batch]);

								//rotate and shift particle
								rotators[batch]->ShiftRotateTwoStep(shift, phi, psi, theta, d_particle[batch]);

								//remove mean particle
								nppSafeCall(nppsSub_32f_I_Ctx((float*)d_meanParticle[batch].GetDevicePtr(), (float*)d_particle[batch].GetDevicePtr(), volSize* volSize* volSize, streamCtx[batch]));

								//apply mask
								nppSafeCall(nppsMul_32f_Ctx((float*)d_mask[batch].GetDevicePtr(), (float*)d_particle[batch].GetDevicePtr(), (float*)d_particleMasked[batch].GetDevicePtr(), volSize * volSize * volSize, streamCtx[batch]));

								//mean free (in masked area)
								nppSafeCall(nppsSum_32f_Ctx((float*)d_particleMasked[batch].GetDevicePtr(), volSize * volSize * volSize, (float*)d_sum[batch].GetDevicePtr(), (unsigned char*)d_buffer[batch].GetDevicePtr(), streamCtx[batch]));
								kernelSubDiv(streams[batch], d_sum[batch], d_particle[batch], volSize * volSize * volSize, (float)maskedVoxels);

								delete part;
								d_particle[batch].CopyDeviceToHostAsync(streams[batch], h_pagelockedBuffer[batch].GetHostPtr());
								cudaSafeCall(cuStreamSynchronize(streams[batch]));

								memcpy(particleBlock2[p2 - pBlock2], h_pagelockedBuffer[batch].GetHostPtr(), d_particle[batch].GetSize());
								});
						}
					}
					//Wait for command queue to finish
					for (size_t batch = 0; batch < batchSize; batch++)
					{
						SingletonThread::Get(batch)->SyncTasks();
					}
					//wait for GPU
					ctx->Synchronize();


					for (int p1 = pBlock1; p1 < pBlock1 + aConfig.BlockSize && p1 < endEntries[mpi_part]; p1++)
					{
						motive m1 = motl.GetAt(motiveListIndeces[p1]);
						int wedgeIdx1 = 0;
						if (!aConfig.SingleWedge)
						{
							wedgeIdx1 = (int)m1.wedgeIdx;
						}

						for (size_t batch = 0; batch < batchSize; batch++)
						{
							d_wedge1[batch].CopyHostToDevice((float*)wedges[wedgeIdx1]->GetData());
							rotators[batch]->Rotate(-m1.psi, -m1.phi, -m1.theta, d_wedge1[batch]);
							d_particle[batch].CopyHostToDevice(particleBlock1[p1 - pBlock1]);
						}

						for (int i = pBlock2; i < pBlock2 + aConfig.BlockSize && i < totalCountPCA; i+=batchSize)
						{
							for (int batch = 0; batch < batchSize; batch++)
							{
								int p2 = i + batch;
								if (p2 >= pBlock2 + aConfig.BlockSize || p2 >= totalCountPCA)
								{
									continue;
								}
								if (p2 > p1) //only upper diagonal part of matrix needed
								{
									//compute CC
									motive m2 = motl.GetAt(motiveListIndeces[p2]);
									float phi = -m2.psi;
									float psi = -m2.phi;
									float theta = -m2.theta;

									int wedgeIdx2 = 0;
									if (!aConfig.SingleWedge)
									{
										wedgeIdx2 = (int)m2.wedgeIdx;
									}
									

									runBatch([&, batch, wedgeIdx2, phi, psi, theta, p1, p2, pBlock2] {
										//GetCCFast syncs with host thread due to copy to host, i.e. here we are for sure in sync with device
										memcpy(h_pagelockedBuffer[batch].GetHostPtr(), wedges[wedgeIdx2]->GetData(), d_wedge2[batch].GetSize());
										d_wedge2[batch].CopyHostToDeviceAsync(streams[batch], h_pagelockedBuffer[batch].GetHostPtr());

										rotators[batch]->Rotate(phi, psi, theta, d_wedge2[batch]);

										nppSafeCall(nppsMul_32f_Ctx((float*)d_wedge1[batch].GetDevicePtr(), (float*)d_wedge2[batch].GetDevicePtr(), (float*)d_wedgeMerge[batch].GetDevicePtr(), volSize* volSize* volSize, streamCtx[batch]));

										//make sure that the previous copy to device has finished:
										cudaSafeCall(cuStreamSynchronize(streams[batch]));
										memcpy(h_pagelockedBuffer[batch].GetHostPtr(), particleBlock2[p2 - pBlock2], d_particleRef[batch].GetSize());
										d_particleRef[batch].CopyHostToDeviceAsync(streams[batch], h_pagelockedBuffer[batch].GetHostPtr());

										correlators[batch]->GettCCFast(d_mask[batch], d_particle[batch], d_particleRef[batch], d_wedgeMerge[batch], &CCMatrix[p1 + p2 * totalCountPCA], aConfig.NormalizeAmplitudes);
										});

								}
							}
						}
						//Wait for command queue to finish
						for (size_t batch = 0; batch < batchSize; batch++)
						{
							SingletonThread::Get(batch)->SyncTasks();
						}
					}
				}
			}
			ctx->Synchronize();

	#ifdef USE_MPI	
			//accumulate partial cc Matrix from MPI nodes
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(CCMatrixMPI, totalCountPCA * totalCountPCA, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

					for (int i = 0; i < totalCountPCA * totalCountPCA; i++)
					{
						CCMatrix[i] += CCMatrixMPI[i];
					}
				}

				//Fill diagonal
				for (int i = 0; i < totalCountPCA; i++)
				{
					CCMatrix[i + totalCountPCA * i] = 1.0f;
				}			
			}
			else
			{
				MPI_Send(CCMatrix, totalCountPCA * totalCountPCA, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}

	#endif
	#ifndef USE_MPI
			//Fill diagonal
			for (int i = 0; i < totalCountPCA; i++)
			{
				CCMatrix[i + totalCountPCA * i] = 1.0f;
			}
	#endif

			if (mpi_part == 0)
			{
				emwrite(aConfig.CovVarMatrixFilename, CCMatrix, totalCountPCA, totalCountPCA);
			}
		} //if compute ccMatrix
		else
		{
			if (mpi_part == 0)
			{
				printf("  Restore matrix from file: %s\n", aConfig.CovVarMatrixFilename.c_str()); fflush(stdout);
			}
			EmFile ccMat(aConfig.CovVarMatrixFilename);
			ccMat.OpenAndRead();

			if (ccMat.GetFileHeader().DimX != ccMat.GetFileHeader().DimY || 
				ccMat.GetFileHeader().DimX != totalCountPCA || 
				ccMat.GetFileHeader().DimZ != 1)
			{
				throw std::invalid_argument("Provided EM file dimensions do not fit dimensions of the covariance matrix");
			}

			for (int i = 0; i < totalCountPCA * totalCountPCA; i++)
			{
				CCMatrix[i] = ((float*)ccMat.GetData())[i];
			}
		}

		/////////////////////////////////////////////////////////////
		/// Step 4: compute Eigen values and vectors of cc matrix ///
		/////////////////////////////////////////////////////////////

		if (mpi_part == 0)
		{
			printf("Step 4: Compute Eigen values and vectors of CC matrix\n"); fflush(stdout);
		}

		d_covVarMat.CopyHostToDevice(CCMatrix);


		if (mpi_part == 0)
		{
			cusolverSafeCall(cusolverDnXsyevdx(
				cusolverH,
				params,
				jobz,
				range,
				uplo,
				totalCountPCA,
				CUDA_R_32F,
				(float*)d_covVarMat.GetDevicePtr(),
				totalCountPCA,
				&something,
				&something,
				totalCountPCA - aConfig.NumberOfEigenVectors + 1,
				totalCountPCA,
				&h_meig,
				CUDA_R_32F,
				(float*)d_eigenVals.GetDevicePtr(),
				CUDA_R_32F,
				(void*)solverBuffer.GetDevicePtr(),
				workspaceInBytesOnDevice,
				hostBuffer,
				workspaceInBytesOnHost,
				(int*)d_info.GetDevicePtr()));

			ctx->Synchronize();


			int info;
			d_info.CopyDeviceToHost(&info);

			if (info < 0)
			{
				stringstream ss;
				ss << "Something went wrong during computation of Eigen vectors. Info returned: " << info;
				throw std::invalid_argument(ss.str());
			}
			if (info > 0)
			{
				stringstream ss;
				ss << "Something went wrong during computation of Eigen vectors. Info returned: " << info;
				//throw std::invalid_argument(ss.str());
			}

			d_covVarMat.CopyDeviceToHost(CCMatrix);
			if (aConfig.EigenVectors.size() > 1)
			{
				emwrite(aConfig.EigenVectors, CCMatrix, totalCountPCA, totalCountPCA);
			}

			d_eigenVals.CopyDeviceToHost(eigenValues, aConfig.NumberOfEigenVectors * sizeof(float));

			printf("  Found Eigenvalues:\n");
			for (int i = aConfig.NumberOfEigenVectors-1; i >= 0; i--)
			{
				printf("  %f\n", eigenValues[i]);
				//save eigenvalues in correct order (large to small)
				eigenValuesSorted[aConfig.NumberOfEigenVectors - 1 - i] = eigenValues[i];
			}
			printf("\n");

			if (aConfig.EigenValues.size() > 1)
			{
				emwrite(aConfig.EigenValues, eigenValuesSorted, aConfig.NumberOfEigenVectors, 1);
			}
		}

	#ifdef USE_MPI	
		//share eigen vectors/values with nodes	
		MPI_Bcast(CCMatrix, totalCountPCA* totalCountPCA, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(eigenValuesSorted, aConfig.NumberOfEigenVectors, MPI_FLOAT, 0, MPI_COMM_WORLD);

		if (mpi_part > 0)
		{
			d_covVarMat.CopyHostToDevice(CCMatrix);
		}
	#endif



		////////////////////////////////////
		/// Step 5: compute Eigen images ///
		////////////////////////////////////

		if (mpi_part == 0)
		{
			printf("Step 5: Compute Eigen images\n"); fflush(stdout);
		}

		d_eigenImages.Memset(0);
		ctx->Synchronize();

		//now all particles of the motive list:
		for (size_t i = startParticle; i < endParticle; i+=batchSize)
		{
			for (size_t batch = 0; batch < batchSize; batch++)
			{
				size_t particle = i + batch;
				if (particle >= endParticle)
				{
					continue;
				}
				motive m = motl.GetAt(particle);
				float3 shift = make_float3(-m.x_Shift, -m.y_Shift, -m.z_Shift);
				float phi = -m.psi;
				float psi = -m.phi;
				float theta = -m.theta;
				stringstream ss;
				ss << aConfig.Particles;
				ss << m.GetIndexCoding(aConfig.NamingConv) << ".em";
				EmFile* part = new EmFile(ss.str());

				runBatch([&, batch, particle, part, shift, phi, psi, theta] {

						part->OpenAndRead();

						cudaSafeCall(cuStreamSynchronize(streams[batch]));
						memcpy(h_pagelockedBuffer[batch].GetHostPtr(), part->GetData(), part->GetDataSize());
						d_particle[batch].CopyHostToDeviceAsync(streams[batch], h_pagelockedBuffer[batch].GetHostPtr());
						correlators[batch]->FourierFilter(d_particle[batch]);

						//rotate and shift particle
						rotators[batch]->ShiftRotateTwoStep(shift, phi, psi, theta, d_particle[batch]);

						//remove mean particle
						nppSafeCall(nppsSub_32f_I_Ctx((float*)d_meanParticle[batch].GetDevicePtr(), (float*)d_particle[batch].GetDevicePtr(), volSize* volSize* volSize, streamCtx[batch]));

						//apply mask (directly to particle)
						nppSafeCall(nppsMul_32f_I_Ctx((float*)d_mask[batch].GetDevicePtr(), (float*)d_particle[batch].GetDevicePtr(), volSize * volSize * volSize, streamCtx[batch]));

						//mean free
						nppSafeCall(nppsSum_32f_Ctx((float*)d_particle[batch].GetDevicePtr(), volSize * volSize * volSize, (float*)d_sum[batch].GetDevicePtr(), (unsigned char*)d_buffer[batch].GetDevicePtr(), streamCtx[batch]));
						kernelSubDiv(streams[batch], d_sum[batch], d_particle[batch], volSize * volSize * volSize, maskedVoxels);

						//apply mask again to have non masked area =0
						nppSafeCall(nppsMul_32f_I_Ctx((float*)d_mask[batch].GetDevicePtr(), (float*)d_particle[batch].GetDevicePtr(), volSize * volSize * volSize, streamCtx[batch]));

						//uses atomic add for eigenImage addition, it is safe to use the same array for all batches
						kernelEigenImages(streams[batch], volSize * volSize * volSize, aConfig.NumberOfEigenVectors, particle, totalCountPCA, d_covVarMat, d_particle[batch], d_eigenImages);
						delete part;
					});
			}
		}
		//Wait for command queue to finish
		for (size_t batch = 0; batch < batchSize; batch++)
		{
			SingletonThread::Get(batch)->SyncTasks();
		}
		//wait for GPU
		ctx->Synchronize();

		d_eigenImages.CopyDeviceToHost(eigenImages);

	#ifdef USE_MPI	
		//accumulate partial eigenImages from MPI nodes
		if (mpi_part == 0)
		{
			for (int mpi = 1; mpi < mpi_size; mpi++)
			{
				MPI_Recv(eigenImagesMPI, aConfig.NumberOfEigenVectors* volSize* volSize* volSize, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				for (int i = 0; i < aConfig.NumberOfEigenVectors * volSize * volSize * volSize; i++)
				{
					eigenImages[i] += eigenImagesMPI[i];
				}
			}
		}
		else
		{
			MPI_Send(eigenImages, aConfig.NumberOfEigenVectors* volSize* volSize* volSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		}
		//share eigen images with nodes	
		MPI_Bcast(eigenImages, aConfig.NumberOfEigenVectors* volSize* volSize* volSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
		d_eigenImages.CopyHostToDevice(eigenImages);
	#endif

		if (aConfig.EigenImages.size() > 1 && mpi_part == 0)
		{
			for (int i = 0; i < aConfig.NumberOfEigenVectors; i++)
			{
				stringstream ss;
				ss << aConfig.EigenImages << i << ".em";

				emwrite(ss.str(), eigenImages + ((size_t)i * volSize * volSize * volSize), volSize, volSize, volSize);
			}
		}



		/////////////////////////////////////
		/// Step 6: compute weight matrix ///
		/////////////////////////////////////

		if (mpi_part == 0)
		{
			printf("Step 6: Compute weight matrix\n"); fflush(stdout);
		}

		ctx->Synchronize();
		for (size_t i = startParticle; i < endParticle; i+=batchSize)
		{
			for (size_t batch = 0; batch < batchSize; batch++)
			{
				size_t particle = i + batch;
				if (particle >= endParticle)
				{
					continue;
				}

				motive m = motl.GetAt(particle);
				float3 shift = make_float3(-m.x_Shift, -m.y_Shift, -m.z_Shift);
				float phi = -m.psi;
				float psi = -m.phi;
				float theta = -m.theta;
				stringstream ss;
				ss << aConfig.Particles;
				ss << m.GetIndexCoding(aConfig.NamingConv) << ".em";
				EmFile* part = new EmFile(ss.str());

				runBatch([&, batch, particle, part, shift, phi, psi, theta] {

					part->OpenAndRead();
					//host is synced later in the loop: here the device is certainly not accessing h_pagelockedBuffer[batch]
					memcpy(h_pagelockedBuffer[batch].GetHostPtr(), part->GetData(), part->GetDataSize());
					d_particle[batch].CopyHostToDeviceAsync(streams[batch], h_pagelockedBuffer[batch].GetHostPtr());
					correlators[batch]->FourierFilter(d_particle[batch]);

					//rotate and shift particle
					rotators[batch]->ShiftRotateTwoStep(shift, phi, psi, theta, d_particle[batch]);

					//remove mean particle
					nppSafeCall(nppsSub_32f_I_Ctx((float*)d_meanParticle[batch].GetDevicePtr(), (float*)d_particle[batch].GetDevicePtr(), volSize* volSize* volSize, streamCtx[batch]));

					//apply mask (directly to particle)
					nppSafeCall(nppsMul_32f_I_Ctx((float*)d_mask[batch].GetDevicePtr(), (float*)d_particle[batch].GetDevicePtr(), volSize* volSize* volSize, streamCtx[batch]));

					//mean free
					nppSafeCall(nppsSum_32f_Ctx((float*)d_particle[batch].GetDevicePtr(), volSize* volSize* volSize, (float*)d_sum[batch].GetDevicePtr(), (unsigned char*)d_buffer[batch].GetDevicePtr(), streamCtx[batch]));

					//apply mask again to have non masked area =0
					nppSafeCall(nppsMul_32f_I_Ctx((float*)d_mask[batch].GetDevicePtr(), (float*)d_particle[batch].GetDevicePtr(), volSize* volSize* volSize, streamCtx[batch]));

					delete part;
					for (int eigenImage = 0; eigenImage < aConfig.NumberOfEigenVectors; eigenImage++)
					{
						nppSafeCall(nppsMul_32f_Ctx(((float*)d_eigenImages.GetDevicePtr()) + ((size_t)eigenImage * volSize * volSize * volSize), (float*)d_particle[batch].GetDevicePtr(), (float*)d_particleRef[batch].GetDevicePtr(), volSize * volSize * volSize, streamCtx[batch]));
						nppSafeCall(nppsSum_32f_Ctx((float*)d_particleRef[batch].GetDevicePtr(), volSize * volSize * volSize, (float*)d_sum[batch].GetDevicePtr(), (Npp8u*)d_buffer[batch].GetDevicePtr(), streamCtx[batch]));

						d_sum[batch].CopyDeviceToHostAsync(streams[batch], h_pagelockedBuffer[batch].GetHostPtr());
						cudaSafeCall(cuStreamSynchronize(streams[batch]));

						float eivo = *((float*)h_pagelockedBuffer[batch].GetHostPtr());

						//scale to number of voxels to keep values in a reasonable range
						eivo /= volSize * volSize * volSize;
						weightMatrix[eigenImage + particle * aConfig.NumberOfEigenVectors] = eivo * eigenValuesSorted[eigenImage] * eigenValuesSorted[eigenImage];
					}
				});
			}
		}
		//Wait for command queue to finish
		for (size_t batch = 0; batch < batchSize; batch++)
		{
			SingletonThread::Get(batch)->SyncTasks();
		}
		//wait for GPU
		ctx->Synchronize();

	#ifdef USE_MPI	
		//accumulate partial eigenImages from MPI nodes
		if (mpi_part == 0)
		{
			for (int mpi = 1; mpi < mpi_size; mpi++)
			{
				MPI_Recv(weightMatrixMPI, aConfig.NumberOfEigenVectors * totalCount, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				for (int i = 0; i < aConfig.NumberOfEigenVectors * totalCount; i++)
				{
					weightMatrix[i] += weightMatrixMPI[i];
				}
			}
		}
		else
		{
			MPI_Send(weightMatrix, aConfig.NumberOfEigenVectors * totalCount, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		}
		//share eigen images with nodes	
		MPI_Bcast(weightMatrix, aConfig.NumberOfEigenVectors * totalCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
	#endif
	
		if (mpi_part == 0)
		{
			emwrite(aConfig.WeightMatrixFile, weightMatrix, aConfig.NumberOfEigenVectors, totalCount);
		}




		//////////////////////
		/// Step 7: KMeans ///
		//////////////////////

		if (mpi_part == 0)
		{
			printf("Step 7: Classify with KMeans\n"); fflush(stdout);
		}


		if (mpi_part == 0)
		{
			KMeans kmeans(aConfig.NumberOfEigenVectors, totalCount, aConfig.NumberOfClasses, weightMatrix);

			int* result = kmeans.GetClasses();

			printf("  Centroids of classes\n");
			printf("  ");
			for (int c = 0; c < aConfig.NumberOfClasses; c++)
			{
				printf("Class %-6d ", c+1);
			}
			printf("\n  ");
			for (int ev = 0; ev < aConfig.NumberOfEigenVectors; ev++)
			{
				for (int c = 0; c < aConfig.NumberOfClasses; c++)
				{
					printf("%12.2f ", kmeans.GetCentroids()[c][ev]);
				}
				printf("\n  ");
			}
			printf("\n");

			vector<int> classCount(aConfig.NumberOfClasses, 0);
		
			for (int i = 0; i < totalCount; i++)
			{
				classCount[result[i]]++;
				classes[i] = result[i] + 1; //switch from index 0 to index 1

				motive m = motl.GetAt(i);
				m.classNo = classes[i];
				motl.SetAt(i, m);
			}
			motl.OpenAndWrite();

			for (int c = 0; c < aConfig.NumberOfClasses; c++)
			{
				printf("  Class %d: %d particles\n", c+1, classCount[c]);
			}
		}

	#ifdef USE_MPI
		//share classes with nodes	
		MPI_Bcast(classes, totalCount, MPI_INT, 0, MPI_COMM_WORLD);
	#endif

		///////////////////////////////////////////////
		/// Step 8: Add particles for found classes ///
		///////////////////////////////////////////////

		if (mpi_part == 0)
		{
			printf("Step 8: Add particles for found classes\n"); fflush(stdout);
		}

		for (int c = 1; c <= aConfig.NumberOfClasses; c++)
		{
			for (size_t batch = 0; batch < batchSize; batch++)
			{
				d_particleRef[batch].Memset(0);
				d_wedgeMerge[batch].Memset(0);
			}
			ctx->Synchronize();

			for (int i = startParticle; i < endParticle; i+=batchSize)
			{
				for (size_t batch = 0; batch < batchSize; batch++)
				{
					size_t particle = i + batch;
					if (particle >= endParticle)
					{
						continue;
					}

					if (classes[particle] == c)
					{
						motive m = motl.GetAt(particle);
						float3 shift = make_float3(-m.x_Shift, -m.y_Shift, -m.z_Shift);
						float phi = -m.psi;
						float psi = -m.phi;
						float theta = -m.theta;
						int wedgeIdx = 0;
						if (!aConfig.SingleWedge)
						{
							wedgeIdx = (int)m.wedgeIdx;
						}
						stringstream ss;
						ss << aConfig.Particles;
						ss << m.GetIndexCoding(aConfig.NamingConv) << ".em";
						EmFile* part = new EmFile(ss.str());

						runBatch([&, batch, part, shift, phi, psi, theta, wedgeIdx] {

							part->OpenAndRead();

							cudaSafeCall(cuStreamSynchronize(streams[batch]));
							memcpy(h_pagelockedBuffer[batch].GetHostPtr(), part->GetData(), part->GetDataSize());
							d_particle[batch].CopyHostToDeviceAsync(streams[batch], h_pagelockedBuffer[batch].GetHostPtr());

							cudaSafeCall(cuStreamSynchronize(streams[batch]));
							memcpy(h_pagelockedBuffer[batch].GetHostPtr(), wedges[wedgeIdx]->GetData(), d_wedge1[batch].GetSize());
							d_wedge1[batch].CopyHostToDeviceAsync(streams[batch], h_pagelockedBuffer[batch].GetHostPtr());

							correlators[batch]->MultiplyWedge(d_particle[batch], d_wedge1[batch]);

							rotators[batch]->ShiftRotateTwoStep(shift, phi, psi, theta, d_particle[batch]);

							nppSafeCall(nppsAdd_32f_I_Ctx((float*)d_particle[batch].GetDevicePtr(), (float*)d_particleRef[batch].GetDevicePtr(), volSize * volSize * volSize, streamCtx[batch]));

							rotators[batch]->Rotate(phi, psi, theta, d_wedge1[batch]);

							nppSafeCall(nppsAdd_32f_I_Ctx((float*)d_wedge1[batch].GetDevicePtr(), (float*)d_wedgeMerge[batch].GetDevicePtr(), volSize * volSize * volSize, streamCtx[batch]));

							delete part;
						});
					}
				}
			}
			//Wait for command queue to finish
			for (size_t batch = 0; batch < batchSize; batch++)
			{
				SingletonThread::Get(batch)->SyncTasks();
			}
			//wait for GPU
			ctx->Synchronize();
			for (size_t batch = 1; batch < batchSize; batch++)
			{
				nppSafeCall(nppsAdd_32f_I((float*)d_particleRef[batch].GetDevicePtr(), (float*)d_particleRef[0].GetDevicePtr(), volSize * volSize * volSize));
				nppSafeCall(nppsAdd_32f_I((float*)d_wedgeMerge[batch].GetDevicePtr(), (float*)d_wedgeMerge[0].GetDevicePtr(), volSize* volSize* volSize));
			}
			ctx->Synchronize();
	#ifdef USE_MPI	
			//accumulate partial sums over MPI nodes
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, volSize * volSize * volSize, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					d_particle[0].CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppsAdd_32f_I_Ctx((float*)d_particle[0].GetDevicePtr(), (float*)d_particleRef[0].GetDevicePtr(), volSize * volSize * volSize, streamCtx[0]));

					MPI_Recv(MPIBuffer, volSize * volSize * volSize, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					d_wedge1[0].CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppsAdd_32f_I_Ctx((float*)d_wedge1[0].GetDevicePtr(), (float*)d_wedgeMerge[0].GetDevicePtr(), volSize * volSize * volSize, streamCtx[0]));
				}
			}
			else
			{
				d_particleRef[0].CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, volSize * volSize * volSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
				d_wedgeMerge[0].CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, volSize * volSize * volSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
	#endif

			if (mpi_part == 0)
			{
				correlators[0]->NormalizeWedge(d_particleRef[0], d_wedgeMerge[0]);
				d_particleRef[0].CopyDeviceToHost(sumOfParticles);


				stringstream ss;
				ss << aConfig.ClassFileName;
				ss << c << ".em";

				emwrite(ss.str(), sumOfParticles, volSize, volSize, volSize);
			}
		}

		//clean up
		if (mpi_part == 0)
		{
			cusolverSafeCall(cusolverDnDestroyParams(params));
			cusolverSafeCall(cusolverDnDestroy(cusolverH));
		}

		delete[] wedge_ids;
		delete[] hostBuffer;

		delete[] sumOfParticles;
		delete[] MPIBuffer;

		delete[] CCMatrix;
		delete[] CCMatrixMPI;

		delete[] eigenValues;
		delete[] eigenValuesSorted;

		delete[] eigenImages;
		delete[] eigenImagesMPI;

		delete[] weightMatrix;
		delete[] weightMatrixMPI;

		delete[] classes;


		for (size_t i = 0; i < aConfig.BlockSize; i++)
		{
			delete[] particleBlock1[i];
			delete[] particleBlock2[i];
		}

		delete[] particleBlock1;
		delete[] particleBlock2;

		delete[] cc_vol;

	}
	catch (const std::exception& e)
	{
		cout << e.what() << endl << endl;
		WaitForInput(-1);
	}


	CudaContext::DestroyInstance(ctx);

#ifdef USE_MPI
	MPI_Finalize();
#endif

}
