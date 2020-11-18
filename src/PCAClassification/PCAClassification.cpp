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

// Self
#include "Kernels/kernels.cu.h"
#include "PCAKernels.h"
#include "utils/Config.h"
#include "KMeans.h"

using namespace std;
using namespace Cuda;

#define LOG(a, ...) (MKLog::Get()->Log(LL_INFO, a, __VA_ARGS__))

#define round(x) (x >= 0 ? (int)(x + 0.5) : (int)(x - 0.5))

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
		Configuration::Config aConfig = Configuration::Config::GetConfig("pca.cfg", argc, argv, mpi_part, NULL);
		ctx = CudaContext::CreateInstance(aConfig.CudaDeviceIDs[mpi_part], CU_CTX_SCHED_SPIN);

		printf("\nLoad motivelist..."); fflush(stdout);
		MotiveList motl(aConfig.MotiveList, 1.0f, aConfig.ScaleMotivelistShift);
		printf("Done\n"); fflush(stdout);
	
		int totalCount = motl.GetParticleCount();
		int partCount = motl.GetParticleCount() / mpi_size;
		int partCountArray = partCount;
		int lastPartCount = totalCount - (partCount * (mpi_size - 1));
		int startParticle = mpi_part * partCount;

		//adjust last part to fit really all particles (rounding errors...)
		if (mpi_part == mpi_size - 1)
		{
			partCount = lastPartCount;
		}

		int endParticle = startParticle + partCount;


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
		ComputeEigenImagesKernel kernelEigenImages(mod);

		CudaDeviceVariable d_particle(volSize* volSize* volSize * sizeof(float));
		CudaDeviceVariable d_particleRef(volSize* volSize* volSize * sizeof(float));
		CudaDeviceVariable d_mask(volSize* volSize* volSize * sizeof(float));
		CudaDeviceVariable d_filter(volSize* volSize* volSize * sizeof(float));
		CudaDeviceVariable d_meanParticle(volSize* volSize* volSize * sizeof(float));

		CudaDeviceVariable d_wedge1((size_t)volSize* volSize* volSize * sizeof(float));
		CudaDeviceVariable d_wedge2((size_t)volSize* volSize* volSize * sizeof(float));
		CudaDeviceVariable d_wedgeMerge((size_t)volSize* volSize* volSize * sizeof(float));
		CudaDeviceVariable d_CC((size_t)volSize* volSize* volSize * sizeof(float));

		cusolverDnHandle_t cusolverH = NULL;
		cusolverDnParams_t params = NULL;
		cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
		cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
		cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;

		CudaDeviceVariable d_eigenImages((size_t)aConfig.NumberOfEigenVectors * volSize * volSize * volSize * sizeof(float));
		CudaDeviceVariable d_covVarMat((size_t)totalCount * totalCount * sizeof(float));
		CudaDeviceVariable d_eigenVecs((size_t)totalCount * totalCount * sizeof(float)); //check size
		CudaDeviceVariable d_eigenVals(totalCount * sizeof(float)); //check size
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
				totalCount,
				CUDA_R_32F,
				(float*)d_covVarMat.GetDevicePtr(),
				totalCount,
				&something,
				&something,
				totalCount - aConfig.NumberOfEigenVectors + 1,
				totalCount,
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
		Cuda::CudaDeviceVariable d_buffer(bufferSize);
		Cuda::CudaDeviceVariable d_sum(sizeof(float));

		float* sumOfParticles = new float[(size_t)volSize * volSize * volSize];
		float* MPIBuffer = new float[(size_t)volSize * volSize * volSize];

		float* CCMatrix = new float[(size_t)totalCount * totalCount];
		memset(CCMatrix, 0, (size_t)totalCount* totalCount * sizeof(float));
		float* CCMatrixMPI = new float[(size_t)totalCount * totalCount];
		memset(CCMatrixMPI, 0, (size_t)totalCount* totalCount * sizeof(float));

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
			d_filter.CopyHostToDevice(filter.GetData());
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

		Rotator rotator(ctx, volSize);
		Correlator3D correlator(ctx, volSize, d_filter, aConfig.HighPass, aConfig.LowPass, aConfig.Sigma, aConfig.UseFilterVolume);

		////////////////////////////////////
		/// Step 2: sum up all particles ///
		////////////////////////////////////
	
		if (mpi_part == 0)
		{
			printf("Step 2: Sum up all particles\n"); fflush(stdout);
		}

		d_mask.CopyHostToDevice(mask.GetData());
		d_meanParticle.Memset(0);

		if (aConfig.ComputeAverageParticle)
		{

			for (size_t particle = startParticle; particle < endParticle; particle++)
			{
				motive m = motl.GetAt(particle);
				stringstream ss;
				ss << aConfig.Particles;
				ss << m.GetIndexCoding(aConfig.NamingConv) << ".em";
				EmFile part(ss.str());
				part.OpenAndRead();

				d_particle.CopyHostToDevice(part.GetData());
				correlator.FourierFilter(d_particle);

				//rotate and shift particle
				float3 shift = make_float3(-m.x_Shift, -m.y_Shift, -m.z_Shift);
				rotator.ShiftRotateTwoStep(shift, -m.psi, -m.phi, -m.theta, d_particle);

				//apply mask
				nppSafeCall(nppsMul_32f_I((float*)d_mask.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

				//mean free
				nppSafeCall(nppsSum_32f((float*)d_particle.GetDevicePtr(), volSize * volSize * volSize, (float*)d_sum.GetDevicePtr(), (unsigned char*)d_buffer.GetDevicePtr()));
				float sum_h;
				d_sum.CopyDeviceToHost(&sum_h);
				nppSafeCall(nppsSubC_32f_I(sum_h / maskedVoxels, (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

				//apply mask again to have non masked area =0
				nppSafeCall(nppsMul_32f_I((float*)d_mask.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

				//sum up
				nppSafeCall(nppsAdd_32f_I((float*)d_particle.GetDevicePtr(), (float*)d_meanParticle.GetDevicePtr(), volSize * volSize * volSize));
			}

	#ifndef USE_MPI
			//If not in MPI mode, scale the sum directly here, otherwise later after gathering all partial results from nodes
			//Scale to number of particles
			nppSafeCall(nppsDivC_32f_I(totalCount, (float*)d_meanParticle.GetDevicePtr(), volSize * volSize * volSize));
	#endif
			d_meanParticle.CopyDeviceToHost(sumOfParticles);


	#ifdef USE_MPI	
			//accumulate partial sums over MPI nodes
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, volSize * volSize * volSize, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					d_particle.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppsAdd_32f_I((float*)d_particle.GetDevicePtr(), (float*)d_meanParticle.GetDevicePtr(), volSize * volSize * volSize));
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
				nppSafeCall(nppsDivC_32f_I(totalCount, (float*)d_meanParticle.GetDevicePtr(), volSize * volSize * volSize));

				d_meanParticle.CopyDeviceToHost(sumOfParticles);
			}
			MPI_Bcast(sumOfParticles, volSize * volSize * volSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

			if (mpi_part != 0)
			{
				d_meanParticle.CopyHostToDevice(sumOfParticles);
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
			d_meanParticle.CopyHostToDevice(partSum.GetData());
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
		correlator.PrepareMask(d_mask, false); //mask must be binary anyway

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

			int totalRows = totalCount;

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

			endEntries[mpi_size - 1] = totalCount;

			if (mpi_size == 1)
			{
				startEntries[0] = 0;
				endEntries[0] = totalCount;
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
				for (int p1 = pBlock1; p1 < pBlock1 + aConfig.BlockSize && p1 < endEntries[mpi_part]; p1++)
				{
					motive m = motl.GetAt(p1);
					stringstream ss;
					ss << aConfig.Particles;
					ss << m.GetIndexCoding(aConfig.NamingConv) << ".em";
					EmFile part(ss.str());
					part.OpenAndRead();

					d_particle.CopyHostToDevice(part.GetData());
					nppSafeCall(nppsSub_32f_I((float*)d_meanParticle.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize* volSize* volSize));
					correlator.FourierFilter(d_particle);

					//rotate and shift particle
					float3 shift = make_float3(-m.x_Shift, -m.y_Shift, -m.z_Shift);
					rotator.ShiftRotateTwoStep(shift, -m.psi, -m.phi, -m.theta, d_particle);

					//apply mask
					nppSafeCall(nppsMul_32f_I((float*)d_mask.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

					//mean free
					nppSafeCall(nppsSum_32f((float*)d_particle.GetDevicePtr(), volSize * volSize * volSize, (float*)d_sum.GetDevicePtr(), (unsigned char*)d_buffer.GetDevicePtr()));
					float sum_h;
					d_sum.CopyDeviceToHost(&sum_h);
					nppSafeCall(nppsSubC_32f_I(sum_h / maskedVoxels, (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

					//apply mask again to have non masked area =0
					nppSafeCall(nppsMul_32f_I((float*)d_mask.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

					d_particle.CopyDeviceToHost(particleBlock1[p1 - pBlock1]);
				}

				//we have to scan the entire line, i.e. all colums = all particles
				for (int pBlock2 = pBlock1; pBlock2 < totalCount; pBlock2 += aConfig.BlockSize)
				{
					//load particles for block 2 from disk and make them mean free
					for (int p2 = pBlock2; p2 < pBlock2 + aConfig.BlockSize && p2 < totalCount; p2++)
					{
						motive m = motl.GetAt(p2);
						stringstream ss;
						ss << aConfig.Particles;
						ss << m.GetIndexCoding(aConfig.NamingConv) << ".em";
						EmFile part(ss.str());
						part.OpenAndRead();

						d_particle.CopyHostToDevice(part.GetData());
						nppSafeCall(nppsSub_32f_I((float*)d_meanParticle.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize* volSize* volSize));
						correlator.FourierFilter(d_particle);

						//rotate and shift particle
						float3 shift = make_float3(-m.x_Shift, -m.y_Shift, -m.z_Shift);
						rotator.ShiftRotateTwoStep(shift, -m.psi, -m.phi, -m.theta, d_particle);

						//apply mask
						nppSafeCall(nppsMul_32f_I((float*)d_mask.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

						//mean free
						nppSafeCall(nppsSum_32f((float*)d_particle.GetDevicePtr(), volSize * volSize * volSize, (float*)d_sum.GetDevicePtr(), (unsigned char*)d_buffer.GetDevicePtr()));
						float sum_h;
						d_sum.CopyDeviceToHost(&sum_h);
						nppSafeCall(nppsSubC_32f_I(sum_h / maskedVoxels, (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

						//apply mask again to have non masked area =0
						nppSafeCall(nppsMul_32f_I((float*)d_mask.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

						d_particle.CopyDeviceToHost(particleBlock2[p2 - pBlock2]);
					}


					for (int p1 = pBlock1; p1 < pBlock1 + aConfig.BlockSize && p1 < endEntries[mpi_part]; p1++)
					{
						motive m1 = motl.GetAt(p1);
						int wedgeIdx1 = 0;
						if (!aConfig.SingleWedge)
						{
							wedgeIdx1 = (int)m1.wedgeIdx;
						}
						d_wedge1.CopyHostToDevice((float*)wedges[wedgeIdx1]->GetData());
						rotator.Rotate(-m1.psi, -m1.phi, -m1.theta, d_wedge1);

						d_particle.CopyHostToDevice(particleBlock1[p1 - pBlock1]);

						for (int p2 = pBlock2; p2 < pBlock2 + aConfig.BlockSize && p2 < totalCount; p2++)
						{
							if (p2 > p1) //only upper diagonal part of matrix needed
							{
								//compute CC
								motive m2 = motl.GetAt(p2);

								int wedgeIdx2 = 0;
								if (!aConfig.SingleWedge)
								{
									wedgeIdx2 = (int)m2.wedgeIdx;
								}

								d_wedge2.CopyHostToDevice((float*)wedges[wedgeIdx2]->GetData());
								rotator.Rotate(-m2.psi, -m2.phi, -m2.theta, d_wedge2);

								nppSafeCall(nppsMul_32f((float*)d_wedge1.GetDevicePtr(), (float*)d_wedge2.GetDevicePtr(), (float*)d_wedgeMerge.GetDevicePtr(), volSize * volSize * volSize));

								d_particleRef.CopyHostToDevice(particleBlock2[p2 - pBlock2]);

								//correlator.PrepareParticle(d_particle, d_wedgeMerge);
								//correlator.GetCC(d_mask, d_particleRef, d_wedgeMerge, d_CC);

								//correlator.PhaseCorrelate(d_particle, d_mask, d_particleRef, d_wedgeMerge, d_CC);

								//d_CC.CopyDeviceToHost(cc_vol);

								//emwrite("J:\\TestData\\Test\\cc.em", cc_vol, volSize, volSize, volSize);

								//int ccIdx = (volSize / 2) * volSize * volSize + (volSize / 2) * volSize + (volSize / 2);

								//float CC = cc_vol[ccIdx];

								float CC = correlator.GettCCFast(d_mask, d_particle, d_particleRef, d_wedgeMerge);
							
								CCMatrix[p1 + p2 * totalCount] = CC;
							}
						}
					}
				}
			}

	#ifdef USE_MPI	
			//accumulate partial cc Matrix from MPI nodes
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(CCMatrixMPI, totalCount * totalCount, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

					for (int i = 0; i < totalCount * totalCount; i++)
					{
						CCMatrix[i] += CCMatrixMPI[i];
					}
					//Fill diagonal
					for (int i = 0; i < totalCount; i++)
					{
						CCMatrix[i + totalCount * i] = 1.0f;
					}
				}

				//Fill diagonal
				for (int i = 0; i < totalCount; i++)
				{
					CCMatrix[i + totalCount * i] = 1.0f;
				}			
			}
			else
			{
				MPI_Send(CCMatrix, totalCount * totalCount, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}

	#endif
	#ifndef USE_MPI
			//Fill diagonal
			for (int i = 0; i < totalCount; i++)
			{
				CCMatrix[i + totalCount * i] = 1.0f;
			}
	#endif

			if (mpi_part == 0)
			{
				emwrite(aConfig.CovVarMatrixFilename, CCMatrix, totalCount, totalCount);
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
			for (int i = 0; i < totalCount * totalCount; i++)
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
				totalCount,
				CUDA_R_32F,
				(float*)d_covVarMat.GetDevicePtr(),
				totalCount,
				&something,
				&something,
				totalCount - aConfig.NumberOfEigenVectors + 1,
				totalCount,
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

			if (info != 0)
			{
				stringstream ss;
				ss << "Something went wrong during computation of Eigen vectors. Info returned: " << info;
				throw std::invalid_argument(ss.str());
			}

			d_covVarMat.CopyDeviceToHost(CCMatrix);
			if (aConfig.EigenVectors.size() > 1)
			{
				emwrite(aConfig.EigenVectors, CCMatrix, totalCount, totalCount);
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
		MPI_Bcast(CCMatrix, totalCount* totalCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
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

		for (size_t particle = startParticle; particle < endParticle; particle++)
		{
			motive m = motl.GetAt(particle);
			stringstream ss;
			ss << aConfig.Particles;
			ss << m.GetIndexCoding(aConfig.NamingConv) << ".em";
			EmFile part(ss.str());
			part.OpenAndRead();

			d_particle.CopyHostToDevice(part.GetData());
			nppSafeCall(nppsSub_32f_I((float*)d_meanParticle.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize* volSize* volSize));
			correlator.FourierFilter(d_particle);

			//rotate and shift particle
			float3 shift = make_float3(-m.x_Shift, -m.y_Shift, -m.z_Shift);
			rotator.ShiftRotateTwoStep(shift, -m.psi, -m.phi, -m.theta, d_particle);

			//apply mask
			nppSafeCall(nppsMul_32f_I((float*)d_mask.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

			//mean free
			nppSafeCall(nppsSum_32f((float*)d_particle.GetDevicePtr(), volSize * volSize * volSize, (float*)d_sum.GetDevicePtr(), (unsigned char*)d_buffer.GetDevicePtr()));
			float sum_h;
			d_sum.CopyDeviceToHost(&sum_h);
			nppSafeCall(nppsSubC_32f_I(sum_h / maskedVoxels, (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

			//apply mask again to have non masked area =0
			nppSafeCall(nppsMul_32f_I((float*)d_mask.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

			//d_particle.CopyDeviceToHost(part.GetData());
			//float* data = (float*)part.GetData();

			//for (int eigenImage = 0; eigenImage < aConfig.NumberOfEigenVectors; eigenImage++)
			//{
			//	for (int voxel = 0; voxel < maskedVoxels; voxel++)
			//	{
			//		float ev = CCMatrix[particle + (aConfig.NumberOfEigenVectors - 1 - eigenImage) * totalCount]; //eigenvectors are in inverse order (small to large eigen value)
			//		int unmaskedIndex = maskIndices[voxel];
			//		float vo = data[unmaskedIndex];
			//		eigenImages[eigenImage * volSize * volSize * volSize + unmaskedIndex] += ev * vo;
			//	}
			//}

			kernelEigenImages(volSize * volSize * volSize, aConfig.NumberOfEigenVectors, particle, totalCount, d_covVarMat, d_particle, d_eigenImages);
		}

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

		for (size_t particle = startParticle; particle < endParticle; particle++)
		{
			motive m = motl.GetAt(particle);
			stringstream ss;
			ss << aConfig.Particles;
			ss << m.GetIndexCoding(aConfig.NamingConv) << ".em";
			EmFile part(ss.str());
			part.OpenAndRead();

			d_particle.CopyHostToDevice(part.GetData());
			nppSafeCall(nppsSub_32f_I((float*)d_meanParticle.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize* volSize* volSize));
			correlator.FourierFilter(d_particle);

			//rotate and shift particle
			float3 shift = make_float3(-m.x_Shift, -m.y_Shift, -m.z_Shift);
			rotator.ShiftRotateTwoStep(shift, -m.psi, -m.phi, -m.theta, d_particle);

			//apply mask
			nppSafeCall(nppsMul_32f_I((float*)d_mask.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

			//mean free
			nppSafeCall(nppsSum_32f((float*)d_particle.GetDevicePtr(), volSize * volSize * volSize, (float*)d_sum.GetDevicePtr(), (unsigned char*)d_buffer.GetDevicePtr()));
			float sum_h;
			d_sum.CopyDeviceToHost(&sum_h);
			nppSafeCall(nppsSubC_32f_I(sum_h / maskedVoxels, (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

			//apply mask again to have non masked area =0
			nppSafeCall(nppsMul_32f_I((float*)d_mask.GetDevicePtr(), (float*)d_particle.GetDevicePtr(), volSize * volSize * volSize));

			//d_particle.CopyDeviceToHost(part.GetData());
			//float* data = (float*)part.GetData();

			for (int eigenImage = 0; eigenImage < aConfig.NumberOfEigenVectors; eigenImage++)
			{
				/*for (int voxel = 0; voxel < maskedVoxels; voxel++)
				{
					int unmaskedIndex = maskIndices[voxel];
					float ei = eigenImages[eigenImage * volSize * volSize * volSize + unmaskedIndex];
					float vo = data[unmaskedIndex];
					weightMatrix[eigenImage + particle * aConfig.NumberOfEigenVectors] += ei * vo * eigenValuesSorted[eigenImage] * eigenValuesSorted[eigenImage];
				}*/
				//d_particleRef.CopyHostToDevice(eigenImages + (eigenImage * volSize * volSize * volSize));
				
				nppSafeCall(nppsMul_32f(((float*)d_eigenImages.GetDevicePtr()) + ((size_t)eigenImage * volSize * volSize * volSize), (float*)d_particle.GetDevicePtr(), (float*)d_particleRef.GetDevicePtr(), volSize* volSize* volSize));
				nppSafeCall(nppsSum_32f((float*)d_particleRef.GetDevicePtr(), volSize* volSize* volSize, (float*)d_sum.GetDevicePtr(), (Npp8u*)d_buffer.GetDevicePtr()));
				float eivo = 0;
				d_sum.CopyDeviceToHost(&eivo);
				eivo /= volSize * volSize * volSize; //scale to number of voxels to keep values in a reasonable range;
				weightMatrix[eigenImage + particle * aConfig.NumberOfEigenVectors] = eivo * eigenValuesSorted[eigenImage] * eigenValuesSorted[eigenImage];
			}
		}


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
				printf("Class %-6d ", c);
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
			d_particleRef.Memset(0);
			d_wedgeMerge.Memset(0);

			for (int particle = startParticle; particle < endParticle; particle++)
			{
				if (classes[particle] == c)
				{
					motive m = motl.GetAt(particle);

					stringstream ss;
					ss << aConfig.Particles;
					ss << m.GetIndexCoding(aConfig.NamingConv) << ".em";
					EmFile part(ss.str());
					part.OpenAndRead();

					d_particle.CopyHostToDevice(part.GetData());

					int wedgeIdx = 0;
					if (!aConfig.SingleWedge)
					{
						wedgeIdx = (int)m.wedgeIdx;
					}

					d_wedge1.CopyHostToDevice((float*)wedges[wedgeIdx]->GetData());

					correlator.MultiplyWedge(d_particle, d_wedge1);

					float3 shift = make_float3(-m.x_Shift, -m.y_Shift, -m.z_Shift);
					rotator.ShiftRotateTwoStep(shift, -m.psi, -m.phi, -m.theta, d_particle);

					nppSafeCall(nppsAdd_32f_I((float*)d_particle.GetDevicePtr(), (float*)d_particleRef.GetDevicePtr(), volSize* volSize* volSize));

					rotator.Rotate(-m.psi, -m.phi, -m.theta, d_wedge1);

					nppSafeCall(nppsAdd_32f_I((float*)d_wedge1.GetDevicePtr(), (float*)d_wedgeMerge.GetDevicePtr(), volSize* volSize* volSize));

				}
			}


	#ifdef USE_MPI	
			//accumulate partial sums over MPI nodes
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, volSize * volSize * volSize, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					d_particle.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppsAdd_32f_I((float*)d_particle.GetDevicePtr(), (float*)d_particleRef.GetDevicePtr(), volSize * volSize * volSize));

					MPI_Recv(MPIBuffer, volSize * volSize * volSize, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					d_wedge1.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppsAdd_32f_I((float*)d_wedge1.GetDevicePtr(), (float*)d_wedgeMerge.GetDevicePtr(), volSize * volSize * volSize));
				}
			}
			else
			{
				d_particleRef.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, volSize * volSize * volSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
				d_wedgeMerge.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, volSize * volSize * volSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
	#endif

			if (mpi_part == 0)
			{
				correlator.NormalizeWedge(d_particleRef, d_wedgeMerge);
				d_particleRef.CopyDeviceToHost(sumOfParticles);


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


	CudaContext::DestroyContext(ctx);

#ifdef USE_MPI
	MPI_Finalize();
#endif

}
