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


#define USE_MPI
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "basics/default.h"
#include <algorithm>
#include "io/MotiveListe.h"
#include "config/Config.h"
#include "io/EMFile.h"
#include "CudaReducer.h"
#include "cuda/CudaContext.h"
#include "cuda/CudaVariables.h"
#include "cuda/CudaKernel.h"
#include "AvgProcess.h"
#include <time.h>
#include <iomanip>
#include <algorithm>
#include <map>

using namespace std;
using namespace Cuda;


#define round(x) (x >= 0 ? (int)(x + 0.5) : (int)(x - 0.5))

void computeRotMat(float phi, float psi, float theta, float rotMat[3][3])
{
	int i, j;
	float sinphi, sinpsi, sintheta;	/* sin of rotation angles */
	float cosphi, cospsi, costheta;	/* cos of rotation angles */


	float angles[] = { 0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330 };
	float angle_cos[16];
	float angle_sin[16];

	angle_cos[0] = 1.0f;
	angle_cos[1] = sqrt(3.0f) / 2.0f;
	angle_cos[2] = sqrt(2.0f) / 2.0f;
	angle_cos[3] = 0.5f;
	angle_cos[4] = 0.0f;
	angle_cos[5] = -0.5f;
	angle_cos[6] = -sqrt(2.0f) / 2.0f;
	angle_cos[7] = -sqrt(3.0f) / 2.0f;
	angle_cos[8] = -1.0f;
	angle_cos[9] = -sqrt(3.0f) / 2.0f;
	angle_cos[10] = -sqrt(2.0f) / 2.0f;
	angle_cos[11] = -0.5f;
	angle_cos[12] = 0.0f;
	angle_cos[13] = 0.5f;
	angle_cos[14] = sqrt(2.0f) / 2.0f;
	angle_cos[15] = sqrt(3.0f) / 2.0f;
	angle_sin[0] = 0.0f;
	angle_sin[1] = 0.5f;
	angle_sin[2] = sqrt(2.0f) / 2.0f;
	angle_sin[3] = sqrt(3.0f) / 2.0f;
	angle_sin[4] = 1.0f;
	angle_sin[5] = sqrt(3.0f) / 2.0f;
	angle_sin[6] = sqrt(2.0f) / 2.0f;
	angle_sin[7] = 0.5f;
	angle_sin[8] = 0.0f;
	angle_sin[9] = -0.5f;
	angle_sin[10] = -sqrt(2.0f) / 2.0f;
	angle_sin[11] = -sqrt(3.0f) / 2.0f;
	angle_sin[12] = -1.0f;
	angle_sin[13] = -sqrt(3.0f) / 2.0f;
	angle_sin[14] = -sqrt(2.0f) / 2.0f;
	angle_sin[15] = -0.5f;

	for (i = 0, j = 0; i<16; i++)
		if (angles[i] == phi)
		{
			cosphi = angle_cos[i];
			sinphi = angle_sin[i];
			j = 1;
		}

	if (j < 1)
	{
		phi = phi * (float)M_PI / 180.0f;
		cosphi = cos(phi);
		sinphi = sin(phi);
	}

	for (i = 0, j = 0; i<16; i++)
		if (angles[i] == psi)
		{
			cospsi = angle_cos[i];
			sinpsi = angle_sin[i];
			j = 1;
		}

	if (j < 1)
	{
		psi = psi * (float)M_PI / 180.0f;
		cospsi = cos(psi);
		sinpsi = sin(psi);
	}

	for (i = 0, j = 0; i<16; i++)
		if (angles[i] == theta)
		{
			costheta = angle_cos[i];
			sintheta = angle_sin[i];
			j = 1;
		}

	if (j < 1)
	{
		theta = theta * (float)M_PI / 180.0f;
		costheta = cos(theta);
		sintheta = sin(theta);
	}

	/* calculation of rotation matrix */

	rotMat[0][0] = cospsi*cosphi - costheta*sinpsi*sinphi;
	rotMat[1][0] = sinpsi*cosphi + costheta*cospsi*sinphi;
	rotMat[2][0] = sintheta*sinphi;
	rotMat[0][1] = -cospsi*sinphi - costheta*sinpsi*cosphi;
	rotMat[1][1] = -sinpsi*sinphi + costheta*cospsi*cosphi;
	rotMat[2][1] = sintheta*cosphi;
	rotMat[0][2] = sintheta*sinpsi;
	rotMat[1][2] = -sintheta*cospsi;
	rotMat[2][2] = costheta;
}

void multiplyRotMatrix(float m1[3][3], float m2[3][3], float out[3][3])
{
	out[0][0] = m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1] + m1[2][0] * m2[0][2];
	out[1][0] = m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1] + m1[2][0] * m2[1][2];
	out[2][0] = m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1] + m1[2][0] * m2[2][2];
	out[0][1] = m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1] + m1[2][1] * m2[0][2];
	out[1][1] = m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1] + m1[2][1] * m2[1][2];
	out[2][1] = m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1] + m1[2][1] * m2[2][2];
	out[0][2] = m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1] + m1[2][2] * m2[0][2];
	out[1][2] = m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1] + m1[2][2] * m2[1][2];
	out[2][2] = m1[0][2] * m2[2][0] + m1[1][2] * m2[2][1] + m1[2][2] * m2[2][2];

}

void multiplyRotMatrixVec(float phi, float psi, float theta, float m2[3], float out[3])
{
	phi = phi / 180.0f * (float)M_PI; psi = psi / 180.0f * (float)M_PI; theta = theta / 180.0f * (float)M_PI;

	float m11[3][3];
	m11[0][0] = cos(psi);
	m11[0][1] = sin(psi);
	m11[0][2] = 0;
	m11[1][0] = -sin(psi);
	m11[1][1] = cos(psi);
	m11[1][2] = 0;
	m11[2][0] = 0;
	m11[2][1] = 0;
	m11[2][2] = 1;

	float m22[3][3];
	m22[0][0] = 1;
	m22[0][1] = 0;
	m22[0][2] = 0;
	m22[1][0] = 0;
	m22[1][1] = cos(theta);
	m22[1][2] = sin(theta);
	m22[2][0] = 0;
	m22[2][1] = -sin(theta);
	m22[2][2] = cos(theta);

	float m33[3][3];
	m33[0][0] = cos(phi);
	m33[0][1] = sin(phi);
	m33[0][2] = 0;
	m33[1][0] = -sin(phi);
	m33[1][1] = cos(phi);
	m33[1][2] = 0;
	m33[2][0] = 0;
	m33[2][1] = 0;
	m33[2][2] = 1;

	float m4[3][3];
	float m5[3][3];

	multiplyRotMatrix(m11, m22, m4);
	multiplyRotMatrix(m4, m33, m5);


	out[0] = m5[0][0] * m2[0] + m5[1][0] * m2[1] + m5[2][0] * m2[2];
	out[1] = m5[0][1] * m2[0] + m5[1][1] * m2[1] + m5[2][1] * m2[2];
	out[2] = m5[0][2] * m2[0] + m5[1][2] * m2[1] + m5[2][2] * m2[2];
	/*out[0][1] = m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1] + m1[2][1] * m2[0][2];
	out[1][1] = m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1] + m1[2][1] * m2[1][2];
	out[2][1] = m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1] + m1[2][1] * m2[2][2];
	out[0][2] = m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1] + m1[2][2] * m2[0][2];
	out[1][2] = m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1] + m1[2][2] * m2[1][2];
	out[2][2] = m1[0][2] * m2[2][0] + m1[1][2] * m2[2][1] + m1[2][2] * m2[2][2];*/

}

void getEulerAngles(float matrix[3][3], float& phi, float& psi, float& theta)
{
	theta = acos(matrix[2][2])*180.0f / (float)M_PI;

	if (matrix[2][2] > 0.999)
	{
		float sign = matrix[1][0] > 0 ? 1.0f : -1.0f;
		phi = sign * acos(matrix[0][0])*180.0f / (float)M_PI;
		psi = 0.0f;
	}
	else
	{
		phi = atan2(matrix[2][0], matrix[2][1]) * 180.0f / (float)M_PI;
		psi = atan2(matrix[0][2], -matrix[1][2]) * 180.0f / (float)M_PI;
	}
}

bool checkIfClassIsToAverage(vector<int>& classes, int aClass)
{
	if (classes.size() == 0)
		return true;

	for (int c : classes)
	{
		if (c == aClass)
			return true;
	}
	return false;
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

	bool onlySumUp = true;

	

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

	Configuration::Config aConfig = Configuration::Config::GetConfig("average.cfg", argc, argv, mpi_part, NULL);

	CudaContext* ctx = CudaContext::CreateInstance(aConfig.CudaDeviceIDs[mpi_part], CU_CTX_SCHED_SPIN);

	int iter = aConfig.StartIteration;
	{
		printf("Load motivelist..."); fflush(stdout);
		stringstream ssml;
		ssml << aConfig.Path << aConfig.MotiveList << iter << ".em";
		MotiveList motl(ssml.str());
		printf("Done\n"); fflush(stdout);
		stringstream ssmlStart;
		ssmlStart << aConfig.Path << aConfig.MotiveList << 1 << ".em";
		MotiveList motlStart(ssmlStart.str());

		int totalCount = motl.DimY;
		int partCount = motl.DimY / mpi_size;
		int partCountArray = partCount;
		int lastPartCount = totalCount - (partCount * (mpi_size - 1));
		int startParticle = mpi_part * partCount;

		//adjust last part to fit really all particles (rounding errors...)
		if (mpi_part == mpi_size - 1)
		{
			partCount = lastPartCount;
		}

		int endParticle = startParticle + partCount;

		if (aConfig.ClearAngles)
		{
			for (int i = startParticle; i < endParticle; i++)
			{
				motive m = motl.GetAt(i);
				motive mStart = motlStart.GetAt(i);

				m.phi = mStart.phi;
				m.psi = mStart.psi;
				m.theta = mStart.theta;

				motl.SetAt(i, m);
			}
		}

		/*stringstream ssref;
		ssref << aConfig.Path << aConfig.Reference[0] << iter << ".em";
		EMFile ref(ssref.str());*/
		map<int, EMFile*> wedges;
		if (aConfig.WedgeIndices.size() < 1)
		{
			wedges.insert(pair<int, EMFile*>(0, new EMFile(aConfig.WedgeFile)));
			wedges[0]->OpenAndRead();
			wedges[0]->ReadHeaderInfo();
		}
		else
		{
			for (size_t i = 0; i < aConfig.WedgeIndices.size(); i++)
			{
				stringstream sswedge;
				sswedge << aConfig.WedgeFile << aConfig.WedgeIndices[i] << ".em";
				wedges.insert(pair<int, EMFile*>(aConfig.WedgeIndices[i], new EMFile(sswedge.str())));
				wedges[aConfig.WedgeIndices[i]]->OpenAndRead();
				wedges[aConfig.WedgeIndices[i]]->ReadHeaderInfo();
			}
		}
		//EMFile wedge(aConfig.WedgeList);
		//EMFile mask(aConfig.Mask);
		//EMFile ccmask(aConfig.MaskCC);

		/*ref.OpenAndRead();
		ref.ReadHeaderInfo();
		if (mpi_part == 0)
			cout << "ref OK" << endl;*/


		/*wedge.OpenAndRead();
		wedge.ReadHeaderInfo();
		if (mpi_part == 0)
		cout << "wedge OK" << endl;*/
		//mask.OpenAndRead();
		//mask.ReadHeaderInfo();
		//if (mpi_part == 0)
		//	cout << "mask OK" << endl;
		/*ccmask.OpenAndRead();
		ccmask.ReadHeaderInfo();
		if (mpi_part == 0)
			cout << "maskcc OK" << endl;*/

		

		std::cout << "Context OK" << std::endl;
		int size;
		{
			int particleSize;

			motive mot = motl.GetAt(0);
			stringstream ss;
			ss << aConfig.Path << aConfig.Particles;

			//ss << mot.partNr << ".em";
			ss << mot.GetIndexCoding(aConfig.NamingConv) << ".em";
			EMFile part(ss.str());
			part.OpenAndRead();
			part.ReadHeaderInfo();
			particleSize = part.DimX;
			size = particleSize;

			if (mpi_part == 0)
			{


				cout << "Checking dimensions of input data:" << endl;
				//cout << "Reference: " << ref.DimX << endl;
				cout << "Particles: " << particleSize << endl;
				cout << "Wedge:     " << wedges.begin()->second->DimX << endl;
				//cout << "Mask:      " << mask.DimX << endl;
				//cout << "MaskCC:    " << ccmask.DimX << endl;
			}

			if (wedges.begin()->second->DimX != particleSize)
			{
				if (mpi_part == 0)
					cout << endl << "ERROR: not all input data dimensions are equal!" << endl;
				MPI_Finalize();
				exit(-1);
			}
		}

		//std::cout << "Starte Avergaing... (part size: " << size << ")" << std::endl;



		ctx->Synchronize();

		///////////////////////////////////////
		/// End of Average on motl fragment ///
		///////////////////////////////////////
		MPI_Barrier(MPI_COMM_WORLD);
		/////////////////////////////////
		/// Merge partial motivelists ///
		/////////////////////////////////

		float meanCCValue = 0;
		if (mpi_part == 0)
		{
			float* buffer = new float[motl.DimX * (partCount > lastPartCount ? partCount : lastPartCount)];
			float* motlBuffer = (float*)motl.GetData();

			for (int mpi = 1; mpi < mpi_size - 1; mpi++)
			{
				//cout << mpi_part << ": " << partCount << endl;
				MPI_Recv(buffer, motl.DimX * partCount, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				memcpy(&motlBuffer[motl.DimX * mpi * partCount], buffer, partCount * motl.DimX * sizeof(float));
				//cout << mpi_part << ": " << buffer[0] << endl;
			}

			if (mpi_size > 1)
				for (int mpi = mpi_size - 1; mpi < mpi_size; mpi++)
				{
					//cout << mpi_part << ": " << lastPartCount << endl;
					MPI_Recv(buffer, motl.DimX * lastPartCount, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					memcpy(&motlBuffer[motl.DimX * mpi * partCount], buffer, lastPartCount * motl.DimX * sizeof(float));
					//cout << mpi_part << ": " << buffer[0] << endl;
				}



			double counter = 0;
			double meanCC = 0;

			float* tempCC = new float[totalCount];

			for (size_t i = 0; i < totalCount; i++)
			{
				motive mot2 = motl.GetAt(i);
				meanCC += mot2.ccCoeff;
				counter++;
				tempCC[i] = mot2.ccCoeff;
			}

			sort(tempCC, tempCC + totalCount);

			size_t idx = (size_t)(totalCount * (1.0f - aConfig.BestParticleRatio));
			if (idx > totalCount - 1)
				idx = totalCount - 1;

			if (idx < 0)
				idx = 0;
			meanCCValue = tempCC[idx];
			delete[] tempCC;

			if (!onlySumUp)
			{
				//save motiveList
				stringstream ssmlNew;
				ssmlNew << aConfig.Path << aConfig.MotiveList << iter + 1 << ".em";
				emwrite(ssmlNew.str(), motlBuffer, motl.DimX, motl.DimY, 1);
			}

			motive* motlMot = (motive*)motlBuffer;
			for (size_t i = 0; i < totalCount; i++)
			{
				//mark bad particles with too low ccCoeff with a negative class number, only if it isn't already negative
				if (motlMot[i].ccCoeff < meanCCValue && motlMot[i].classNo < 0)
				{
					motlMot[i].classNo *= -1;
				}
				//if ccCoeff got better in that iteration, remove the negative mark to re-integrate the particle. This keeps the total amount of particle constant!
				if (motlMot[i].ccCoeff >= meanCCValue && motlMot[i].classNo < 0)
				{
					motlMot[i].classNo *= -1;
				}
			}

			
			delete[] buffer;

			/*meanCC /= counter;
			meanCCValue = (float)meanCC;*/
			for (int mpi = 1; mpi < mpi_size; mpi++)
			{
				MPI_Send(&meanCCValue, 1, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD);
			}
		}
		else
		{
			float* motlBuffer = (float*)motl.GetData();
			//cout << mpi_part << ": " << partCount << ": " << startParticle << endl;
			//in last part, partCount is equal to lastPartCount!
			MPI_Send(&motlBuffer[motl.DimX * mpi_part * partCountArray], motl.DimX * partCount, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			MPI_Recv(&meanCCValue, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		/////////////////////////////////
		/// End of motivelist merging ///
		/////////////////////////////////





		MPI_Barrier(MPI_COMM_WORLD);



		/////////////////////
		/// Add particles ///
		/////////////////////

		{

			cufftHandle ffthandle;

			int n[] = { size, size, size };
			cufftSafeCall(cufftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 1));

			CUstream stream = 0;
			cufftSafeCall(cufftSetStream(ffthandle, stream));

			CudaRot rot(size, stream, ctx, aConfig.LinearInterpolation);
			CudaRot rotWedge(size, stream, ctx, aConfig.LinearInterpolation);
			CudaSub sub(size, stream, ctx);
			CudaMakeCplxWithSub makecplx(size, stream, ctx);
			CudaBinarize binarize(size, stream, ctx);
			CudaMul mul(size, stream, ctx);
			CudaFFT fft(size, stream, ctx);
			CudaReducer max(size*size*size, stream, ctx);
			CudaWedgeNorm wedgeNorm(size, stream, ctx);

			CudaDeviceVariable partReal(size*size*size*sizeof(float));
			CudaDeviceVariable partRot(size*size*size*sizeof(float));
			CudaDeviceVariable partCplx(size*size*size*sizeof(float2));
			CudaDeviceVariable wedge_d(size*size*size*sizeof(float));
			CudaDeviceVariable wedgeSum(size*size*size*sizeof(float));
			CudaDeviceVariable wedgeSumO(size*size*size*sizeof(float));
			CudaDeviceVariable wedgeSumE(size*size*size*sizeof(float));
			CudaDeviceVariable wedgeSumA(size*size*size*sizeof(float));
			CudaDeviceVariable wedgeSumB(size*size*size*sizeof(float));
			CudaDeviceVariable tempCplx(size*size*size*sizeof(float2));
			CudaDeviceVariable temp(size*size*size*sizeof(float));
			CudaDeviceVariable partSum(size*size*size*sizeof(float));
			CudaDeviceVariable partSumEven(size*size*size*sizeof(float));
			CudaDeviceVariable partSumOdd(size*size*size*sizeof(float));
			CudaDeviceVariable partSumA(size*size*size*sizeof(float));
			CudaDeviceVariable partSumB(size*size*size*sizeof(float));


			int skipped = 0;
			vector<int> partsPerRef;

			for (size_t ref = 0; ref < aConfig.Reference.size(); ref++)
			{
				float currentReference = ref + 1;

				partSum.Memset(0);
				partSumOdd.Memset(0);
				partSumEven.Memset(0);
				partSumA.Memset(0);
				partSumB.Memset(0);

				wedgeSum.Memset(0);
				wedgeSumO.Memset(0);
				wedgeSumE.Memset(0);
				wedgeSumA.Memset(0);
				wedgeSumB.Memset(0);




				int sumCount = 0;
				int motCount = partCount;

				float limit = 0;

				limit = meanCCValue;
				int oldWedgeIdx = -1;

				for (int i = startParticle; i < endParticle; i++)
				{
					motive mot = motl.GetAt(i);
					stringstream ss;

					ss << aConfig.Path << aConfig.Particles;
					if (mot.classNo != currentReference && aConfig.Reference.size() > 1)
					{
						continue;
					}
					if (mot.ccCoeff < limit)
					{
						skipped++;
						continue;
					}

					if (oldWedgeIdx != mot.wedgeIdx)
					{
						oldWedgeIdx = 0;
						if (aConfig.WedgeIndices.size() > 0)
						{
							oldWedgeIdx = mot.wedgeIdx;
						}

						wedge_d.CopyHostToDevice((float*)wedges[oldWedgeIdx]->GetData());
						rotWedge.SetTexture(wedge_d);
					}
					sumCount++;

					ss << mot.GetIndexCoding(aConfig.NamingConv) << ".em";

					cout << mpi_part << ": " << "Part nr: " << mot.partNr << " ref: " << currentReference << " summed up: " << sumCount << " skipped: " << skipped << " = " << sumCount + skipped << " of " << motCount << endl;

					EMFile part(ss.str());
					part.OpenAndRead();
					part.ReadHeaderInfo();

					int size = part.DimX;

					partReal.CopyHostToDevice(part.GetData());

					float3 shift;
					shift.x = -mot.x_Shift;
					shift.y = -mot.y_Shift;
					shift.z = -mot.z_Shift;

					rot.SetTextureShift(partReal);
					rot.Shift(partRot, shift);

					rot.SetTexture(partRot);
					rot.Rot(partReal, -mot.psi, -mot.phi, -mot.theta);

					rotWedge.Rot(wedge_d, -mot.psi, -mot.phi, -mot.theta);
					sub.Add(wedge_d, wedgeSum);

					makecplx.MakeCplxWithSub(partReal, partCplx, 0);


					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)partCplx.GetDevicePtr(), (cufftComplex*)tempCplx.GetDevicePtr(), CUFFT_FORWARD));
					fft.FFTShift2(tempCplx, partCplx);
					mul.MulVolCplx(wedge_d, partCplx);

					fft.FFTShift2(partCplx, tempCplx);

					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)tempCplx.GetDevicePtr(), (cufftComplex*)partCplx.GetDevicePtr(), CUFFT_INVERSE));

					mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);

					makecplx.MakeReal(partCplx, partReal);

					sub.Add(partReal, partSum);

					if (sumCount % 2 == 0)
					{
						sub.Add(partReal, partSumEven);
						sub.Add(wedge_d, wedgeSumE);
					}
					else
					{
						sub.Add(partReal, partSumOdd);
						sub.Add(wedge_d, wedgeSumO);
					}

					if (i < motCount / 2)
					{
						sub.Add(partReal, partSumA);
						sub.Add(wedge_d, wedgeSumA);
					}
					else
					{
						sub.Add(partReal, partSumB);
						sub.Add(wedge_d, wedgeSumB);
					}

				}

				partsPerRef.push_back(sumCount);

				if (mpi_part == 0)
				{
					float* buffer = new float[size*size*size];
					for (int mpi = 1; mpi < mpi_size; mpi++)
					{
						MPI_Recv(buffer, size*size*size, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						partReal.CopyHostToDevice(buffer);
						sub.Add(partReal, partSum);
						MPI_Recv(buffer, size*size*size, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						partReal.CopyHostToDevice(buffer);
						sub.Add(partReal, wedgeSum);

						MPI_Recv(buffer, size*size*size, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						partReal.CopyHostToDevice(buffer);
						sub.Add(partReal, partSumEven);
						MPI_Recv(buffer, size*size*size, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						partReal.CopyHostToDevice(buffer);
						sub.Add(partReal, wedgeSumE);

						MPI_Recv(buffer, size*size*size, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						partReal.CopyHostToDevice(buffer);
						sub.Add(partReal, partSumOdd);
						MPI_Recv(buffer, size*size*size, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						partReal.CopyHostToDevice(buffer);
						sub.Add(partReal, wedgeSumO);

						MPI_Recv(buffer, size*size*size, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						partReal.CopyHostToDevice(buffer);
						sub.Add(partReal, partSumA);
						MPI_Recv(buffer, size*size*size, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						partReal.CopyHostToDevice(buffer);
						sub.Add(partReal, wedgeSumA);

						MPI_Recv(buffer, size*size*size, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						partReal.CopyHostToDevice(buffer);
						sub.Add(partReal, partSumB);
						MPI_Recv(buffer, size*size*size, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						partReal.CopyHostToDevice(buffer);
						sub.Add(partReal, wedgeSumB);
					}
					delete[] buffer;
				}
				else
				{
					float* buffer = new float[size*size*size];

					partSum.CopyDeviceToHost(buffer);
					MPI_Send(buffer, size*size*size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
					wedgeSum.CopyDeviceToHost(buffer);
					MPI_Send(buffer, size*size*size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

					partSumEven.CopyDeviceToHost(buffer);
					MPI_Send(buffer, size*size*size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
					wedgeSumE.CopyDeviceToHost(buffer);
					MPI_Send(buffer, size*size*size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

					partSumOdd.CopyDeviceToHost(buffer);
					MPI_Send(buffer, size*size*size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
					wedgeSumO.CopyDeviceToHost(buffer);
					MPI_Send(buffer, size*size*size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

					partSumA.CopyDeviceToHost(buffer);
					MPI_Send(buffer, size*size*size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
					wedgeSumA.CopyDeviceToHost(buffer);
					MPI_Send(buffer, size*size*size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

					partSumB.CopyDeviceToHost(buffer);
					MPI_Send(buffer, size*size*size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
					wedgeSumB.CopyDeviceToHost(buffer);
					MPI_Send(buffer, size*size*size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

					delete[] buffer;
				}


				if (mpi_part == 0)
				{
					max.MaxIndex(wedgeSum, temp, tempCplx);
					/*float* test2 = new float[128 * 128 * 128];
					partSum.CopyDeviceToHost(test2);
					emwrite("Z:\\kunz\\Documents\\TestSubTomogramAveraging\\testPart.em", test2, 128, 128, 128);*/

					/* TODO AS Write Particle into EM file */
					float* testParticle = new float[size*size*size];
					partSum.CopyDeviceToHost(testParticle);
					emwrite("testParticle.em", testParticle, size, size, size);
					delete[] testParticle;

					makecplx.MakeCplxWithSub(partSum, partCplx, 0);
					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)partCplx.GetDevicePtr(), (cufftComplex*)tempCplx.GetDevicePtr(), CUFFT_FORWARD));
					fft.FFTShift2(tempCplx, partCplx);

					/*float* test = new float[128 * 128 * 128];
					wedgeSum.CopyDeviceToHost(test);
					emwrite("Z:\\kunz\\Documents\\TestSubTomogramAveraging\\testWedge.em", test, 128, 128, 128);*/
					
					/* TODO AS Write Out wedge to file */
					float* testWedge = new float[size*size*size];
					wedgeSum.CopyDeviceToHost(testWedge);
					emwrite("testWedge.em", testWedge, size, size, size);
					delete[] testWedge;

					wedgeNorm.WedgeNorm(wedgeSum, partCplx, temp, 0);

					fft.FFTShift2(partCplx, tempCplx);
					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)tempCplx.GetDevicePtr(), (cufftComplex*)partCplx.GetDevicePtr(), CUFFT_INVERSE));
					mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);
					makecplx.MakeReal(partCplx, partSum);

					max.MaxIndex(wedgeSumO, temp, tempCplx);

					makecplx.MakeCplxWithSub(partSumOdd, partCplx, 0);
					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)partCplx.GetDevicePtr(), (cufftComplex*)tempCplx.GetDevicePtr(), CUFFT_FORWARD));
					fft.FFTShift2(tempCplx, partCplx);

					wedgeNorm.WedgeNorm(wedgeSumO, partCplx, temp, 0);

					fft.FFTShift2(partCplx, tempCplx);
					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)tempCplx.GetDevicePtr(), (cufftComplex*)partCplx.GetDevicePtr(), CUFFT_INVERSE));
					mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);
					makecplx.MakeReal(partCplx, partSumOdd);



					max.MaxIndex(wedgeSumE, temp, tempCplx);

					makecplx.MakeCplxWithSub(partSumEven, partCplx, 0);
					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)partCplx.GetDevicePtr(), (cufftComplex*)tempCplx.GetDevicePtr(), CUFFT_FORWARD));
					fft.FFTShift2(tempCplx, partCplx);

					wedgeNorm.WedgeNorm(wedgeSumE, partCplx, temp, 0);

					fft.FFTShift2(partCplx, tempCplx);
					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)tempCplx.GetDevicePtr(), (cufftComplex*)partCplx.GetDevicePtr(), CUFFT_INVERSE));
					mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);
					makecplx.MakeReal(partCplx, partSumEven);



					max.MaxIndex(wedgeSumA, temp, tempCplx);

					makecplx.MakeCplxWithSub(partSumA, partCplx, 0);
					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)partCplx.GetDevicePtr(), (cufftComplex*)tempCplx.GetDevicePtr(), CUFFT_FORWARD));
					fft.FFTShift2(tempCplx, partCplx);

					wedgeNorm.WedgeNorm(wedgeSumA, partCplx, temp, 0);

					fft.FFTShift2(partCplx, tempCplx);
					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)tempCplx.GetDevicePtr(), (cufftComplex*)partCplx.GetDevicePtr(), CUFFT_INVERSE));
					mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);
					makecplx.MakeReal(partCplx, partSumA);



					max.MaxIndex(wedgeSumB, temp, tempCplx);

					makecplx.MakeCplxWithSub(partSumB, partCplx, 0);
					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)partCplx.GetDevicePtr(), (cufftComplex*)tempCplx.GetDevicePtr(), CUFFT_FORWARD));
					fft.FFTShift2(tempCplx, partCplx);

					wedgeNorm.WedgeNorm(wedgeSumB, partCplx, temp, 0);

					fft.FFTShift2(partCplx, tempCplx);
					cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)tempCplx.GetDevicePtr(), (cufftComplex*)partCplx.GetDevicePtr(), CUFFT_INVERSE));
					mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);
					makecplx.MakeReal(partCplx, partSumB);



					float* sum = new float[size*size*size];

					if (aConfig.ApplySymmetry == Configuration::Symmetry_Rotate180)
					{
						float* nowedge = new float[size*size*size];
						float* part = new float[size*size*size];
						for (size_t i = 0; i < size*size*size; i++)
						{
							nowedge[i] = 1;
						}
						partSum.CopyDeviceToHost(sum);
						rot.SetOldAngles(0, 0, 0);
						rot.SetTexture(partSum);
						rot.Rot(temp, 180.0f, 0, 0);
						temp.CopyDeviceToHost(part);

						/*emwrite("testpart.em", part, size, size, size);
						emwrite("testSum.em", sum, size, size, size);*/


						AvgProcess p(size, 0, ctx, sum, nowedge, nowedge, 1, 0, 5, 3, aConfig.BinarizeMask, aConfig.RotateMaskCC, false, aConfig.LinearInterpolation);

						maxVals_t v = p.execute(part, nowedge, NULL, 0, 0, 0, (float)aConfig.HighPass, (float)aConfig.LowPass, (float)aConfig.Sigma, make_float3(0, 0, 0), aConfig.CouplePhiToPsi, false, 0);
						int sx, sy, sz;
						v.getXYZ(size, sx, sy, sz);

						cout << mpi_part << ": " << "Found shift for symmetry: " << sx << ", " << sy << ", " << sz << v.ccVal << endl;
						cout << mpi_part << ": " << "Found PSI/Theta for symmetry: " << v.rphi << " / " << v.rthe << " CC-Val: " << v.ccVal << endl;

						float3 shift;
						shift.x = -sx;
						shift.y = -sy;
						shift.z = -sz;

						rot.SetTextureShift(temp);
						rot.Shift(partRot, shift);

						rot.SetTexture(partRot);
						rot.Rot(partReal, 0, -v.rphi, 0);

						sub.Add(partReal, partSum);
						delete[] nowedge;
						delete[] part;
					}


					if (aConfig.ApplySymmetry == Configuration::Symmetry_Shift)
					{
						//partSum is now the averaged Particle without symmetry
						rot.SetTextureShift(partSum);

						if (!(aConfig.ShiftSymmetryVector[0].x == 0 && aConfig.ShiftSymmetryVector[0].y == 0 && aConfig.ShiftSymmetryVector[0].z == 0))
						{
							rot.Shift(partReal, aConfig.ShiftSymmetryVector[0]);
							sub.Add(partReal, partSum);

							rot.Shift(partReal, -aConfig.ShiftSymmetryVector[0]);
							sub.Add(partReal, partSum);
						}

						if (!(aConfig.ShiftSymmetryVector[1].x == 0 && aConfig.ShiftSymmetryVector[1].y == 0 && aConfig.ShiftSymmetryVector[1].z == 0))
						{
							rot.Shift(partReal, aConfig.ShiftSymmetryVector[1]);
							sub.Add(partReal, partSum);

							rot.Shift(partReal, -aConfig.ShiftSymmetryVector[1]);
							sub.Add(partReal, partSum);
						}

						if (!(aConfig.ShiftSymmetryVector[2].x == 0 && aConfig.ShiftSymmetryVector[2].y == 0 && aConfig.ShiftSymmetryVector[2].z == 0))
						{
							rot.Shift(partReal, aConfig.ShiftSymmetryVector[2]);
							sub.Add(partReal, partSum);

							rot.Shift(partReal, -aConfig.ShiftSymmetryVector[2]);
							sub.Add(partReal, partSum);
						}
					}


					if (aConfig.ApplySymmetry == Configuration::Symmetry_Helical)
					{
						partSum.CopyDeviceToHost(sum);
						stringstream ss1;
						string outName = aConfig.Path + aConfig.Reference[ref] + "noSymm_";
						ss1 << outName << iter << ".em";
						emwrite(ss1.str(), sum, size, size, size);

						//partSum is now the averaged Particle without symmetry
						rot.SetTexture(partSum);


						/*float rise = 22.92f / (49.0f / 3.0f) / (1.1f * 2);
						float twist = 360.0f / 49.0f * 3.0f;*/
						float rise = aConfig.HelicalRise;
						float twist = aConfig.HelicalTwist;

						for (int i = aConfig.HelicalRepeatStart; i <= aConfig.HelicalRepeatEnd; i++)
						{
							if (i != 0)
							{
								float angPhi = twist * i;
								float shift = rise * i;

								rot.Rot(partReal, angPhi, 0, 0);
								rot.SetTextureShift(partReal);
								rot.Shift(partReal, make_float3(0, 0, shift));
								sub.Add(partReal, partSum);
							}
						}
					}

					if (aConfig.ApplySymmetry == Configuration::Symmetry_Rotational)
					{
						partSum.CopyDeviceToHost(sum);
						stringstream ss1;
						string outName = aConfig.Path + aConfig.Reference[ref] + "noSymm_";
						ss1 << outName << iter << ".em";
						emwrite(ss1.str(), sum, size, size, size);

						//partSum is now the averaged Particle without symmetry
						rot.SetTexture(partSum);

						float angle = aConfig.RotationalAngleStep;

						for (int i = 1; i < aConfig.RotationalCount; i++) //i=0 is the average itself
						{
							float angPhi = angle * i;

							rot.Rot(partReal, angPhi, 0, 0);
							sub.Add(partReal, partSum);
						}
					}


					if (aConfig.BFactor != 0)
					{
						cout << "Apply B-factor of " << aConfig.BFactor << "..." << endl;
						partSum.CopyDeviceToHost(sum);
						stringstream ss1;
						string outName = aConfig.Path + aConfig.Reference[ref] + "noBfac_";
						ss1 << outName << iter + 1 << ".em";
						emwrite(ss1.str(), sum, size, size, size);


						makecplx.MakeCplxWithSub(partSum, partCplx, 0);
						cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)partCplx.GetDevicePtr(), (cufftComplex*)tempCplx.GetDevicePtr(), CUFFT_FORWARD));
						fft.FFTShift2(tempCplx, partCplx);

						float2* particle = new float2[size * size * size];
						partCplx.CopyDeviceToHost(particle);


						for (int z = 0; z < size; z++)
						{
							for (int y = 0; y < size; y++)
							{
								for (int x = 0; x < size; x++)
								{
									int dz = (z - size / 2);
									int dy = (y - size / 2);
									int dx = (x - size / 2);

									float d = sqrt(dx * dx + dy * dy + dz * dz);

									d = round(d);
									d = d >(size / 2 - 1) ? (size / 2 - 1) : d;

									float res = size / (d + 1) * aConfig.PixelSize;

									float value = expf(-aConfig.BFactor / (4.0f * res * res));

									size_t idx = z * size * size + y * size + x;
									float2 pixel = particle[idx];
									pixel.x *= value;
									pixel.y *= value;
									particle[idx] = pixel;
								}
							}
						}


						partCplx.CopyHostToDevice(particle);
						delete[] particle;
						fft.FFTShift2(partCplx, tempCplx);
						cufftSafeCall(cufftExecC2C(ffthandle, (cufftComplex*)tempCplx.GetDevicePtr(), (cufftComplex*)partCplx.GetDevicePtr(), CUFFT_INVERSE));
						mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);
						makecplx.MakeReal(partCplx, partSum);
					}


					partSum.CopyDeviceToHost(sum);
					stringstream ss1;
					string outName = aConfig.Path + aConfig.Reference[ref];
					ss1 << outName << iter << ".em";
					emwrite(ss1.str(), sum, size, size, size);

					partSumEven.CopyDeviceToHost(sum);
					stringstream ss2;
					ss2 << outName << iter << "Even.em";
					emwrite(ss2.str(), sum, size, size, size);
					partSumOdd.CopyDeviceToHost(sum);
					stringstream ss3;
					ss3 << outName << iter << "Odd.em";
					emwrite(ss3.str(), sum, size, size, size);

					partSumA.CopyDeviceToHost(sum);
					stringstream ss5;
					ss5 << outName << iter << "A.em";
					emwrite(ss5.str(), sum, size, size, size);
					partSumB.CopyDeviceToHost(sum);
					stringstream ss4;
					ss4 << outName << iter << "B.em";
					emwrite(ss4.str(), sum, size, size, size);
					delete[] sum;
				}

			}



			if (mpi_part == 0)
			{
				int* buffer = new int[aConfig.Reference.size()];
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(buffer, aConfig.Reference.size(), MPI_INT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					for (size_t i = 0; i < aConfig.Reference.size(); i++)
					{
						partsPerRef[i] += buffer[i];
					}
				}

				int totalUsed = 0;
				for (size_t i = 0; i < aConfig.Reference.size(); i++)
				{
					totalUsed += partsPerRef[i];
				}

				//Output statistics:
				cout << "Total particles:   " << totalCount << endl;
				cout << "Ignored particles: " << totalCount - totalUsed << endl;
				cout << "Used particles:    " << totalUsed << endl;

				if (aConfig.MultiReference)
				{
					for (size_t i = 0; i < aConfig.Reference.size(); i++)
					{
						cout << "Used for ref" << i + 1 << ":     " << partsPerRef[i] << endl;
					}
				}

				delete[] buffer;
			}
			else
			{
				MPI_Send(&partsPerRef[0], aConfig.Reference.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
			}

			cufftDestroy(ffthandle);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		////////////////////////////
		/// End of Add particles ///
		////////////////////////////

		
	}


	MPI_Finalize();


	return 0;
}

