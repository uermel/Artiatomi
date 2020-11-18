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

#include "KMeans.h"
#include <random>
#include <cfloat>

KMeans::KMeans(int aDimensions, int aLength, int aClasses, float* aData)
	: dimensions(aDimensions),
	length(aLength),
	classes(aClasses),
	computed(false)
{
	isTaken.resize(length, false);
	data.resize(length);
	distances.resize(length, 0);
	classAssigned.resize(length, -1);

	for (int i = 0; i < length; i++)
	{
		data[i].resize(dimensions);

		for (int j = 0; j < dimensions; j++)
		{
			data[i][j] = aData[j + i * dimensions];
		}
	}

	std::default_random_engine e1(aLength-aDimensions-aClasses); //make it a bit deterministic
	std::uniform_int_distribution<int> uniform_dist(0, length-1);
	std::uniform_real_distribution<float> uniform_dist_float(0.0f, 1.0f);
	int firstPointIndex = uniform_dist(e1);

	isTaken[firstPointIndex] = true;
	centroids.push_back(data[firstPointIndex]);

	for (int i = 0; i < length; i++)
	{
		if (i != firstPointIndex)
		{
			float d = computeDistance(i, firstPointIndex);
			distances[i] = d * d;
		}
	}

	while (centroids.size() < classes)
	{
		float distSqSum = 0.0f;

		for (int i = 0; i < length; i++) {
			if (!isTaken[i]) {
				distSqSum += distances[i];
			}
		}
		
		float r = uniform_dist_float(e1) * distSqSum;

		int nextPointIndex = -1;

		double sum = 0.0;
		for (int i = 0; i < length; i++) 
		{
			if (!isTaken[i]) 
			{
				sum += distances[i];
				if (sum >= r) 
				{
					nextPointIndex = i;
					break;
				}
			}
		}

		if (nextPointIndex == -1) 
		{
			for (int i = length - 1; i >= 0; i--) 
			{
				if (!isTaken[i]) 
				{
					nextPointIndex = i;
					break;
				}
			}
		}

		if (nextPointIndex >= 0) 
		{
			isTaken[nextPointIndex] = true;
			centroids.push_back(data[nextPointIndex]);


			if (centroids.size() < classes) 
			{
				for (int j = 0; j < length; j++) 
				{
					if (!isTaken[j]) 
					{
						double d = computeDistance(nextPointIndex, j);
						double d2 = d * d;
						if (d2 < distances[j]) {
							distances[j] = d2;
						}
					}
				}
			}

		}
		else {
			break;
		}
	}
}

float KMeans::computeDistance(int index1, int index2)
{
	float res = 0;
	for (int i = 0; i < dimensions; i++)
	{
		float diff = data[index1][i] - data[index2][i];
		res += sqrtf(diff * diff);
	}
	return res;
}

float KMeans::computeDistanceCentroid(int index, int indexCentroid)
{
	float res = 0;
	for (int i = 0; i < dimensions; i++)
	{
		float diff = data[index][i] - centroids[indexCentroid][i];
		res += sqrtf(diff * diff);
	}
	return res;
}

int* KMeans::GetClasses()
{
	if (computed)
	{
		return classAssigned.data();
	}

	assignPointsToClusters();

	int maxIter = 10000;
	for (int i = 0; i < maxIter; i++)
	{
		recomputeCentroids();
		int changes = assignPointsToClusters();
		if (changes == 0)
		{
			break;
		}
	}	
	computed = true;
	return classAssigned.data();
}

vector<vector<float> >& KMeans::GetCentroids()
{
	return centroids;
}

int KMeans::getNearestCluster(int index)
{
	int clusterIndex = -1;
	float minDist = FLT_MAX;

	for (int i = 0; i < classes; i++)
	{
		float d = computeDistanceCentroid(index, i);

		if (d < minDist)
		{
			clusterIndex = i;
			minDist = d;
		}
	}
	return clusterIndex;
}

int KMeans::assignPointsToClusters()
{
	int assignedDifferently = 0;

	for (int p = 0; p < length; p++)
	{
		int clusterIndex = getNearestCluster(p);
		if (clusterIndex != classAssigned[p])
		{
			assignedDifferently++;
			classAssigned[p] = clusterIndex;
		}
	}

	return assignedDifferently;
}

void KMeans::recomputeCentroids()
{
	vector<int> countElements(classes, 0);

	for (int i = 0; i < classes; i++)
	{
		for (int j = 0; j < dimensions; j++)
		{
			centroids[i][j] = 0;
		}
	}

	for (int i = 0; i < length; i++)
	{
		int c = classAssigned[i];
		countElements[c]++;

		for (int j = 0; j < dimensions; j++)
		{
			centroids[c][j] += data[i][j];
		}
	}

	for (int c = 0; c < classes; c++)
	{
		for (int j = 0; j < dimensions; j++)
		{
			centroids[c][j] /= countElements[c];
		}
	}
}