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


#ifndef KMEANS_H
#define KMEANS_H

#include <vector>

using namespace std;

//Implements KMeans using KMeans++ initialization.
//Loosely based on the implementation given at https://commons.apache.org/proper/commons-math/userguide/ml.html

class KMeans
{
private:
	int dimensions;
	int length;
	int classes;
	bool computed;
		
	vector<bool> isTaken;

	vector<vector<float> > data;

	vector<vector<float> > centroids;

	vector<float> distances;

	vector<int> classAssigned;

	float computeDistance(int index1, int index2);

	float computeDistanceCentroid(int index1, int index2);

	int getNearestCluster(int index);

	int assignPointsToClusters();

	void recomputeCentroids();


public:

	KMeans(int aDimensions, int aLength, int aClasses, float* aData);

	int* GetClasses();
};

#endif