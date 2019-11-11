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


#ifndef VOLUME_H
#define VOLUME_H

#include "EmSartDefault.h"
#define MEMSIZE (4UL * 1024UL * 1024UL * 1024UL) //2 GB

template<typename tVoxel>
class Volume
{
private:
	int			_countSubVolumes;
	tVoxel**	_data;
	float3		_positionVolume;
	float3*		_positionSubVolume;
	uint3		_dimension;
	uint3*		_dimensionSubVolume;
	float3		_voxelSize;
	int			_bitResolution;

public:
	Volume(uint3 aDim);
	Volume(uint3 aDim, int aSubVolCount, int aSubVol);
	~Volume();

	int GetSubVolumeCount();
	float3 GetVoxelSize();
	float GetSubVolumeZShift(uint aIndex);
	float3 GetSubVolumePosition(uint aIndex);
	float3 GetDimension();
	float3 GetSubVolumeDimension(uint aIndex);
	size_t GetSubVolumeSizeInBytes(uint aIndex);
	size_t GetSubVolumeSizeInVoxels(uint aIndex);
	float3 GetVolumeBBoxRcp();
	float3 GetVolumeBBox();
	float3 GetVolumeBBoxMin();
	float3 GetVolumeBBoxMax();
	float3 GetSubVolumeBBox(uint aIndex);
	float3 GetSubVolumeBBoxMin(uint aIndex);
	float3 GetSubVolumeBBoxMax(uint aIndex);
	float3 GetSubVolumeBBoxRcp(uint aIndex);
	tVoxel* GetPtrToSubVolume(uint aIndex);
	void PositionInSpace(float3 aVoxelSize, float3 aShiftZ, float2 aShiftXY = make_float2(0,0));
	void PositionInSpace(float3 aVoxelSize, float aVoxelSizeClick, Volume<tVoxel>& vol, float3 aSubVolumePosition, float3 aSubVolumeShift);
	void Invert();
	void InvertMakeFloat();
	void InvertMirrorMakeFloat();
	void InvertMirror();
	tVoxel GetMax();
	void WriteToFile(std::string aFilename, int part);
	void MemsetSubVol(uint aIndex, int val);
	void LoadFromFile(std::string aFilename, int part);
	void LoadFromVolume(Volume<tVoxel>& srcVol, int part);

	void RemoveSubVolume(int x, int y, int z, int radius, tVoxel value, int part);
	void PasteSubVolume(int x, int y, int z, int radius, Volume<tVoxel>* subVol, int part);

};

#endif
