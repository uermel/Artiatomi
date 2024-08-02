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


#include "Volume.h"
#include <limits>

template<typename tVoxel>
Volume<tVoxel>::Volume(uint3 aDim, bool alloc):
    _countSubVolumes(0),
    _data(NULL),
    _positionVolume(make_float3(0)),
    _positionSubVolume(NULL),
    _dimension(aDim),
	_dimensionSubVolume(NULL),
    _voxelSize(make_float3(1)),
    _bitResolution(sizeof(tVoxel)),
    _volumeMatrix(4, 4),
    _volumeMatrixInv(4, 4),
    _halfset(0)
{
	_countSubVolumes = 1;
	int sizeZ = (int)ceil((double)aDim.z / (double)_countSubVolumes);
	int rest =  aDim.z - sizeZ * (_countSubVolumes-1);

	_dimensionSubVolume = new uint3[_countSubVolumes];
	for (int i = 0; i < _countSubVolumes; i++)
	{
		uint3 dim;
		dim.x = aDim.x;
		dim.y = aDim.y;
		dim.z = sizeZ;
		_dimensionSubVolume[i] = dim;
	}
	_dimensionSubVolume[_countSubVolumes-1].z = rest;

	_positionSubVolume = new float3[_countSubVolumes];
	_data = new tVoxel*[_countSubVolumes];

    if (alloc) {
        _data[0] = new tVoxel[GetSubVolumeSizeInVoxels(0)];
        memset(_data[0], 0, GetSubVolumeSizeInBytes(0));
    } else {
        _data[0] = NULL;
    }
}

float GetNoise()
{
	const double epsilon = std::numeric_limits<double>::min();
	const double two_pi = 2.0*3.14159265358979323846;

	static double z0, z1;
	static bool generate = false;
	generate = !generate;

	if (!generate)
	   return (float)z1;

	double u1, u2;
	do
	 {
	   u1 = rand() * (1.0 / RAND_MAX);
	   u2 = rand() * (1.0 / RAND_MAX);
	 }
	while ( u1 <= epsilon );

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return (float)z0;
}


template<typename tVoxel>
Volume<tVoxel>::Volume(uint3 aDim, int aSubVolCount, int aSubVol):
    _countSubVolumes(aSubVolCount),
    _data(NULL),
    _positionVolume(make_float3(0)),
    _positionSubVolume(NULL),
    _dimension(aDim),
	_dimensionSubVolume(NULL),
    _voxelSize(make_float3(1)),
    _bitResolution(sizeof(tVoxel)),
    _volumeMatrix(4, 4),
    _volumeMatrixInv(4, 4)
{
	int sizeZ = (int)ceil((double)aDim.z / (double)_countSubVolumes);
	int rest =  aDim.z - sizeZ * (_countSubVolumes-1);

	_dimensionSubVolume = new uint3[_countSubVolumes];
	for (int i = 0; i < _countSubVolumes; i++)
	{
		uint3 dim;
		dim.x = aDim.x;
		dim.y = aDim.y;
		dim.z = sizeZ;
		_dimensionSubVolume[i] = dim;
	}
	_dimensionSubVolume[_countSubVolumes-1].z = rest;

	_positionSubVolume = new float3[_countSubVolumes];
	_data = new tVoxel*[_countSubVolumes];
	
	for (int i = 0; i < _countSubVolumes; i++)
	{
		_data[i] = NULL;
	}

	//a negative value for aSubVol can be used as a dummy volume without storage allocation!
	if (aSubVol >= 0)
	{
		_data[aSubVol] = new tVoxel[GetSubVolumeSizeInVoxels(aSubVol)];
		memset(_data[aSubVol], 0, GetSubVolumeSizeInBytes(aSubVol));
	}
}


template<typename tVoxel>
Volume<tVoxel>::~Volume()
{
	delete[] _dimensionSubVolume;
	delete[] _positionSubVolume;
	for(int i = 0; i < _countSubVolumes; i++)
	{
		if (_data[i] != NULL)
			delete[] _data[i];
	}
	delete[] _data;
}

template<typename tVoxel>
int Volume<tVoxel>::GetSubVolumeCount()
{
	return _countSubVolumes;
}

template<typename tVoxel>
float3 Volume<tVoxel>::GetVoxelSize()
{
	return _voxelSize;
}

template<typename tVoxel>
float Volume<tVoxel>::GetSubVolumeZShift(uint aIndex)
{
	float zShiftForPartialVolume = 0.0f;
	for (unsigned int i = 0; i < aIndex; i++)
	{
		zShiftForPartialVolume += _dimensionSubVolume[i].z;
	}
	return zShiftForPartialVolume;
}

template<typename tVoxel>
float3 Volume<tVoxel>::GetSubVolumePosition(uint aIndex)
{
	return _positionSubVolume[aIndex];
}

template<typename tVoxel>
float3 Volume<tVoxel>::GetDimension()
{
	return make_float3(_dimension);
}

template<typename tVoxel>
float3 Volume<tVoxel>::GetSubVolumeDimension(uint aIndex)
{
	return make_float3(_dimensionSubVolume[aIndex]);
}

template<typename tVoxel>
size_t Volume<tVoxel>::GetSubVolumeSizeInBytes(uint aIndex)
{
	size_t size = (size_t)_dimensionSubVolume[aIndex].x *
				  (size_t)_dimensionSubVolume[aIndex].y *
				  (size_t)_dimensionSubVolume[aIndex].z *
				  (size_t)_bitResolution;
	return size;
}

template<typename tVoxel>
size_t Volume<tVoxel>::GetSubVolumeSizeInVoxels(uint aIndex)
{
	size_t size = (size_t)_dimensionSubVolume[aIndex].x *
				  (size_t)_dimensionSubVolume[aIndex].y *
				  (size_t)_dimensionSubVolume[aIndex].z;
	return size;
}

template<typename tVoxel>
float3 Volume<tVoxel>::GetVolumeBBox()
{
	float3 floatDim;
	floatDim.x = (float)_dimension.x * _voxelSize.x;
	floatDim.y = (float)_dimension.y * _voxelSize.y;
	floatDim.z = (float)_dimension.z * _voxelSize.z;

	return floatDim;
}

template<typename tVoxel>
float3 Volume<tVoxel>::GetVolumeBBoxMin()
{
	//position of the whole volume is the position of first sub-volume
	return GetSubVolumePosition(0);
}

template<typename tVoxel>
float3 Volume<tVoxel>::GetVolumeBBoxMax()
{
	float3 bBoxMin = GetVolumeBBoxMin();
	float3 bBox = GetVolumeBBox();
	return bBoxMin + bBox;
}

template<typename tVoxel>
float3 Volume<tVoxel>::GetVolumeBBoxRcp()
{
	float3 floatDim;
	floatDim.x = 1.0f / ((float)_dimension.x * _voxelSize.x);
	floatDim.y = 1.0f / ((float)_dimension.y * _voxelSize.y);
	floatDim.z = 1.0f / ((float)_dimension.z * _voxelSize.z);

	return floatDim;
}

template<typename tVoxel>
float3 Volume<tVoxel>::GetSubVolumeBBox(uint aIndex)
{
	float3 floatDim;
	floatDim.x = (float)_dimensionSubVolume[aIndex].x * _voxelSize.x;
	floatDim.y = (float)_dimensionSubVolume[aIndex].y * _voxelSize.y;
	floatDim.z = (float)_dimensionSubVolume[aIndex].z * _voxelSize.z;

	return floatDim;
}

template<typename tVoxel>
float3 Volume<tVoxel>::GetSubVolumeBBoxRcp(uint aIndex)
{
	float3 floatDim;
	floatDim.x = 1.0f / ((float)_dimensionSubVolume[aIndex].x * _voxelSize.x);
	floatDim.y = 1.0f / ((float)_dimensionSubVolume[aIndex].y * _voxelSize.y);
	floatDim.z = 1.0f / ((float)_dimensionSubVolume[aIndex].z * _voxelSize.z);

	return floatDim;
}

template<typename tVoxel>
float3 Volume<tVoxel>::GetSubVolumeBBoxMin(uint aIndex)
{
	return GetSubVolumePosition(aIndex);
}

template<typename tVoxel>
float3 Volume<tVoxel>::GetSubVolumeBBoxMax(uint aIndex)
{
	float3 bBoxMin = GetSubVolumeBBoxMin(aIndex);
	float3 bBox = GetSubVolumeBBox(aIndex);
	return bBoxMin + bBox;
}

template<typename tVoxel>
tVoxel* Volume<tVoxel>::GetPtrToSubVolume(uint aIndex)
{
	return _data[aIndex];
}

template<typename tVoxel>
void Volume<tVoxel>::PositionInSpace(float3 aVoxelSize, float3 aVolShift, float2 aShiftXY)
{
	_voxelSize = aVoxelSize;

	_positionVolume.x = _voxelSize.x * (-0.5f * _dimension.x) + aShiftXY.x - aVolShift.x;
	// The projection is flipped in y direction, so change sign!
	_positionVolume.y = _voxelSize.y * (-0.5f * _dimension.y) + aShiftXY.y - aVolShift.y;
	_positionVolume.z = _voxelSize.z * (-0.5f * _dimension.z) - aVolShift.z;

	int old_volumeParts = 0;
	for (int i = 0; i < _countSubVolumes; i++)
	{
		_positionSubVolume[i] = _positionVolume;
		_positionSubVolume[i].z += _voxelSize.z * old_volumeParts;
		old_volumeParts += _dimensionSubVolume[i].z;
	}
}

template<typename tVoxel>
void Volume<tVoxel>::PositionInSpace(float3 aVoxelSize, float3 aVolShift, float2 aShiftXY, float phi, float theta, float psi)
{
    // Voxelsize
    Matrix<double> Mscale = Matrix<double>::AffineScale3D(aVoxelSize.x, aVoxelSize.y, aVoxelSize.z);

    // Center
    Matrix<double> Mcenter = Matrix<double>::AffineShift3D(-0.5 * aVoxelSize.x *_dimension.x,
                                                           -0.5 * aVoxelSize.y *_dimension.y,
                                                           -0.5 * aVoxelSize.z *_dimension.z);

    // Rotation
    Matrix<double> Mphi = Matrix<double>::AffineRotation3DZ(DEG2RAD(phi));
    Matrix<double> Mtheta = Matrix<double>::AffineRotation3DX(DEG2RAD(theta));
    Matrix<double> Mpsi = Matrix<double>::AffineRotation3DZ(DEG2RAD(psi));

    // ShiftXY
    Matrix<double> MshiftXY = Matrix<double>::AffineShift3D(aShiftXY.x,
                                                            aShiftXY.y,
                                                            0);

    // Shift
    Matrix<double> Mshift = Matrix<double>::AffineShift3D(-aVolShift.x,
                                                          -aVolShift.y,
                                                          -aVolShift.z);

    // Final
    Matrix<double> Mvol = (Mshift * (MshiftXY * (Mpsi * (Mtheta * (Mphi * (Mcenter * Mscale))))));

    _volumeMatrix = Mvol;
    _volumeMatrixInv = GetFromSpace(Mvol);

    _voxelSize = aVoxelSize;

    _positionVolume.x = _voxelSize.x * (-0.5f * _dimension.x) + aShiftXY.x - aVolShift.x;
    // The projection is flipped in y direction, so change sign!
    _positionVolume.y = _voxelSize.y * (-0.5f * _dimension.y) + aShiftXY.y - aVolShift.y;
    _positionVolume.z = _voxelSize.z * (-0.5f * _dimension.z) - aVolShift.z;

    int old_volumeParts = 0;
    for (int i = 0; i < _countSubVolumes; i++)
    {
        _positionSubVolume[i] = _positionVolume;
        _positionSubVolume[i].z += _voxelSize.z * old_volumeParts;
        old_volumeParts += _dimensionSubVolume[i].z;
    }
}

template<typename tVoxel>
void Volume<tVoxel>::PositionInSpace(Volume<tVoxel>* parentVol,
                                     float3 aVoxelSize,
                                     float3 aVolShift,
                                     motive particle)
{
    // Halfset
    _halfset = (int)particle.partNr - 1;

    // Pixel coordinates in parent vol to global coords
    // Parent transform
    Matrix<double> Mparent = parentVol->VolumeMatrix();

    // Pixel coords
    Matrix<double> partPos(4, 1);
    partPos(0,0) = particle.x_Coord - 1;
    partPos(1,0) = particle.y_Coord - 1;
    partPos(2,0) = particle.z_Coord - 1;
    partPos(3,0) = 1;

    // Global coords
    partPos = Mparent * partPos;

    // Voxelsize
    Matrix<double> Mscale = Matrix<double>::AffineScale3D(aVoxelSize.x,
                                                          aVoxelSize.y,
                                                          aVoxelSize.z);

    // Center
    Matrix<double> Mcenter = Matrix<double>::AffineShift3D(-0.5 * aVoxelSize.x *_dimension.x,
                                                           -0.5 * aVoxelSize.y *_dimension.y,
                                                           -0.5 * aVoxelSize.z *_dimension.z);

    // Rotation
    Matrix<double> Mphi = Matrix<double>::AffineRotation3DZ(DEG2RAD(particle.phi));
    Matrix<double> Mtheta = Matrix<double>::AffineRotation3DX(DEG2RAD(particle.theta));
    Matrix<double> Mpsi = Matrix<double>::AffineRotation3DZ(DEG2RAD(particle.psi));

    // ShiftXY
    Matrix<double> MshiftXY = Matrix<double>::AffineShift3D(0,0,0);

    // Shift
    Matrix<double> Mshift = Matrix<double>::AffineShift3D(-aVolShift.x + partPos(0, 0),
                                                          -aVolShift.y + partPos(1, 0),
                                                          -aVolShift.z + partPos(2, 0));

    // Final
    Matrix<double> Mvol = (Mshift * (MshiftXY * (Mpsi * (Mtheta * (Mphi * (Mcenter * Mscale))))));

    _volumeMatrix = Mvol;
    _volumeMatrixInv = GetFromSpace(Mvol);

    _voxelSize = aVoxelSize;

    _positionVolume.x = _voxelSize.x * (-0.5f * _dimension.x) + 0 - aVolShift.x;
    // The projection is flipped in y direction, so change sign!
    _positionVolume.y = _voxelSize.y * (-0.5f * _dimension.y) + 0 - aVolShift.y;
    _positionVolume.z = _voxelSize.z * (-0.5f * _dimension.z) - aVolShift.z;

    int old_volumeParts = 0;
    for (int i = 0; i < _countSubVolumes; i++)
    {
        _positionSubVolume[i] = _positionVolume;
        _positionSubVolume[i].z += _voxelSize.z * old_volumeParts;
        old_volumeParts += _dimensionSubVolume[i].z;
    }
}

template<typename tVoxel>
Matrix<double> Volume<tVoxel>::GetFromSpace(Matrix<double> matrix)
{
    Matrix<double> ret(4, 4);
    ////////////////////////////////////////////////////////////////////////////////
    //from http://www.geometrictools.com//LibFoundation/Mathematics/Wm4Matrix4.inl:
    ////////////////////////////////////////////////////////////////////////////////

    double m_afEntry[16];
    double aMatrix[16];
    m_afEntry[0] =  matrix(0, 0);
    m_afEntry[1] =  matrix(1, 0);
    m_afEntry[2] =  matrix(2, 0);
    m_afEntry[3] =  matrix(3, 0);
    m_afEntry[4] =  matrix(0, 1);
    m_afEntry[5] =  matrix(1, 1);
    m_afEntry[6] =  matrix(2, 1);
    m_afEntry[7] =  matrix(3, 1);
    m_afEntry[8] =  matrix(0, 2);
    m_afEntry[9] =  matrix(1, 2);
    m_afEntry[10] = matrix(2, 2);
    m_afEntry[11] = matrix(3, 2);
    m_afEntry[12] = matrix(0, 3);
    m_afEntry[13] = matrix(1, 3);
    m_afEntry[14] = matrix(2, 3);
    m_afEntry[15] = matrix(3, 3);


    double fA0 = m_afEntry[ 0]*m_afEntry[ 5] - m_afEntry[ 1]*m_afEntry[ 4];
    double fA1 = m_afEntry[ 0]*m_afEntry[ 6] - m_afEntry[ 2]*m_afEntry[ 4];
    double fA2 = m_afEntry[ 0]*m_afEntry[ 7] - m_afEntry[ 3]*m_afEntry[ 4];
    double fA3 = m_afEntry[ 1]*m_afEntry[ 6] - m_afEntry[ 2]*m_afEntry[ 5];
    double fA4 = m_afEntry[ 1]*m_afEntry[ 7] - m_afEntry[ 3]*m_afEntry[ 5];
    double fA5 = m_afEntry[ 2]*m_afEntry[ 7] - m_afEntry[ 3]*m_afEntry[ 6];
    double fB0 = m_afEntry[ 8]*m_afEntry[13] - m_afEntry[ 9]*m_afEntry[12];
    double fB1 = m_afEntry[ 8]*m_afEntry[14] - m_afEntry[10]*m_afEntry[12];
    double fB2 = m_afEntry[ 8]*m_afEntry[15] - m_afEntry[11]*m_afEntry[12];
    double fB3 = m_afEntry[ 9]*m_afEntry[14] - m_afEntry[10]*m_afEntry[13];
    double fB4 = m_afEntry[ 9]*m_afEntry[15] - m_afEntry[11]*m_afEntry[13];
    double fB5 = m_afEntry[10]*m_afEntry[15] - m_afEntry[11]*m_afEntry[14];

    double fDet = fA0*fB5-fA1*fB4+fA2*fB3+fA3*fB2-fA4*fB1+fA5*fB0;
    if (fDet == 0)
    {
        printf("Determinante of detector matrix is not 0! Can't inverse matrix.\n");
        exit(1);
    }

    aMatrix[ 0] =
            + m_afEntry[ 5]*fB5 - m_afEntry[ 6]*fB4 + m_afEntry[ 7]*fB3;
    aMatrix[ 1] =
            - m_afEntry[ 4]*fB5 + m_afEntry[ 6]*fB2 - m_afEntry[ 7]*fB1;
    aMatrix[ 2] =
            + m_afEntry[ 4]*fB4 - m_afEntry[ 5]*fB2 + m_afEntry[ 7]*fB0;
    aMatrix[ 3] =
            - m_afEntry[ 4]*fB3 + m_afEntry[ 5]*fB1 - m_afEntry[ 6]*fB0;
    aMatrix[ 4] =
            - m_afEntry[ 1]*fB5 + m_afEntry[ 2]*fB4 - m_afEntry[ 3]*fB3;
    aMatrix[ 5] =
            + m_afEntry[ 0]*fB5 - m_afEntry[ 2]*fB2 + m_afEntry[ 3]*fB1;
    aMatrix[ 6] =
            - m_afEntry[ 0]*fB4 + m_afEntry[ 1]*fB2 - m_afEntry[ 3]*fB0;
    aMatrix[ 7] =
            + m_afEntry[ 0]*fB3 - m_afEntry[ 1]*fB1 + m_afEntry[ 2]*fB0;
    aMatrix[ 8] =
            + m_afEntry[13]*fA5 - m_afEntry[14]*fA4 + m_afEntry[15]*fA3;
    aMatrix[ 9] =
            - m_afEntry[12]*fA5 + m_afEntry[14]*fA2 - m_afEntry[15]*fA1;
    aMatrix[10] =
            + m_afEntry[12]*fA4 - m_afEntry[13]*fA2 + m_afEntry[15]*fA0;
    aMatrix[11] =
            - m_afEntry[12]*fA3 + m_afEntry[13]*fA1 - m_afEntry[14]*fA0;
    aMatrix[12] =
            - m_afEntry[ 9]*fA5 + m_afEntry[10]*fA4 - m_afEntry[11]*fA3;
    aMatrix[13] =
            + m_afEntry[ 8]*fA5 - m_afEntry[10]*fA2 + m_afEntry[11]*fA1;
    aMatrix[14] =
            - m_afEntry[ 8]*fA4 + m_afEntry[ 9]*fA2 - m_afEntry[11]*fA0;
    aMatrix[15] =
            + m_afEntry[ 8]*fA3 - m_afEntry[ 9]*fA1 + m_afEntry[10]*fA0;

    double fInvDet = (1.0f)/fDet;
    aMatrix[ 0] *= fInvDet;
    aMatrix[ 1] *= fInvDet;
    aMatrix[ 2] *= fInvDet;
    aMatrix[ 3] *= fInvDet;
    aMatrix[ 4] *= fInvDet;
    aMatrix[ 5] *= fInvDet;
    aMatrix[ 6] *= fInvDet;
    aMatrix[ 7] *= fInvDet;
    aMatrix[ 8] *= fInvDet;
    aMatrix[ 9] *= fInvDet;
    aMatrix[10] *= fInvDet;
    aMatrix[11] *= fInvDet;
    aMatrix[12] *= fInvDet;
    aMatrix[13] *= fInvDet;
    aMatrix[14] *= fInvDet;
    aMatrix[15] *= fInvDet;

    ret(0, 0) = aMatrix[ 0];
    ret(0, 1) = aMatrix[ 1];
    ret(0, 2) = aMatrix[ 2];
    ret(0, 3) = aMatrix[ 3];
    ret(1, 0) = aMatrix[ 4];
    ret(1, 1) = aMatrix[ 5];
    ret(1, 2) = aMatrix[ 6];
    ret(1, 3) = aMatrix[ 7];
    ret(2, 0) = aMatrix[ 8];
    ret(2, 1) = aMatrix[ 9];
    ret(2, 2) = aMatrix[10];
    ret(2, 3) = aMatrix[11];
    ret(3, 0) = aMatrix[12];
    ret(3, 1) = aMatrix[13];
    ret(3, 2) = aMatrix[14];
    ret(3, 3) = aMatrix[15];

    return ret;
}

template<typename tVoxel>
int Volume<tVoxel>::GetHalfSet()
{
    return _halfset;
}


template<typename tVoxel>
Matrix<double> Volume<tVoxel>::VolumeMatrix()
{
    return _volumeMatrix;
}

template<typename tVoxel>
Matrix<double> Volume<tVoxel>::VolumeMatrixNorm()
{
    return _volumeMatrix * (1/GetVoxelSize().x);
}


template<typename tVoxel>
Matrix<double> Volume<tVoxel>::VolumeMatrixInv()
{
    return _volumeMatrixInv;
}

template<typename tVoxel>
Matrix<double> Volume<tVoxel>::VolumeMatrixInvNorm()
{
    return _volumeMatrixInv * GetVoxelSize().x;
}

template<typename tVoxel>
Matrix<double> Volume<tVoxel>::GetCorners()
{
    // Result
    Matrix<double> corn(4, 8);

    // 8 corners
    for (int x = 0; x <= 1; x++){
        for (int y = 0; y <= 1; y++){
            for (int z = 0; z <= 1; z++){
                // Index
                int idx = x + y*2 + z*4;

                // Voxel corners
                corn(0, idx) = _dimension.x * x;
                corn(1, idx) = _dimension.y * y;
                corn(2, idx) = _dimension.z * z;
                corn(3, idx) = 1;
            }
        }
    }

    // To global frame
    corn = _volumeMatrix * corn;

    return corn;
}

template<typename tVoxel>
Matrix<double> Volume<tVoxel>::GetSubVolumeCorners(uint aIndex)
{
    // Result
    Matrix<double> corn(4, 8);

    // 8 corners
    for (int x = 0; x <= 1; x++){
        for (int y = 0; y <= 1; y++){
            for (int z = 0; z <= 1; z++){
                // Index
                int idx = x + y*2 + z*4;

                // Voxel corners
                corn(0, idx) = _dimensionSubVolume[aIndex].x * x;
                corn(1, idx) = _dimensionSubVolume[aIndex].y * y;
                corn(2, idx) = _dimensionSubVolume[aIndex].z * z;
                corn(3, idx) = 1;
            }
        }
    }

    // To global frame
    //TODO: implement
    //corn = _volumeMatrixSubVolume[aIndex] * corn;

    return corn;
}


template<typename tVoxel>
void Volume<tVoxel>::PositionInSpace(float3 aVoxelSize, float aVoxelSizeSubVolume, Volume<tVoxel>& vol, float3 aSubVolumePosition, float3 aSubVolumeShift)
{
	_voxelSize = make_float3(aVoxelSizeSubVolume, aVoxelSizeSubVolume, aVoxelSizeSubVolume);

	//Motivelist is scaled to entire volume voxelsize.
	//position -1 for compensaiting numbering from 1!
	_positionVolume.x = aVoxelSizeSubVolume * (-0.5f * _dimension.x) + aVoxelSize.x * (aSubVolumePosition.x - 1) + aSubVolumeShift.x * aVoxelSize.x + vol.GetVolumeBBoxMin().x;
	_positionVolume.y = aVoxelSizeSubVolume * (-0.5f * _dimension.y) + aVoxelSize.y * (aSubVolumePosition.y - 1) + aSubVolumeShift.y * aVoxelSize.y + vol.GetVolumeBBoxMin().y;
	_positionVolume.z = aVoxelSizeSubVolume * (-0.5f * _dimension.z) + aVoxelSize.z * (aSubVolumePosition.z - 1) + aSubVolumeShift.z * aVoxelSize.z + vol.GetVolumeBBoxMin().z;

	int old_volumeParts = 0;
	for (int i = 0; i < _countSubVolumes; i++)
	{
		_positionSubVolume[i] = _positionVolume;
		_positionSubVolume[i].z += aVoxelSize.z * old_volumeParts;
		old_volumeParts += _dimensionSubVolume[i].z;
	}
}

template<typename tVoxel>
void Volume<tVoxel>::Invert()
{
    /*tVoxel min = 64535;
    tVoxel max = -64535;*/

//	for (int i = 0; i < _countSubVolumes; i++)
//	{
//		tVoxel* voxels = _data[i];
//
//		for (size_t v = 0; v < (size_t)_dimensionSubVolume[i].x * (size_t)_dimensionSubVolume[i].y * (size_t)_dimensionSubVolume[i].z; v++)
//		{
//		    min = std::min(min, voxels[v]);
//		    max = std::max(max, voxels[v]);
//        }
//	}
//
//	printf("Volume max: %f, min: %f ... ", max, min);

	for (int i = 0; i < _countSubVolumes; i++)
	{
		tVoxel* voxels = _data[i];
		if (voxels != NULL)
		{
			for (size_t v = 0; v < (size_t)_dimensionSubVolume[i].x * (size_t)_dimensionSubVolume[i].y * (size_t)_dimensionSubVolume[i].z; v++)
			{
				voxels[v] = -voxels[v];
			}
		}
	}

}

template<typename tVoxel>
void Volume<tVoxel>::InvertMakeFloat()
{
    /*tVoxel min = 64535;
    tVoxel max = 0;

	for (int i = 0; i < _countSubVolumes; i++)
	{
		tVoxel* voxels = _data[i];

		for (size_t v = 0; v < (size_t)_dimensionSubVolume[i].x * (size_t)_dimensionSubVolume[i].y * (size_t)_dimensionSubVolume[i].z; v++)
		{
		    min = std::min(min, voxels[v]);
		    max = std::max(max, voxels[v]);
        }
	}

	printf("Volume max: %i, min: %i\n", (int)max, (int)min);

	for (int i = 0; i < _countSubVolumes; i++)
	{
		tVoxel* voxels = _data[i];
		float* voxelsf = new float[(size_t)_dimensionSubVolume[i].x * (size_t)_dimensionSubVolume[i].y * (size_t)_dimensionSubVolume[i].z];


		for (size_t v = 0; v < (size_t)_dimensionSubVolume[i].x * (size_t)_dimensionSubVolume[i].y * (size_t)_dimensionSubVolume[i].z; v++)
		{
		    voxelsf[v] = float(max - voxels[v]);
        }
        delete[] voxels;
        _data[i] = (tVoxel*)voxelsf;
	}*/

}

template<typename tVoxel>
void Volume<tVoxel>::InvertMirrorMakeFloat()
{
    tVoxel min = 64535;
    tVoxel max = 0;

	for (int i = 0; i < _countSubVolumes; i++)
	{
		tVoxel* voxels = _data[i];

		for (size_t v = 0; v < (size_t)_dimensionSubVolume[i].x * (size_t)_dimensionSubVolume[i].y * (size_t)_dimensionSubVolume[i].z; v++)
		{
		    /*min = std::min(min, voxels[v]);
		    max = std::max(max, voxels[v]);*/
		    //max = 1078;
        }
	}

	printf("Volume max: %i, min: %i\n", (int)max, (int)min);

	for (int i = 0; i < _countSubVolumes; i++)
	{
		tVoxel* voxelsOld = _data[i];
        float* voxels = new float[(size_t)_dimensionSubVolume[i].x * (size_t)_dimensionSubVolume[i].y * (size_t)_dimensionSubVolume[i].z];


        for (size_t z = 0; z < (size_t)_dimensionSubVolume[i].z; z++)
        for (size_t y = 0; y < (size_t)_dimensionSubVolume[i].y; y++)
        for (size_t x = 0; x < (size_t)_dimensionSubVolume[i].x; x++)
        {
            size_t iIn  = z * (size_t)_dimensionSubVolume[i].x  * (size_t)_dimensionSubVolume[i].y  + y * (size_t)_dimensionSubVolume[i].x  + x;
            size_t iOut  = (_dimensionSubVolume[i].z-z-1) * (size_t)_dimensionSubVolume[i].x  * (size_t)_dimensionSubVolume[i].y  + (_dimensionSubVolume[i].y-y-1) * (size_t)_dimensionSubVolume[i].x  + (_dimensionSubVolume[i].x-x-1);

            voxels[iOut] = (float)(max - voxelsOld[iIn]);
        }
        delete[] voxelsOld;
        _data[i] = (tVoxel*)voxels;
	}

}

template<typename tVoxel>
void Volume<tVoxel>::InvertMirror()
{
    tVoxel min = 64535;
    tVoxel max = 0;

	for (int i = 0; i < _countSubVolumes; i++)
	{
		tVoxel* voxels = _data[i];

		for (size_t v = 0; v < (size_t)_dimensionSubVolume[i].x * (size_t)_dimensionSubVolume[i].y * (size_t)_dimensionSubVolume[i].z; v++)
		{
		    /*min = std::min(min, voxels[v]);
		    max = std::max(max, voxels[v]);*/
		    //max = 1078;
        }
	}

	printf("Volume max: %i, min: %i\n", (int)max, (int)min);

	for (int i = 0; i < _countSubVolumes; i++)
	{
		tVoxel* voxelsOld = _data[i];
        tVoxel* voxels = new tVoxel[(size_t)_dimensionSubVolume[i].x * (size_t)_dimensionSubVolume[i].y * (size_t)_dimensionSubVolume[i].z];


        for (size_t z = 0; z < (size_t)_dimensionSubVolume[i].z; z++)
        for (size_t y = 0; y < (size_t)_dimensionSubVolume[i].y; y++)
        for (size_t x = 0; x < (size_t)_dimensionSubVolume[i].x; x++)
        {
            size_t iIn  = z * (size_t)_dimensionSubVolume[i].x  * (size_t)_dimensionSubVolume[i].y  + y * (size_t)_dimensionSubVolume[i].x  + x;
            size_t iOut  = (_dimensionSubVolume[i].z-z-1) * (size_t)_dimensionSubVolume[i].x  * (size_t)_dimensionSubVolume[i].y  + (_dimensionSubVolume[i].y-y-1) * (size_t)_dimensionSubVolume[i].x  + (_dimensionSubVolume[i].x-x-1);

            voxels[iOut] = (tVoxel)(max - voxelsOld[iIn]);
        }
        _data[i] = (tVoxel*)voxels;
        delete[] voxelsOld;
	}

}

template<typename tVoxel>
tVoxel Volume<tVoxel>::GetMax()
{
    tVoxel max = -999999999;

	for (int i = 0; i < _countSubVolumes; i++)
	{
		tVoxel* voxels = _data[i];
		if (voxels != NULL)
			for (size_t v = 0; v < (size_t)_dimensionSubVolume[i].x * (size_t)_dimensionSubVolume[i].y * (size_t)_dimensionSubVolume[i].z; v++)
			{
				/*max = std::max(max, voxels[v]);*/
			}
	}
    return max;
}

template<typename tVoxel>
void Volume<tVoxel>::WriteToFile(std::string aFilename, int part)
{
    std::ofstream* mVol = new std::ofstream();
	mVol->open(aFilename.c_str(), std::ios_base::out | std::ios_base::binary | std::ios_base::app);
	if (!(mVol->is_open() && mVol->good()))
	{
		delete mVol;
		printf("Cannot open file (%s) for writing!\n", aFilename.c_str());fflush(stdout);
	}
	else
	{
		size_t sizeDataType = sizeof(tVoxel);
		float3 dim = GetSubVolumeDimension(part);
		size_t dimI = (size_t)dim.x * (size_t)dim.y * (size_t)dim.z * sizeDataType;
		mVol->write((char*)GetPtrToSubVolume(part), dimI);
		mVol->flush();
		mVol->close();
		delete mVol;
	}
}

template<typename tVoxel>
void Volume<tVoxel>::LoadFromFile(std::string aFilename, int part)
{
	std::ifstream* mVol = new std::ifstream();
	mVol->open(aFilename.c_str(), std::ios_base::in | std::ios_base::binary);
	if (!(mVol->is_open() && mVol->good()))
	{
		printf("Cannot open file (%s) for reading!\n", aFilename.c_str()); fflush(stdout);
		delete mVol;
	}
	else
	{
		if (GetPtrToSubVolume(part) == NULL)
		{
			printf("SubVol is ZERO!!!");
			fflush(stdout);
			return;
		}
		
		char header[512];
		mVol->read(header, 512);
		int pos = mVol->tellg();
		printf("\nPos = %d, countSubVol %d", pos, _countSubVolumes);
		fflush(stdout);
		size_t sizeDataType = sizeof(tVoxel);

		size_t skip = 0;
		for (size_t i = 0; i < part; i++)
		{
			float3 dimSkip = GetSubVolumeDimension(i);
			size_t dimISkip = (size_t)dimSkip.x * (size_t)dimSkip.y * (size_t)dimSkip.z * sizeDataType;
			//printf("Dims: %f, %f, %f\n", dimSkip.x, dimSkip.y, dimSkip.z);
			skip += dimISkip;
		}

		mVol->seekg(skip, std::ios::cur);
		float3 dim = GetSubVolumeDimension(part);
		size_t dimI = (size_t)dim.x * (size_t)dim.y * (size_t)dim.z * sizeDataType;
		mVol->read((char*)GetPtrToSubVolume(part), dimI);
		mVol->close();
		delete mVol;

		tVoxel* p = GetPtrToSubVolume(part);
		for (size_t i = 0; i < (size_t)dim.x * (size_t)dim.y * (size_t)dim.z; i++)
		{
			p[i] = -p[i];
		}
	}
}

template<typename tVoxel>
void Volume<tVoxel>::LoadFromVolume(Volume<tVoxel>& srcVol, int part)
{
	if (srcVol.GetSubVolumeCount() != GetSubVolumeCount() || part >= GetSubVolumeCount())
		return;

	float3 thisSize = GetSubVolumeDimension(part);
	float3 srcSize = srcVol.GetSubVolumeDimension(part);

	if (thisSize.x != srcSize.x || thisSize.y != srcSize.y || thisSize.z != srcSize.z)
		return;

	size_t totalSizeInBytes = GetSubVolumeSizeInBytes(part);

	memcpy(GetPtrToSubVolume(part), srcVol.GetPtrToSubVolume(part), totalSizeInBytes);
}

template<typename tVoxel>
void Volume<tVoxel>::RemoveSubVolume(int x, int y, int z, int radius, tVoxel value, int part)
{
	int xStart = x - 1 - radius;
	int yStart = y - 1 - radius;
	int zStart = z - 1 - radius;

	int xEnd = xStart + 2 * radius;
	int yEnd = yStart + 2 * radius;
	int zEnd = zStart + 2 * radius;

	if (part >= _countSubVolumes)
		return;

	int zMin = 0;
	int zMax = _dimensionSubVolume[0].z;
	for (int i = 1; i <= part; i++)
	{
		zMin += _dimensionSubVolume[i].z;
		zMax += _dimensionSubVolume[i].z;
	}

	//particle is not in sub-volume
	if (zStart >= zMax || zEnd <= zMin)
		return;


	int zStartInPartVol = zStart;
	int zEndInPartVol = zEnd;
	for (int i = 1; i <= part; i++)
	{
		int dimZ = _dimensionSubVolume[i].z;
		zStartInPartVol -= dimZ;
		zEndInPartVol -= dimZ;
	}

	if (zEndInPartVol > _dimensionSubVolume[part].z)
	{
		zEndInPartVol = _dimensionSubVolume[part].z;
	}

	if (zStartInPartVol < 0)
	{
		zStartInPartVol = 0;
	}

	if (yEnd > _dimensionSubVolume[part].y)
	{
		yEnd = _dimensionSubVolume[part].y;
	}

	if (yStart < 0)
	{
		yStart = 0;
	}

	if (xEnd > _dimensionSubVolume[part].x)
	{
		xEnd = _dimensionSubVolume[part].x;
	}

	if (xStart < 0)
	{
		xStart = 0;
	}

	//printf("part: %d, xStart: %d, xEnd: %d, yStart: %d, yEnd: %d, zStart: %d, zEnd: %d\n", part, zStart, zEnd, yStart, yEnd, zStartInPartVol, zEndInPartVol); fflush(stdout);

	size_t sizeSubVol = 2 * radius;
	tVoxel* ptr = GetPtrToSubVolume(part);

	for (size_t z = zStartInPartVol; z < zEndInPartVol; z++)
	{
		for (size_t y = yStart; y < yEnd; y++)
		{
			for (size_t x = xStart; x < xEnd; x++)
			{
				size_t idx = z * _dimensionSubVolume[part].x * _dimensionSubVolume[part].y
					+ y * _dimensionSubVolume[part].x + x;
				
				ptr[idx] = value;
			}
		}
	}
}

template<typename tVoxel>
void Volume<tVoxel>::PasteSubVolume(int x, int y, int z, int radius, Volume<tVoxel>* subVol, int part)
{
	size_t xStart = x - 1 - radius;
	size_t yStart = y - 1 - radius;
	size_t zStart = z - 1 - radius;

	size_t xEnd = xStart + 2 * radius;
	size_t yEnd = yStart + 2 * radius;
	size_t zEnd = zStart + 2 * radius;

	if (part >= _countSubVolumes)
		return;

	size_t zStartInPartVol = zStart;
	size_t zEndInPartVol = zEnd;
	for (size_t i = 0; i < part; i++)
	{
		size_t dimZ = _dimensionSubVolume[i].z;
		zStartInPartVol -= dimZ;
		zEndInPartVol -= dimZ;
	}

	if (zEndInPartVol > _dimensionSubVolume[part].z)
	{
		zEndInPartVol = _dimensionSubVolume[part].z;
	}

	if (zStartInPartVol < 0)
	{
		zStartInPartVol = 0;
	}

	if (yEnd > _dimensionSubVolume[part].y)
	{
		yEnd = _dimensionSubVolume[part].y;
	}

	if (yStart < 0)
	{
		yStart = 0;
	}

	if (xEnd > _dimensionSubVolume[part].x)
	{
		xEnd = _dimensionSubVolume[part].x;
	}

	if (xStart < 0)
	{
		xStart = 0;
	}

	size_t sizeSubVol = 2 * radius;
	tVoxel* d = subVol->GetPtrToSubVolume(0); //subVolumes are not sliced!
	tVoxel* ptr = GetPtrToSubVolume(part);

	for (size_t z = zStart; z < zEnd; z++)
	{
		for (size_t y = yStart; y < yEnd; y++)
		{
			for (size_t x = xStart; x < xEnd; x++)
			{
				size_t idx = z * _dimensionSubVolume[part].x * _dimensionSubVolume[part].y
					+ y * _dimensionSubVolume[part].x + x;

				size_t idxSubVol = (z - zStart) * sizeSubVol * sizeSubVol + (y - yStart) * sizeSubVol + (x - xStart);

				ptr[idx] += d[idxSubVol];
			}
		}
	}
}

template<typename tVoxel>
void Volume<tVoxel>::MemsetSubVol(uint aIndex, int val)
{
	memset(_data[aIndex], val, (size_t)_dimensionSubVolume[aIndex].x * (size_t)_dimensionSubVolume[aIndex].y * (size_t)_dimensionSubVolume[aIndex].z * (size_t)_bitResolution);
}

template class Volume<float>;
template class Volume<unsigned short>;
