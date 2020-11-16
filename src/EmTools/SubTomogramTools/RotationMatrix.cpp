

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include "RotationMatrix.h"


RotationMatrix::RotationMatrix()
{
	_data[0][0] = 1.0f;
	_data[0][1] = 0.0f;
	_data[0][2] = 0.0f;

	_data[1][0] = 0.0f;
	_data[1][1] = 1.0f;
	_data[1][2] = 0.0f;

	_data[2][0] = 0.0f;
	_data[2][1] = 0.0f;
	_data[2][2] = 1.0f;
}


RotationMatrix::RotationMatrix(float aMatrix[3][3])
{
	_data[0][0] = aMatrix[0][0];
	_data[0][1] = aMatrix[0][1];
	_data[0][2] = aMatrix[0][2];

	_data[1][0] = aMatrix[1][0];
	_data[1][1] = aMatrix[1][1];
	_data[1][2] = aMatrix[1][2];

	_data[2][0] = aMatrix[2][0];
	_data[2][1] = aMatrix[2][1];
	_data[2][2] = aMatrix[2][2];
}

RotationMatrix::RotationMatrix(float phi, float psi, float theta)
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

	for (i = 0, j = 0; i < 16; i++)
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

	for (i = 0, j = 0; i < 16; i++)
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

	for (i = 0, j = 0; i < 16; i++)
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

	_data[0][0] = cospsi * cosphi - costheta * sinpsi * sinphi;
	_data[1][0] = sinpsi * cosphi + costheta * cospsi * sinphi;
	_data[2][0] = sintheta * sinphi;
	_data[0][1] = -cospsi * sinphi - costheta * sinpsi * cosphi;
	_data[1][1] = -sinpsi * sinphi + costheta * cospsi * cosphi;
	_data[2][1] = sintheta * cosphi;
	_data[0][2] = sintheta * sinpsi;
	_data[1][2] = -sintheta * cospsi;
	_data[2][2] = costheta;
}

RotationMatrix::RotationMatrix(const RotationMatrix& rotMat)
{
	_data[0][0] = rotMat._data[0][0];
	_data[0][1] = rotMat._data[0][1];
	_data[0][2] = rotMat._data[0][2];

	_data[1][0] = rotMat._data[1][0];
	_data[1][1] = rotMat._data[1][1];
	_data[1][2] = rotMat._data[1][2];

	_data[2][0] = rotMat._data[2][0];
	_data[2][1] = rotMat._data[2][1];
	_data[2][2] = rotMat._data[2][2];
}

void RotationMatrix::GetEulerAngles(float& phi, float& psi, float& theta)
{
	theta = acos(_data[2][2]) * 180.0f / (float)M_PI;

	if (_data[2][2] > 0.999f)
	{
		float sign = _data[1][0] > 0 ? 1.0f : -1.0f;
		phi = sign * acos(_data[0][0]) * 180.0f / (float)M_PI;
		psi = 0.0f;
	}
	else
	{
		phi = atan2(_data[2][0], _data[2][1]) * 180.0f / (float)M_PI;
		psi = atan2(_data[0][2], -_data[1][2]) * 180.0f / (float)M_PI;
	}
}

void RotationMatrix::GetData(float data[3][3])
{
	data[0][0] = _data[0][0];
	data[0][1] = _data[0][1];
	data[0][2] = _data[0][2];
	
	data[1][0] = _data[1][0];
	data[1][1] = _data[1][1];
	data[1][2] = _data[1][2];

	data[2][0] = _data[2][0];
	data[2][1] = _data[2][1];
	data[2][2] = _data[2][2];
}

float& RotationMatrix::operator()(int i, int j)
{
	return _data[i][j];
}

float RotationMatrix::operator()(int i, int j) const
{
	return _data[i][j];
}

RotationMatrix RotationMatrix::operator*(const RotationMatrix& other)
{
	RotationMatrix out;
	out._data[0][0] = _data[0][0] * other._data[0][0] + _data[1][0] * other._data[0][1] + _data[2][0] * other._data[0][2];
	out._data[1][0] = _data[0][0] * other._data[1][0] + _data[1][0] * other._data[1][1] + _data[2][0] * other._data[1][2];
	out._data[2][0] = _data[0][0] * other._data[2][0] + _data[1][0] * other._data[2][1] + _data[2][0] * other._data[2][2];
	out._data[0][1] = _data[0][1] * other._data[0][0] + _data[1][1] * other._data[0][1] + _data[2][1] * other._data[0][2];
	out._data[1][1] = _data[0][1] * other._data[1][0] + _data[1][1] * other._data[1][1] + _data[2][1] * other._data[1][2];
	out._data[2][1] = _data[0][1] * other._data[2][0] + _data[1][1] * other._data[2][1] + _data[2][1] * other._data[2][2];
	out._data[0][2] = _data[0][2] * other._data[0][0] + _data[1][2] * other._data[0][1] + _data[2][2] * other._data[0][2];
	out._data[1][2] = _data[0][2] * other._data[1][0] + _data[1][2] * other._data[1][1] + _data[2][2] * other._data[1][2];
	out._data[2][2] = _data[0][2] * other._data[2][0] + _data[1][2] * other._data[2][1] + _data[2][2] * other._data[2][2];

	return out;
}


std::ostream& operator<< (std::ostream& stream, const RotationMatrix& matrix)
{
	stream << matrix(0, 0) << " " << matrix(1, 0) << " " << matrix(2, 0) << std::endl;
	stream << matrix(0, 1) << " " << matrix(1, 1) << " " << matrix(2, 1) << std::endl;
	stream << matrix(0, 2) << " " << matrix(1, 2) << " " << matrix(2, 2) << std::endl;
	return stream;
}
