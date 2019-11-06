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


#include "Matrix.h"

using namespace std;

#pragma region Constructors

template<class T>
Matrix<T>::Matrix(uint aRows, uint aColumns)
	: rows(aRows), cols(aColumns)
{
	values = new T[rows * cols];
	memset(values, 0, sizeof(T) * rows * cols);
}

template<class T>
Matrix<T>::Matrix(const Matrix& aCopy)
	: rows(aCopy.rows), cols(aCopy.cols)
{
    size_t size = rows * cols;
	values = new T[size];
	memcpy(values, aCopy.values, sizeof(T) * rows * cols);
}

//template<class T>
//Matrix<T>::Matrix (float4 aVals, MatrixVectorType_enum aType)
//{
//	switch (aType)
//	{
//	case MVT_ColumnVector:
//		cols = 1;
//		rows = 4;
//		values = new T[cols * rows];
//		values[0] = (T)aVals.x;
//		values[1] = (T)aVals.y;
//		values[2] = (T)aVals.z;
//		values[3] = (T)aVals.w;
//		break;
//	case MVT_RowVector:
//		cols = 4;
//		rows = 1;
//		values = new T[cols * rows];
//		values[0] = (T)aVals.x;
//		values[1] = (T)aVals.y;
//		values[2] = (T)aVals.z;
//		values[3] = (T)aVals.w;
//		break;
//	}
//}
//
//template<class T>
//Matrix<T>::Matrix (float3 aVals, MatrixVectorType_enum aType)
//{
//	switch (aType)
//	{
//	case MVT_ColumnVector:
//		cols = 1;
//		rows = 3;
//		values = new T[cols * rows];
//		values[0] = (T)aVals.x;
//		values[1] = (T)aVals.y;
//		values[2] = (T)aVals.z;
//		break;
//	case MVT_RowVector:
//		cols = 3;
//		rows = 1;
//		values = new T[cols * rows];
//		values[0] = (T)aVals.x;
//		values[1] = (T)aVals.y;
//		values[2] = (T)aVals.z;
//		break;
//	}
//}
//
//template<class T>
//Matrix<T>::Matrix (float2 aVals, MatrixVectorType_enum aType)
//{
//	switch (aType)
//	{
//	case MVT_ColumnVector:
//		cols = 1;
//		rows = 2;
//		values = new T[cols * rows];
//		values[0] = (T)aVals.x;
//		values[1] = (T)aVals.y;
//		break;
//	case MVT_RowVector:
//		cols = 2;
//		rows = 1;
//		values = new T[cols * rows];
//		values[0] = (T)aVals.x;
//		values[1] = (T)aVals.y;
//		break;
//	}
//}
//
//template<class T>
//Matrix<T>::Matrix (int4 aVals, MatrixVectorType_enum aType)
//{
//	switch (aType)
//	{
//	case MVT_ColumnVector:
//		cols = 1;
//		rows = 4;
//		values = new T[cols * rows];
//		values[0] = (T)aVals.x;
//		values[1] = (T)aVals.y;
//		values[2] = (T)aVals.z;
//		values[3] = (T)aVals.w;
//		break;
//	case MVT_RowVector:
//		cols = 4;
//		rows = 1;
//		values = new T[cols * rows];
//		values[0] = (T)aVals.x;
//		values[1] = (T)aVals.y;
//		values[2] = (T)aVals.z;
//		values[3] = (T)aVals.w;
//		break;
//	}
//}
//
//template<class T>
//Matrix<T>::Matrix (int3 aVals, MatrixVectorType_enum aType)
//{
//	switch (aType)
//	{
//	case MVT_ColumnVector:
//		cols = 1;
//		rows = 3;
//		values = new T[cols * rows];
//		values[0] = (T)aVals.x;
//		values[1] = (T)aVals.y;
//		values[2] = (T)aVals.z;
//		break;
//	case MVT_RowVector:
//		cols = 3;
//		rows = 1;
//		values = new T[cols * rows];
//		values[0] = (T)aVals.x;
//		values[1] = (T)aVals.y;
//		values[2] = (T)aVals.z;
//		break;
//	}
//}
//
//template<class T>
//Matrix<T>::Matrix (int2 aVals, MatrixVectorType_enum aType)
//{
//	switch (aType)
//	{
//	case MVT_ColumnVector:
//		cols = 1;
//		rows = 2;
//		values = new T[cols * rows];
//		values[0] = (T)aVals.x;
//		values[1] = (T)aVals.y;
//		break;
//	case MVT_RowVector:
//		cols = 2;
//		rows = 1;
//		values = new T[cols * rows];
//		values[0] = (T)aVals.x;
//		values[1] = (T)aVals.y;
//		break;
//	}
//}


template<class T>
Matrix<T>::Matrix(string aValues)
{
	stringstream ss(aValues);
	vector<string> vrows;
	vector<class vector<string> > vvcols;  // Changed so that gcc accepts it...

	string item;
	char delim = ';';
	while (getline(ss, item, delim))
	{
		vrows.push_back(item);
	}

	for (size_t i = 0; i < vrows.size(); i++)
	{
		stringstream ss2(vrows[i]);
		vector<string> vcol;
		string item2;
		char delim2 = ' ';

		while (getline(ss2, item2, delim2))
		{
			if (item2.size() > 0)
				vcol.push_back(item2);
		}
		vvcols.push_back(vcol);
	}

	if (vvcols.size() < 1) return;
	if (vvcols[0].size() < 1) return;

	cols = (uint)vvcols[0].size();
	rows = (uint)vvcols.size();
	for (uint i = 0; i < rows; i++)
	{
		if (cols != vvcols[i].size()) return;
	}

	values = new T[cols * rows];

	for (uint i = 0; i < rows; i++)
	{
		for (uint j = 0; j < cols; j++)
		{
			stringstream ssT(vvcols[i][j]);
			T val;
			ssT >> val;
			uint index = GetIndex(i, j);
			values[index] = val;
		}
	}
}


template<class T>
Matrix<T>::~Matrix()
{
	delete[] values;
	values = NULL;
}
#pragma endregion

#pragma region Getter
template<class T>
uint Matrix<T>::GetIndex(const uint r, const uint c) const
{
	return c + r * cols;
}

template<class T>
bool Matrix<T>::CheckDimensions(const Matrix<T>& mat)
{
	return cols == mat.cols && rows == mat.rows;
}

template<class T>
T* Matrix<T>::GetData()
{
	return values;
}

//template<class T>
//float3 Matrix<T>::GetAsFloat3()
//{
//	float3 ret;
//
//	if (rows == 1 && cols == 3)
//	{
//		ret.x = (float)(*this)(0,0);
//		ret.y = (float)(*this)(0,1);
//		ret.z = (float)(*this)(0,2);
//	}
//
//	if (rows == 3 && cols == 1)
//	{
//		ret.x = (float)(*this)(0,0);
//		ret.y = (float)(*this)(1,0);
//		ret.z = (float)(*this)(2,0);
//	}
//	return ret;
//}
//
//template<class T>
//float2 Matrix<T>::GetAsFloat2()
//{
//	float2 ret;
//
//	if (rows == 1 && cols == 2)
//	{
//		ret.x = (float)(*this)(0,0);
//		ret.y = (float)(*this)(0,1);
//	}
//
//	if (rows == 2 && cols == 1)
//	{
//		ret.x = (float)(*this)(0,0);
//		ret.y = (float)(*this)(1,0);
//	}
//	return ret;
//}
//
//template<class T>
//float4 Matrix<T>::GetAsFloat4()
//{
//	float4 ret;
//
//	if (rows == 1 && cols == 4)
//	{
//		ret.x = (float)(*this)(0,0);
//		ret.y = (float)(*this)(0,1);
//		ret.z = (float)(*this)(0,2);
//		ret.w = (float)(*this)(0,3);
//	}
//
//	if (rows == 4 && cols == 1)
//	{
//		ret.x = (float)(*this)(0,0);
//		ret.y = (float)(*this)(1,0);
//		ret.z = (float)(*this)(2,0);
//		ret.w = (float)(*this)(3,0);
//	}
//	return ret;
//}
#pragma endregion

#pragma region Operators
template<class T>
T& Matrix<T>::operator()(const uint aRowIndex, const uint aColIndex) const
{
	return values[GetIndex(aRowIndex, aColIndex)];
}

template<class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& aValue)
{
	return this->Add(aValue);
}

template<class T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& aValue)
{
	return this->Sub(aValue);
}

template<class T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& aValue)
{
	return this->Mul(aValue);
}

template<class T>
Matrix<T> Matrix<T>::operator/(const Matrix<T>& aValue)
{
	return this->DivComp(aValue);
}

template<class T>
Matrix<T> Matrix<T>::operator+(const T& aValue)
{
	return this->Add(aValue);
}

template<class T>
Matrix<T> Matrix<T>::operator-(const T& aValue)
{
	return this->Sub(aValue);
}

template<class T>
Matrix<T> Matrix<T>::operator*(const T& aValue)
{
	return this->MulComp(aValue);
}

template<class T>
Matrix<T> Matrix<T>::operator/(const T& aValue)
{
	return this->DivComp(aValue);
}



template<class T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& aValue)
{
	if (!CheckDimensions(aValue)) return *this;
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		(*this)(x,y) += aValue(x,y);
	}
	return (*this);
}

template<class T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& aValue)
{
	if (!CheckDimensions(aValue)) return *this;
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		(*this)(x,y) -= aValue(x,y);
	}
	return (*this);
}

template<class T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& aValue)
{
	if (!CheckDimensions(aValue)) return *this;
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		(*this)(x,y) *= aValue(x,y);
	}
	return (*this);
}

template<class T>
Matrix<T>& Matrix<T>::operator/=(const Matrix<T>& aValue)
{
	if (!CheckDimensions(aValue)) return *this;
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		(*this)(x,y) /= aValue(x,y);
	}
	return (*this);
}

template<class T>
Matrix<T>& Matrix<T>::operator+=(const T& aValue)
{
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		(*this)(x,y) += aValue;
	}
	return (*this);
}

template<class T>
Matrix<T>& Matrix<T>::operator-=(const T& aValue)
{
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		(*this)(x,y) -= aValue;
	}
	return (*this);
}

template<class T>
Matrix<T>& Matrix<T>::operator*=(const T& aValue)
{
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		(*this)(x,y) *= aValue;
	}
	return (*this);
}

template<class T>
Matrix<T>& Matrix<T>::operator/=(const T& aValue)
{
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		(*this)(x,y) /= aValue;
	}
	return (*this);
}

template<class T>
bool Matrix<T>::operator==(const Matrix<T>& aValue)
{
	if (!CheckDimensions(aValue)) return false;
	bool ret = true;
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		ret &= (*this)(x,y) == aValue(x,y);
	}
	return ret;
}

template<class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& aValue)
{
    T* temp = new T[aValue.cols * aValue.rows];
    rows = aValue.rows;
    cols = aValue.cols;

    if (values != NULL)
        delete[] values;

    values = temp;
	memcpy(values, aValue.values, sizeof(T) * rows * cols);

	return *this;
}

template<class T>
bool Matrix<T>::operator!=(const Matrix<T>& aValue)
{
	return !((*this) == aValue);
}

#pragma endregion

#pragma region Operator Methods
template<class T>
Matrix<T> Matrix<T>::Add(const Matrix<T>& aValue)
{
	if (!CheckDimensions(aValue)) return *this;
	Matrix<T> ret(rows, cols);
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		ret(x,y) = (*this)(x,y) + aValue(x,y);
	}
	return ret;
}

template<class T>
Matrix<T> Matrix<T>::Sub(const Matrix<T>& aValue)
{
	if (!CheckDimensions(aValue)) return *this;
	Matrix<T> ret(rows, cols);
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		ret(x,y) = (*this)(x,y) - aValue(x,y);
	}
	return ret;
}

template<class T>
Matrix<T> Matrix<T>::MulComp(const Matrix<T>& aValue)
{
	if (!CheckDimensions(aValue)) return *this;
	Matrix<T> ret(rows, cols);
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		ret(x,y) = (*this)(x,y) * aValue(x,y);
	}
	return ret;
}

template<class T>
Matrix<T> Matrix<T>::DivComp(const Matrix<T>& aValue)
{
	if (!CheckDimensions(aValue)) return *this;
	Matrix<T> ret(rows, cols);
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		ret(x,y) = (*this)(x,y) / aValue(x,y);
	}
	return ret;
}
template<class T>
Matrix<T> Matrix<T>::Add(const T& aValue)
{
	Matrix<T> ret(rows, cols);
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		ret(x,y) = (*this)(x,y) + aValue;
	}
	return ret;
}

template<class T>
Matrix<T> Matrix<T>::Sub(const T& aValue)
{
	Matrix<T> ret(rows, cols);
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		ret(x,y) = (*this)(x,y) - aValue;
	}
	return ret;
}

template<class T>
Matrix<T> Matrix<T>::MulComp(const T& aValue)
{
	Matrix<T> ret(rows, cols);
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		ret(x,y) = (*this)(x,y) * aValue;
	}
	return ret;
}

template<class T>
Matrix<T> Matrix<T>::DivComp(const T& aValue)
{
	Matrix<T> ret(rows, cols);
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		ret(x,y) = (*this)(x,y) / aValue;
	}
	return ret;
}

template<class T>
Matrix<T> Matrix<T>::Mul(const Matrix<T>& aValue)
{
	if (cols != aValue.rows) return *this;
	Matrix<T> ret(rows, aValue.cols);
	for (uint retx = 0; retx < ret.rows; retx++)
	for (uint rety = 0; rety < ret.cols; rety++)
	{
		T val = 0;
		for (uint i = 0; i < cols; i++)
		{
			val += (*this)(retx, i) * aValue(i, rety);
		}
		ret(retx,rety) = val;
	}
	return ret;
}

#pragma endregion

#pragma region UtileMethods

template<class T>
Matrix<T> Matrix<T>::Transpose()
{
	Matrix<T> ret(rows, cols);
	for (uint x = 0; x < rows; x++)
	for (uint y = 0; y < cols; y++)
	{
		ret(y,x) = (*this)(x,y);
	}
	return ret;
}

template<class T>
Matrix<T> Matrix<T>::Inverse()
{
	return *this;
}

template<>
Matrix<float> Matrix<float>::Inverse()
{
	if (rows != cols) return *this;
	uint n = rows;
	Matrix<float> inv(n * 2, n);
	uint i, j;
	uint s;
	uint pivotLine;
	uint error = 0;
	float mulFac;
	const float Epsilon = 0.001f;
	float Maximum;
	uint pivot = 1;

	//Append unit matrix
	for (uint x = 0; x < rows; x++)
	{
		for (uint y = 0; y < cols; y++)
		{
			inv(x,y) = (*this)(x,y);
			if (x == y)
				inv(x, n + y) = 1;
		}
	}

	s = 0;
	do
	{
		Maximum = fabs(inv(s, s));
		if (pivot)
		{
			pivotLine = s;
			for (i = s+1; i < n; i++)
			{
				if (fabs(inv(i, s)) > Maximum)
				{
					Maximum = fabs(inv(i,s));
					pivotLine = i;
				}
			}
		}
		error = Maximum < Epsilon;

		if (error) break;

		if (pivot)
		{
			if (pivotLine != s) // if necessary, swap lines
			{
				float h;
				for (j = s; j < 2 * n; j++)
				{
					h = inv(s,j);
					inv(s,j) = inv(pivotLine, j);
					inv(pivotLine, j) = h;
				}
			}
		}

		mulFac = inv(s,s);
		for (j = s; j < 2 * n; j++)
			inv(s,j) = inv(s,j) / mulFac;

		for (i = 0; i < n; i++)
		{
			if (i != s)
			{
				mulFac = -inv(i,s);
				for (j = s; j < 2 * n; j++)
					inv(i,j) += mulFac * inv(s,j);
			}
		}
		s++;
	}
	while (s < n);

	if (error)
	{
		return *this;
	}

	Matrix<float> ret(n,n);

	for (uint x = 0; x < cols; x++)
	for (uint y = 0; y < rows; y++)
	{
		ret(x,y) = inv(x,n+y);
	}

	return ret;
}

template <class T>
Matrix<T> Matrix<T>::Diagonalize()
{
	Matrix<T> diag = (*this);
	T mult;
	uint i,j,k;

	for (i = 0; i < cols; i++)
	{
		for (j = 0; j < rows; j++)
		{
			mult = diag(i,j) / diag(i,i);
			for (k = 0; k < cols; k++)
			{
				if(i == j) break;
				diag(k,j) = diag(k,j) - diag(k,i) * mult;
			}
		}
	}

	return diag;
}

template <class T>
T Matrix<T>::Det()
{
	if (cols != rows) return 0;

	T deter = 1;

	Matrix<T> temp = Diagonalize();

	for (uint i = 0; i < cols; i++)
	{
		deter = deter * temp(i,i);
	}

	return deter;
}
#pragma endregion

#pragma region Rotate methods
template <class T>
Matrix<float> Matrix<T>::GetRotationMatrix2D(float aAngle)
{
	Matrix<float> rot(2,2);

	rot(0,0) = cos(aAngle); rot(0,1) = -sin(aAngle);
	rot(1,0) = sin(aAngle); rot(1,1) = cos(aAngle);

	return rot;
}

template <class T>
Matrix<float> Matrix<T>::GetRotationMatrix3DX(float aAngle)
{
	Matrix<float> rot(3,3);

	rot(0,0) = 1; rot(0,1) = 0;           rot(0,2) = 0;
	rot(1,0) = 0; rot(1,1) = cos(aAngle); rot(1,2) = -sin(aAngle);
	rot(2,0) = 0; rot(2,1) = sin(aAngle); rot(2,2) = cos(aAngle);

	return rot;
}

template <class T>
Matrix<float> Matrix<T>::GetRotationMatrix3DY(float aAngle)
{
	Matrix<float> rot(3,3);

	rot(0,0) = cos(aAngle) ; rot(0,1) = 0; rot(0,2) = sin(aAngle);
	rot(1,0) = 0           ; rot(1,1) = 1; rot(1,2) = 0;
	rot(2,0) = -sin(aAngle); rot(2,1) = 0; rot(2,2) = cos(aAngle);

	return rot;
}

template <class T>
Matrix<float> Matrix<T>::GetRotationMatrix3DZ(float aAngle)
{
	Matrix<float> rot(3,3);

	rot(0,0) = cos(aAngle); rot(0,1) = -sin(aAngle); rot(0,2) = 0;
	rot(1,0) = sin(aAngle); rot(1,1) = cos(aAngle) ; rot(1,2) = 0;
	rot(2,0) = 0          ; rot(2,1) = 0           ; rot(2,2) = 1;

	return rot;
}

//template <class T>
//float2 Matrix<T>::Rotate(const float2& aVec, float aAngle)
//{
//	Matrix<float> rot = GetRotationMatrix2D(aAngle);
//	Matrix<float> vec(aVec);
//
//	return (rot * vec).GetAsFloat2();
//}
//
//template <class T>
//float3 Matrix<T>::RotateX(const float3& aVec, float aAngle)
//{
//	Matrix<float> rot = GetRotationMatrix3DX(aAngle);
//	Matrix<float> vec(aVec);
//
//	return (rot * vec).GetAsFloat3();
//}
//
//template <class T>
//float3 Matrix<T>::RotateY(const float3& aVec, float aAngle)
//{
//	Matrix<float> rot = GetRotationMatrix3DY(aAngle);
//	Matrix<float> vec(aVec);
//
//	return (rot * vec).GetAsFloat3();
//}
//
//template <class T>
//float3 Matrix<T>::RotateZ(const float3& aVec, float aAngle)
//{
//	Matrix<float> rot = GetRotationMatrix3DZ(aAngle);
//	Matrix<float> vec(aVec);
//
//	return (rot * vec).GetAsFloat3();
//}

#pragma endregion

template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;

//std::ostream& operator<<(std::ostream& out, const Matrix<int>& mat)
//{
//	out << "[";
//	for (uint r = 0; r < mat.rows; r++)
//	{
//		for (uint c = 0; c < mat.cols; c++)
//		{
//			out << mat(r, c);
//			if (c != mat.cols -1)
//				out << " ";
//		}
//		if (r != mat.rows -1)
//			out << "; ";
//	}
//	out << "]";
//	return out;
//}
//
//std::ostream& operator<<(std::ostream& out, const Matrix<float>& mat)
//{
//	out << "[";
//	for (uint r = 0; r < mat.rows; r++)
//	{
//		for (uint c = 0; c < mat.cols; c++)
//		{
//			out << mat(r, c);
//			if (c != mat.cols -1)
//				out << " ";
//		}
//		if (r != mat.rows -1)
//			out << "; ";
//	}
//	out << "]";
//	return out;
//}
