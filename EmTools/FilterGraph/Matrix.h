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


#ifndef MATRIX_H
#define MATRIX_H

#include "../Basics/Default.h"
#include <sstream>
#include <cmath>

enum MatrixVectorType_enum
{
	MVT_ColumnVector = 0,
	MVT_RowVector = 1
};

//! Small vbut powerfull matrix class
/*!
	Matrix provides constructors for CUDA based vector types and parses strings in Matlab syntax.

	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
template <class T>
class Matrix
{
private:
	inline uint GetIndex(uint r, uint c) const;
	bool CheckDimensions(const Matrix<T>& mat);
protected:
	uint cols;
	uint rows;
	T* values;

public:
	//! Create a new matrix with dimensions \p aRows x \p aColumns filled with zeros.
	Matrix (uint aRows, uint aColumns);
	//! Copy constructor.
	Matrix (const Matrix& aCopy);
	//! Create a new matrix and parse the values from \p aValues.
	Matrix (std::string aValues);
	////! Convert a float4 to a Matrix (row or column vector).
	//Matrix (float4 aVals, MatrixVectorType_enum aType = MVT_ColumnVector);
	////! Convert a float3 to a Matrix (row or column vector).
	//Matrix (float3 aVals, MatrixVectorType_enum aType = MVT_ColumnVector);
	////! Convert a float2 to a Matrix (row or column vector).
	//Matrix (float2 aVals, MatrixVectorType_enum aType = MVT_ColumnVector);
	////! Convert a int4 to a Matrix (row or column vector).
	//Matrix (int4 aVals, MatrixVectorType_enum aType = MVT_ColumnVector);
	////! Convert a int3 to a Matrix (row or column vector).
	//Matrix (int3 aVals, MatrixVectorType_enum aType = MVT_ColumnVector);
	////! Convert a int to a Matrix (row or column vector).
	//Matrix (int2 aVals, MatrixVectorType_enum aType = MVT_ColumnVector);
	virtual ~Matrix();

	//! Returns a pointer to the inner data array.
	T* GetData();

	//! User friendly output.
	//template<>
	//friend std::ostream& operator<<(std::ostream& out, const Matrix<T>& mat);


	//! Add the values of two matrices.
	Matrix<T> Add(const Matrix<T>& aValue);
	//! Sub the values of two matrices.
	Matrix<T> Sub(const Matrix<T>& aValue);
	//! Matrix multiplication.
	Matrix<T> Mul(const Matrix<T>& aValue);
	//! Component wise multiplication.
	Matrix<T> MulComp(const Matrix<T>& aValue);
	//! Component wise division.
	Matrix<T> DivComp(const Matrix<T>& aValue);

	//! Add a scalar value to each matrix element.
	Matrix<T> Add(const T& aValue);
	//! Sub a scalar value to each matrix element.
	Matrix<T> Sub(const T& aValue);
	//! Mul a scalar value to each matrix element.
	Matrix<T> MulComp(const T& aValue);
	//! Div a scalar value to each matrix element.
	Matrix<T> DivComp(const T& aValue);

	//! Computes the inverse.
	Matrix<T> Inverse();
	//! Transposes the matrix
	Matrix<T> Transpose();
	//! Computes the determinant.
	T Det();
	//! Computes the diagonal representation of the matrix.
	Matrix<T> Diagonalize();

	//! Returns a reference to the value at [\p aRowIndex, \p aColIndex].
	T& operator() (const uint aRowIndex, const uint aColIndex) const;
	//! Add the values of two matrices.
	Matrix<T> operator+ (const Matrix<T>& aValue);
	//! Sub the values of two matrices.
	Matrix<T> operator- (const Matrix<T>& aValue);
	//! Matrix multiplication.
	Matrix<T> operator* (const Matrix<T>& aValue);
	//! Div the values of two matrices.
	Matrix<T> operator/ (const Matrix<T>& aValue);
	//! Add the values of two matrices.
	Matrix<T>& operator+= (const Matrix<T>& aValue);
	//! Sub the values of two matrices.
	Matrix<T>& operator-= (const Matrix<T>& aValue);
	//! Component wise Mul of the values of two matrices.
	Matrix<T>& operator*= (const Matrix<T>& aValue);
	//! Component wise Div of the values of two matrices.
	Matrix<T>& operator/= (const Matrix<T>& aValue);

	//! Add a scalar value to each matrix element.
	Matrix<T> operator+ (const T& aValue);
	//! Sub a scalar value to each matrix element.
	Matrix<T> operator- (const T& aValue);
	//! Mul a scalar value to each matrix element.
	Matrix<T> operator* (const T& aValue);
	//! Div a scalar value to each matrix element.
	Matrix<T> operator/ (const T& aValue);
	//! Add a scalar value to each matrix element.
	Matrix<T>& operator+= (const T& aValue);
	//! Sub a scalar value to each matrix element.
	Matrix<T>& operator-= (const T& aValue);
	//! Mul a scalar value to each matrix element.
	Matrix<T>& operator*= (const T& aValue);
	//! Div a scalar value to each matrix element.
	Matrix<T>& operator/= (const T& aValue);

	//! Checks equality for each element
	bool operator== (const Matrix<T>& aValue);
	//! Checks inequality for each element
	bool operator!= (const Matrix<T>& aValue);
	//! Assignment operator
	Matrix<T>& operator= (const Matrix<T>& aValue);

	////! Returns the values as a float2
	//float2 GetAsFloat2();
	////! Returns the values as a float3
	//float3 GetAsFloat3();
	////! Returns the values as a float4
	//float4 GetAsFloat4();

	//! Returns a matrix representing a rotation by \p aAngle.
	static Matrix<float> GetRotationMatrix2D(float aAngle);
	//! Returns a matrix representing a rotation by \p aAngle around the X axis.
	static Matrix<float> GetRotationMatrix3DX(float aAngle);
	//! Returns a matrix representing a rotation by \p aAngle around the Y axis.
	static Matrix<float> GetRotationMatrix3DY(float aAngle);
	//! Returns a matrix representing a rotation by \p aAngle around the Z axis.
	static Matrix<float> GetRotationMatrix3DZ(float aAngle);

	////! Rotate the float2 vector \p aVec by \p aAngle degrees.
	//static float2 Rotate(const float2& aVec, float aAngle);
	////! Rotate the float3 vector \p aVec by \p aAngle degrees around the X axis.
	//static float3 RotateX(const float3& aVec, float aAngle);
	////! Rotate the float3 vector \p aVec by \p aAngle degrees around the Y axis.
	//static float3 RotateY(const float3& aVec, float aAngle);
	////! Rotate the float3 vector \p aVec by \p aAngle degrees around the Z axis.
	//static float3 RotateZ(const float3& aVec, float aAngle);
};

typedef Matrix<float> matrix;

#endif
