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


#include <math.h>
#include "singleframecontroller.h"
#include <QMessageBox>
#include <QDebug>
#include <QApplication>


SingleFrameController* SingleFrameController::myController = NULL;

SingleFrameController::SingleFrameController(QObject *parent) :
    BaseFrameController(parent),
    mImage(NULL)
{
    myController = this;
}

SingleFrameController::~SingleFrameController()
{
    if (mImage)
        delete mImage;
    mImage = NULL;
}

int SingleFrameController::GetWidth()
{
    if (mImage)
        return mDimX;
    return 2048;
}

int SingleFrameController::GetHeight()
{
    if (mImage)
        return mDimY;
    return 2048;
}

bool SingleFrameController::GetIsMultiChannel()
{
    if (mImage)
        return mIsRGB;
    return false;
}

float SingleFrameController::GetPixelSize()
{
    if (mImage)
        return mImage->GetPixelSize();
    return 0;
}

void SingleFrameController::CloseFile()
{
    mIsFileLoaded = false;
    if (mImage)
        delete mImage;
    mImage = NULL;
    mFilename = "";
}

QMatrix4x4 SingleFrameController::GetAlignmentMatrix()
{
    QMatrix4x4 m;
    m.setToIdentity();
    if (mImage)
    {
        float mirrorY = mImage->NeedsFlipOnYAxis() ? -1.0f : 1.0f;
        m.scale(1, mirrorY, 1);
    }
    return m;
}

int SingleFrameController::openFile(QString filename)
{
    mIsFileLoaded = false;
    mFilename = "";
    if (mCtx == 0)
    {
        mCtx = OpenCL::OpenCLThreadBoundContext::GetCtxOpenGL();
    }

    if (mProgram == 0)
    {
        OpenCL::OpenCLThreadBoundContext::GetProgramAndBuild(&mProgram, openCLSource, strlen(openCLSource));
    }

    if (mImage)
    {
        delete mImage;
        mImage = NULL;
    }
    FileType_enum fileType;

    mIsRGB = false;
    if (SingleFrame::CanReadFile(filename.toLocal8Bit().constData(), fileType, mDimX, mDimY, mDatatype))
    {
        LoadKernel();
    }
    else
    {
        //QMessageBox::warning(NULL, tr("Clicker"),
        //                               tr("Can't read the following file:\n'") + filename + "'\nMake sure it is a single frame image.", QMessageBox::Ok, QMessageBox::Ok);
        return -1;
    }

    mIsLockedForLoading = true;
    mImage = new SingleFrame(filename.toLocal8Bit().constData(), &FileLoadStatusUpdate);
    mIsLockedForLoading = false;

    mIsFileLoaded = true;
    mFilename = filename;
    emit StartPreparingData();

    size_t dataTypeSize = GetDataTypeSize(mDatatype);
    if (mDevVar) delete mDevVar;
    mDevVar = new OpenCL::OpenCLDeviceVariable(mDimX * mDimY * dataTypeSize);
    mDevVar->CopyHostToDevice(mImage->GetData());

    computeImageStatisticsOpenCL();

    //Mean values after normalization to 0..1
    mMeanValues[0] = (float)((mMeanValues[0] - mMinValues[0]) / (mMaxValues[0] - mMinValues[0]));
    mMeanValues[1] = (float)((mMeanValues[1] - mMinValues[1]) / (mMaxValues[1] - mMinValues[1]));
    mMeanValues[2] = (float)((mMeanValues[2] - mMinValues[2]) / (mMaxValues[2] - mMinValues[2]));

    //Std values after normalization to 0..1
    mStdValues[0] = (float)sqrt(mStdValues[0]) / (mMaxValues[0] - mMinValues[0]);
    mStdValues[1] = (float)sqrt(mStdValues[1]) / (mMaxValues[1] - mMinValues[1]);
    mStdValues[2] = (float)sqrt(mStdValues[2]) / (mMaxValues[2] - mMinValues[2]);

    OpenCL::OpenCLDeviceVariable histDeviceA(mHistogramBinCount*4);
    OpenCL::OpenCLDeviceVariable histDeviceB(mHistogramBinCount*4);
    OpenCL::OpenCLDeviceVariable histDeviceC(mHistogramBinCount*4);
    computeHistogramOpenCL(&histDeviceA, &histDeviceB, &histDeviceC);

    data.devPtr = mDevVar->GetDevicePtr();
    data.scale = 1.0f;
    data.useFilter = false;
    if (mIsRGB)
    {
        data.maxVal = max(max(mMaxValues[0], mMaxValues[1]), mMaxValues[2]);
        data.minVal = min(min(mMinValues[0], mMinValues[1]), mMinValues[2]);
    }
    else
    {
        data.maxVal = mMaxValues[0];
        data.minVal = mMinValues[0];
    }

    emit MinValueChanged(data.minVal);
    emit MaxValueChanged(data.maxVal);
    emit ValueRangeChanged(data.maxVal - data.minVal);
    emit NegValueRangeChanged(data.minVal - data.maxVal);

    if (mIsRGB)
    {
        emit MeanValueChanged((mMeanValues[0] + mMeanValues[1] + mMeanValues[2]) / 3.0f);
        emit StdValueChanged((mStdValues[0] + mStdValues[1] + mStdValues[2]) / 3.0f);
        //we have to add 0.5, as in normalized range is from 0..1 and not -0.5..0.5
        emit Std3ValueChanged((mStdValues[0] + mStdValues[1] + mStdValues[2]) + 0.5f);
    }
    else
    {
        emit MeanValueChanged(mMeanValues[0]);
        emit StdValueChanged(mStdValues[0]);
        //we have to add 0.5, as in normalized range is from 0..1 and not -0.5..0.5
        emit Std3ValueChanged(mStdValues[0] * 3.0f+0.5f);

        //qDebug() << "Std: " << mStdValues[0] << "Std3: " << mStdValues[0] * 3.0f;
    }

    emit MinValuesChanged(mMinValues);
    emit MaxValuesChanged(mMaxValues);
    emit MeanValuesChanged(mMeanValues);
    emit StdValuesChanged(mStdValues);
    emit IsRGBChanged(mIsRGB);
    emit PixelSizeChanged(mImage->GetPixelSize());
    emit DimensionsChanged(mDimX, mDimY);
    emit DimensionXChanged(mDimX);
    emit DimensionYChanged(mDimY);

    if (mIsRGB)
    {
        emit HistogramChanged(&mHistogramRorGray, &mHistogramG, &mHistogramB);
    }
    else
    {
        emit HistogramChanged(&mHistogramRorGray, NULL, NULL);
    }

    switch (mDatatype)
    {
    case DT_UCHAR:
    case DT_UCHAR2:
    case DT_UCHAR3:
    case DT_UCHAR4:
        emit MinMaxDatatypeChanged(0, 255);
    break;
    case DT_USHORT:
        emit MinMaxDatatypeChanged(0, 65535);
    break;
    case DT_SHORT:
        emit MinMaxDatatypeChanged(-32767, 32767);
    break;
    case DT_INT:
        emit MinMaxDatatypeChanged(data.minVal, data.maxVal);
    break;
    case DT_FLOAT:
        emit MinMaxDatatypeChanged(data.minVal, data.maxVal);
    break;
    }

    emit AlignmentMatrixChanged(GetAlignmentMatrix());

    return 0;
}


void SingleFrameController::FileLoadStatusUpdate(FileReader::FileReaderStatus status)
{
    float progress = (float)status.bytesRead / (float)status.bytesToRead * 100.0f;

    QApplication::postEvent(myController, new QProgressUpdateEvent((int)progress));
}
