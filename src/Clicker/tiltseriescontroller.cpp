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


#include "tiltseriescontroller.h"
#include <QMessageBox>
#include <QDebug>
#include "mymainwindow.h"
#include <QApplication>
#include <QFileInfo>


TiltSeriesController::TiltSeriesController(QObject *parent) :
    BaseFrameController(parent),
    mTiltSeries(NULL),
    mTiltCount(0),
    mMaxValPerTilt(NULL),
    mMinValPerTilt(NULL),
    mStdValPerTilt(NULL),
    mMeanValPerTilt(NULL),
    mMaxTotal(0),
    mMinTotal(0),
    mMeanTotal(0),
    mMeanStdTotal(0),
    mMarkerfile(NULL),
    mActiveMarker(0),
    mCurrentTiltIdx(0),
    mActualTiltIdx(1),

    mBeamDeclination(0),
    markerX(NULL), markerY(NULL), markerZ(NULL),
    mAlignmentErrorScore(0),
    errorPerProjection(NULL),
    errorPerMarker(NULL),

    mFixOne(false),
    mFixActive(false),
    mShowAlignment(false),
    mMarkerSize(10),
    mCropSize(256),
    mSortByTiltAngle(false)

    //readStatusCallback(NULL)
{
    myController = this;
}

TiltSeriesController* TiltSeriesController::myController = NULL;

TiltSeriesController::~TiltSeriesController()
{
    if (mTiltSeries)
        delete mTiltSeries;
    mTiltSeries = NULL;

    if (mMaxValPerTilt)
        delete mMaxValPerTilt;
    mMaxValPerTilt = NULL;
    if (mMinValPerTilt)
        delete mMinValPerTilt;
    mMinValPerTilt = NULL;

    if (mStdValPerTilt)
        delete mStdValPerTilt;
    mStdValPerTilt = NULL;

    if (mMeanValPerTilt)
        delete mMeanValPerTilt;
    mMeanValPerTilt = NULL;

    if (mMarkerfile)
        delete mMarkerfile;
    mMarkerfile = NULL;

    if (markerX)
    {
        delete[] markerX;
    }
    markerX = NULL;
    if (markerY)
    {
        delete[] markerY;
    }
    markerY = NULL;
    if (markerZ)
    {
        delete[] markerZ;
    }
    markerZ = NULL;
    if (errorPerMarker)
    {
        delete[] errorPerMarker;
    }
    errorPerMarker = NULL;
    if (errorPerProjection)
    {
        delete[] errorPerProjection;
    }
    errorPerProjection = NULL;
}

int TiltSeriesController::GetWidth()
{
    if (mIsLockedForLoading) return 2048;
    if (mTiltSeries)
        return mDimX;
    return 2048;
}

int TiltSeriesController::GetHeight()
{
    if (mIsLockedForLoading) return 2048;
    if (mTiltSeries)
        return mDimY;
    return 2048;
}

int TiltSeriesController::GetImageCount()
{
    if (mIsLockedForLoading) return 1;
    if (mTiltCount)
        return mTiltCount;
    return 1;
}

bool TiltSeriesController::GetIsMultiChannel()
{
    return false;
}

float TiltSeriesController::GetPixelSize()
{
    if (mIsLockedForLoading) return 0;
    if (mTiltSeries)
        return mTiltSeries->GetPixelSize();
    return 0;
}

void TiltSeriesController::CloseFile()
{
    if (mIsLockedForLoading) return;
    mIsFileLoaded = false;
    if (mTiltSeries)
        delete mTiltSeries;
    mTiltSeries = NULL;
    mCurrentTiltIdx = 0;

    if (mMarkerfile)
        delete mMarkerfile;
    mMarkerfile = NULL;
    mFilename = "";
    EmitAlignmentInfo();
    emit MarkerCountChanged(0);
}

int TiltSeriesController::openFile(QString filename)
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

    if (mTiltSeries)
    {
        delete mTiltSeries;
        mTiltSeries = NULL;
    }

    if (mMarkerfile)
    {
        delete mMarkerfile;
        mMarkerfile = NULL;
    }

    FileType_enum fileType;

    mIsRGB = false;
    if (TiltSeries::CanReadFile(filename.toStdString(), fileType, mDimX, mDimY, mTiltCount, mDatatype))
    {
        LoadKernel();
    }
    else
    {
        mIsLockedForLoading = false;
        return -1;
    }
    mFilename = filename;
    mIsLockedForLoading = true;
    mTiltSeries = new TiltSeries(filename.toLocal8Bit().constData(), &FileLoadStatusUpdate);
    mIsLockedForLoading = false;

    emit StartPreparingData();

    mIsFileLoaded = true;

    size_t dataTypeSize = GetDataTypeSize(mDatatype);
    if (mDevVar) delete mDevVar;
    mDevVar = new OpenCL::OpenCLDeviceVariable(mDimX * mDimY * dataTypeSize);

    mMinTotal = FLT_MAX;
    mMaxTotal = -FLT_MAX;
    mMeanTotal = 0;
    mMeanStdTotal = 0;

    if (mMaxValPerTilt)
        delete[] mMaxValPerTilt;

    if (mMinValPerTilt)
        delete[] mMinValPerTilt;

    if (mStdValPerTilt)
        delete[] mStdValPerTilt;

    if (mMeanValPerTilt)
        delete[] mMeanValPerTilt;


    mMaxValPerTilt = new float[mTiltCount];
    mMinValPerTilt = new float[mTiltCount];
    mStdValPerTilt = new float[mTiltCount];
    mMeanValPerTilt = new float[mTiltCount];

    mHistograms.clear();

    emit ProgressStatusChanged(0);
    QVector<float> stds;
    QVector<float> maxValForSort;
    float meanOfMax = 0;

    for (size_t idx = 0; idx < mTiltCount; idx++)
    {
        mDevVar->CopyHostToDevice(mTiltSeries->GetData(idx));

        computeImageStatisticsOpenCL();
        mMaxTotal = max(mMaxValues[0], mMaxTotal);
        mMinTotal = min(mMinValues[0], mMinTotal);
        mMeanTotal += mMeanValues[0];
        stds.push_back((float)sqrt(mStdValues[0]));
        mMeanStdTotal += (float)sqrt(mStdValues[0]);

        mMaxValPerTilt[idx] = mMaxValues[0];
        mMinValPerTilt[idx] = mMinValues[0];
        mMeanValPerTilt[idx] = mMeanValues[0];
        mStdValPerTilt[idx] = (float)sqrt(mStdValues[0]);
        emit ProgressStatusChanged((int)((float)(idx + 1)/(float)mTiltCount * 50.0f));
        meanOfMax += mMaxValues[0];
        maxValForSort.push_back(mMaxValues[0]);
    }

    meanOfMax /= mTiltCount;
    if (mMaxTotal > meanOfMax * 3) //we will have problems with image contrast then...
    {
        qSort(maxValForSort);
        mMaxTotal = maxValForSort[mTiltCount / 2];
//        for (int i = mTiltCount - 1; i >= 0; i--)
//        {
//            if (maxValForSort[i] < meanOfMax * 10)
//            {
//                mMaxTotal = maxValForSort[i];
//            }
//        }
    }

    qSort(stds); //Use Median and not mean of standard deviations to avoid problems with bad projections...

    mMeanTotal /= mTiltCount;
    //mMeanStdTotal /= mTiltCount * (mMaxTotal - mMinTotal);
    mMeanStdTotal = stds[mTiltCount / 2] / (mMaxTotal - mMinTotal);

    data.devPtr = mDevVar->GetDevicePtr();
    data.scale = 1;
    data.useFilter = false;
    if (mIsRGB)
    {
        data.maxVal = mMaxTotal;
        data.minVal = mMinTotal;
    }
    else
    {
        data.maxVal = mMaxTotal;
        data.minVal = mMinTotal;
    }

    emit MinValueChanged(data.minVal);
    emit MaxValueChanged(data.maxVal);
    emit ValueRangeChanged(data.maxVal - data.minVal);
    emit NegValueRangeChanged(data.minVal - data.maxVal);

    mMinValues[0] = mMinTotal;
    mMaxValues[0] = mMaxTotal;
    mMeanValues[0] = mMeanTotal;
    mStdValues[0] = mMeanStdTotal;


    OpenCL::OpenCLDeviceVariable histDeviceA(mHistogramBinCount*4);
    OpenCL::OpenCLDeviceVariable histDeviceB(mHistogramBinCount*4);
    OpenCL::OpenCLDeviceVariable histDeviceC(mHistogramBinCount*4);
    for (size_t idx = 0; idx < mTiltCount; idx++)
    {
        mDevVar->CopyHostToDevice(mTiltSeries->GetData(idx));
        computeHistogramOpenCL(&histDeviceA, &histDeviceB, &histDeviceC);
        mHistograms.push_back(mHistogramRorGray);
        emit ProgressStatusChanged((int)((float)(idx + 1)/(float)mTiltCount * 50.0f) + 50);
    }
    mDevVar->CopyHostToDevice(mTiltSeries->GetData(0));


    //Mean values after normalization to 0..1
    mMeanValues[0] = (float)((mMeanValues[0] - mMinValues[0]) / (mMaxValues[0] - mMinValues[0]));

    if (mIsRGB)
    {
        //Tilt series is always B/W
    }
    else
    {
        emit MeanValueChanged(mMeanValues[0]);
        emit StdValueChanged(mMeanStdTotal * (mMaxTotal - mMinTotal));
        //we have to add 0.5, as in normalized range is from 0..1 and not -0.5..0.5
        emit Std3ValueChanged(mMeanStdTotal * 3.0f+0.5f);
    }

    emit MinValuesChanged(mMinValues);
    emit MaxValuesChanged(mMaxValues);
    emit MeanValuesChanged(mMeanValues);
    emit StdValuesChanged(mStdValues);
    emit IsRGBChanged(mIsRGB);
    emit PixelSizeChanged(mTiltSeries->GetPixelSize());
    emit DimensionsChanged(mDimX, mDimY);
    emit DimensionXChanged(mDimX);
    emit DimensionYChanged(mDimY);
    emit ImageCountChanged(1, mTiltCount);

    if (mIsRGB)
    {
        //Tilt series is always B/W
    }
    else
    {
        emit HistogramChanged(&mHistograms[0], NULL, NULL);
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
        emit MinMaxDatatypeChanged(mMinTotal, mMaxTotal);
    break;
    case DT_FLOAT:
        emit MinMaxDatatypeChanged(mMinTotal, mMaxTotal);
    break;
    }
    mCurrentTiltIdx = 0;
    EmitAlignmentInfo();
    emit MarkerCountChanged(0);

    emit TiltMeanValueChanged(mMeanValPerTilt[mCurrentTiltIdx]);
    emit TiltMinValueChanged(mMinValPerTilt[mCurrentTiltIdx]);
    emit TiltMaxValueChanged(mMaxValPerTilt[mCurrentTiltIdx]);
    emit TiltStdValueChanged(mStdValPerTilt[mCurrentTiltIdx]);

    emit MarkerFilenameChanged("");
    emit MarkerShortFilenameChanged("");
    return 0;
}

int TiltSeriesController::setImageNr(int idx)
{
    if (mIsLockedForLoading) return 0;
    mActualTiltIdx = idx;
    idx--; //idx starts at one, numbering here starts at 0...

    if (mSortByTiltAngle)
    {
        vector<pair<float, int> > sortTemp;
        if (mMarkerfile)
        {
            for (int i = 0; i < mMarkerfile->GetTotalProjectionCount(); i++)
            {
                sortTemp.push_back(pair<float, int>((*mMarkerfile)(MFI_TiltAngle, i, 0), i));
            }
            sort(sortTemp.begin(), sortTemp.end());
            mCurrentTiltIdx = sortTemp[idx].second;
        }
        else
        {
            //if tilt angles are stored in tilt series:
            bool tiltSeriesHasThetas = false;
            for (int proj = 0; proj < mTiltCount - 1; proj++)
            {
                if (mTiltSeries->GetTiltAngle(proj) != mTiltSeries->GetTiltAngle(proj) + 1)
                {
                    tiltSeriesHasThetas = true;
                    break;
                }
            }

            if (tiltSeriesHasThetas)
            {
                for (int proj = 0; proj < mTiltCount; proj++)
                {
                    float theta = mTiltSeries->GetTiltAngle(proj);
                    sortTemp.push_back(pair<float, int>(theta, proj));
                }
                sort(sortTemp.begin(), sortTemp.end());
                mCurrentTiltIdx = sortTemp[idx].second;
            }
            else
            {
                mCurrentTiltIdx = idx; //No tilt angles available to sort...
            }
        }
    }
    else
    {
        mCurrentTiltIdx = idx;
    }

    if (mCurrentTiltIdx < 0 || mCurrentTiltIdx >= mTiltCount)
        return 0;

    mDevVar->CopyHostToDevice(mTiltSeries->GetData(mCurrentTiltIdx));

    float scale = mMeanTotal / mMeanValPerTilt[mCurrentTiltIdx];

    data.minVal = mMinTotal;
    data.maxVal = mMaxTotal;
    data.scale = scale;

    emit HistogramChanged(&mHistograms[mCurrentTiltIdx], NULL, NULL);
    emit ImageChanged();


    emit ProjectionParametersChanged(GetProjectionParameters(mCurrentTiltIdx));
    EmitAlignmentInfo();
    emit TiltMeanValueChanged(mMeanValPerTilt[mCurrentTiltIdx]);
    emit TiltMinValueChanged(mMinValPerTilt[mCurrentTiltIdx]);
    emit TiltMaxValueChanged(mMaxValPerTilt[mCurrentTiltIdx]);
    emit TiltStdValueChanged(mStdValPerTilt[mCurrentTiltIdx]);

    //qDebug() << mMaxValPerTilt[idx] << mMinValPerTilt[idx] << mMeanValPerTilt[idx] << mMeanTotal << scale;
    return 0;
}

void TiltSeriesController::handleClickOnImage(int x, int y, bool isHit, Qt::MouseButtons button)
{
    if (isHit && button == Qt::MouseButton::RightButton)
    {
        if (mMarkerfile)
        {
            emit StartClickOperation();
            float* data = new float[mCropSize*mCropSize];
            if (CopyPatch(data, x, mDimY - y))
            {
                unsigned char* imgcrop = new unsigned char[mCropSize*mCropSize];
                unsigned char* imgcc = new unsigned char[mCropSize*mCropSize];
                FilterPoint2D p = cc.GetShiftGauss(data, imgcc, mCropSize, mCropSize, mMarkerSize);
                //qDebug() << "Found Shift: " << p.x << "; " << p.y;
                (*mMarkerfile)(MFI_X_Coordinate, mCurrentTiltIdx, mActiveMarker) = x - p.x;
                if (mTiltSeries->NeedsFlipOnYAxis())
                {
                    (*mMarkerfile)(MFI_Y_Coordinate, mCurrentTiltIdx, mActiveMarker) = y + p.y;
                }
                else
                {
                    (*mMarkerfile)(MFI_Y_Coordinate, mCurrentTiltIdx, mActiveMarker) = mDimY - y - p.y - 1;
                }
                EmitAlignmentInfo();
                for (int i = 0; i < mCropSize*mCropSize; i++)
                    imgcrop[i] = (unsigned char)(data[i] * 255.0f);
                QImage cc(imgcc, mCropSize, mCropSize, QImage::Format_Grayscale8);
                QImage crop(imgcrop, mCropSize, mCropSize, QImage::Format_Grayscale8);
                emit CCImageChanged(cc, crop);
                delete[] imgcc;
                delete[] imgcrop;
            }
            delete[] data;
            emit EndClickOperation();
        }
        else
        {
            //qDebug() << "Hit: " << x << "; " << y;
        }
    }

    if (isHit && button == Qt::MouseButton::MiddleButton)
    {
        if (mMarkerfile)
        {
            (*mMarkerfile)(MFI_X_Coordinate, mCurrentTiltIdx, mActiveMarker) = x;
            if (mTiltSeries->NeedsFlipOnYAxis())
            {
                (*mMarkerfile)(MFI_Y_Coordinate, mCurrentTiltIdx, mActiveMarker) = y;
            }
            else
            {
                (*mMarkerfile)(MFI_Y_Coordinate, mCurrentTiltIdx, mActiveMarker) = mDimY - y - 1;
            }
            EmitAlignmentInfo();
        }
    }
}

void TiltSeriesController::removeCurrentMarker()
{
    if (mMarkerfile)
    {
        (*mMarkerfile)(MFI_X_Coordinate, mCurrentTiltIdx, mActiveMarker) = -1;
        (*mMarkerfile)(MFI_Y_Coordinate, mCurrentTiltIdx, mActiveMarker) = -1;
        EmitAlignmentInfo();
    }
}

int TiltSeriesController::openMarkerFile(QString filename)
{
    if (mMarkerfile)
        delete mMarkerfile;
    mMarkerfile = NULL;

    emit MarkerFilenameChanged("");
    emit MarkerShortFilenameChanged("");

    if (MarkerFile::CanReadAsMarkerfile(filename.toStdString()))
    {
        try
        {
            mMarkerfile = new MarkerFile(filename.toStdString());
            if (mMarkerfile->GetTotalProjectionCount() != mTiltCount)
            {
                delete mMarkerfile;
                mMarkerfile = NULL;
                EmitAlignmentInfo();
                emit MarkerCountChanged(0);
                return -1;
            }
        }
        catch (std::exception ex)
        {
            mMarkerfile = NULL;

            EmitAlignmentInfo();
            emit MarkerCountChanged(0);
            return -1;
        }

        if (markerX)
        {
            delete[] markerX;
        }
        if (markerY)
        {
            delete[] markerY;
        }
        if (markerZ)
        {
            delete[] markerZ;
        }

        markerX = new float[mMarkerfile->GetMarkerCount()];
        markerY = new float[mMarkerfile->GetMarkerCount()];
        markerZ = new float[mMarkerfile->GetMarkerCount()];

        mMarkerfile->GetMarkerPositions(markerX, markerY, markerZ);

        mMarkerfile->GetBeamDeclination(mBeamDeclination);
        float magFactor;
        float magAngle;
        mMarkerfile->GetMagAnisotropy(magFactor, magAngle);
        emit MagAnisotropyFactorChanged(magFactor);
        emit MagAnisotropyAngleChanged(magAngle);
        EmitAlignmentInfo();
        emit MarkerCountChanged(mMarkerfile->GetMarkerCount());
        emit MarkerFilenameChanged(filename);
        emit MarkerShortFilenameChanged(GetShortFilename(filename));
        return 0;
    }
    EmitAlignmentInfo();
    emit MarkerCountChanged(0);
    return -1;
}

int TiltSeriesController::importMarkerFromImod(QString filename)
{
    if (mMarkerfile)
        delete mMarkerfile;
    mMarkerfile = NULL;

    emit MarkerFilenameChanged("");
    emit MarkerShortFilenameChanged("");

    try
    {
        mMarkerfile = MarkerFile::ImportFromIMOD(filename.toStdString());
        if (mMarkerfile->GetTotalProjectionCount() != mTiltCount)
        {
            delete mMarkerfile;
            mMarkerfile = NULL;
            EmitAlignmentInfo();
            emit MarkerCountChanged(0);
            return -1;
        }
    }
    catch (std::exception ex)
    {
        mMarkerfile = NULL;

        EmitAlignmentInfo();
        emit MarkerCountChanged(0);
        return -1;
    }

    if (markerX)
    {
        delete[] markerX;
        markerX = NULL;
    }
    if (markerY)
    {
        delete[] markerY;
        markerY = NULL;
    }
    if (markerZ)
    {
        delete[] markerZ;
        markerZ = NULL;
    }

    mMarkerfile->GetBeamDeclination(mBeamDeclination);
    float magFactor;
    float magAngle;
    mMarkerfile->GetMagAnisotropy(magFactor, magAngle);
    emit MagAnisotropyFactorChanged(magFactor);
    emit MagAnisotropyAngleChanged(magAngle);
    EmitAlignmentInfo();
    emit MarkerCountChanged(mMarkerfile->GetMarkerCount());
    return 0;

}

int TiltSeriesController::saveMarkerFile(QString filename)
{
    if (!mMarkerfile)
        return -1;

    try
    {
        if (mMarkerfile->Save(filename.toStdString()))
        {
            emit MarkerFilenameChanged(filename);
            emit MarkerShortFilenameChanged(GetShortFilename(filename));
            return 0;
        }
    }
    catch (FileIOException& ex)
    {
        emit MarkerFilenameChanged("");
        emit MarkerShortFilenameChanged("");
        return -1;
    }

    emit MarkerFilenameChanged("");
    emit MarkerShortFilenameChanged("");
    return -1;
}

void TiltSeriesController::createNewMarkerFile()
{
    if (mMarkerfile)
        delete mMarkerfile;

    float* tilts = new float[mTiltCount];
    for (int i = 0; i < mTiltCount; i++)
    {
        tilts[i] = mTiltSeries->GetTiltAngle(i);
    }

    mMarkerfile = new MarkerFile(mTiltCount, tilts);
    delete[] tilts;
    EmitAlignmentInfo();
    emit MarkerCountChanged(mMarkerfile->GetMarkerCount());
    emit MarkerFilenameChanged("");
    emit MarkerShortFilenameChanged("");
}

void TiltSeriesController::setActiveMarker(int idx)
{
    if (idx != mActiveMarker && mMarkerfile)
    {
        if (idx >= 0 && idx < mMarkerfile->GetMarkerCount())
        {
            mActiveMarker = idx;
            emit ActiveMarkerChanged(mActiveMarker);
            EmitAlignmentInfo();
        }
    }
}

void TiltSeriesController::setShowAligned(bool showAsAligned)
{
    if (mShowAlignment != showAsAligned)
    {
        mShowAlignment = showAsAligned;
        emit ShowAsAlignedChanged(mShowAlignment);
        EmitAlignmentInfo();
    }
}

void TiltSeriesController::setFixFirstMarker(bool fixFirst)
{
    if (mFixOne != fixFirst)
    {
        mFixOne = fixFirst;
        EmitAlignmentInfo();
    }
}

void TiltSeriesController::setFixActiveMarker(bool fixActive)
{
    if (mFixActive != fixActive)
    {
        mFixActive = fixActive;
        EmitAlignmentInfo();
    }
}

void TiltSeriesController::addMarker()
{
    if (mMarkerfile)
    {
        mMarkerfile->AddMarker();
        emit MarkerCountChanged(mMarkerfile->GetMarkerCount());
        EmitAlignmentInfo();
    }
}

void TiltSeriesController::removeMarker()
{
    if (mMarkerfile)
    {
        mMarkerfile->RemoveMarker(mActiveMarker);
        emit MarkerCountChanged(mMarkerfile->GetMarkerCount());
        EmitAlignmentInfo();
        mActiveMarker = mMarkerfile->GetMarkerCount() - 1;
        emit ActiveMarkerChanged(mActiveMarker);
    }
}

void TiltSeriesController::SetMarkerSize(int aSize)
{
    mMarkerSize = aSize;
}

void TiltSeriesController::SetCropSize(int aSize)
{
    mCropSize = aSize;
}

void TiltSeriesController::SetBeamDeclination(float aValue)
{
    if (aValue != mBeamDeclination)
    {
        mBeamDeclination = aValue;
        emit BeamDeclinationChanged(aValue);
    }
}

void TiltSeriesController::SetFilterImage(bool aValue)
{
    if (aValue != data.useFilter)
    {
        data.useFilter = aValue;
        emit SetFilterImageChanged(data.useFilter);
        setImageNr(mActualTiltIdx);
    }
}

void TiltSeriesController::SetSortByTiltAngle(bool aValue)
{
    if (aValue)
    {
        if (!mMarkerfile)
        {
            bool tiltSeriesHasThetas = false;
            for (int proj = 0; proj < mTiltCount - 1; proj++)
            {
                if (mTiltSeries->GetTiltAngle(proj) != mTiltSeries->GetTiltAngle(proj) + 1)
                {
                    tiltSeriesHasThetas = true;
                    break;
                }
            }
            if (!tiltSeriesHasThetas)
            {
                mSortByTiltAngle = false;
                emit SortByTiltAngleChanged(false);
                return;
            }
        }
    }

    if (aValue != mSortByTiltAngle)
    {
        mSortByTiltAngle = aValue;
        emit SortByTiltAngleChanged(aValue);
        setImageNr(mActualTiltIdx);
    }
}

void TiltSeriesController::DeleteAllMarkersFromCurrentImage()
{
    if (mMarkerfile)
    {
        mMarkerfile->DeleteAllMarkersInProjection(mCurrentTiltIdx);
        EmitAlignmentInfo();
    }
}

bool TiltSeriesController::CopyPatch(float *data, int aX, int aY)
{
    int px = aX - mCropSize / 2;
    int py = aY - mCropSize / 2;

    if (px < 0 || py < 0 || px + mCropSize >= mDimX || py + mCropSize >= mDimY)
    {
        return false;
    }

    void* ptr = mTiltSeries->GetData(mCurrentTiltIdx);
    switch (mTiltSeries->GetFileDataType())
    {
        case DT_FLOAT:
        {
            float* d = (float*)ptr;
            for (int y = py; y < py + mCropSize; y++)
            {
                for (int x = px; x < px + mCropSize; x++)
                {
                    data[(y-py) * mCropSize + (x - px)] = d[y * mDimX + x];
                }
            }
        }
        return true;
        case DT_INT:
        {
            int* d = (int*)ptr;
            for (int y = py; y < py + mCropSize; y++)
            {
                for (int x = px; x < px + mCropSize; x++)
                {
                    data[(y-py) * mCropSize + (x - px)] = d[y * mDimX + x];
                }
            }
        }
        return true;
        case DT_UINT:
        {
            uint* d = (uint*)ptr;
            for (int y = py; y < py + mCropSize; y++)
            {
                for (int x = px; x < px + mCropSize; x++)
                {
                    data[(y-py) * mCropSize + (x - px)] = d[y * mDimX + x];
                }
            }
        }
        return true;
        case DT_SHORT:
        {
            short* d = (short*)ptr;
            for (int y = py; y < py + mCropSize; y++)
            {
                for (int x = px; x < px + mCropSize; x++)
                {
                    data[(y-py) * mCropSize + (x - px)] = d[y * mDimX + x];
                }
            }
        }
        return true;
        case DT_USHORT:
        {
            ushort* d = (ushort*)ptr;
            for (int y = py; y < py + mCropSize; y++)
            {
                for (int x = px; x < px + mCropSize; x++)
                {
                    data[(y-py) * mCropSize + (x - px)] = d[y * mDimX + x];
                }
            }
        }
        return true;
        case DT_UCHAR:
        {
            uchar* d = (uchar*)ptr;
            for (int y = py; y < py + mCropSize; y++)
            {
                for (int x = px; x < px + mCropSize; x++)
                {
                    data[(y-py) * mCropSize + (x - px)] = d[y * mDimX + x];
                }
            }
        }
        return true;
        case DT_CHAR:
        {
            char* d = (char*)ptr;
            for (int y = py; y < py + mCropSize; y++)
            {
                for (int x = px; x < px + mCropSize; x++)
                {
                    data[(y-py) * mCropSize + (x - px)] = d[y * mDimX + x];
                }
            }
        }
        return true;
    }
    return false;
}

void TiltSeriesController::EmitAlignmentInfo()
{
    int proj = mCurrentTiltIdx;

    ProjectionParameters params = GetProjectionParameters(proj);
    emit ProjectionParametersChanged(params);
    emit AlignmentMatrixChanged(GetAlignmentMatrix());
    if (mMarkerfile)
    {
        emit AlignedTiltAngleChanged(params.tiltAngleAligned);
        emit BeamDeclinationChanged(mBeamDeclination);
        emit ImageRotationChanged(params.imageRotation);
        emit ScalingChanged(params.stretchFactor);

        EmitMarkerPositions();
        emit ActiveMarkerChanged(mActiveMarker);
    }
    else
    {
        emit AlignedTiltAngleChanged(0);
        emit BeamDeclinationChanged(0);
        emit ImageRotationChanged(0);
        emit ScalingChanged(1);
    }
}



ProjectionParameters TiltSeriesController::GetProjectionParameters(int idx)
{
    ProjectionParameters params;
    memset(&params, 0, sizeof(params));

    if (mIsLockedForLoading) return params;
    if (!mTiltSeries) return params;

    params.index = idx+1;
    params.totalNumber = mTiltCount;
    params.isAligned = false;
    params.stretchFactor = 1;
    params.tiltAngleUnaligned = mTiltSeries->GetTiltAngle(idx);
    params.tiltAngleAligned = 0;
    params.imageRotation = 0;

    if (mMarkerfile)
    {
        params.isAligned = true;
        params.stretchFactor = (*mMarkerfile)(MFI_Magnifiaction, idx, 0);
        params.tiltAngleAligned = (*mMarkerfile)(MFI_TiltAngle, idx, 0);
        params.imageRotation = (*mMarkerfile)(MFI_RotationPsi, idx, 0);
    }
    return params;
}

QMatrix4x4 TiltSeriesController::GetAlignmentMatrix()
{
    QMatrix4x4 m;
    m.setToIdentity();

    if (mMarkerfile)
    {
        {
            float shiftX = 0, shiftY = 0;
            float rot = 0;
            float scale = 1;
            if (mFixOne && (*mMarkerfile)(MFI_X_Coordinate, mCurrentTiltIdx, 0) >= 0)
            {
                shiftX = (*mMarkerfile)(MFI_X_Coordinate, mCurrentTiltIdx, 0) - mDimX / 2.0f;
                shiftY = (*mMarkerfile)(MFI_Y_Coordinate, mCurrentTiltIdx, 0) - mDimY / 2.0f;
            }
            else if(mFixActive && (*mMarkerfile)(MFI_X_Coordinate, mCurrentTiltIdx, mActiveMarker) >= 0)
            {
                shiftX = (*mMarkerfile)(MFI_X_Coordinate, mCurrentTiltIdx, mActiveMarker) - mDimX / 2.0f;
                shiftY = (*mMarkerfile)(MFI_Y_Coordinate, mCurrentTiltIdx, mActiveMarker) - mDimY / 2.0f;
            }
            else if (mShowAlignment)
            {
                shiftX = (*mMarkerfile)(MFI_X_Shift, mCurrentTiltIdx, mActiveMarker);
                shiftY = (*mMarkerfile)(MFI_Y_Shift, mCurrentTiltIdx, mActiveMarker);
                rot = (*mMarkerfile)(MFI_RotationPsi, mCurrentTiltIdx, 0);
                scale = (*mMarkerfile)(MFI_Magnifiaction, mCurrentTiltIdx, 0);

                //remove close to 0 values in case of an incorrect marker file
                if (abs(scale) < 0.01)
                {
                    scale = 1; //This is just to
                }
            }

            m.rotate(rot, 0, 0, 1);
            m.scale(scale);
            m.translate(-shiftX, shiftY, 0);
        }
    }

    if (mTiltSeries)
    {
        float mirrorY = mTiltSeries->NeedsFlipOnYAxis() ? -1.0f : 1.0f;
        m.scale(1, mirrorY, 1);
    }
    return m;
}

int TiltSeriesController::AlignMarker(bool DoPsi, bool DoPsiFixed, bool DoTheta, bool DoPhi, bool DoMags, bool normMin, bool normZeroTilt, bool magsFirst, int iterSwitch, int iterations, float ansiFactor, float anisoAngle, int addZShift)
{
    if (mMarkerfile)
    {
        //mBeamDeclination = 0;
        mAlignmentErrorScore = 0;

        if (markerX)
        {
            delete[] markerX;
        }
        if (markerY)
        {
            delete[] markerY;
        }
        if (markerZ)
        {
            delete[] markerZ;
        }
        if (errorPerMarker)
        {
            delete[] errorPerMarker;
        }
        if (errorPerProjection)
        {
            delete[] errorPerProjection;
        }
        markerX = new float[mMarkerfile->GetMarkerCount()];
        markerY = new float[mMarkerfile->GetMarkerCount()];
        markerZ = new float[mMarkerfile->GetMarkerCount()];
        errorPerMarker = new float[mMarkerfile->GetMarkerCount()];
        errorPerProjection = new float[mMarkerfile->GetTotalProjectionCount()];

        //Reset tilt angles to default:
        //if tilt angles are stored in tilt series:
        bool tiltSeriesHasThetas = false;
        for (int proj = 0; proj < mTiltCount - 1; proj++)
        {
            if (mTiltSeries->GetTiltAngle(proj) != mTiltSeries->GetTiltAngle(proj) + 1)
            {
                tiltSeriesHasThetas = true;
                break;
            }
        }

        if (tiltSeriesHasThetas)
        {
            for (int proj = 0; proj < mTiltCount; proj++)
            {
                float theta = mTiltSeries->GetTiltAngle(proj);
                for (int m = 0; m < mMarkerfile->GetMarkerCount(); m++)
                {
                    (*mMarkerfile)(MFI_TiltAngle, proj, m) = theta;
                }
            }
        }
//        float x1 = 1000;
//        float y1 = 1000;
//        MarkerFile::MoveXYToMagDistort(x1, y1, 1.016f, 42.7f / 180.0f * M_PI, 4096, 4096);
        mMarkerfile->SetMagAnisotropy(ansiFactor, anisoAngle, mDimX, mDimY);
        mMarkerfile->Align3D(mActiveMarker, mDimX, mDimY, mAlignmentErrorScore, mBeamDeclination, DoPsi, DoPsiFixed, DoTheta, DoPhi, DoMags, normMin, normZeroTilt, magsFirst, iterSwitch, iterations, addZShift, errorPerProjection, errorPerMarker, markerX, markerY, markerZ, &AlignmentStatusUpdate);
        EmitAlignmentInfo();
    }
    return 0;
}

bool TiltSeriesController::IsMarkerfileLoaded()
{
    return mMarkerfile != NULL;
}

bool TiltSeriesController::IsProjectionAligned(int projIdx)
{
    if (mMarkerfile)
    {
        return mMarkerfile->CheckIfProjIndexIsGood(projIdx);
    }
    return true; //if no marker file loader, assume all good...
}

QString TiltSeriesController::GetAlignmentReport()
{
    if (!mMarkerfile)
    {
        return QString();
    }

    if (!markerX)
    {
        return QString();
    }

    QString result;

    result += "Alignment score: " + QString::number(mAlignmentErrorScore) + " [pixels]\n";
    result += "Beam declination: " + QString::number(mBeamDeclination) + "°\n\n";
    result += "Error per projection:\n";

    for (int i = 0; i < mMarkerfile->GetTotalProjectionCount(); i++)
    {
        result += "Projection " + QString::number(i+1) + ":\t" + QString::number(errorPerProjection[i]) + "\n";
    }

    result += "\n\nError per marker:\n";

    for (int i = 0; i < mMarkerfile->GetMarkerCount(); i++)
    {
        result += "Marker " + QString::number(i+1) + ":\t" + QString::number(errorPerMarker[i]) + "\n";
    }


    result += "\n\nMarker positions:\n";

    for (int i = 0; i < mMarkerfile->GetMarkerCount(); i++)
    {
        result += "Marker " + QString::number(i+1) + ":\t" + QString::number(markerX[i]) + ";\t" + QString::number(markerY[i]) + ";\t" + QString::number(markerZ[i]) + "\n";
    }
    return result;
}

void TiltSeriesController::EmitMarkerPositions()
{
    int proj = mCurrentTiltIdx;
    int markerCount = mMarkerfile->GetMarkerCount();

    QVector<float> x;
    QVector<float> y;
    QVector<float> xAlig;
    QVector<float> yAlig;

    for (int i = 0; i < markerCount; i++)
    {
        x.push_back((*mMarkerfile)(MFI_X_Coordinate, proj, i));
        xAlig.push_back((*mMarkerfile)(MFI_ProjectedCoordinateX, proj, i));
        if (mTiltSeries->NeedsFlipOnYAxis())
        {
            y.push_back((*mMarkerfile)(MFI_Y_Coordinate, proj, i));
            yAlig.push_back((*mMarkerfile)(MFI_ProjectedCoordinateY, proj, i));
        }
        else
        {
            y.push_back(mDimY - (*mMarkerfile)(MFI_Y_Coordinate, proj, i) - 1);
            yAlig.push_back(mDimY - (*mMarkerfile)(MFI_ProjectedCoordinateY, proj, i) - 1);
        }
    }
    emit CurrentMarkerPositionsChanged(x, y, xAlig, yAlig);
}

int TiltSeriesController::GetZeroTiltIndex()
{
    if (mMarkerfile)
    {
        return mMarkerfile->GetMinTiltIndex();
    }

    //if tilt angles are stored in tilt series:
    bool tiltSeriesHasThetas = false;
    for (int proj = 0; proj < mTiltCount - 1; proj++)
    {
        if (mTiltSeries->GetTiltAngle(proj) != mTiltSeries->GetTiltAngle(proj) + 1)
        {
            tiltSeriesHasThetas = true;
            break;
        }
    }

    if (tiltSeriesHasThetas)
    {
        int idx = -1;
        float theta = FLT_MAX;
        for (int proj = 0; proj < mTiltCount; proj++)
        {
            if (fabs(mTiltSeries->GetTiltAngle(proj)) < theta)
            {
                idx = proj;
                theta = fabs(mTiltSeries->GetTiltAngle(proj));
            }
        }
        return idx;
    }
    else
    {
        return mTiltCount / 2; //assume the central projection as min tilt
    }
}

int TiltSeriesController::GetMarkerCount()
{
    if (mMarkerfile)
    {
        return mMarkerfile->GetMarkerCount();
    }
    return 0;
}

QString TiltSeriesController::GetMarkerFilename()
{
    if (mMarkerfile)
    {
        return QString::fromUtf8(mMarkerfile->GetFilename().c_str());
    }
    return "";
}

void TiltSeriesController::GetMagAnisotropy(float &aAmount, float &aAngle)
{
    aAmount = 1;
    aAngle = 0;
    if (mMarkerfile)
        mMarkerfile->GetMagAnisotropy(aAmount, aAngle);
}

float TiltSeriesController::GetMeanStd()
{
    return mMeanStdTotal * (mMaxTotal - mMinTotal);
}

float TiltSeriesController::GetBeamDeclination()
{
    return mBeamDeclination;
}

void TiltSeriesController::GetMarkerMinMaxZ(float &zMin, float &zMax)
{
    zMin = 0;
    zMax = 1;
    if (mMarkerfile && markerZ)
    {
        zMin = FLT_MAX;
        zMax = -FLT_MAX;
        for (int i = 0; i < mMarkerfile->GetMarkerCount(); i++)
        {
            if (markerZ[i] < zMin) zMin = markerZ[i];
            if (markerZ[i] > zMax) zMax = markerZ[i];
        }
    }
}


void TiltSeriesController::FileLoadStatusUpdate(FileReader::FileReaderStatus status)
{
    float progress = (float)status.bytesRead / (float)status.bytesToRead * 100.0f;

    QApplication::postEvent(myController, new QProgressUpdateEvent((int)progress));
}

void TiltSeriesController::AlignmentStatusUpdate(int percent)
{
    if (percent < 0) percent = 0;
    if (percent > 100) percent = 100;
    QApplication::postEvent(myController, new QProgressUpdateEvent(percent));
}

QString TiltSeriesController::GetShortFilename(QString filename)
{
    QFileInfo fileInfo(filename);
    return fileInfo.fileName();
}
