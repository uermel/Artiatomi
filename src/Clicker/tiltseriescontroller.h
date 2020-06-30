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


#ifndef TILTSERIESCONTROLLER_H
#define TILTSERIESCONTROLLER_H

#include <QObject>
#include <TiltSeries.h>
#include "baseframecontroller.h"
#include "FileReader.h"
#include "MarkerFile.h"
#include "CrossCorrelator.h"
#include <QImage>



struct ProjectionParameters
{
    int index;
    int totalNumber;
    float tiltAngleUnaligned;
    float tiltAngleAligned;
    float imageRotation;
    float stretchFactor;
    bool isAligned;
};
Q_DECLARE_METATYPE(ProjectionParameters)

class TiltSeriesController : public BaseFrameController
{
    Q_OBJECT

public:
    explicit TiltSeriesController(QObject *parent = 0);
    ~TiltSeriesController();
    int GetWidth();
    int GetHeight();
    bool GetIsMultiChannel();
    float GetPixelSize();
    void CloseFile();

    int GetImageCount();
    //void SetOpenFileStatusCallback(void(*_readStatusCallback)(FileReader::FileReaderStatus));

    virtual QMatrix4x4 GetAlignmentMatrix();
    int AlignMarker(bool DoPsi, bool DoPsiFixed, bool DoTheta, bool DoPhi, bool DoMags, bool normMin, bool normZeroTilt, bool magsFirst, int iterSwitch, int iterations, float ansiFactor, float anisoAngle, int addZShift);
    bool IsMarkerfileLoaded();
    bool IsProjectionAligned(int projIdx);
    QString GetAlignmentReport();
    void EmitMarkerPositions();
    int GetZeroTiltIndex();
    int GetMarkerCount();
    QString GetMarkerFilename();
    void GetMagAnisotropy(float& aAmount, float& aAngle);
    float GetMeanStd();
    float GetBeamDeclination();
    void GetMarkerMinMaxZ(float& zMin, float& zMax);
    ProjectionParameters GetProjectionParameters(int idx);

public slots:
    int openFile(QString filename);
    int setImageNr(int idx);
    void handleClickOnImage(int x, int y, bool isHit, Qt::MouseButtons button);
    void removeCurrentMarker();
    int openMarkerFile(QString filename);
    int importMarkerFromImod(QString filename);
    int saveMarkerFile(QString filename);
    void createNewMarkerFile();
    void setActiveMarker(int idx);
    void setShowAligned(bool showAsAligned);
    void setFixFirstMarker(bool fixFirst);
    void setFixActiveMarker(bool fixActive);
    void addMarker();
    void removeMarker();
    void SetMarkerSize(int aSize);
    void SetCropSize(int aSize);
    void SetBeamDeclination(float aValue);
    void SetSortByTiltAngle(bool aValue);
    void SetFilterImage(bool aValue);
    void DeleteAllMarkersFromCurrentImage();



signals:
    void ImageCountChanged(int startIsOne, int imageCount);
    void ImageChanged();
    void ProjectionParametersChanged(ProjectionParameters params);
    void ShowAsAlignedChanged(bool showAsAligned);
    void MarkerCountChanged(int count);
    void ActiveMarkerChanged(int idx);
    void CurrentMarkerPositionsChanged(QVector<float>& x, QVector<float>& y, QVector<float>& xAlig, QVector<float>& yAlig);
    void AlignedTiltAngleChanged(float value);
    void BeamDeclinationChanged(float value);
    void ImageRotationChanged(float valueInDeg);
    void ScalingChanged(float value);
    void MarkerfileLoadedChanged(bool isLoaded);
    void CCImageChanged(QImage& CCImage, QImage& CropImage);
    void StartClickOperation();
    void EndClickOperation();
    void TiltMeanValueChanged(float aValue);
    void TiltMinValueChanged(float aValue);
    void TiltMaxValueChanged(float aValue);
    void TiltStdValueChanged(float aValue);
    void MagAnisotropyFactorChanged(float aValue);
    void MagAnisotropyAngleChanged(float aValue);
    void SortByTiltAngleChanged(bool aValue);
    void MarkerFilenameChanged(QString aValue);
    void MarkerShortFilenameChanged(QString aValue);
    void SetFilterImageChanged(bool aValue);


private:
    MarkerFile* mMarkerfile;
    int mActiveMarker;
    int mTiltCount;
    int mCurrentTiltIdx;
    int mActualTiltIdx; //not affected by sorting or 0 numbering (as given by GUI)
    TiltSeries* mTiltSeries;
    float* mMaxValPerTilt;
    float* mMinValPerTilt;
    float* mStdValPerTilt;
    float* mMeanValPerTilt;
    float mMaxTotal;
    float mMinTotal;
    float mMeanTotal;
    float mMeanStdTotal;
    QVector<QVector<int> > mHistograms;
    float mBeamDeclination;
    float *markerX, *markerY, *markerZ;
    float mAlignmentErrorScore;
    float* errorPerProjection;
    float* errorPerMarker;
    bool mFixOne;
    bool mFixActive;
    bool mShowAlignment;
    CrossCorrelator cc;
    int mMarkerSize;
    int mCropSize;
    bool mSortByTiltAngle;
    bool CopyPatch(float* data, int x, int y);

    //void(*readStatusCallback)(FileReader::FileReaderStatus );

    void EmitAlignmentInfo();
    static void FileLoadStatusUpdate(FileReader::FileReaderStatus status);
    static void AlignmentStatusUpdate(int percent);
    static TiltSeriesController* myController;
    QString GetShortFilename(QString filename);

};

#endif // TILTSERIESCONTROLLER_H
