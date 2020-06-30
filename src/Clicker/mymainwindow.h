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


#ifndef MYMAINWINDOW_H
#define MYMAINWINDOW_H

#include <QMainWindow>
#include <QShortcut>
#include <CL/cl.h>
#include <OpenCLHelpers.h>
#include <OpenCLKernel.h>
#include <OpenCLDeviceVariable.h>
#include <mutex>
#include <singleframecontroller.h>
#include <tiltseriescontroller.h>
#include <QProgressBar>
#include <QTimer>

namespace Ui {
class MyMainWindow;
}

class TiltSeriesAligner : public QObject {
    Q_OBJECT
public:

    TiltSeriesAligner(float phi, bool DoPsi, bool DoPsiFixed, bool DoTheta, bool DoPhi, bool DoMags, bool normMin, bool normZeroTilt, bool magsFirst, int iterSwitch, int iterations, float ansiFactor, float ansiAngle, int addZShift, TiltSeriesController* aTS);
    //~TiltSeriesLoader();
public slots:
    void process();
signals:
    void finished();
    void error(QString err);
    void alignmentDone(int code);
    void alignmentReport(QString report);
private:
    TiltSeriesController* ts;
    float mPhi;
    bool mDoPsi;
    bool mDoPsiFixed;
    bool mDoTheta;
    bool mDoPhi;
    bool mDoMags;
    bool mNormMin;
    bool mNormZeroTilt;
    bool mMagsFirst;
    int mIterSwitch;
    int mIterations;
    float mAnsiFactor;
    float mAnsiAngle;
    int mAddZShift;
    // add your variables here
};

class TiltSeriesLoader : public QObject {
    Q_OBJECT
public:

    TiltSeriesLoader(QString aFilename, TiltSeriesController* aTS);
    //~TiltSeriesLoader();
public slots:
    void process();
signals:
    void finished();
    void error(QString err);
    void loadingDone(int code, QString filename);
private:
    QString filename;
    TiltSeriesController* ts;
    // add your variables here
};

class SingleFrameLoader : public QObject {
    Q_OBJECT
public:

    SingleFrameLoader(QString aFilename, SingleFrameController* aImage);
public slots:
    void process();
signals:
    void finished();
    void error(QString err);
    void loadingDone(int code, QString filename);
private:
    QString filename;
    SingleFrameController* image;
    // add your variables here
};

class MyMainWindow : public QMainWindow
{
    Q_OBJECT



public:
    explicit MyMainWindow(QWidget *parent = 0);
    ~MyMainWindow();
    bool CanLoadFile(QString fileName);
    bool LoadFile(QString fileName);
    void closeEvent(QCloseEvent *event);

private slots:
    void openSingleFrame();
    void openSingleFrame(QString fileName);
    void openTiltSeries();
    void openTiltSeries(QString fileName);
    void AlignTiltSeries();
    void CreateNewMarkerfile();
    void LoadMarkerfile();
    void SaveMarkerfile();
    void HandleSaveImage();
    void HandleSaveImageSeries();
    void HandlePageUpEvent();
    void HandlePageDownEvent();
    void HandleFileLoadUpdate(int value);
    void HandleFileLoadDoneTiltSeries(int statusCode, QString filename);
    void HandleFileLoadDoneSingleFrame(int statusCode, QString filename);
    void HandlePrepareData();
    void HandleProjectionsParametersChanged(ProjectionParameters params);
    void HandleProgressUpdate(int value);
    void HandleStartProgress(QString message);
    void HandleEndProgress(int statusCode);
    void HandleCCImageChanged(QImage& img1, QImage& img2);
    void HandleTimerStartStop(bool start);
    void HandleTimerTick();
    void HandleTimerIntervalChanged(int);
    void HandleAlignmentReport(QString report);
    void HandleIsNotBusyChanged(bool isBusy);
    void HandleStartClickOperation();
    void HandleStopClickOperation();
    void HandleTiltIndexDoubleClick();
    void HandleRemoveMarker();
    void HandleBeamDeclinationEnabled(bool value);
    void HandleReconstructionAssistant();
    void HandleQuit();

signals:
    void IsNotBusyChanged(bool isBusy);
    void BeamDeclinationEnabled(bool value);

private:
    Ui::MyMainWindow *ui;

    SingleFrameController* sf;
    TiltSeriesController* ts;
    QShortcut pageUp;
    QShortcut pageDown;
    QShortcut plus;
    QShortcut minus;
    QShortcut deleteButton;
    QShortcut deleteAllMarkers;
    QProgressBar* fileLoadProgress;
    QThread* backgroundThread;
    SingleFrameLoader* singleFrameLoader;
    TiltSeriesLoader* tiltSeriesLoader;
    TiltSeriesAligner* tiltSeriesAligner;
    QTimer* mTimer;
    int mAutoScrollDirection;
    bool mIsNotBusy;
    bool mClickInProgress;

    void LoadTiltSeries(QString filename);
    void CloseAllFiles();

    std::mutex _mutex;
};

#endif // MYMAINWINDOW_H
