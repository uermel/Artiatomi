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


#include "mymainwindow.h"
//#include <QtCharts/QChartView>
//#include <QtCharts/QBarSeries>
//#include <QtCharts/QBarSet>
//#include <QtCharts/QLegend>

#include "ui_mymainwindow.h"
#include "OpenCLHelpers.h"
#include "MKLog.h"
#include <CL/cl_gl.h>
#include <QOpenGLTexture>
#include <QOpenGLFunctions>
#include <QThread>
#include <QOpenGLContext>
#include <QOffscreenSurface>
#include <SingleFrame.h>
#include <QFileDialog>
#include <QtDebug>
#include <QMessageBox>
//#include <QtCharts>
#include <QThread>
#include "alignmentreport.h"
#include "reconstructionassistant.h"




MyMainWindow::MyMainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MyMainWindow),
    sf(new SingleFrameController(NULL)),
    ts(new TiltSeriesController(NULL)),
    pageUp(QKeySequence(Qt::Key_PageUp), this),
    pageDown(QKeySequence(Qt::Key_PageDown), this),
    plus(QKeySequence(Qt::Key_Plus), this),
    minus(QKeySequence(Qt::Key_Minus), this),
    deleteButton(QKeySequence(Qt::Key_Delete), this),
    deleteAllMarkers(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_D), this),
    fileLoadProgress(new QProgressBar(this)),
    backgroundThread(NULL),
    singleFrameLoader(NULL),
    tiltSeriesLoader(NULL),
    tiltSeriesAligner(NULL),
    mTimer(new QTimer()),
    mAutoScrollDirection(1),
    mIsNotBusy(true),
    mClickInProgress(false)
{
    ui->setupUi(this);
    statusBar()->show();
    pageUp.setContext(Qt::ApplicationShortcut);
    pageDown.setContext(Qt::ApplicationShortcut);
    plus.setContext(Qt::ApplicationShortcut);
    minus.setContext(Qt::ApplicationShortcut);
    deleteButton.setContext(Qt::ApplicationShortcut);
    deleteAllMarkers.setContext(Qt::ApplicationShortcut);

    connect(ui->actionOpen_single_frame_image, SIGNAL(triggered(bool)), this, SLOT(openSingleFrame()));
    connect(ui->actionOpen_tilt_series, SIGNAL(triggered(bool)), this, SLOT(openTiltSeries()));
    connect(ui->actionAlign_tilt_series, SIGNAL(triggered(bool)), this, SLOT(AlignTiltSeries()));
    connect(ui->btn_StartAlignment, SIGNAL(clicked(bool)), ui->actionAlign_tilt_series, SLOT(trigger()));
    connect(ui->btn_NewMarkerFile, SIGNAL(clicked(bool)), ui->actionCreate_new_marker_file, SLOT(trigger()));
    connect(ui->actionCreate_new_marker_file, SIGNAL(triggered(bool)), this, SLOT(CreateNewMarkerfile()));
    connect(ui->btn_LoadMarkerFile, SIGNAL(clicked(bool)), ui->actionOpen_marker_file, SLOT(trigger()));
    connect(ui->actionOpen_marker_file, SIGNAL(triggered(bool)), this, SLOT(LoadMarkerfile()));
    connect(ui->btn_SaveMarkerfile, SIGNAL(clicked(bool)), ui->actionSave_marker_file, SLOT(trigger()));
    connect(ui->actionSave_marker_file, SIGNAL(triggered(bool)), this, SLOT(SaveMarkerfile()));
    connect(ts, SIGNAL(MarkerCountChanged(int)), ui->listMarkers, SLOT(SetMarkerCount(int)));
    connect(ui->actionReconstruction, SIGNAL(triggered(bool)), this, SLOT(HandleReconstructionAssistant()));
    connect(ui->actionQuit, SIGNAL(triggered()), this, SLOT(HandleQuit()));
    connect(ui->actionSave_image, SIGNAL(triggered()), this, SLOT(HandleSaveImage()));
    connect(ui->actionSave_image_series,SIGNAL(triggered()), this, SLOT(HandleSaveImageSeries()));
    connect(ui->chk_showAlignedMarkers, SIGNAL(toggled(bool)), ui->chk_ShowAlignedAsArrow, SLOT(setEnabled(bool)));


    connect(sf, SIGNAL(PixelSizeChanged(float)), ui->openGLWidget, SLOT(setPixelSize(float)));
    connect(sf, SIGNAL(HistogramChanged(QVector<int>*,QVector<int>*,QVector<int>*)), ui->histogram, SLOT(setHistogram(QVector<int>*,QVector<int>*,QVector<int>*)));

    connect(sf, SIGNAL(MinValueChanged(float)), ui->txt_MinValue, SLOT(setValue(float)));
    connect(sf, SIGNAL(MaxValueChanged(float)), ui->txt_MaxValue, SLOT(setValue(float)));
    connect(sf, SIGNAL(MinValueChanged(float)), ui->txt_ContrastCenter, SLOT(setMinRange(float)));
    connect(sf, SIGNAL(MinValueChanged(float)), ui->txt_MeanValue, SLOT(setMinRange(float)));
//    connect(sf, SIGNAL(MinValueChanged(float)), ui->txt_StdValue, SLOT(setMinRange(float)));
    connect(sf, SIGNAL(MaxValueChanged(float)), ui->txt_MeanValue, SLOT(setMaxRange(float)));
//    connect(sf, SIGNAL(MaxValueChanged(float)), ui->txt_StdValue, SLOT(setMaxRange(float)));
    connect(sf, SIGNAL(MeanValueChanged(float)), ui->txt_MeanValue, SLOT(setValueNormalized(float)));
    connect(sf, SIGNAL(StdValueChanged(float)), ui->txt_StdValue, SLOT(setValue(float)));
    connect(sf, SIGNAL(NegValueRangeChanged(float)), ui->txt_ContrastWidth, SLOT(setMinRange(float)));
    connect(sf, SIGNAL(MaxValueChanged(float)), ui->txt_ContrastCenter, SLOT(setMaxRange(float)));
    connect(sf, SIGNAL(ValueRangeChanged(float)), ui->txt_ContrastWidth, SLOT(setMaxRange(float)));
    connect(sf, SIGNAL(Std3ValueChanged(float)), ui->sld_ContrastWidth, SLOT(setValue(float)));
    connect(sf, SIGNAL(MeanValueChanged(float)), ui->sld_ContrastCenter, SLOT(setValue(float)));
    connect(sf, SIGNAL(Std3ValueChanged(float)), ui->sld_ContrastWidth, SLOT(setDefaultValue(float)));
    connect(sf, SIGNAL(MeanValueChanged(float)), ui->sld_ContrastCenter, SLOT(setDefaultValue(float)));

    connect(sf, SIGNAL(PixelSizeChanged(float)), ui->txt_ImagePixelSize, SLOT(setValue(float)));
    connect(sf, SIGNAL(DimensionXChanged(int)), ui->txt_ImageWidth, SLOT(setValue(int)));
    connect(sf, SIGNAL(DimensionYChanged(int)), ui->txt_ImageHeight, SLOT(setValue(int)));
    connect(sf, SIGNAL(StartPreparingData()), this, SLOT(HandlePrepareData()));


    connect(ts, SIGNAL(PixelSizeChanged(float)), ui->openGLWidget, SLOT(setPixelSize(float)));
    connect(ts, SIGNAL(HistogramChanged(QVector<int>*,QVector<int>*,QVector<int>*)), ui->histogram, SLOT(setHistogram(QVector<int>*,QVector<int>*,QVector<int>*)));

    connect(ts, SIGNAL(MinValueChanged(float)), ui->txt_MinValue, SLOT(setValue(float)));
    connect(ts, SIGNAL(MaxValueChanged(float)), ui->txt_MaxValue, SLOT(setValue(float)));
    connect(ts, SIGNAL(MinValueChanged(float)), ui->txt_ContrastCenter, SLOT(setMinRange(float)));
    connect(ts, SIGNAL(MinValueChanged(float)), ui->txt_MeanValue, SLOT(setMinRange(float)));
//    connect(ts, SIGNAL(MinValueChanged(float)), ui->txt_StdValue, SLOT(setMinRange(float)));
    connect(ts, SIGNAL(MaxValueChanged(float)), ui->txt_MeanValue, SLOT(setMaxRange(float)));
//    connect(ts, SIGNAL(MaxValueChanged(float)), ui->txt_StdValue, SLOT(setMaxRange(float)));
    connect(ts, SIGNAL(MeanValueChanged(float)), ui->txt_MeanValue, SLOT(setValueNormalized(float)));
    connect(ts, SIGNAL(StdValueChanged(float)), ui->txt_StdValue, SLOT(setValue(float)));
    connect(ts, SIGNAL(NegValueRangeChanged(float)), ui->txt_ContrastWidth, SLOT(setMinRange(float)));
    connect(ts, SIGNAL(MaxValueChanged(float)), ui->txt_ContrastCenter, SLOT(setMaxRange(float)));
    connect(ts, SIGNAL(ValueRangeChanged(float)), ui->txt_ContrastWidth, SLOT(setMaxRange(float)));
    connect(ts, SIGNAL(Std3ValueChanged(float)), ui->sld_ContrastWidth, SLOT(setValue(float)));
    connect(ts, SIGNAL(MeanValueChanged(float)), ui->sld_ContrastCenter, SLOT(setValue(float)));
    connect(ts, SIGNAL(Std3ValueChanged(float)), ui->sld_ContrastWidth, SLOT(setDefaultValue(float)));
    connect(ts, SIGNAL(MeanValueChanged(float)), ui->sld_ContrastCenter, SLOT(setDefaultValue(float)));

    connect(ts, SIGNAL(PixelSizeChanged(float)), ui->txt_ImagePixelSize, SLOT(setValue(float)));
    connect(ts, SIGNAL(DimensionXChanged(int)), ui->txt_ImageWidth, SLOT(setValue(int)));
    connect(ts, SIGNAL(DimensionYChanged(int)), ui->txt_ImageHeight, SLOT(setValue(int)));
    connect(ts, SIGNAL(ImageCountChanged(int, int)), ui->sld_tiltIndex, SLOT(setRange(int,int)));
    connect(ui->sld_tiltIndex, SIGNAL(valueChanged(int)), ts, SLOT(setImageNr(int)));
    connect(ui->sld_tiltIndex, SIGNAL(doubleClicked()), this, SLOT(HandleTiltIndexDoubleClick()));
    connect(ts, SIGNAL(ImageChanged()), ui->openGLWidget, SLOT(updateImage()));
    connect(ts, SIGNAL(StartPreparingData()), this, SLOT(HandlePrepareData()));
    connect(ui->openGLWidget, SIGNAL(mouseClicked(int,int,bool,Qt::MouseButtons)), ts, SLOT(handleClickOnImage(int,int,bool,Qt::MouseButtons)));
    connect(ts, SIGNAL(CCImageChanged(QImage&, QImage&)), this, SLOT(HandleCCImageChanged(QImage&, QImage&)));
    connect(ts, SIGNAL(TiltMinValueChanged(float)), ui->txt_TiltMinValue, SLOT(setValue(float)));
    connect(ts, SIGNAL(TiltMaxValueChanged(float)), ui->txt_TiltMaxValue, SLOT(setValue(float)));
    connect(ts, SIGNAL(TiltMeanValueChanged(float)), ui->txt_TiltMeanValue, SLOT(setValue(float)));
    connect(ts, SIGNAL(TiltStdValueChanged(float)), ui->txt_TiltStdValue, SLOT(setValue(float)));

    connect(ui->sld_ContrastCenter, SIGNAL(valueChanged(float)), ui->histogram, SLOT(setVisualCenter(float)));
    connect(ui->sld_ContrastWidth, SIGNAL(valueChanged(float)), ui->histogram, SLOT(setVisualRangeWidth(float)));

    connect(&pageUp, SIGNAL(activated()), this, SLOT(HandlePageUpEvent()));
    connect(&pageDown, SIGNAL(activated()), this, SLOT(HandlePageDownEvent()));
    connect(&plus, SIGNAL(activated()), this, SLOT(HandlePageUpEvent()));
    connect(&minus, SIGNAL(activated()), this, SLOT(HandlePageDownEvent()));
    connect(&deleteButton, SIGNAL(activated()), ts, SLOT(removeCurrentMarker()));
    connect(&deleteAllMarkers, SIGNAL(activated()), ts, SLOT(DeleteAllMarkersFromCurrentImage()));

    connect(ts, SIGNAL(ProgressStatusChanged(int)), this, SLOT(HandleProgressUpdate(int)));
    connect(ts, SIGNAL(ProjectionParametersChanged(ProjectionParameters)), this, SLOT(HandleProjectionsParametersChanged(ProjectionParameters)));


    connect(ts, SIGNAL(ActiveMarkerChanged(int)), this->ui->listMarkers, SLOT(SetActiveMarker(int)));
    connect(this->ui->listMarkers, SIGNAL(ActiveMarkerChanged(int)), ts, SLOT(setActiveMarker(int)));
    connect(ts, SIGNAL(MarkerCountChanged(int)), this->ui->listMarkers, SLOT(SetMarkerCount(int)));
    connect(ts, SIGNAL(ScalingChanged(float)), this->ui->txt_alignedScaling, SLOT(setValue(float)));
    connect(ts, SIGNAL(BeamDeclinationChanged(float)), this->ui->txt_alignedBeamDecl, SLOT(setValue(float)));
    connect(ts, SIGNAL(ImageRotationChanged(float)), this->ui->txt_alignedImageRot, SLOT(setValue(float)));
    connect(ts, SIGNAL(ImageRotationChanged(float)), this->ui->txt_alignedImageRot, SLOT(setValue(float)));
    connect(ts, SIGNAL(AlignedTiltAngleChanged(float)), this->ui->txt_alignedTiltAngle, SLOT(setValue(float)));
    connect(ts, SIGNAL(SortByTiltAngleChanged(bool)), this->ui->chk_SortByTiltAngle, SLOT(setChecked(bool)));
    connect(this->ui->chk_SortByTiltAngle, SIGNAL(toggled(bool)), ts, SLOT(SetSortByTiltAngle(bool)));


    connect(sf, SIGNAL(AlignmentMatrixChanged(QMatrix4x4)), ui->openGLWidget, SLOT(setViewMatrix(QMatrix4x4)));
    connect(ts, SIGNAL(AlignmentMatrixChanged(QMatrix4x4)), ui->openGLWidget, SLOT(setViewMatrix(QMatrix4x4)));
    connect(ui->sld_markerSize, SIGNAL(valueChanged(int)), ui->openGLWidget, SLOT(setMarkerSize(int)));
    connect(ui->openGLWidget, SIGNAL(markerSizeChanged(int)), ui->sld_markerSize, SLOT(setValue(int)));
    connect(ui->sld_markerSize, SIGNAL(valueChanged(int)), ts, SLOT(SetMarkerSize(int)));
    connect(ui->cmb_CropSize, SIGNAL(valueChanged(int)), ts, SLOT(SetCropSize(int)));
    connect(ts, SIGNAL(CurrentMarkerPositionsChanged(QVector<float>&,QVector<float>&,QVector<float>&,QVector<float>&)), ui->openGLWidget, SLOT(setMarkerPositions(QVector<float>&,QVector<float>&,QVector<float>&,QVector<float>&)));
    connect(ts, SIGNAL(ActiveMarkerChanged(int)), ui->openGLWidget, SLOT(setActiveMarker(int)));
    connect(ui->rb_aligShow, SIGNAL(toggled(bool)), ts, SLOT(setShowAligned(bool)));
    connect(ui->rb_fixFirst, SIGNAL(toggled(bool)), ts, SLOT(setFixFirstMarker(bool)));
    connect(ui->rb_fixSelected, SIGNAL(toggled(bool)), ts, SLOT(setFixActiveMarker(bool)));
    connect(ui->chk_showAlignedMarkers, SIGNAL(toggled(bool)), ui->openGLWidget, SLOT(SetShowAlignedMarkers(bool)));
    connect(ui->chk_ShowAlignedAsArrow, SIGNAL(toggled(bool)), ui->openGLWidget, SLOT(SetShowAlignedMarkersAsArrows(bool)));

    connect(ui->btn_AddMarker, SIGNAL(clicked(bool)), ui->actionAdd_marker, SLOT(trigger()));
    connect(ui->btn_RemoveMarker, SIGNAL(clicked(bool)), ui->actionRemove_marker, SLOT(trigger()));
    connect(ui->actionAdd_marker, SIGNAL(triggered(bool)), ts, SLOT(addMarker()));
    connect(ui->actionRemove_marker, SIGNAL(triggered(bool)), this, SLOT(HandleRemoveMarker()));

    connect(ui->chk_autoLoop, SIGNAL(toggled(bool)), this, SLOT(HandleTimerStartStop(bool)));
    connect(ui->sld_autoScrollSpeed, SIGNAL(valueChanged(int)), this, SLOT(HandleTimerIntervalChanged(int)));
    //connect(ui->actionDelete_all_markers_in_projection, SIGNAL(triggered(bool)), ts, SLOT(DeleteAllMarkersFromCurrentImage()));
    connect(mTimer, SIGNAL(timeout()), this, SLOT(HandleTimerTick()));
    mTimer->setSingleShot(false);
    // Set to corresponding interval of default slider value
    mTimer->setInterval(1);

    ui->txt_ContrastCenter->setMinRange(0);
    ui->txt_ContrastCenter->setMaxRange(1);
    ui->txt_ContrastWidth->setMinRange(-0.5f);
    ui->txt_ContrastWidth->setMaxRange(0.5f);
    ui->sld_ContrastWidth->setDefaultValue(1.0f);
    ui->dockTiltSettings->setVisible(false);
    ui->dockAlignment->setVisible(false);

    //enable/disable controls if busy:
    connect(this, SIGNAL(IsNotBusyChanged(bool)), ui->dockAlignment, SLOT(setEnabled(bool)));
    connect(this, SIGNAL(IsNotBusyChanged(bool)), ui->dockTiltSettings, SLOT(setEnabled(bool)));
    connect(this, SIGNAL(IsNotBusyChanged(bool)), this, SLOT(HandleIsNotBusyChanged(bool)));
    connect(this, SIGNAL(IsNotBusyChanged(bool)), ui->actionAlign_tilt_series, SLOT(setEnabled(bool)));
    connect(this, SIGNAL(IsNotBusyChanged(bool)), ui->actionOpen_single_frame_image, SLOT(setEnabled(bool)));
    connect(this, SIGNAL(IsNotBusyChanged(bool)), ui->actionOpen_tilt_series, SLOT(setEnabled(bool)));
    connect(this, SIGNAL(IsNotBusyChanged(bool)), ui->actionOpen_marker_file, SLOT(setEnabled(bool)));

    connect(ts, SIGNAL(StartClickOperation()), this, SLOT(HandleStartClickOperation()));
    connect(ts, SIGNAL(EndClickOperation()), this, SLOT(HandleStopClickOperation()));

    connect(ts, SIGNAL(BeamDeclinationChanged(float)), ui->txt_BeamDeclination, SLOT(setValue(float)));
    connect(ui->txt_BeamDeclination, SIGNAL(valueChanged(float)), ts, SLOT(SetBeamDeclination(float)));
    connect(ui->chk_AlignBeacmDecl, SIGNAL(toggled(bool)), this, SLOT(HandleBeamDeclinationEnabled(bool)));
    connect(ts, SIGNAL(MagAnisotropyFactorChanged(float)), ui->txt_MagAnisoFactor, SLOT(setValue(float)));
    connect(ts, SIGNAL(MagAnisotropyAngleChanged(float)), ui->txt_MagAnisoAngle, SLOT(setValue(float)));

    connect(ts, SIGNAL(MarkerShortFilenameChanged(QString)), ui->txt_Markerfilename, SLOT(setText(QString)));

    connect(ui->chk_FilterImage, SIGNAL(toggled(bool)), ts, SLOT(SetFilterImage(bool)));
    connect(ts, SIGNAL(SetFilterImageChanged(bool)), ui->chk_FilterImage, SLOT(setChecked(bool)));
}

MyMainWindow::~MyMainWindow()
{
    delete ui;

    if (sf)
        delete sf;
    sf = NULL;

    if (ts)
        delete ts;
    ts = NULL;

    delete mTimer;
}

bool MyMainWindow::CanLoadFile(QString fileName)
{
    bool res = SingleFrame::CanReadFile(fileName.toStdString());
    res |= TiltSeries::CanReadFile(fileName.toStdString());

    return res;
}

bool MyMainWindow::LoadFile(QString fileName)
{
    if (SingleFrame::CanReadFile(fileName.toStdString()))
    {
        openSingleFrame(fileName);
        return true;
    }
    if (TiltSeries::CanReadFile(fileName.toStdString()))
    {
        openTiltSeries(fileName);
        return true;
    }
    return false;
}

void MyMainWindow::closeEvent(QCloseEvent *event)
{
    if (ts)
    {
        if (ts->IsMarkerfileLoaded())
        {
            int res = QMessageBox::warning(NULL, tr("Clicker"),
                                           tr("There is a marker file loaded. Discard possibly unsaved changes?"), QMessageBox::Yes, QMessageBox::No | QMessageBox::Default);

            if (res == QMessageBox::No)
            {
                event->ignore();
                return;
            }
        }
    }
    event->accept();
}



void MyMainWindow::openSingleFrame()
{
    if (!mIsNotBusy)
        return;

    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open Image"), "", tr("Image Files (*.dm3 *.dm4 *.tif *.ser *.mrc *.st *.lsm)"));


    if (fileName.length() == 0)
        return;

    openSingleFrame(fileName);

}

void MyMainWindow::openSingleFrame(QString fileName)
{
    CloseAllFiles();
    fileLoadProgress->setRange(0, 100);
    fileLoadProgress->setValue(0);
    fileLoadProgress->show();
    fileLoadProgress->setMaximumHeight(10);
    statusBar()->addPermanentWidget(fileLoadProgress);
    statusBar()->showMessage("Loading file: " + fileName);

    backgroundThread = new QThread();
    singleFrameLoader = new SingleFrameLoader(fileName, sf);
    singleFrameLoader->moveToThread(backgroundThread);

    connect(backgroundThread, SIGNAL (started()), singleFrameLoader, SLOT (process()));
    connect(singleFrameLoader, SIGNAL (finished()), backgroundThread, SLOT (quit()));
    connect(singleFrameLoader, SIGNAL (finished()), singleFrameLoader, SLOT (deleteLater()));
    connect(backgroundThread, SIGNAL (finished()), backgroundThread, SLOT (deleteLater()));
    connect(singleFrameLoader, SIGNAL(loadingDone(int, QString)), this, SLOT(HandleFileLoadDoneSingleFrame(int,QString)));
    backgroundThread->start();
    emit IsNotBusyChanged(false);
}

void MyMainWindow::openTiltSeries()
{
    if (!mIsNotBusy)
        return;

    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open tilt series"), "", tr("Tilt series files (*.dm3 *.dm4 *.mrc *.mrcs *.st)"));


    if (fileName.length() == 0)
        return;

    openTiltSeries(fileName);
}

void MyMainWindow::openTiltSeries(QString fileName)
{
    CloseAllFiles();
    fileLoadProgress->setRange(0, 100);
    fileLoadProgress->setValue(0);
    fileLoadProgress->show();
    fileLoadProgress->setMaximumHeight(10);
    statusBar()->addPermanentWidget(fileLoadProgress);
    statusBar()->showMessage("Loading file: " + fileName);

    backgroundThread = new QThread();
    tiltSeriesLoader = new TiltSeriesLoader(fileName, ts);
    tiltSeriesLoader->moveToThread(backgroundThread);
    connect(backgroundThread, SIGNAL (started()), tiltSeriesLoader, SLOT (process()));
    connect(tiltSeriesLoader, SIGNAL (finished()), backgroundThread, SLOT (quit()));
    connect(tiltSeriesLoader, SIGNAL (finished()), tiltSeriesLoader, SLOT (deleteLater()));
    connect(backgroundThread, SIGNAL (finished()), backgroundThread, SLOT (deleteLater()));
    connect(tiltSeriesLoader, SIGNAL(loadingDone(int, QString)), this, SLOT(HandleFileLoadDoneTiltSeries(int,QString)));
    backgroundThread->start();
    emit IsNotBusyChanged(false);
}

void MyMainWindow::AlignTiltSeries()
{
    if (!mIsNotBusy)
        return;

    HandleTimerStartStop(false);
    fileLoadProgress->setRange(0, 100);
    fileLoadProgress->setValue(0);
    fileLoadProgress->show();
    fileLoadProgress->setMaximumHeight(10);
    statusBar()->addPermanentWidget(fileLoadProgress);
    statusBar()->showMessage("Aligning tilt series...");

    backgroundThread = new QThread();
    tiltSeriesAligner = new TiltSeriesAligner(ui->txt_BeamDeclination->GetValue(), ui->chk_AlignImagRotation->isChecked(), ui->chk_FixedImageRot->isChecked(), ui->chk_AlignTiltAngles->isChecked(),
                                             ui->chk_AlignBeacmDecl->isChecked(), ui->chk_AlignMag->isChecked(),
                                             ui->rb_NormMinimum->isChecked(), ui->rb_NormZeroTilt->isChecked(),
                                             ui->rb_MagsFirst->isChecked(), ui->txt_IterationSwitch->GetValue(),
                                             ui->txt_Iterations->GetValue(), ui->txt_MagAnisoFactor->GetValue(),
                                             ui->txt_MagAnisoAngle->GetValue(), ui->txt_AddZShift->GetValue(), ts);
    tiltSeriesAligner->moveToThread(backgroundThread);
    connect(backgroundThread, SIGNAL (started()), tiltSeriesAligner, SLOT (process()));
    connect(tiltSeriesAligner, SIGNAL (finished()), backgroundThread, SLOT (quit()));
    connect(tiltSeriesAligner, SIGNAL (finished()), tiltSeriesAligner, SLOT (deleteLater()));
    connect(backgroundThread, SIGNAL (finished()), backgroundThread, SLOT (deleteLater()));
    connect(tiltSeriesAligner, SIGNAL(alignmentDone(int)), this, SLOT(HandleEndProgress(int)));
    connect(tiltSeriesAligner, SIGNAL(alignmentReport(QString)), this, SLOT(HandleAlignmentReport(QString)));
    backgroundThread->start();
    emit IsNotBusyChanged(false);
}

void MyMainWindow::CreateNewMarkerfile()
{
    if (!mIsNotBusy)
        return;

    if (ts->IsMarkerfileLoaded())
    {
        int res = QMessageBox::warning(NULL, tr("Clicker"),
                                       tr("There is already a marker file loaded. Discard unsaved changes?"), QMessageBox::Yes | QMessageBox::Default, QMessageBox::No);

        if (res != QMessageBox::Yes)
            return;
    }
    ts->createNewMarkerFile();
}

void MyMainWindow::LoadMarkerfile()
{
    if (!mIsNotBusy)
        return;

    if (ts->IsMarkerfileLoaded())
    {
        int res = QMessageBox::warning(NULL, tr("Clicker"),
                                       tr("There is already a marker file loaded. Discard unsaved changes?"), QMessageBox::Yes | QMessageBox::Default, QMessageBox::No);

        if (res != QMessageBox::Yes)
            return;
    }
    QString filename = QFileDialog::getOpenFileName(this,
        tr("Open marker file"), "", tr("Marker files (*.em);;IMOD fiducial files (*.fid)"));


    if (filename.length() == 0)
        return;

    int res = -1;

    if (filename.endsWith(".em"))
    {
        res = ts->openMarkerFile(filename);
    }
    else if (filename.endsWith(".fid"))
    {
        res = ts->importMarkerFromImod(filename);
    }

    if (res != 0)
    {
        QMessageBox::warning(NULL, tr("Clicker"),
                                       tr("Can't read the following file:\n'") + filename + "'\nMake sure it is a marker file for this tilt series.", QMessageBox::Ok);
    }
}

void MyMainWindow::SaveMarkerfile()
{
    if (!mIsNotBusy)
        return;

    if (ts->IsMarkerfileLoaded())
    {
        QString filename = QFileDialog::getSaveFileName(this,
            tr("Save marker file"), "", tr("Marker Files (*.em)"));


        if (filename.length() == 0)
            return;

        if (!filename.endsWith(".em"))
        {
            filename += ".em";
        }

        int res = ts->saveMarkerFile(filename);
        if (res != 0)
        {
            QMessageBox::warning(NULL, tr("Clicker"),
                                           tr("Could not save the marker file \n'") + filename + "'", QMessageBox::Ok);
        }
    }
}

void MyMainWindow::HandleSaveImage()
{
    QImage fb = ui->openGLWidget->grabFramebuffer();

    QString filename = QFileDialog::getSaveFileName(this,
        tr("Save image"), "", tr("PNG Image file (*.png)"));

    if (filename.length() == 0)
        return;

    if (!filename.endsWith(".png"))
    {
        filename += ".png";
    }

    fb.save(filename);
}

void MyMainWindow::HandleSaveImageSeries()
{
    if (!ts) return;


    if (ts->GetIsLoaded())
    {
        QString filename = QFileDialog::getSaveFileName(this,
            tr("Save image"), "", tr("PNG Image file (*.png)"));

        if (filename.length() == 0)
            return;

        if (filename.endsWith(".png"))
            filename = filename.remove(filename.length() - 4, 4);

        int count = ts->GetImageCount();

        statusBar()->showMessage("Saving image series...");

        for (int i = 0; i < count; i++)
        {
            ui->sld_tiltIndex->setValue(i+1);
            //this->update();
            this->repaint();
            QImage fb = ui->openGLWidget->grabFramebuffer();
            fb.save(filename + QString::asprintf("_%03d", i) + ".png");
        }

        statusBar()->clearMessage();
    }
}

void MyMainWindow::HandlePageUpEvent()
{
    if (!mIsNotBusy || mClickInProgress)
        return;

    if (ts->GetIsLoaded())
    {
        int currentValue = ui->sld_tiltIndex->value();
        currentValue++;
        ui->sld_tiltIndex->setValue(currentValue);
    }
}

void MyMainWindow::HandlePageDownEvent()
{
    if (!mIsNotBusy || mClickInProgress)
        return;

    if (ts->GetIsLoaded())
    {
        int currentValue = ui->sld_tiltIndex->value();
        currentValue--;
        ui->sld_tiltIndex->setValue(currentValue);
    }
}

void MyMainWindow::HandleFileLoadUpdate(int value)
{
    fileLoadProgress->setValue(value);
}

void MyMainWindow::HandleFileLoadDoneTiltSeries(int statusCode, QString filename)
{
    if (statusCode < 0)
    {
        ui->openGLWidget->SetOpenCLProcessor(NULL);
        ui->openGLWidget->setNewImageDimensions(ts->GetWidth(), ts->GetHeight(), ts->GetIsMultiChannel());
        ui->openGLWidget->setPixelSize(0);
        ui->dockTiltSettings->setVisible(false);
        ui->dockAlignment->setVisible(false);
        ui->actionSave_image_series->setEnabled(false);
        this->setWindowTitle("Clicker");
        QMessageBox::warning(NULL, tr("Clicker"),
                                       tr("Can't read the following file:\n'") + filename + "'\nMake sure it is a tilt series.", QMessageBox::Ok, QMessageBox::Ok);

    }
    else
    {
        ui->openGLWidget->SetOpenCLProcessorUserData(ts->GetUserData());
        ui->openGLWidget->SetOpenCLProcessor(&TiltSeriesController::processWithOpenCL);
        ui->openGLWidget->setNewImageDimensions(ts->GetWidth(), ts->GetHeight(), ts->GetIsMultiChannel());
        ui->openGLWidget->setPixelSize(ts->GetPixelSize());
        ui->openGLWidget->setShowScaleBar(true);
        ui->dockTiltSettings->setVisible(true);
        ui->dockAlignment->setVisible(true);
        ui->actionSave_image_series->setEnabled(true);
        this->setWindowTitle("Clicker - " + filename);

        HandleProjectionsParametersChanged(ts->GetProjectionParameters(0));
    }
    statusBar()->removeWidget(fileLoadProgress);
    statusBar()->clearMessage();
    emit IsNotBusyChanged(true);
    ui->sld_tiltIndex->setValue(1);
}

void MyMainWindow::HandleFileLoadDoneSingleFrame(int statusCode, QString filename)
{
    if (statusCode < 0)
    {
        ui->openGLWidget->SetOpenCLProcessor(NULL);
        ui->openGLWidget->setNewImageDimensions(sf->GetWidth(), sf->GetHeight(), sf->GetIsMultiChannel());
        ui->openGLWidget->setPixelSize(0);
        ui->dockTiltSettings->setVisible(false);
        ui->dockAlignment->setVisible(false);
        ui->actionSave_image_series->setEnabled(false);
        this->setWindowTitle("Clicker");
        QMessageBox::warning(NULL, tr("Clicker"),
                                       tr("Can't read the following file:\n'") + filename + "'\nMake sure it is a single frame image.", QMessageBox::Ok, QMessageBox::Ok);

    }
    else
    {
        ui->openGLWidget->SetOpenCLProcessorUserData(sf->GetUserData());
        ui->openGLWidget->SetOpenCLProcessor(&SingleFrameController::processWithOpenCL);
        ui->openGLWidget->setNewImageDimensions(sf->GetWidth(), sf->GetHeight(), sf->GetIsMultiChannel());
        ui->openGLWidget->setPixelSize(sf->GetPixelSize());
        ui->openGLWidget->setShowScaleBar(true);
        ui->dockTiltSettings->setVisible(false);
        ui->dockAlignment->setVisible(false);
        ui->actionSave_image_series->setEnabled(false);
        this->setWindowTitle("Clicker - " + filename);
    }
    statusBar()->removeWidget(fileLoadProgress);
    statusBar()->clearMessage();
    emit IsNotBusyChanged(true);
}

void MyMainWindow::HandlePrepareData()
{
    statusBar()->showMessage("Preparing data...");
}

void MyMainWindow::HandleProjectionsParametersChanged(ProjectionParameters params)
{
    QString index = QStringLiteral("%1 / %2").arg(params.index).arg(params.totalNumber);
    ui->txt_TiltNr->setText(index);
    QString angle = QString::asprintf("%0.2f%c", params.tiltAngleUnaligned, 0x00B0);
    ui->txt_TiltAngle->setText(angle);
}

void MyMainWindow::HandleProgressUpdate(int value)
{
    fileLoadProgress->setValue(value);
}

void MyMainWindow::HandleStartProgress(QString message)
{
    fileLoadProgress->setRange(0, 100);
    fileLoadProgress->setValue(0);
    fileLoadProgress->show();
    fileLoadProgress->setMaximumHeight(10);
    statusBar()->addPermanentWidget(fileLoadProgress);
    statusBar()->showMessage(message);
}

void MyMainWindow::HandleEndProgress(int statusCode)
{
    if (statusCode < 0)
    {
        QMessageBox::warning(NULL, tr("Clicker"),
                                       tr("Error during operation. Aborting..."), QMessageBox::Ok, QMessageBox::Ok);

    }
    statusBar()->removeWidget(fileLoadProgress);
    statusBar()->clearMessage();
    emit IsNotBusyChanged(true);
}

void MyMainWindow::HandleCCImageChanged(QImage &img1, QImage &img2)
{
    ui->img_cc->setScaledContents(true);
    ui->img_cc->setPixmap(QPixmap::fromImage(img1));

    ui->img_Crop->setScaledContents(true);
    ui->img_Crop->setPixmap(QPixmap::fromImage(img2));

}

void MyMainWindow::HandleTimerStartStop(bool start)
{
    if (start)
    {   
        mTimer->start();
    }
    else
    {
        mTimer->stop();
        ui->chk_autoLoop->setChecked(false);
    }
}

void MyMainWindow::HandleTimerTick()
{
    if (!mIsNotBusy)
        return;

    if (ui->chk_SkipUnaligned->isChecked())
    {
        int current = ui->sld_tiltIndex->value()-1;
        int next = current + mAutoScrollDirection;

        int safety = 0;
        while (true)
        {
            if (next >= ts->GetImageCount() || next < 0)
            {
                mAutoScrollDirection *= -1;
                next = current + mAutoScrollDirection;
            }
            else
            {
                if (ts->IsProjectionAligned(next))
                {
                    ui->sld_tiltIndex->setValue(next+1);
                    break;
                }
                else
                {
                    next = next + mAutoScrollDirection;
                }
            }
            safety++;
            if (safety > ts->GetImageCount() * 2)
            {
                break;
            }
        }
    }
    else
    {
        int current = ui->sld_tiltIndex->value()-1;
        int next = current + mAutoScrollDirection;

        int safety = 0;
        while (true)
        {
            if (next >= ts->GetImageCount() || next < 0)
            {
                mAutoScrollDirection *= -1;
                next = current + mAutoScrollDirection;
            }
            else
            {
                ui->sld_tiltIndex->setValue(next+1);
                break;
            }
            safety++;
            if (safety > ts->GetImageCount() * 2)
            {
                break;
            }
        }
    }
}

void MyMainWindow::HandleTimerIntervalChanged(int value)
{
    // Convert from slider speed to interval value
    int maxSpeed = 500;
    int minSpeed = 1;
    int maxInterval = 500;
    int interval = (int)(maxInterval * (1 - (float)value / maxSpeed)) + 1;

    mTimer->setInterval(interval);
}

void MyMainWindow::HandleAlignmentReport(QString report)
{
    AlignmentReport* ar = new AlignmentReport(this);
    ar->setWindowModality(Qt::WindowModal);
    ar->open();
    ar->SetText(report);
    ts->EmitMarkerPositions();
}

void MyMainWindow::HandleIsNotBusyChanged(bool isNotBusy)
{
    if (mIsNotBusy != isNotBusy)
    {
        mIsNotBusy = isNotBusy;
        emit IsNotBusyChanged(isNotBusy);
    }
}

void MyMainWindow::HandleStartClickOperation()
{
    mClickInProgress = true;
}

void MyMainWindow::HandleStopClickOperation()
{
    mClickInProgress = false;
}

void MyMainWindow::HandleTiltIndexDoubleClick()
{
    if (ts->GetIsLoaded())
    {
        int idx = ts->GetZeroTiltIndex();
        ui->sld_tiltIndex->setValue(idx + 1);
    }
}

void MyMainWindow::HandleRemoveMarker()
{
    if (ts->IsMarkerfileLoaded())
    {
        if (ts->GetMarkerCount() == 1)
        {
            QMessageBox::information(NULL, tr("Clicker"),
                                           tr("The first marker cannot be removed."), QMessageBox::Ok, QMessageBox::Ok);
            return;
        }

        int res = QMessageBox::question(NULL, tr("Clicker"),
                                       tr("Remove the currently active marker from the entire tilt series?"), QMessageBox::Yes | QMessageBox::Default, QMessageBox::No);

        if (res != QMessageBox::Yes)
            return;
        ts->removeMarker();
    }
}

void MyMainWindow::HandleBeamDeclinationEnabled(bool value)
{
    ui->txt_BeamDeclination->setEnabled(!value);
}

void MyMainWindow::HandleReconstructionAssistant()
{
    if (ts)
    {
        ReconstructionAssistant* ra = new ReconstructionAssistant(ts, this);
        ra->setWindowModality(Qt::WindowModal);
        ra->open();
    }
}

void MyMainWindow::HandleQuit()
{
    if (ts)
    {
        if (ts->IsMarkerfileLoaded())
        {
            int res = QMessageBox::warning(NULL, tr("Clicker"),
                                           tr("There is a marker file loaded. Discard possibly unsaved changes?"), QMessageBox::Yes, QMessageBox::No | QMessageBox::Default);

            if (res == QMessageBox::No)
            {
                return;
            }
        }
    }
    QApplication::quit();
}

void MyMainWindow::CloseAllFiles()
{
    HandleTimerStartStop(false); //stop looping through tilt series, if it does...
    ts->CloseFile();
    sf->CloseFile();

}


TiltSeriesAligner::TiltSeriesAligner(float phi, bool DoPsi, bool DoPsiFixed, bool DoTheta, bool DoPhi, bool DoMags, bool normMin, bool normZeroTilt, bool magsFirst, int iterSwitch, int iterations, float ansiFactor, float ansiAngle, int addZShift, TiltSeriesController *aTS)
    : mPhi(phi),
      mDoPsi(DoPsi),
      mDoPsiFixed(DoPsiFixed),
      mDoTheta(DoTheta),
      mDoPhi(DoPhi),
      mDoMags(DoMags),
      mNormMin(normMin),
      mNormZeroTilt(normZeroTilt),
      mMagsFirst(magsFirst),
      mIterSwitch(iterSwitch),
      mIterations(iterations),
      mAnsiFactor(ansiFactor),
      mAnsiAngle(ansiAngle),
      mAddZShift(addZShift),
      ts(aTS)
{

}

void TiltSeriesAligner::process()
{
    try
    {
        ts->SetBeamDeclination(mPhi);
        int code = ts->AlignMarker(mDoPsi, mDoPsiFixed, mDoTheta, mDoPhi, mDoMags, mNormMin, mNormZeroTilt, mMagsFirst, mIterSwitch, mIterations, mAnsiFactor, mAnsiAngle, mAddZShift);
        QString report = ts->GetAlignmentReport();
        //qDebug() << report;
        if (report.size() > 2)
        {
            emit alignmentReport(report);
        }
        emit alignmentDone(code);
    }
    catch (std::exception&)
    {
        emit alignmentDone(-1);
    }

}


TiltSeriesLoader::TiltSeriesLoader(QString aFilename, TiltSeriesController *aTS) :
    ts(aTS), filename(aFilename)
{

}

void TiltSeriesLoader::process()
{
    try
    {
        int code = ts->openFile(filename);
        emit loadingDone(code, filename);
    }
    catch (std::exception&)
    {
        emit loadingDone(-1, filename);
        //QMessageBox::warning(this, tr("Clicker"), tr("Cannot open file:\n'") + ex.what(), QMessageBox::Ok, QMessageBox::Ok);

    }
}

SingleFrameLoader::SingleFrameLoader(QString aFilename, SingleFrameController *aImage) :
    image(aImage), filename(aFilename)
{

}

void SingleFrameLoader::process()
{
    try
    {
        int code = image->openFile(filename);
        emit loadingDone(code, filename);
    }
    catch (std::exception&)
    {
        emit loadingDone(-1, filename);
    }
}

