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


#include "reconstructionassistant.h"
#include "ui_reconstructionassistant.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QClipboard>

ReconstructionAssistant::ReconstructionAssistant(TiltSeriesController* ts, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ReconstructionAssistant)
{
    ui->setupUi(this);
    mProjWidth = ts->GetWidth();
    mProjHeight = ts->GetHeight();
    int maxSize = max(mProjWidth, mProjHeight);
    mPixelSize = ts->GetPixelSize();

    connect(ui->cmb_Binning, SIGNAL(currentIndexChanged(int)), this, SLOT(BinningChanged(int)));
    connect(ui->chk_FP16, SIGNAL(toggled(bool)), this, SLOT(FP16Changed(bool)));

    connect(ui->actionOpen_CTF_file, SIGNAL(triggered(bool)), this, SLOT(OpenCTFFile()));
    connect(ui->actionOpen_marker_file, SIGNAL(triggered(bool)), this, SLOT(OpenMarkerfile()));
    connect(ui->actionOpen_projection_file, SIGNAL(triggered(bool)), this, SLOT(OpenProjectionfile()));
    connect(ui->actionSet_output_file, SIGNAL(triggered(bool)), this, SLOT(SetVolumeFile()));

    ui->txt_LP->setMaximum(maxSize / 2);
    ui->txt_LPS->setMaximum(maxSize / 2);
    ui->txt_HP->setMaximum(maxSize / 2);
    ui->txt_HPS->setMaximum(maxSize / 2);

    ui->cmb_Binning->addItem("1");
    ui->cmb_Binning->addItem("2");
    ui->cmb_Binning->addItem("4");
    ui->cmb_Binning->addItem("8");
    ui->cmb_Binning->addItem("16");
    ui->cmb_Binning->setCurrentIndex(1);

    ui->cmb_NomalizationMethod->addItem("Mean value");
    ui->cmb_NomalizationMethod->addItem("Standard deviation");
    ui->cmb_NomalizationMethod->setCurrentIndex(1);

    ui->txt_Projection->setText(ts->GetFilename());
    ui->txt_markerFile->setText(ts->GetMarkerFilename());
    ui->txt_beamDeclination->setValue(ts->GetBeamDeclination());
    float amount, angle;
    ts->GetMagAnisotropy(amount, angle);
    ui->txt_MagAnisoAmount->setValue(amount);
    ui->txt_MagAnisoAngle->setValue(angle);

    ui->txt_SIRTCount->setMaximum(ts->GetImageCount());
    ui->txt_BadPixelValue->setValue(4 * roundf(ts->GetMeanStd()));

    ts->GetMarkerMinMaxZ(mZMin, mZMax);
    ui->tabWidget->setCurrentIndex(0);
    BinningChanged(1);
}

ReconstructionAssistant::~ReconstructionAssistant()
{
    delete ui;
}

void ReconstructionAssistant::accept()
{
    //Make some basic validity tests:
    if (ui->txt_OutputVolumefile->text().length() == 0)
    {
        QMessageBox::warning(NULL, tr("Clicker"),
                                       tr("Output volume filename is not set."), QMessageBox::Ok | QMessageBox::Default);
        return;
    }
    if (ui->txt_Projection->text().length() == 0)
    {
        QMessageBox::warning(NULL, tr("Clicker"),
                                       tr("Projection filename is not set."), QMessageBox::Ok | QMessageBox::Default);
        return;
    }
    if (ui->txt_markerFile->text().length() == 0)
    {
        QMessageBox::warning(NULL, tr("Clicker"),
                                       tr("Marker filename is not set."), QMessageBox::Ok | QMessageBox::Default);
        return;
    }
    if (ui->txt_CTFFilename->text().length() == 0 && ui->chk_CTFCorrection->isChecked())
    {
        QMessageBox::warning(NULL, tr("Clicker"),
                                       tr("CTF filename is not set."), QMessageBox::Ok | QMessageBox::Default);
        return;
    }

    QString filename = QFileDialog::getSaveFileName(this,
        tr("Save configuration as"), "", tr("EmSART config file (*.cfg)"));

    int deviceCount = GetDeviceCount();


    if (filename.length() != 0)
    {

        if (!(filename.endsWith(".cfg")) )
        {
            filename += ".cfg";
        }
        QFile file(filename);
        if (file.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&file);
            writeToFile(stream);
            file.close();

            QString command;
            if (deviceCount < 2)
            {
                command = "EmSART -u \"" + filename + "\"";
            }
            else
            {
                command = "mpiexec -n " + QString::asprintf("%d", deviceCount) + " EmSART -u \"" + filename + "\"";
            }

            QApplication::clipboard()->setText(command);
            QMessageBox::information(NULL, tr("Clicker"),
                                           ("Copied following command to clipboard:\n" + command), QMessageBox::Ok | QMessageBox::Default);
        }
        else
        {
            QMessageBox::warning(NULL, tr("Clicker"),
                                           tr("Could not write EmSART config file. Check permissions..."), QMessageBox::Ok | QMessageBox::Default);
            return;
        }
        QDialog::accept();
    }
    return;
}

void ReconstructionAssistant::writeToFile(QTextStream &stream)
{
    stream << "CudaDeviceID = " << ui->txt_CudaDeviceIDs->text() << "\r\n";
    stream << "Lambda = 1\r\n";
    stream << "Iterations = 1\r\n";

    stream << "ProjectionFile = " << ui->txt_Projection->text() << "\r\n";
    stream << "OutVolumeFile = " << ui->txt_OutputVolumefile->text() << "\r\n";
    stream << "MarkerFile = " << ui->txt_markerFile->text() << "\r\n";
    stream << "CtfFile = " << (ui->txt_CTFFilename->text().length() == 0 ? "nothing" : ui->txt_CTFFilename->text()) << "\r\n";

    stream << "RecDimesions = " << ui->txt_VolDimX->text() << " " << ui->txt_VolDimY->text() << " "  << ui->txt_VolDimZ->text() << "\r\n";
    stream << "VolumeShift = " << ui->txt_VolumeShiftX->text() << " " << ui->txt_VolumeShiftY->text() << " "  << ui->txt_VolumeShiftZ->text() << "\r\n";
    stream << "VoxelSize = " << ui->cmb_Binning->currentText() << "\r\n";
    stream << "AddTiltAngle = " << ui->txt_AddTiltAngle->text() << "\r\n";
    stream << "AddTiltXAngle = " << ui->txt_AddTiltXAngle->text() << "\r\n";

    stream << "UseFixPsiAngle = false\r\n";
    stream << "PsiAngle = 0\r\n";
    stream << "PhiAngle = " << ui->txt_beamDeclination->text() << "\r\n";
    stream << "OverSampling = " << ui->txt_OverSampling->text() << "\r\n";

    stream << "CtfMode = " << (ui->chk_CTFCorrection->isChecked() ? "true" : "false") << "\r\n";
    stream << "CTFBetaFac = " << ui->txt_FirstMaximum->text() << " 0 " << ui->txt_BetaFac->text() << " 0\r\n";
    stream << "Cs = " << ui->txt_Cs->text() << "\r\n";
    stream << "Voltage = " << ui->txt_Voltage->text() << "\r\n";
    stream << "IgnoreZShiftForCTF = " << (ui->chk_IgnoreZShift->isChecked() ? "true" : "false") << "\r\n";
    stream << "CTFSliceThickness = " << ui->txt_SliceThickness->text() << "\r\n";

    stream << "SkipFilter = " << (!(ui->chk_Fourier->isChecked()) ? "true" : "false") << "\r\n";
    stream << "fourFilterLP = " << (ui->txt_LP->text().length() == 0 ? "0" : ui->txt_LP->text()) << "\r\n";
    stream << "fourFilterLPS = " << (ui->txt_LPS->text().length() == 0 ? "0" : ui->txt_LPS->text()) << "\r\n";
    stream << "fourFilterHP = " << (ui->txt_HP->text().length() == 0 ? "0" : ui->txt_HP->text()) << "\r\n";
    stream << "fourFilterHPS = " << (ui->txt_HPS->text().length() == 0 ? "0" : ui->txt_HPS->text()) << "\r\n";
    stream << "SIRTCount = " << ui->txt_SIRTCount->text() << "\r\n";
    stream << "CorrectBadPixels = " << (ui->chk_BadPixels->isChecked() ? "true" : "false") << "\r\n";
    stream << "BadPixelValue = " << ui->txt_BadPixelValue->text() << "\r\n";

    stream << "Crop = 50 50 50 50\r\n";
    stream << "CropDim = 10 10 10 10\r\n";
    stream << "DimLength = 50 50\r\n";
    stream << "CutLength = 10 10\r\n";

    stream << "FP16Volume = " << (ui->chk_FP16->isChecked() ? "true" : "false") << "\r\n";
    stream << "WriteVolumeAsFP16 = " << (ui->chk_WriteAsFP16->isChecked() ? "true" : "false") << "\r\n";
    stream << "ProjectionScaleFactor = " << ui->txt_ScaleFactor->text() << "\r\n";

    if (ui->cmb_NomalizationMethod->currentIndex() == 0)
    {
        stream << "ProjectionNormalization = mean\r\n";
    }
    else
    {
        stream << "ProjectionNormalization = std\r\n";
    }
    if (ui->chk_SART->isChecked())
    {
        stream << "WBP = false\r\n";
    }
    else
    {
        stream << "WBP = true\r\n";
        if (ui->chk_WBPContrast->isChecked())
        {
            stream << "WBPFilter = Contrast10\r\n";
        }
        else
        {
            stream << "WBPFilter = Ramp\r\n";
        }
    }

    stream << "MagAnisotropy = " << ui->txt_MagAnisoAmount->text() << " " << ui->txt_MagAnisoAngle->text() << "\r\n";
}

int ReconstructionAssistant::GetDeviceCount()
{
    QString devices = ui->txt_CudaDeviceIDs->text();

    QStringList temp = devices.split(" ");

    return temp.size();
}

void ReconstructionAssistant::BinningChanged(int idx)
{
    int minSize = min(mProjWidth, mProjHeight);
    float thickness = mZMax - mZMin;
    int slices = floor(thickness / 100.0f) * 100 + 100;

    switch (idx)
    {
    case 0:
        ui->txt_VolDimX->setText(QString::number((minSize / 1) - (minSize / 1) % 4 + 4));
        ui->txt_VolDimY->setText(QString::number((minSize / 1) - (minSize / 1) % 4 + 4));
        ui->txt_VolDimZ->setText(QString::number((slices / 1) + 100 - (slices / 1) % 100));
        break;
    case 1:
        ui->txt_VolDimX->setText(QString::number((minSize / 2) - (minSize / 2) % 4 + 4));
        ui->txt_VolDimY->setText(QString::number((minSize / 2) - (minSize / 2) % 4 + 4));
        ui->txt_VolDimZ->setText(QString::number((slices / 2) + 100 - (slices / 2) % 100));
        break;
    case 2:
        ui->txt_VolDimX->setText(QString::number((minSize / 4) - (minSize / 4) % 4 + 4));
        ui->txt_VolDimY->setText(QString::number((minSize / 4) - (minSize / 4) % 4 + 4));
        ui->txt_VolDimZ->setText(QString::number((slices / 4) + 100 - (slices / 4) % 100));
        break;
    case 3:
        ui->txt_VolDimX->setText(QString::number((minSize / 8) - (minSize / 8) % 4 + 4));
        ui->txt_VolDimY->setText(QString::number((minSize / 8) - (minSize / 8) % 4 + 4));
        ui->txt_VolDimZ->setText(QString::number((slices / 8) + 100 - (slices / 8) % 100));
        break;
    case 4:
        ui->txt_VolDimX->setText(QString::number((minSize / 16) - (minSize / 16) % 4 + 4));
        ui->txt_VolDimY->setText(QString::number((minSize / 16) - (minSize / 16) % 4 + 4));
        ui->txt_VolDimZ->setText(QString::number((slices / 16) + 100 - (slices / 16) % 100));
        break;
    }
}

void ReconstructionAssistant::FP16Changed(bool value)
{
    if (value)
    {
        ui->txt_ScaleFactor->setValue(1000.0f);
    }
    else
    {
        ui->txt_ScaleFactor->setValue(1.0f);
    }
}

void ReconstructionAssistant::OpenMarkerfile()
{
    QString filename = QFileDialog::getOpenFileName(this,
        tr("Open marker file"), "", tr("Marker files (*.em)"));


    if (filename.length() != 0)
    {
        ui->txt_markerFile->setText(filename);
    }
}

void ReconstructionAssistant::OpenProjectionfile()
{
    QString filename = QFileDialog::getOpenFileName(this,
        tr("Open tilt series"), "", tr("Tilt series files (*.dm3 *.dm4 *.mrc *.st)"));


    if (filename.length() != 0)
    {
        ui->txt_Projection->setText(filename);
    }
}

void ReconstructionAssistant::OpenCTFFile()
{
    QString filename = QFileDialog::getOpenFileName(this,
        tr("Open CTF file"), "", tr("CTF files (*.em)"));


    if (filename.length() != 0)
    {
        ui->txt_CTFFilename->setText(filename);
    }
}

void ReconstructionAssistant::SetVolumeFile()
{
    QString selectedFilter = "EM file (*.em)";
    QString filename = QFileDialog::getSaveFileName(this,
        tr("Save reconstruction as"), "", tr("EM file (*.em);;MRC files (*.mrc *.rec)"), &selectedFilter);


    if (filename.length() != 0)
    {
        if (!(filename.endsWith(".em") || filename.endsWith(".mrc") || filename.endsWith("*.rec")) )
        {
            filename += ".em";
        }
        ui->txt_OutputVolumefile->setText(filename);
    }
}
