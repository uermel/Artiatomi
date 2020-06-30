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


#ifndef RECONSTRUCTIONASSISTANT_H
#define RECONSTRUCTIONASSISTANT_H

#include <QDialog>
#include <MarkerFile.h>
#include <tiltseriescontroller.h>

namespace Ui {
class ReconstructionAssistant;
}

class ReconstructionAssistant : public QDialog
{
    Q_OBJECT

public:
    //ReconstructionAssistant(QWidget *parent = 0);
    explicit ReconstructionAssistant(TiltSeriesController* ts, QWidget* parent = 0);

    ~ReconstructionAssistant();

protected:
    void accept();

private:
    Ui::ReconstructionAssistant *ui;
    int mProjWidth;
    int mProjHeight;
    float mPixelSize;
    float mZMin;
    float mZMax;
    void writeToFile(QTextStream& file);
    int GetDeviceCount();

private slots:
    void BinningChanged(int idx);
    void FP16Changed(bool value);
    void OpenMarkerfile();
    void OpenProjectionfile();
    void OpenCTFFile();
    void SetVolumeFile();

};

#endif // RECONSTRUCTIONASSISTANT_H
