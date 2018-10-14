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


#ifndef ALIGNMENTREPORT_H
#define ALIGNMENTREPORT_H

#include <QDialog>

namespace Ui {
class AlignmentReport;
}

class AlignmentReport : public QDialog
{
    Q_OBJECT

public:
    explicit AlignmentReport(QWidget *parent = 0);
    ~AlignmentReport();
    void SetText(QString text);

private:
    Ui::AlignmentReport *ui;
};

#endif // ALIGNMENTREPORT_H
