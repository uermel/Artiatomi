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
#include <QApplication>
#include <QSurfaceFormat>
#include <QFont>
#include <QMessageBox>
#include "tiltseriescontroller.h"

int main(int argc, char *argv[])
{
    qRegisterMetaType<ProjectionParameters>("ProjectionParameters");

    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);

    QApplication a(argc, argv);
//    QFont font = a.font();
//    int size = font.pointSize();
//    font.setPointSize(size - 2);
//    a.setFont(font);


    MyMainWindow w;

    QSurfaceFormat f;
    f.setVersion(4,1);
    f.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(f);
    w.setCorner(Qt::TopLeftCorner, Qt::LeftDockWidgetArea);
    w.setCorner(Qt::TopRightCorner, Qt::RightDockWidgetArea);
    w.show();

    if (argc > 1) //open image file?
    {
        QString filename = QString::fromUtf8(argv[1]);
        try
        {
            if (!w.LoadFile(filename))
            {
                QMessageBox::warning(NULL, "Clicker",
                                               "Cannot open file:\n" + filename, QMessageBox::Ok | QMessageBox::Default);
            }
        }
        catch (FileIOException& ex)
        {
            QMessageBox::warning(NULL, "Clicker",
                                           QString::fromUtf8(ex.GetMessage().c_str()), QMessageBox::Ok | QMessageBox::Default);
        }
    }

    return a.exec();
}
