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


#include "cropsizeselector.h"

CropSizeSelector::CropSizeSelector(QWidget *parent)
    : QComboBox(parent)
{
    this->addItem("64");
    this->addItem("128");
    this->addItem("256");
    this->addItem("512");

    this->setCurrentIndex(2);
    this->setEditable(false);
    //this->setEnabled(false);
    connect(this, SIGNAL(currentTextChanged(QString)), this, SLOT(convertFromString(QString)));
}

void CropSizeSelector::convertFromString(QString value)
{
    emit valueChanged(value.toInt());
}
