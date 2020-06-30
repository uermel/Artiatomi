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


#include "intlineedit.h"

IntLineEdit::IntLineEdit(QWidget *parent) :
    QLineEdit(parent), mValue(0)
{
    connect(this, SIGNAL(textEdited(QString)), this, SLOT(setText(QString)));
}

int IntLineEdit::GetValue()
{
    return mValue;
}

void IntLineEdit::setValue(const int &aValue)
{
    if (aValue != mValue)
    {
        mValue = aValue;
        QLineEdit::setText(QString::asprintf("%d", mValue));
        emit valueChanged(mValue);
        update();
    }
}

void IntLineEdit::setText(const QString& text)
{
    bool ok;
    QString newText = text;
    int val = newText.toInt(& ok);

    if (ok)
    {
        this->setStyleSheet("background-color:white;");
    }
    else
    {
        this->setStyleSheet("background-color:red;");
    }
    if (ok && mValue != val)
    {
        QLineEdit::setText(newText);
        mValue = val;
        emit valueChanged(mValue);
        update();
    }
    else
    {
        mValue = 0;
        QLineEdit::setText(newText);
        update();
    }
}
