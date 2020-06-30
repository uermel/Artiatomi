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


#include "floatlineedit.h"

FloatLineEdit::FloatLineEdit(QWidget *parent) :
    QLineEdit(parent), mValue(-1000), mMinValue(0), mMaxValue(1)
{
    connect(this, SIGNAL(textEdited(QString)), this, SLOT(setText(QString)));
}

float FloatLineEdit::GetValue()
{
    return mValue;
}

void FloatLineEdit::setValue(const float &aValue)
{
    if (aValue != mValue)
    {
        mValue = aValue;
        QLineEdit::setText(QString::asprintf("%0.3f", mValue));
        emit valueChanged(mValue);
        emit valueChangedNormalized((mValue - mMinValue) / (mMaxValue - mMinValue));
        update();
    }
}

void FloatLineEdit::setValueNormalized(const float &aValue)
{
    float val = aValue * (mMaxValue - mMinValue) + mMinValue;
    if (mValue != val)
    {
        mValue = val;
        QLineEdit::setText(QString::asprintf("%0.3f", mValue));
        emit valueChanged(mValue);
        emit valueChangedNormalized((mValue - mMinValue) / (mMaxValue - mMinValue));
        update();
    }
}

void FloatLineEdit::setMinRange(float aValue)
{
    if (aValue != mMinValue)
    {
        mMinValue = aValue;
        emit valueChanged(mValue);
        emit valueChangedNormalized((mValue - mMinValue) / (mMaxValue - mMinValue));
        update();
    }
}

void FloatLineEdit::setMaxRange(float aValue)
{
    if (aValue != mMaxValue)
    {
        mMaxValue = aValue;
        emit valueChanged(mValue);
        emit valueChangedNormalized((mValue - mMinValue) / (mMaxValue - mMinValue));
        update();
    }
}

void FloatLineEdit::setText(const QString& text)
{
    bool ok;
    QString newText = text;
    newText = newText.replace(",", ".");
    float val = newText.toFloat(& ok);

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
