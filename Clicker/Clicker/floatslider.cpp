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


#include "floatslider.h"

void FloatSlider::mouseDoubleClickEvent(QMouseEvent *e)
{
    setValue(mDefaultValue);
}

void FloatSlider::setValue(int aValue)
{
    emit valueChanged(aValue / mFactor);
    QSlider::setValue(aValue);
    if (value() != aValue)
    {
    }
}

void FloatSlider::setValue(float value)
{
    setValue((int)round(value * mFactor));
}

void FloatSlider::setDefaultValue(float value)
{
    if (mDefaultValue != value)
    {
        mDefaultValue = value;
    }
}

FloatSlider::FloatSlider(QWidget *parent) :
    QSlider(parent), mFactor(50000.0f), mDefaultValue(0.5f)
{
    setMinimum(0);
    setMaximum((int)mFactor);
    connect(this, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));
}
