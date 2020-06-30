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


#ifndef FLOATLINEEDIT_H
#define FLOATLINEEDIT_H

#include <QLineEdit>


class FloatLineEdit : public QLineEdit
{
    Q_OBJECT

public:
    explicit FloatLineEdit(QWidget *parent = 0);
    float GetValue();
public slots:
    void setText(const QString& text);
    void setValue(const float& aValue);
    void setValueNormalized(const float& aValue);
    void setMinRange(float aValue);
    void setMaxRange(float aValue);

signals:
    void valueChanged(float value);
    void valueChangedNormalized(float value);

private:
    float mValue;
    float mMinValue;
    float mMaxValue;
};
#endif // FLOATLINEEDIT_H
