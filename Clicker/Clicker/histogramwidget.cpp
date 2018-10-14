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


#include "histogramwidget.h"
#include <QPainter>

HistogramWidget::HistogramWidget(QWidget *parent) : QWidget(parent),
  mLevels(256), mIsRGB(false), mCenter(-1), mWidth(0)
{

}

void HistogramWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);

    QPainter painter(this);

    painter.fillRect(0, 0, width(), height(), QColor::fromRgb(0, 0, 0));
    if (mHistogramROrGray.isEmpty()) {
        return;
    }
    painter.fillRect(0, 0, width(), height(), QColor::fromRgb(255, 255, 255));

    qreal barWidth = (width() / (qreal)mLevels);
    QBrush RedOrGray(QColor::fromRgb(255, 0, 0, 255));
    QBrush Green(QColor::fromRgb(0, 255, 0, 255));
    QBrush Blue(QColor::fromRgb(0, 0, 255, 255));
    if (!mIsRGB)
        RedOrGray = QBrush(QColor::fromRgb(0, 0, 0, 128));

    for (int i = 0; i < mLevels; i++)
    {
        if (mIsRGB)
        {
            float hR = mHistogramROrGray[i] * height();
            float hG = mHistogramG[i] * height();
            float hB = mHistogramB[i] * height();

            if (hR > hG && hR > hB)
            {
                painter.fillRect(floor(barWidth * i), height() - hR, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
                if (hG > hB)
                {
                    painter.fillRect(floor(barWidth * i), height() - hG, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Green);
                    painter.fillRect(floor(barWidth * i), height() - hB, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Blue);
                }
                else
                {
                    painter.fillRect(floor(barWidth * i), height() - hB, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Blue);
                    painter.fillRect(floor(barWidth * i), height() - hG, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Green);
                }
            }
            else if (hG > hR && hG > hB)
            {
                painter.fillRect(floor(barWidth * i), height() - hG, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Green);
                if (hR > hB)
                {
                    painter.fillRect(floor(barWidth * i), height() - hR, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
                    painter.fillRect(floor(barWidth * i), height() - hB, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Blue);
                }
                else
                {
                    painter.fillRect(floor(barWidth * i), height() - hB, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Blue);
                    painter.fillRect(floor(barWidth * i), height() - hR, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
                }
            }
            else if (hB > hR && hB > hG)
            {
                painter.fillRect(floor(barWidth * i), height() - hB, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Blue);
                if (hR > hG)
                {
                    painter.fillRect(floor(barWidth * i), height() - hR, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
                    painter.fillRect(floor(barWidth * i), height() - hG, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Green);
                }
                else
                {
                    painter.fillRect(floor(barWidth * i), height() - hG, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Green);
                    painter.fillRect(floor(barWidth * i), height() - hR, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
                }
            }
            else
            {
                if (hB == hR && hR == hG)
                {
                    painter.fillRect(floor(barWidth * i), height() - hR, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
                }
                if (hB == hR && hR > hG)
                {
                    painter.fillRect(floor(barWidth * i), height() - hR, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
                    painter.fillRect(floor(barWidth * i), height() - hG, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Green);
                }
                else
                {
                    painter.fillRect(floor(barWidth * i), height() - hG, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Green);
                    painter.fillRect(floor(barWidth * i), height() - hR, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
                }
                if (hG == hR && hR > hB)
                {
                    painter.fillRect(floor(barWidth * i), height() - hR, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
                    painter.fillRect(floor(barWidth * i), height() - hB, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Blue);
                }
                else
                {
                    painter.fillRect(floor(barWidth * i), height() - hB, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Blue);
                    painter.fillRect(floor(barWidth * i), height() - hR, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
                }
                if (hG == hB && hR > hB)
                {
                    painter.fillRect(floor(barWidth * i), height() - hR, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
                    painter.fillRect(floor(barWidth * i), height() - hB, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Blue);
                }
                else
                {
                    painter.fillRect(floor(barWidth * i), height() - hB, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), Blue);
                    painter.fillRect(floor(barWidth * i), height() - hR, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
                }
            }
        }
        else
        {
            qreal h = mHistogramROrGray[i] * height();
            painter.fillRect(floor(barWidth * i), height() - h, floor(barWidth * (i + 1)) - floor(barWidth * i), height(), RedOrGray);
        }
    }

    if (mWidth >= 0 && mCenter >= 0)
    {
        float imageMinValue = std::max(std::min(mCenter - mWidth, 1.0f), 0.0f);
        float imageMaxValue = std::max(std::min(mCenter + mWidth, 1.0f), 0.0f);
        QBrush brush(QColor::fromRgb(128, 128, 128, 200));
        float w = ceil((imageMaxValue - imageMinValue) * width());
        if (w == 0)
            w = 1;
        painter.fillRect(imageMinValue * width(), 0, (imageMaxValue - imageMinValue) * width(), height(), brush);
    }
}

void HistogramWidget::setHistogram(QVector<int> *histogramR, QVector<int> *histogramG, QVector<int> *histogramB)
{
    mHistogramROrGray.clear();
    mHistogramG.clear();
    mHistogramB.clear();
    mHistogramROrGray.reserve(mLevels);
    mHistogramG.reserve(mLevels);
    mHistogramB.reserve(mLevels);

    mIsRGB = true;
    if (histogramG == NULL && histogramB == NULL)
    {
        mIsRGB = false;
    }

    if (histogramR->size() != mLevels)
    {
        return;
    }
    if (mIsRGB)
    {
        if (histogramG->size() != mLevels || histogramB->size() != mLevels)
        {
            return;
        }
    }

    mMaxVal = 0;
    for (int i = 0; i < mLevels; i++)
    {
        mMaxVal = std::max((float)(*histogramR)[i], mMaxVal);
        if (mIsRGB)
        {
            mMaxVal = std::max((float)(*histogramG)[i], mMaxVal);
            mMaxVal = std::max((float)(*histogramB)[i], mMaxVal);
        }
    }

    for (int i = 0; i < mLevels; i++)
    {
        mHistogramROrGray.push_back((*histogramR)[i] / mMaxVal);
        if (mIsRGB)
        {
            mHistogramG.push_back((*histogramG)[i] / mMaxVal);
            mHistogramB.push_back((*histogramB)[i] / mMaxVal);
        }
    }
    update();
}

void HistogramWidget::setVisualCenter(float value)
{
    if (value != mCenter)
    {
        mCenter = value;
        update();
    }
}

void HistogramWidget::setVisualRangeWidth(float value)
{
    if (abs(value - 0.5f) != mWidth)
    {
        mWidth = abs(value - 0.5f);
        update();
    }
}
