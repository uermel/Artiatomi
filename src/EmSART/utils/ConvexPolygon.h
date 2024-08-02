//
// Created by uermel on 9/17/23.
//

#ifndef ARTIATOMI_CONVEXPOLYGON_H
#define ARTIATOMI_CONVEXPOLYGON_H

#include "EmSartDefault.h"
#include <cmath>
#include <tuple>
#include <stack>
#include <algorithm>
#include "Matrix.h"

class ConvexPolygon {
private:
    std::vector<double2> pts;

public:
    ConvexPolygon(Matrix<double>& points);
    ConvexPolygon(std::vector<double2>& points);

    void ConvexHull();
    bool Inside(double2& aPoint);
    ConvexPolygon Intersect(ConvexPolygon& aPolygon);

    std::vector<double2> GetPointsVec();
    void GetPointsVecFloat(std::vector<float2>& aOut);
    void GetNormValsFloat(std::vector<float>& aOut);

};


#endif //ARTIATOMI_CONVEXPOLYGON_H
