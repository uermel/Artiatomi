//
// Created by uermel on 9/17/23.
//

#include "ConvexPolygon.h"

bool line_line_intersect(const double2& P1, const double2& P2, const double2& P3, const double2& P4, double2& Pout)
{
    double2 A = P2 - P1;
    double2 B = P3 - P4;
    double2 C = P1 - P3;

    double den = ((A.y * B.x) - (A.x * B.y));

    double alpha = ((B.y * C.x) - (B.x * C.y)) / den;
    double beta = ((A.x * C.y) - (A.y * C.x)) / den;

    if (alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1){
        Pout = P1 + alpha * (P2 - P1);
        return true;
    } else {
        Pout = make_double2(0, 0);
        return false;
    }
}

int orientation(const double2& p1, const double2& p2, const double2& p3)
{
    int val = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);

    if (val == 0){
        return 0;
    }

    return (val > 0) ? 1 : 2;
}

int orientationz(const double2& p2, const double2& p3)
{
    int val = (p2.x) * (p3.y) - (p2.y) * (p3.x);

    if (val == 0){
        return 0;
    }

    return (val > 0) ? 1 : 2;
}

struct lowestYthenX
{
    bool operator()(const double2& p1, const double2& p2) const
    {
        return (p1.y < p2.y) || (p1.y == p2.y && p1.x < p2.x);
    }
};

struct lowestPolarAngleFarthest
{
    bool operator()(const double2& p1, const double2& p2) const
    {
        int o = orientationz(p1, p2);
        return (o == 0) ? hypot(p1.x, p1.y) < hypot(p2.x, p2.y) : (o == 1);
    }
};

ConvexPolygon::ConvexPolygon(Matrix<double>& aPoints)
{
    // Get points
    for (int i=0; i < aPoints.GetColNum(); i++){
        pts.emplace_back(make_double2(aPoints(0, i), aPoints(1, i)));
    }

    // Compute Hull
    ConvexHull();
}

ConvexPolygon::ConvexPolygon(std::vector<double2>& aPoints)
{
    // This is not a polygon, wdym??
    if (aPoints.size() < 3){
        throw std::invalid_argument( "Tried initializing a ConvexPolygon with 2 or less points." );
    }

    // Get points
    for (auto & aPoint : aPoints){
        pts.emplace_back(make_double2(aPoint.x, aPoint.y));
    }

    // Compute Hull
    ConvexHull();
}

std::vector<double2> ConvexPolygon::GetPointsVec()
{
    return pts;
}

void ConvexPolygon::GetPointsVecFloat(std::vector<float2>& aOut)
{
    for (auto pt : pts){
        float2 p = make_float2((float)pt.x, (float)pt.y);
        aOut.push_back(p);
    }
}

void ConvexPolygon::GetNormValsFloat(std::vector<float>& aOut)
{
    for (auto pt = pts.begin(); pt != pts.end(); pt++){
        //float2 p = make_float2((float)pt.x, (float)pt.y);

        double2 p1 = *pt;
        double2 p2 = (pt == pts.end()-1) ? *(pts.begin()) : *(pt+1);

        double2 d = p2 - p1;

        double val = sqrt(d.x * d.x + d.y * d.y);

        aOut.push_back((float)val);
    }
}

void ConvexPolygon::ConvexHull()
{
    // Sort to find lowest y, x
    std::sort(pts.begin(), pts.end(), lowestYthenX());

    // Subtract lowest point from each
    double2 lp = pts[0];

    for(auto pt = pts.begin(); pt < pts.end(); pt++){
        (*pt) = (*pt) - lp;
    }

    // Sort by polar angle, farthest point.
    std::sort(pts.begin(), pts.end(), lowestPolarAngleFarthest());

    // Graham Scan
    std::vector<double2> stack;

    for (auto pt = pts.begin(); pt != pts.end(); pt++){
        while (stack.size() > 1 && orientation(*(stack.end()-2), *(stack.end()-1), *pt) != 1)
        {
            stack.pop_back();
        }
        stack.emplace_back(*pt);
    }

    // The hull
    pts.clear();
    for (auto pt = stack.end(); pt == stack.begin(); --pt){
        pts.emplace_back(*pt);
    }

    pts = stack;

    // Add back the lowest point
    for (auto& pt: pts){
        pt = pt + lp;
    }
}

bool ConvexPolygon::Inside(double2 &aPoint)
{
    int pos = 0;
    int neg = 0;

    //auto pt ;
    int i = 1;
    for(auto pt = pts.begin(); pt != pts.end(); ++pt, i++){

        // Point on corner
        if ((aPoint.x == (*pt).x) && (aPoint.y == (*pt).y)){
            return true;
        }

        // Poly line seg
        double2 p1 = *pt;
        double2 p2 = (i == pts.size()) ? *(pts.begin()) : *(pt+1);

        double d = (aPoint.x - p1.x) * (p2.y - p1.y) - (aPoint.y - p1.y) * (p2.x - p1.x);

        if (d > 0) pos++;
        if (d < 0) neg++;

        if (pos > 0 && neg > 0)
            return false;
    }

    return true;
}

ConvexPolygon ConvexPolygon::Intersect(ConvexPolygon& aPolygon)
{
    // Non overlapping points and points that are definitely in
    std::vector<std::tuple<double2, double2, double2>> problemSegsInput;
    std::vector<std::tuple<double2, double2, double2>> problemSegsThis;

    std::vector<double2> goodPoints;

    // Check if all points of query poly are inside this one and record problematic segs if not
    std::vector<double2> aPts = aPolygon.GetPointsVec();
    bool allIn = true;
    for (auto pt = aPts.begin(); pt != aPts.end(); pt++){
        bool isin = Inside(*pt);

        if (isin){
            goodPoints.push_back(*pt);
        } else {
            double2 prev = (pt == aPts.begin()) ? *(aPts.end()-1) : *(pt-1);
            double2 next = (pt == aPts.end()-1) ? *(aPts.begin()) : *(pt+1);

            problemSegsInput.emplace_back(prev, *pt, next);
        }

        allIn &= isin;
    }

    // If all query points are inside this one, so the intersection is the query
    if (allIn){
        return aPolygon;
    }

    // Now check if all points of this one are in parameter poly, the intersections would be the same, so we don't need
    // to keep track again
    allIn = true;
    for (auto pt = pts.begin(); pt != pts.end(); pt++) {
        bool isin = aPolygon.Inside(*pt);

        if (isin){
            goodPoints.emplace_back(*pt);
       }
//        else {
//            double2 prev = (pt == pts.begin()) ? *(pts.end()-1) : *(pt-1);
//            double2 next = (pt == pts.end()-1) ? *(pts.begin()) : *(pt+1);
//
//            problemSegsThis.emplace_back(prev, *pt, next);
//        }

        allIn &= isin;
    }

    // All our points are inside the query, we are the intersection.
    if (allIn){
        return *this;
    }

    // Now if necessary intersect each line seg and compute new Convex Polygon

    // Problematic line segments of the input. These are points of the query outside us.
    for (auto & seg : problemSegsInput){
        for (auto pt = pts.begin(); pt != pts.end(); pt++){
            // Line of this poly
            double2 P1 = *pt;
            double2 P2 = (pt == pts.end()-1) ? *(pts.begin()) : *(pt+1);
            double2 out;

            // First problematic segment
            double2 P3 = std::get<0>(seg);
            double2 P4 = std::get<1>(seg);
            bool does_intersect = line_line_intersect(P1, P2, P3, P4, out);
            if (does_intersect) goodPoints.emplace_back(out);

            // Second problematic segment
            P3 = std::get<1>(seg);
            P4 = std::get<2>(seg);
            does_intersect = line_line_intersect(P1, P2, P3, P4, out);
            if (does_intersect) goodPoints.emplace_back(out);
        }
    }

    // Second problematic line segments of the input. These are points of self outside the query.
//    for (auto & seg : problemSegsThis){
//        for (auto pt = aPts.begin(); pt != aPts.end(); pt++){
//            // Line of query poly
//            double2 P1 = *pt;
//            double2 P2 = (pt == aPts.end()-1) ? *(aPts.begin()) : *(pt+1);
//            double2 out;
//
//            // First problematic segment
//            double2 P3 = std::get<0>(seg);
//            double2 P4 = std::get<1>(seg);
//            bool does_intersect = line_line_intersect(P1, P2, P3, P4, out);
//            if (does_intersect) goodPoints.emplace_back(out);
//
//            // Second problematic segment
//            P3 = std::get<1>(seg);
//            P4 = std::get<2>(seg);
//            does_intersect = line_line_intersect(P1, P2, P3, P4, out);
//            if (does_intersect) goodPoints.emplace_back(out);
//        }
//    }

    // Generate output
    ConvexPolygon output(goodPoints);
    return output;
}

