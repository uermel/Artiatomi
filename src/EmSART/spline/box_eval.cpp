//
// Created by uermel on 9/23/21.
//

#include <stdio.h>      /* printf */
#include <math.h>       /* pow */

void box_eval(double* X, double* nu, int k, double plow, double phigh, double pstep)
{
    // Points at which to eval
    int N = (phigh - plow)/pstep + 1;
    auto p = new double[2*N*N];

    for (int x = 0; x < N; x++){
        for (int y = 0; y < N; y++){
            int idx_x = y * N * 2 + x * 2;
            int idx_y = y * N * 2 + x * 2 + 1;
            p[idx_x] = x;
            p[idx_y] = y;
        } // for y
    } // for x

    // Domain space dimension
    int s = 2;

    // Hashing function
    auto u = new double[k];

    for (int i = 0; i < k; i++){
        u[i] = pow(2., (double)i);
    }



}


int main(int argc, char *argv[]) {

}