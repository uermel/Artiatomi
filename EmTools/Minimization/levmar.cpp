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


#include "levmar.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>



#ifdef _USE_WINDOWS_COMPILER_SETTINGS
#define LM_FINITE _finite // MSVC
#else
#ifdef _USE_LINUX_COMPILER_SETTINGS
#define LM_FINITE finite // gcc
#endif //_USE_LINUX_COMPILER_SETTINGS
#endif //ELSE
#ifdef _USE_APPLE_COMPILER_SETTINGS
#define LM_FINITE finite // gcc
#endif

#define __BLOCKSZ__       32 /* block size for cache-friendly matrix-matrix multiply. It should be
							  * such that __BLOCKSZ__^2*sizeof(float) is smaller than the CPU (L1)
							  * data cache size. Notice that a value of 32 when float=double assumes
							  * an 8Kb L1 data cache (32*32*8=8K). This is a concervative choice since
							  * newer Pentium 4s have a L1 data cache of size 16K, capable of holding
							  * up to 45x45 double blocks.
							  */
#define __BLOCKSZ__SQ    (__BLOCKSZ__)*(__BLOCKSZ__)

#define LM_DIF_WORKSZ(npar, nmeas) (4*(nmeas) + 4*(npar) + (nmeas)*(npar) + (npar)*(npar))
#define ONE_THIRD     0.3333333334f /* 1.0/3.0 */



							  /* Compute e=x-y for two n-vectors x and y and return the squared L2 norm of e.
							  * e can coincide with either x or y; x can be NULL, in which case it is assumed
							  * to be equal to the zero vector.
							  * Uses loop unrolling and blocking to reduce bookkeeping overhead & pipeline
							  * stalls and increase instruction-level parallelism; see http://www.abarnett.demon.co.uk/tutorial.html
							  */

float LEVMAR_L2NRMXMY(float *e, float *x, float *y, int n)
{
	const int blocksize = 8, bpwr = 3; /* 8=2^3 */
	register int i;
	int j1, j2, j3, j4, j5, j6, j7;
	int blockn;
	register float sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

	/* n may not be divisible by blocksize,
	* go as near as we can first, then tidy up.
	*/
	blockn = (n >> bpwr) << bpwr; /* (n / blocksize) * blocksize; */

								  /* unroll the loop in blocks of `blocksize'; looping downwards gains some more speed */
	if (x) {
		for (i = blockn - 1; i>0; i -= blocksize) {
			e[i] = x[i] - y[i]; sum0 += e[i] * e[i];
			j1 = i - 1; e[j1] = x[j1] - y[j1]; sum1 += e[j1] * e[j1];
			j2 = i - 2; e[j2] = x[j2] - y[j2]; sum2 += e[j2] * e[j2];
			j3 = i - 3; e[j3] = x[j3] - y[j3]; sum3 += e[j3] * e[j3];
			j4 = i - 4; e[j4] = x[j4] - y[j4]; sum0 += e[j4] * e[j4];
			j5 = i - 5; e[j5] = x[j5] - y[j5]; sum1 += e[j5] * e[j5];
			j6 = i - 6; e[j6] = x[j6] - y[j6]; sum2 += e[j6] * e[j6];
			j7 = i - 7; e[j7] = x[j7] - y[j7]; sum3 += e[j7] * e[j7];
		}

		/*
		* There may be some left to do.
		* This could be done as a simple for() loop,
		* but a switch is faster (and more interesting)
		*/

		i = blockn;
		if (i<n) {
			/* Jump into the case at the place that will allow
			* us to finish off the appropriate number of items.
			*/

			switch (n - i) {
			case 7: e[i] = x[i] - y[i]; sum0 += e[i] * e[i]; ++i;
			case 6: e[i] = x[i] - y[i]; sum1 += e[i] * e[i]; ++i;
			case 5: e[i] = x[i] - y[i]; sum2 += e[i] * e[i]; ++i;
			case 4: e[i] = x[i] - y[i]; sum3 += e[i] * e[i]; ++i;
			case 3: e[i] = x[i] - y[i]; sum0 += e[i] * e[i]; ++i;
			case 2: e[i] = x[i] - y[i]; sum1 += e[i] * e[i]; ++i;
			case 1: e[i] = x[i] - y[i]; sum2 += e[i] * e[i]; //++i;
			}
		}
	}
	else { /* x==0 */
		for (i = blockn - 1; i>0; i -= blocksize) {
			e[i] = -y[i]; sum0 += e[i] * e[i];
			j1 = i - 1; e[j1] = -y[j1]; sum1 += e[j1] * e[j1];
			j2 = i - 2; e[j2] = -y[j2]; sum2 += e[j2] * e[j2];
			j3 = i - 3; e[j3] = -y[j3]; sum3 += e[j3] * e[j3];
			j4 = i - 4; e[j4] = -y[j4]; sum0 += e[j4] * e[j4];
			j5 = i - 5; e[j5] = -y[j5]; sum1 += e[j5] * e[j5];
			j6 = i - 6; e[j6] = -y[j6]; sum2 += e[j6] * e[j6];
			j7 = i - 7; e[j7] = -y[j7]; sum3 += e[j7] * e[j7];
		}

		/*
		* There may be some left to do.
		* This could be done as a simple for() loop,
		* but a switch is faster (and more interesting)
		*/

		i = blockn;
		if (i<n) {
			/* Jump into the case at the place that will allow
			* us to finish off the appropriate number of items.
			*/

			switch (n - i) {
			case 7: e[i] = -y[i]; sum0 += e[i] * e[i]; ++i;
			case 6: e[i] = -y[i]; sum1 += e[i] * e[i]; ++i;
			case 5: e[i] = -y[i]; sum2 += e[i] * e[i]; ++i;
			case 4: e[i] = -y[i]; sum3 += e[i] * e[i]; ++i;
			case 3: e[i] = -y[i]; sum0 += e[i] * e[i]; ++i;
			case 2: e[i] = -y[i]; sum1 += e[i] * e[i]; ++i;
			case 1: e[i] = -y[i]; sum2 += e[i] * e[i]; //++i;
			}
		}
	}

	return sum0 + sum1 + sum2 + sum3;
}



/* forward finite difference approximation to the Jacobian of func */
void LEVMAR_FDIF_FORW_JAC_APPROX(
	void(*func)(float *p, float *hx, int m, int n, void *adata),
	/* function to differentiate */
	float *p,              /* I: current parameter estimate, mx1 */
	float *hx,             /* I: func evaluated at p, i.e. hx=func(p), nx1 */
	float *hxx,            /* W/O: work array for evaluating func(p+delta), nx1 */
	float delta,           /* increment for computing the Jacobian */
	float *jac,            /* O: array for storing approximated Jacobian, nxm */
	int m,
	int n,
	void *adata)
{
	register int i, j;
	float tmp;
	register float d;

	for (j = 0; j<m; ++j) {
		/* determine d=max(1E-04*|p[j]|, delta), see HZ */
		d = (1E-04f)*p[j]; // force evaluation
		d = fabs(d);
		if (d<delta)
			d = delta;

		tmp = p[j];
		p[j] += d;

		(*func)(p, hxx, m, n, adata);

		p[j] = tmp; /* restore */

		d = (1.0f) / d; /* invert so that divisions can be carried out faster as multiplications */
		for (i = 0; i<n; ++i) {
			jac[i*m + j] = (hxx[i] - hx[i])*d;
		}
	}
}


/* central finite difference approximation to the Jacobian of func */
void LEVMAR_FDIF_CENT_JAC_APPROX(
	void(*func)(float *p, float *hx, int m, int n, void *adata),
	/* function to differentiate */
	float *p,              /* I: current parameter estimate, mx1 */
	float *hxm,            /* W/O: work array for evaluating func(p-delta), nx1 */
	float *hxp,            /* W/O: work array for evaluating func(p+delta), nx1 */
	float delta,           /* increment for computing the Jacobian */
	float *jac,            /* O: array for storing approximated Jacobian, nxm */
	int m,
	int n,
	void *adata)
{
	register int i, j;
	float tmp;
	register float d;

	for (j = 0; j<m; ++j) {
		/* determine d=max(1E-04*|p[j]|, delta), see HZ */
		d = (1E-04f)*p[j]; // force evaluation
		d = fabs(d);
		if (d<delta)
			d = delta;

		tmp = p[j];
		p[j] -= d;
		(*func)(p, hxm, m, n, adata);

		p[j] = tmp + d;
		(*func)(p, hxp, m, n, adata);
		p[j] = tmp; /* restore */

		d = (0.5f) / d; /* invert so that divisions can be carried out faster as multiplications */
		for (i = 0; i<n; ++i) {
			jac[i*m + j] = (hxp[i] - hxm[i])*d;
		}
	}
}


/* blocked multiplication of the transpose of the nxm matrix a with itself (i.e. a^T a)
* using a block size of bsize. The product is returned in b.
* Since a^T a is symmetric, its computation can be sped up by computing only its
* upper triangular part and copying it to the lower part.
*
* More details on blocking can be found at
* http://www-2.cs.cmu.edu/afs/cs/academic/class/15213-f02/www/R07/section_a/Recitation07-SectionA.pdf
*/
void LEVMAR_TRANS_MAT_MAT_MULT(float *a, float *b, int n, int m)
{
	register int i, j, k, jj, kk;
	register float sum, *bim, *akm;
	const int bsize = __BLOCKSZ__;

#define __MIN__(x, y) (((x)<=(y))? (x) : (y))
#define __MAX__(x, y) (((x)>=(y))? (x) : (y))

	/* compute upper triangular part using blocking */
	for (jj = 0; jj<m; jj += bsize) {
		for (i = 0; i<m; ++i) {
			bim = b + i*m;
			for (j = __MAX__(jj, i); j<__MIN__(jj + bsize, m); ++j)
				bim[j] = 0.0; //b[i*m+j]=0.0;
		}

		for (kk = 0; kk<n; kk += bsize) {
			for (i = 0; i<m; ++i) {
				bim = b + i*m;
				for (j = __MAX__(jj, i); j<__MIN__(jj + bsize, m); ++j) {
					sum = 0.0;
					for (k = kk; k<__MIN__(kk + bsize, n); ++k) {
						akm = a + k*m;
						sum += akm[i] * akm[j]; //a[k*m+i]*a[k*m+j];
					}
					bim[j] += sum; //b[i*m+j]+=sum;
				}
			}
		}
	}

	/* copy upper triangular part to the lower one */
	for (i = 0; i<m; ++i)
		for (j = 0; j<i; ++j)
			b[i*m + j] = b[j*m + i];

#undef __MIN__
#undef __MAX__

}



/*
* This function returns the solution of Ax = b
*
* The function employs LU decomposition followed by forward/back substitution (see
* also the LAPACK-based LU solver above)
*
* A is mxm, b is mx1
*
* The function returns 0 in case of error, 1 if successful
*
* This function is often called repetitively to solve problems of identical
* dimensions. To avoid repetitive malloc's and free's, allocated memory is
* retained between calls and free'd-malloc'ed when not of the appropriate size.
* A call with NULL as the first argument forces this memory to be released.
*/
int AX_EQ_B_LU(float *A, float *B, float *x, int m)
{
	void *buf = NULL;
	int buf_sz = 0;

	register int i, j, k;
	int *idx, maxi = -1, idx_sz, a_sz, work_sz, tot_sz;
	float *a, *work, max, sum, tmp;

	if (!A)
		return 1; /* NOP */

	/* calculate required memory size */
	idx_sz = m;
	a_sz = m*m;
	work_sz = m;
	tot_sz = (a_sz + work_sz) * sizeof(float) + idx_sz * sizeof(int); /* should be arranged in that order for proper doubles alignment */

	buf_sz = tot_sz;
	buf = (void *)malloc(tot_sz);
	if (!buf) {
		//fprintf(stderr, RCAT("memory allocation in ", AX_EQ_B_LU) "() failed!\n");
		return 0;
	}

	a = (float*)buf;
	work = a + a_sz;
	idx = (int *)(work + work_sz);

	/* avoid destroying A, B by copying them to a, x resp. */
	memcpy(a, A, a_sz * sizeof(float));
	memcpy(x, B, m * sizeof(float));

	/* compute the LU decomposition of a row permutation of matrix a; the permutation itself is saved in idx[] */
	for (i = 0; i<m; ++i) {
		max = 0.0;
		for (j = 0; j<m; ++j)
			if ((tmp = fabs(a[i*m + j]))>max)
				max = tmp;
		if (max == 0.0) {
			//fprintf(stderr, RCAT("Singular matrix A in ", AX_EQ_B_LU) "()!\n");

			free(buf);

			return 0;
		}
		work[i] = (1.0f) / max;
	}

	for (j = 0; j<m; ++j) {
		for (i = 0; i<j; ++i) {
			sum = a[i*m + j];
			for (k = 0; k<i; ++k)
				sum -= a[i*m + k] * a[k*m + j];
			a[i*m + j] = sum;
		}
		max = 0.0;
		for (i = j; i<m; ++i) {
			sum = a[i*m + j];
			for (k = 0; k<j; ++k)
				sum -= a[i*m + k] * a[k*m + j];
			a[i*m + j] = sum;
			if ((tmp = work[i] * fabs(sum)) >= max) {
				max = tmp;
				maxi = i;
			}
		}
		if (j != maxi) {
			for (k = 0; k<m; ++k) {
				tmp = a[maxi*m + k];
				a[maxi*m + k] = a[j*m + k];
				a[j*m + k] = tmp;
			}
			work[maxi] = work[j];
		}
		idx[j] = maxi;
		if (a[j*m + j] == 0.0)
			a[j*m + j] = EPSILON;
		if (j != m - 1) {
			tmp = (1.0f) / (a[j*m + j]);
			for (i = j + 1; i<m; ++i)
				a[i*m + j] *= tmp;
		}
	}

	/* The decomposition has now replaced a. Solve the linear system using
	* forward and back substitution
	*/
	for (i = k = 0; i<m; ++i) {
		j = idx[i];
		sum = x[j];
		x[j] = x[i];
		if (k != 0)
			for (j = k - 1; j<i; ++j)
				sum -= a[i*m + j] * x[j];
		else
			if (sum != 0.0)
				k = i + 1;
		x[i] = sum;
	}

	for (i = m - 1; i >= 0; --i) {
		sum = x[i];
		for (j = i + 1; j<m; ++j)
			sum -= a[i*m + j] * x[j];
		x[i] = sum / a[i*m + i];
	}

	free(buf);

	return 1;
}

/*
* This function computes the inverse of A in B. A and B can coincide
*
* The function employs LAPACK-free LU decomposition of A to solve m linear
* systems A*B_i=I_i, where B_i and I_i are the i-th columns of B and I.
*
* A and B are mxm
*
* The function returns 0 in case of error, 1 if successful
*
*/
static int LEVMAR_LUINVERSE(float *A, float *B, int m)
{
	float *buf = NULL;
	int buf_sz = 0;

	register int i, j, k, l;
	int *idx, maxi = -1, idx_sz, a_sz, x_sz, work_sz, tot_sz;
	float *a, *x, *work, max, sum, tmp;

	/* calculate required memory size */
	idx_sz = m;
	a_sz = m*m;
	x_sz = m;
	work_sz = m;
	tot_sz = (a_sz + x_sz + work_sz) * sizeof(float) + idx_sz * sizeof(int); /* should be arranged in that order for proper doubles alignment */

	buf_sz = tot_sz;
	buf = (float *)malloc(tot_sz);
	if (!buf) {
		//fprintf(stderr, RCAT("memory allocation in ", LEVMAR_LUINVERSE) "() failed!\n");
		return 0; /* error */
	}

	a = buf;
	x = a + a_sz;
	work = x + x_sz;
	idx = (int *)(work + work_sz);

	/* avoid destroying A by copying it to a */
	for (i = 0; i<a_sz; ++i) a[i] = A[i];

	/* compute the LU decomposition of a row permutation of matrix a; the permutation itself is saved in idx[] */
	for (i = 0; i<m; ++i) {
		max = 0.0;
		for (j = 0; j<m; ++j)
			if ((tmp = fabs(a[i*m + j]))>max)
				max = tmp;
		if (max == 0.0) {
			//fprintf(stderr, RCAT("Singular matrix A in ", LEVMAR_LUINVERSE) "()!\n");
			free(buf);

			return 0;
		}
		work[i] = (1.0f) / max;
	}

	for (j = 0; j<m; ++j) {
		for (i = 0; i<j; ++i) {
			sum = a[i*m + j];
			for (k = 0; k<i; ++k)
				sum -= a[i*m + k] * a[k*m + j];
			a[i*m + j] = sum;
		}
		max = 0.0;
		for (i = j; i<m; ++i) {
			sum = a[i*m + j];
			for (k = 0; k<j; ++k)
				sum -= a[i*m + k] * a[k*m + j];
			a[i*m + j] = sum;
			if ((tmp = work[i] * fabs(sum)) >= max) {
				max = tmp;
				maxi = i;
			}
		}
		if (j != maxi) {
			for (k = 0; k<m; ++k) {
				tmp = a[maxi*m + k];
				a[maxi*m + k] = a[j*m + k];
				a[j*m + k] = tmp;
			}
			work[maxi] = work[j];
		}
		idx[j] = maxi;
		if (a[j*m + j] == 0.0)
			a[j*m + j] = EPSILON;
		if (j != m - 1) {
			tmp = (1.0f) / (a[j*m + j]);
			for (i = j + 1; i<m; ++i)
				a[i*m + j] *= tmp;
		}
	}

	/* The decomposition has now replaced a. Solve the m linear systems using
	* forward and back substitution
	*/
	for (l = 0; l<m; ++l) {
		for (i = 0; i<m; ++i) x[i] = 0.0;
		x[l] = (1.0f);

		for (i = k = 0; i<m; ++i) {
			j = idx[i];
			sum = x[j];
			x[j] = x[i];
			if (k != 0)
				for (j = k - 1; j<i; ++j)
					sum -= a[i*m + j] * x[j];
			else
				if (sum != 0.0)
					k = i + 1;
			x[i] = sum;
		}

		for (i = m - 1; i >= 0; --i) {
			sum = x[i];
			for (j = i + 1; j<m; ++j)
				sum -= a[i*m + j] * x[j];
			x[i] = sum / a[i*m + i];
		}

		for (i = 0; i<m; ++i)
			B[i*m + l] = x[i];
	}

	free(buf);

	return 1;
}

/*
* This function computes in C the covariance matrix corresponding to a least
* squares fit. JtJ is the approximate Hessian at the solution (i.e. J^T*J, where
* J is the Jacobian at the solution), sumsq is the sum of squared residuals
* (i.e. goodnes of fit) at the solution, m is the number of parameters (variables)
* and n the number of observations. JtJ can coincide with C.
*
* if JtJ is of full rank, C is computed as sumsq/(n-m)*(JtJ)^-1
* otherwise and if LAPACK is available, C=sumsq/(n-r)*(JtJ)^+
* where r is JtJ's rank and ^+ denotes the pseudoinverse
* The diagonal of C is made up from the estimates of the variances
* of the estimated regression coefficients.
* See the documentation of routine E04YCF from the NAG fortran lib
*
* The function returns the rank of JtJ if successful, 0 on error
*
* A and C are mxm
*
*/
int LEVMAR_COVAR(float *JtJ, float *C, float sumsq, int m, int n)
{
	register int i;
	int rnk;
	float fact;

		rnk = LEVMAR_LUINVERSE(JtJ, C, m);
	if (!rnk) return 0;

	rnk = m; /* assume full rank */


	fact = sumsq / (float)(n - rnk);
	for (i = 0; i<m*m; ++i)
		C[i] *= fact;

	return rnk;
}



/* Secant version of the LEVMAR_DER() function above: the Jacobian is approximated with
* the aid of finite differences (forward or central, see the comment for the opts argument)
*/
int LEVMAR_DIF(
	void(*func)(float *p, float *hx, int m, int n, void *adata), /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
	float *p,         /* I/O: initial parameter estimates. On output has the estimated solution */
	float *x,         /* I: measurement vector. NULL implies a zero vector */
	int m,              /* I: parameter vector dimension (i.e. #unknowns) */
	int n,              /* I: measurement vector dimension */
	int itmax,          /* I: maximum number of iterations */
	float opts[5],    /* I: opts[0-4] = minim. options [\mu, \epsilon1, \epsilon2, \epsilon3, \delta]. Respectively the
						* scale factor for initial \mu, stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2 and
						* the step used in difference approximation to the Jacobian. Set to NULL for defaults to be used.
						* If \delta<0, the Jacobian is approximated with central differences which are more accurate
						* (but slower!) compared to the forward differences employed by default.
						*/
	float info[LM_INFO_SZ],
	/* O: information regarding the minimization. Set to NULL if don't care
	* info[0]= ||e||_2 at initial p.
	* info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
	* info[5]= # iterations,
	* info[6]=reason for terminating: 1 - stopped by small gradient J^T e
	*                                 2 - stopped by small Dp
	*                                 3 - stopped by itmax
	*                                 4 - singular matrix. Restart from current p with increased mu
	*                                 5 - no further error reduction is possible. Restart with increased mu
	*                                 6 - stopped by small ||e||_2
	*                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
	* info[7]= # function evaluations
	* info[8]= # Jacobian evaluations
	* info[9]= # linear systems solved, i.e. # attempts for reducing error
	*/
	float *work,     /* working memory at least LM_DIF_WORKSZ() reals large, allocated if NULL */
	float *covar,    /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
	void *adata)       /* pointer to possibly additional data, passed uninterpreted to func.
					   * Set to NULL if not needed
					   */
{
	register int i, j, k, l;
	int worksz, freework = 0, issolved;
	/* temp work arrays */
	float *e,          /* nx1 */
		*hx,         /* \hat{x}_i, nx1 */
		*jacTe,      /* J^T e_i mx1 */
		*jac,        /* nxm */
		*jacTjac,    /* mxm */
		*Dp,         /* mx1 */
		*diag_jacTjac,   /* diagonal of J^T J, mx1 */
		*pDp,        /* p + Dp, mx1 */
		*wrk,        /* nx1 */
		*wrk2;       /* nx1, used only for holding a temporary e vector and when differentiating with central differences */

	int using_ffdif = 1;

	register float mu,  /* damping constant */
		tmp; /* mainly used in matrix & vector multiplications */
	float p_eL2, jacTe_inf, pDp_eL2; /* ||e(p)||_2, ||J^T e||_inf, ||e(p+Dp)||_2 */
	float p_L2, Dp_L2 = FLT_MAX, dF, dL;
	float tau, eps1, eps2, eps2_sq, eps3, delta;
	float init_p_eL2;
	int nu, nu2, stop = 0, nfev, njap = 0, nlss = 0, K = (m >= 10) ? m : 10, updjac, updp = 1, newjac;
	const int nm = n*m;
	int(*linsolver)(float *A, float *B, float *x, int m) = NULL;

	mu = jacTe_inf = p_L2 = 0.0; /* -Wall */
	updjac = newjac = 0; /* -Wall */

	if (n<m) {
		//fprintf(stderr, LCAT(LEVMAR_DIF, "(): cannot solve a problem with fewer measurements [%d] than unknowns [%d]\n"), n, m);
		return LM_ERROR;
	}

	if (opts) {
		tau = opts[0];
		eps1 = opts[1];
		eps2 = opts[2];
		eps2_sq = opts[2] * opts[2];
		eps3 = opts[3];
		delta = opts[4];
		if (delta<0.0) {
			delta = -delta; /* make positive */
			using_ffdif = 0; /* use central differencing */
		}
	}
	else { // use default values
		tau = (LM_INIT_MU);
		eps1 = (LM_STOP_THRESH);
		eps2 = (LM_STOP_THRESH);
		eps2_sq = (LM_STOP_THRESH)*(LM_STOP_THRESH);
		eps3 = (LM_STOP_THRESH);
		delta = (LM_DIFF_DELTA);
	}

	if (!work) {
		worksz = LM_DIF_WORKSZ(m, n); //4*n+4*m + n*m + m*m;
		work = (float *)malloc(worksz * sizeof(float)); /* allocate a big chunk in one step */
		if (!work) {
			//fprintf(stderr, LCAT(LEVMAR_DIF, "(): memory allocation request failed\n"));
			return LM_ERROR;
		}
		freework = 1;
	}

	/* set up work arrays */
	e = work;
	hx = e + n;
	jacTe = hx + n;
	jac = jacTe + m;
	jacTjac = jac + nm;
	Dp = jacTjac + m*m;
	diag_jacTjac = Dp + m;
	pDp = diag_jacTjac + m;
	wrk = pDp + m;
	wrk2 = wrk + n;

	/* compute e=x - f(p) and its L2 norm */
	(*func)(p, hx, m, n, adata); nfev = 1;
	/* ### e=x-hx, p_eL2=||e|| */

	p_eL2 = LEVMAR_L2NRMXMY(e, x, hx, n);

	init_p_eL2 = p_eL2;
	if (!LM_FINITE(p_eL2)) stop = 7;

	nu = 20; /* force computation of J */

	for (k = 0; k<itmax && !stop; ++k) {
		/* Note that p and e have been updated at a previous iteration */

		if (p_eL2 <= eps3) { /* error is small */
			stop = 6;
			break;
		}

		/* Compute the Jacobian J at p,  J^T J,  J^T e,  ||J^T e||_inf and ||p||^2.
		* The symmetry of J^T J is again exploited for speed
		*/

		if ((updp && nu>16) || updjac == K) { /* compute difference approximation to J */
			if (using_ffdif) { /* use forward differences */
				LEVMAR_FDIF_FORW_JAC_APPROX(func, p, hx, wrk, delta, jac, m, n, adata);
				++njap; nfev += m;
			}
			else { /* use central differences */
				LEVMAR_FDIF_CENT_JAC_APPROX(func, p, wrk, wrk2, delta, jac, m, n, adata);
				++njap; nfev += 2 * m;
			}
			nu = 2; updjac = 0; updp = 0; newjac = 1;
		}

		if (newjac) { /* Jacobian has changed, recompute J^T J, J^t e, etc */
			newjac = 0;

			/* J^T J, J^T e */
			if (nm <= __BLOCKSZ__SQ) { // this is a small problem
									   /* J^T*J_ij = \sum_l J^T_il * J_lj = \sum_l J_li * J_lj.
									   * Thus, the product J^T J can be computed using an outer loop for
									   * l that adds J_li*J_lj to each element ij of the result. Note that
									   * with this scheme, the accesses to J and JtJ are always along rows,
									   * therefore induces less cache misses compared to the straightforward
									   * algorithm for computing the product (i.e., l loop is innermost one).
									   * A similar scheme applies to the computation of J^T e.
									   * However, for large minimization problems (i.e., involving a large number
									   * of unknowns and measurements) for which J/J^T J rows are too large to
									   * fit in the L1 cache, even this scheme incures many cache misses. In
									   * such cases, a cache-efficient blocking scheme is preferable.
									   *
									   * Thanks to John Nitao of Lawrence Livermore Lab for pointing out this
									   * performance problem.
									   *
									   * Note that the non-blocking algorithm is faster on small
									   * problems since in this case it avoids the overheads of blocking.
									   */
				register int l;
				register float alpha, *jaclm, *jacTjacim;

				/* looping downwards saves a few computations */
				for (i = m*m; i-->0; )
					jacTjac[i] = 0.0;
				for (i = m; i-->0; )
					jacTe[i] = 0.0;

				for (l = n; l-->0; ) {
					jaclm = jac + l*m;
					for (i = m; i-->0; ) {
						jacTjacim = jacTjac + i*m;
						alpha = jaclm[i]; //jac[l*m+i];
						for (j = i + 1; j-->0; ) /* j<=i computes lower triangular part only */
							jacTjacim[j] += jaclm[j] * alpha; //jacTjac[i*m+j]+=jac[l*m+j]*alpha

															  /* J^T e */
						jacTe[i] += alpha*e[l];
					}
				}

				for (i = m; i-->0; ) /* copy to upper part */
					for (j = i + 1; j<m; ++j)
						jacTjac[i*m + j] = jacTjac[j*m + i];
			}
			else { // this is a large problem
				   /* Cache efficient computation of J^T J based on blocking
				   */
				LEVMAR_TRANS_MAT_MAT_MULT(jac, jacTjac, n, m);

				/* cache efficient computation of J^T e */
				for (i = 0; i<m; ++i)
					jacTe[i] = 0.0;

				for (i = 0; i<n; ++i) {
					register float *jacrow;

					for (l = 0, jacrow = jac + i*m, tmp = e[i]; l<m; ++l)
						jacTe[l] += jacrow[l] * tmp;
				}
			}

			/* Compute ||J^T e||_inf and ||p||^2 */
			for (i = 0, p_L2 = jacTe_inf = 0.0; i<m; ++i) {
				if (jacTe_inf < (tmp = fabs(jacTe[i]))) jacTe_inf = tmp;

				diag_jacTjac[i] = jacTjac[i*m + i]; /* save diagonal entries so that augmentation can be later canceled */
				p_L2 += p[i] * p[i];
			}
			//p_L2=sqrt(p_L2);
		}


		/* check for convergence */
		if ((jacTe_inf <= eps1)) {
			Dp_L2 = 0.0; /* no increment for p in this case */
			stop = 1;
			break;
		}

		/* compute initial damping factor */
		if (k == 0) {
			for (i = 0, tmp = -FLT_MAX; i<m; ++i)
				if (diag_jacTjac[i]>tmp) tmp = diag_jacTjac[i]; /* find max diagonal element */
			mu = tau*tmp;
		}

		/* determine increment using adaptive damping */

		/* augment normal equations */
		for (i = 0; i<m; ++i)
			jacTjac[i*m + i] += mu;

		/* solve augmented equations */

		/* use the LU included with levmar */
		issolved = AX_EQ_B_LU(jacTjac, jacTe, Dp, m); ++nlss; linsolver = AX_EQ_B_LU;

		if (issolved) {
			/* compute p's new estimate and ||Dp||^2 */
			for (i = 0, Dp_L2 = 0.0; i<m; ++i) {
				pDp[i] = p[i] + (tmp = Dp[i]);
				Dp_L2 += tmp*tmp;
			}
			//Dp_L2=sqrt(Dp_L2);

			if (Dp_L2 <= eps2_sq*p_L2) { /* relative change in p is small, stop */
										 //if(Dp_L2<=eps2*(p_L2 + eps2)){ /* relative change in p is small, stop */
				stop = 2;
				break;
			}

			if (Dp_L2 >= (p_L2 + eps2) / ((EPSILON)*(EPSILON))) { /* almost singular */
																				//if(Dp_L2>=(p_L2+eps2)/LM_CNST(EPSILON)){ /* almost singular */
				stop = 4;
				break;
			}

			(*func)(pDp, wrk, m, n, adata); ++nfev; /* evaluate function at p + Dp */
													/* compute ||e(pDp)||_2 */
													/* ### wrk2=x-wrk, pDp_eL2=||wrk2|| */
#if 1
			pDp_eL2 = LEVMAR_L2NRMXMY(wrk2, x, wrk, n);
#else
			for (i = 0, pDp_eL2 = 0.0; i<n; ++i) {
				wrk2[i] = tmp = x[i] - wrk[i];
				pDp_eL2 += tmp*tmp;
			}
#endif
			if (!LM_FINITE(pDp_eL2)) { /* sum of squares is not finite, most probably due to a user error.
									   * This check makes sure that the loop terminates early in the case
									   * of invalid input. Thanks to Steve Danauskas for suggesting it
									   */

				stop = 7;
				break;
			}

			dF = p_eL2 - pDp_eL2;
			if (updp || dF>0) { /* update jac */
				for (i = 0; i<n; ++i) {
					for (l = 0, tmp = 0.0; l<m; ++l)
						tmp += jac[i*m + l] * Dp[l]; /* (J * Dp)[i] */
					tmp = (wrk[i] - hx[i] - tmp) / Dp_L2; /* (f(p+dp)[i] - f(p)[i] - (J * Dp)[i])/(dp^T*dp) */
					for (j = 0; j<m; ++j)
						jac[i*m + j] += tmp*Dp[j];
				}
				++updjac;
				newjac = 1;
			}

			for (i = 0, dL = 0.0; i<m; ++i)
				dL += Dp[i] * (mu*Dp[i] + jacTe[i]);

			if (dL>0.0 && dF>0.0) { /* reduction in error, increment is accepted */
				tmp = ((2.0)*dF / dL - (1.0));
				tmp = (1.0) - tmp*tmp*tmp;
				mu = mu*((tmp >= (ONE_THIRD)) ? tmp : (ONE_THIRD));
				nu = 2;

				for (i = 0; i<m; ++i) /* update p's estimate */
					p[i] = pDp[i];

				for (i = 0; i<n; ++i) { /* update e, hx and ||e||_2 */
					e[i] = wrk2[i]; //x[i]-wrk[i];
					hx[i] = wrk[i];
				}
				p_eL2 = pDp_eL2;
				updp = 1;
				continue;
			}
		}

		/* if this point is reached, either the linear system could not be solved or
		* the error did not reduce; in any case, the increment must be rejected
		*/

		mu *= nu;
		nu2 = nu << 1; // 2*nu;
		if (nu2 <= nu) { /* nu has wrapped around (overflown). Thanks to Frank Jordan for spotting this case */
			stop = 5;
			break;
		}
		nu = nu2;

		for (i = 0; i<m; ++i) /* restore diagonal J^T J entries */
			jacTjac[i*m + i] = diag_jacTjac[i];
	}

	if (k >= itmax) stop = 3;

	for (i = 0; i<m; ++i) /* restore diagonal J^T J entries */
		jacTjac[i*m + i] = diag_jacTjac[i];

	if (info) {
		info[0] = init_p_eL2;
		info[1] = p_eL2;
		info[2] = jacTe_inf;
		info[3] = Dp_L2;
		for (i = 0, tmp = -FLT_MAX; i<m; ++i)
			if (tmp<jacTjac[i*m + i]) tmp = jacTjac[i*m + i];
		info[4] = mu / tmp;
		info[5] = (float)k;
		info[6] = (float)stop;
		info[7] = (float)nfev;
		info[8] = (float)njap;
		info[9] = (float)nlss;
	}

	/* covariance matrix */
	if (covar) {
		LEVMAR_COVAR(jacTjac, covar, p_eL2, m, n);
	}


	if (freework) free(work);


	return (stop != 4 && stop != 7) ? k : LM_ERROR;
}