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


#ifndef LEVMAR_H
#define LEVMAR_H

//Adopted from: http://users.ics.forth.gr/~lourakis/levmar/

#define LM_INFO_SZ    	 10
#define LM_ERROR -1
#define LM_INIT_MU    	 1E-03f
#define LM_STOP_THRESH	 1E-17f
#define LM_DIFF_DELTA    1E-06f
#define EPSILON       1E-12f
#include "../Basics/Default.h"

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
	void *adata);       /* pointer to possibly additional data, passed uninterpreted to func.
					   * Set to NULL if not needed
					   */

#endif