/*=================================================================
*
* rot2dc.c    Performs 2D Rotation 
* The syntax is:
*
*        rot2dc(IN,OUT,PHI,INTERP,[CENTER])
*
*
* Last changes: Oct. 20, 2003
* M. Riedlberger
*
*=================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mex.h"
#include "splines_3.h"
#define	 PI ((double)3.14159265358979323846264338327950288419716939937510)

/* Input Arguments */
#define    INP    prhs[0]
#define    OUT    prhs[1]
#define    PHI    prhs[2]
#define    INT    prhs[3]
#define    CENT   prhs[4]

/* 3D Rotation */
void rot2d (
	float *image,
	float *rotimg,
	long  sx,
	long  sy,
	float phi,
	char  ip,
	float px,
	float py)
{
float rm00, rm01, rm10, rm11;	/* rot matrix */
float pi, pj;			/* coordinates according to pivot */
float r_x, r_y;			/* rotated pixel */
long  i, j;			/* loop variables */
long  sx_1, sy_1;	/* highest pixels */
long  floorx, floory;		/* rotated coordinates as integer */
float vx1, vx2;		/* interp. parameter */
float vy1, vy2;		/*  ... */
float AA, BB;	        /*  ... */
float *imgp;

phi = phi * PI / 180;

/* rotation matrix */
rm00 = cos(phi);
rm01 = -sin(phi);
rm10 = sin(phi);
rm11 = cos(phi);

if (ip=='s') {

   /* Modified by Achilleas -> Check linux version
    if (SamplesToCoefficients2D(image, sx, sy))*/
   if (SamplesToCoefficients2D(image, sx, sy, 3))
	mexErrMsgTxt("Splines: Change of basis failed\n");
  for (pj=-py, j=0; j<sy; j++, pj++)
        for (pi=-px, i=0; i<sx; i++, pi++) {

	    /* transformation of coordinates */
             r_x = px + rm00 * pi + rm10 * pj;
             if (r_x <= -0.5 || r_x >= sx - 0.5 ) {
	         *rotimg++  = 0;
		 continue;
		 } 
             r_y = py + rm01 * pi + rm11 * pj;
             if (r_y <= -0.5 || r_y >= sy - 0.5 ) {
	         *rotimg++ = 0;
		 continue;
		 } 

 	     *rotimg++ = (float) InterpolatedValue2D (image,sx, sy, r_x, r_y);
  }
 }

if (ip=='l') {

  sx_1 = sx - 1;
  sy_1 = sy - 1;
  for (pj=-py, j=0; j<sy; j++, pj++)
      for (pi=-px, i=0; i<sx; i++, pi++) {
      
	    /* transformation of coordinates */
             r_x = px + rm00 * pi + rm10 * pj;
             if (r_x < 0 || r_x > sx_1 ) {
	         *rotimg++ = 0;    /* pixel not inside */
		 continue;
		 } 
             r_y = py + rm01 * pi + rm11 * pj;
             if (r_y < 0 || r_y > sy_1 ) {
	         *rotimg++ = 0;
		 continue;
		 } 
		 
	     /* Interpolation */
	     floorx = r_x;
	     vx2 = r_x - floorx;
	     vx1 = 1 - vx2;
	     floory = r_y;
	     vy2 = r_y - floory;
	     vy1 = 1 - vy2;
	     
/*	     imgp = &image[floorx + sx*floory];
	     AA = *imgp+(imgp[1]-*imgp)*vx2;
	     BB = imgp[sx]+(imgp[sx+1]-imgp[sx])*vx2;
	     *rotimg++ = AA + (BB - AA) * vy2;*/
		 imgp = &image[floorx + sx*floory];
		 if (r_x<sx-1)    			/* not last x pixel, */
	             AA = *imgp+(imgp[1]-*imgp)*vx2;	/* interpolation */
		  else           	/* last x pixel, no interpolation in x possible */
	             AA = *imgp;
		 if (r_y<sy-1) {				/* not last y pixel, */
  		     if (r_x<sx-1)			/* not last x pixel, */
     	                 BB = imgp[sx]+(imgp[sx+1]-imgp[sx])*vx2;/* interpolation */
	              else 
     	                 BB = imgp[sx];  /* last x pixel, no interpolation in x possible */
			 }
		  else
	             BB = 0; /* last y pixel, no interpolation in y possible */
	         *rotimg++ = AA + (BB - AA) * vy2;
  }
 }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   const int *dims_i, *dims_o;
   char  *ip;
   float *center;

   /* Check for proper number of arguments */
   if (nrhs != 5) { printf("%d",nrhs);
       mexErrMsgTxt("5 input arguments required.\n Syntax: rot2d(input_image, output_image, Phi, Interpolation_type, center)");    }
   else if (nlhs > 1) {
       mexErrMsgTxt("Too many output arguments.");    }
 
   /* Check data types */
   if (!mxIsSingle(INP) || !mxIsSingle(OUT)) {
       mexErrMsgTxt("Input volumes must be single.\n"); }

   if (mxGetNumberOfDimensions(INP)!= mxGetNumberOfDimensions(OUT)) {
       mexErrMsgTxt("Image volumes must have same dimensions.\n");    }

   dims_i=mxGetDimensions(INP);
   dims_o=mxGetDimensions(OUT);

    if (dims_o[0]!=dims_i[0] || dims_o[1]!=dims_i[1]) {
       mexErrMsgTxt("Image volumes must have same size.\n"); }

    ip = mxArrayToString(INT);
    if (strcmp ("linear",ip) != 0 && strcmp ("splines",ip) != 0)
	mexErrMsgTxt("Unknown interpolation type\n");

   center=mxGetData(CENT);
   /* Take into consideration that in matlab the image starts at 1
    and in C at 0. Hence the center needs to be put one pixel back */
 
   center[0]=center[0]-1;
   center[1]=center[1]-1;
   
   
   /* Do the actual computations in a subroutine */
   rot2d(mxGetData(INP),mxGetData(OUT),dims_i[0],dims_i[1],mxGetScalar(PHI),ip[0],center[0],center[1]);

   }

