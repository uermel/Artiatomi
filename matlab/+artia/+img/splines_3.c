/*****************************************************************************
 *	Date: January 29, 2002
 *----------------------------------------------------------------------------
 *	This C program is based on the following three papers:
 *		[1]	M. Unser,
 *			"Splines: A Perfect Fit for Signal and Image Processing,"
 *			IEEE Signal Processing Magazine, vol. 16, no. 6, pp. 22-38,
 *			November 1999.
 *		[2]	M. Unser, A. Aldroubi and M. Eden,
 *			"B-Spline Signal Processing: Part I--Theory,"
 *			IEEE Transactions on Signal Processing, vol. 41, no. 2, pp. 821-832,
 *			February 1993.
 *		[3]	M. Unser, A. Aldroubi and M. Eden,
 *			"B-Spline Signal Processing: Part II--Efficient Design and Applications,"
 *			IEEE Transactions on Signal Processing, vol. 41, no. 2, pp. 834-848,
 *			February 1993.
 *----------------------------------------------------------------------------
 *	
 *	EPFL/STI/IOA/BIG
 *	Philippe Thevenaz
 *	Bldg. BM-Ecublens 4.137
 *	CH-1015 Lausanne
 *----------------------------------------------------------------------------
 *	phone (CET):	+41(21)693.51.61
 *	fax:			+41(21)693.37.01
 *	RFC-822:		philippe.thevenaz@epfl.ch
 *	X-400:			/C=ch/A=400net/P=switch/O=epfl/S=thevenaz/G=philippe/
 *	URL:			http://bigwww.epfl.ch/
 *----------------------------------------------------------------------------
 *	Date Sep. 23, 2003
 *	Extension to 3D
 *	Martin Riedlberger
 *----------------------------------------------------------------------------
 *	This file is best viewed with 4-space tabs (the bars below should be aligned)
 *	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|
 *  |...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|
 ****************************************************************************/

/*****************************************************************************
 *	System includes
 ****************************************************************************/
#include	<float.h>
#include	<math.h>
#include	<stddef.h>
#include	<stdio.h>
#include	<stdlib.h>

/*****************************************************************************
 *	Other includes
 ****************************************************************************/
#include	"splines_3.h"

/*****************************************************************************
 *	Declaration of static procedures
 ****************************************************************************/
/*--------------------------------------------------------------------------*/
static void		ConvertToInterpolationCoefficients
				(
					double	c[],		/* input samples --> output coefficients */
					long	DataLength,	/* number of samples or coefficients */
					double	z[],		/* poles */
					long	NbPoles,	/* number of poles */
					double	Tolerance	/* admissible relative error */
				);

/*--------------------------------------------------------------------------*/
static void		GetColumn
				(
					float	*Image,		/* input image array */
					long	x,		/* x coordinate of the selected line */
					long	z,		/* z coordinate of the selected line */
					double	Line[],		/* output linear array */
					long	Width,		/* width of the image */
					long	Height		/* length of the line and height of the image */
				);

/*--------------------------------------------------------------------------*/
static void		GetRow
				(
					float	*Image,		/* input image array */
					long	y,		/* y coordinate of the selected line */
					long	z,		/* z coordinate of the selected line */
					double	Line[],		/* output linear array */
					long	Width,		/* length of the line and width of the image */
					long	Height		/* length of the line and height of the image */
				);

/*--------------------------------------------------------------------------*/
static void		GetZLine
				(
					float	*Image,		/* output image array */
					long	x,		/* x coordinate of the selected line */
					long	y,		/* y coordinate of the selected line */
					double	Line[],		/* input linear array */
					long	Width,		/* length of the line and width of the image */
					long	Height,		/* length of the line and height of the image */
					long	Depth
				);

/*--------------------------------------------------------------------------*/
static double	InitialCausalCoefficient
				(
					double	c[],		/* coefficients */
					long	DataLength,	/* number of coefficients */
					double	z,			/* actual pole */
					double	Tolerance	/* admissible relative error */
				);

/*--------------------------------------------------------------------------*/
static double	InitialAntiCausalCoefficient
				(
					double	c[],		/* coefficients */
					long	DataLength,	/* number of samples or coefficients */
					double	z			/* actual pole */
				);

/*--------------------------------------------------------------------------*/
static void		PutColumn
				(
					float	*Image,		/* output image array */
					long	x,		/* x coordinate of the selected line */
					long	z,		/* z coordinate of the selected line */
					double	Line[],		/* input linear array */
					long	Width,		/* width of the image */
					long	Height		/* length of the line and height of the image */
				);

/*--------------------------------------------------------------------------*/
static void		PutRow
				(
					float	*Image,		/* output image array */
					long	y,		/* y coordinate of the selected line */
					long	z,		/* z coordinate of the selected line */
					double	Line[],		/* input linear array */
					long	Width,		/* length of the line and width of the image */
					long	Height		/* length of the line and height of the image */
				);

/*--------------------------------------------------------------------------*/
static void		PutZline
				(
					float	*Image,		/* output image array */
					long	x,		/* x coordinate of the selected line */
					long	y,		/* y coordinate of the selected line */
					double	Line[],		/* input linear array */
					long	Width,		/* length of the line and width of the image */
					long	Height,		/* length of the line and height of the image */
					long	Depth
				);

/*****************************************************************************
 *	Definition of static procedures
 ****************************************************************************/
/*--------------------------------------------------------------------------*/
static void		ConvertToInterpolationCoefficients
				(
					double	c[],		/* input samples --> output coefficients */
					long	DataLength,	/* number of samples or coefficients */
					double	z[],		/* poles */
					long	NbPoles,	/* number of poles */
					double	Tolerance	/* admissible relative error */
				)

{ /* begin ConvertToInterpolationCoefficients */

	double	Lambda = 1.0;
	long	n, k;

	/* special case required by mirror boundaries */
	if (DataLength == 1L) {
		return;
	}
	/* compute the overall gain */
	for (k = 0L; k < NbPoles; k++) {
		Lambda = Lambda * (1.0 - z[k]) * (1.0 - 1.0 / z[k]);
	}
	/* apply the gain */
	for (n = 0L; n < DataLength; n++) {
		c[n] *= Lambda;
	}
	/* loop over all poles */
	for (k = 0L; k < NbPoles; k++) {
		/* causal initialization */
		c[0] = InitialCausalCoefficient(c, DataLength, z[k], Tolerance);
		/* causal recursion */
		for (n = 1L; n < DataLength; n++) {
			c[n] += z[k] * c[n - 1L];
		}
		/* anticausal initialization */
		c[DataLength - 1L] = InitialAntiCausalCoefficient(c, DataLength, z[k]);
		/* anticausal recursion */
		for (n = DataLength - 2L; 0 <= n; n--) {
			c[n] = z[k] * (c[n + 1L] - c[n]);
		}
	}
} /* end ConvertToInterpolationCoefficients */

/*--------------------------------------------------------------------------*/
static double	InitialCausalCoefficient
				(
					double	c[],		/* coefficients */
					long	DataLength,	/* number of coefficients */
					double	z,			/* actual pole */
					double	Tolerance	/* admissible relative error */
				)

{ /* begin InitialCausalCoefficient */

	double	Sum, zn, z2n, iz;
	long	n, Horizon;

	/* this initialization corresponds to mirror boundaries */
	Horizon = DataLength;
	if (Tolerance > 0.0) {
		Horizon = (long)ceil(log(Tolerance) / log(fabs(z)));
	}
	if (Horizon < DataLength) {
		/* accelerated loop */
		zn = z;
		Sum = c[0];
		for (n = 1L; n < Horizon; n++) {
			Sum += zn * c[n];
			zn *= z;
		}
		return(Sum);
	}
	else {
		/* full loop */
		zn = z;
		iz = 1.0 / z;
		z2n = pow(z, (double)(DataLength - 1L));
		Sum = c[0] + z2n * c[DataLength - 1L];
		z2n *= z2n * iz;
		for (n = 1L; n <= DataLength - 2L; n++) {
			Sum += (zn + z2n) * c[n];
			zn *= z;
			z2n *= iz;
		}
		return(Sum / (1.0 - zn * zn));
	}
} /* end InitialCausalCoefficient */

/*--------------------------------------------------------------------------*/
static void		GetColumn
				(
					float	*Image,		/* input image array */
					long	x,		/* x coordinate of the selected line */
					long	z,		/* z coordinate of the selected line */
					double	Line[],		/* output linear array */
					long	Width,		/* length of the line */
					long	Height		
				)

{ /* begin GetColumn */

	long	y;

	Image = Image + (ptrdiff_t)x + (ptrdiff_t)(z * Width * Height);
	for (y = 0L; y < Height; y++) {
		Line[y] = (double)*Image;
		Image += (ptrdiff_t)Width;
	}
} /* end GetColumn */

/*--------------------------------------------------------------------------*/
static void		GetRow
				(
					float	*Image,		/* input image array */
					long	y,		/* y coordinate of the selected line */
					long	z,		/* z coordinate of the selected line */
					double	Line[],		/* output linear array */
					long	Width,		/* length of the line */
					long	Height		
				)

{ /* begin GetRow */

	long	x;

	Image = Image + (ptrdiff_t)(y * Width) + (ptrdiff_t)(z * Width * Height);
	for (x = 0L; x < Width; x++) {
		Line[x] = (double)*Image++;
	}
} /* end GetRow */

/*--------------------------------------------------------------------------*/
static void		GetZLine
				(
					float	*Image,		/* input image array */
					long	x,		/* x coordinate of the selected line */
					long	y,		/* y coordinate of the selected line */
					double	Line[],		/* output linear array */
					long	Width,		/* length of the line */
					long	Height,		
					long	Depth
				)

{ /* begin GetZline */

	long	z;

	Image = Image + (ptrdiff_t)x + (ptrdiff_t)(y * Width);
	for (z = 0L; z < Depth; z++) {
		Line[z] = (double)*Image;
		Image += (ptrdiff_t)(Width*Height);
	}
} /* end GetZline */

/*--------------------------------------------------------------------------*/
static double	InitialAntiCausalCoefficient
				(
					double	c[],		/* coefficients */
					long	DataLength,	/* number of samples or coefficients */
					double	z			/* actual pole */
				)

{ /* begin InitialAntiCausalCoefficient */

	/* this initialization corresponds to mirror boundaries */
	return((z / (z * z - 1.0)) * (z * c[DataLength - 2L] + c[DataLength - 1L]));
} /* end InitialAntiCausalCoefficient */

/*--------------------------------------------------------------------------*/
static void		PutColumn
				(
					float	*Image,		/* output image array */
					long	x,		/* x coordinate of the selected line */
					long	z,		/* z coordinate of the selected line */
					double	Line[],		/* input linear array */
					long	Width,		/* length of the line and height of the image */
					long	Height
				)

{ /* begin PutColumn */

	long	y;

	Image = Image + (ptrdiff_t)x + (ptrdiff_t)(z * Width * Height);
	for (y = 0L; y < Height; y++) {
		*Image = (float)Line[y];
		Image += (ptrdiff_t)Width;
	}
} /* end PutColumn */

/*--------------------------------------------------------------------------*/
static void		PutRow
				(
					float	*Image,		/* output image array */
					long	y,		/* y coordinate of the selected line */
					long	z,		/* z coordinate of the selected line */
					double	Line[],		/* input linear array */
					long	Width,
					long	Height
				)

{ /* begin PutRow */

	long	x;

	Image = Image + (ptrdiff_t)(y * Width) + (ptrdiff_t)(z * Width * Height);
	for (x = 0L; x < Width; x++) {
		*Image++ = (float)Line[x];
	}
} /* end PutRow */

/*--------------------------------------------------------------------------*/
static void		PutZLine
				(
					float	*Image,		/* output image array */
					long	x,		/* x coordinate of the selected line */
					long	y,		/* y coordinate of the selected line */
					double	Line[],		/* input linear array */
					long	Width,
					long	Height,
					long	Depth
				)

{ /* begin PutZline */

	long	z;

	Image = Image + (ptrdiff_t)x + (ptrdiff_t)(y * Width);
	for (z = 0L; z < Depth; z++) {
		*Image = (float)Line[z];
		Image += (ptrdiff_t)(Width*Height);
	}
} /* end PutZLine */

/*****************************************************************************
 *	Definition of extern procedures
 ****************************************************************************/
/*--------------------------------------------------------------------------*/
extern int		SamplesToCoefficients
				(
					float	*Image,		/* in-place processing */
					long	Width,		/* width of the image */
					long	Height,		/* height of the image */
					long	Depth		/* depth  of the image */
				)

{ /* begin SamplesToCoefficients */

	double	*Line;
	double	Pole[2];
	long	x, y, z;

	/* recover the poles from a lookup table */
			Pole[0] = sqrt(3.0) - 2.0;

	/* convert the image samples into interpolation coefficients */
	Line = (double *)malloc((size_t)(Height * (long)sizeof(double)));
	if (Line == (double *)NULL) {
		printf("Column allocation failed\n");
		return(1);
	}

for (x = 0L; x < Width; x++) {
	for (z = 0L; z < Depth; z++) {
		GetColumn(Image, x, z, Line, Width, Height);
		ConvertToInterpolationCoefficients(Line, Height, Pole, 1, DBL_EPSILON);
		PutColumn(Image, x, z, Line, Width, Height);
	}
	}
	free(Line);

	Line = (double *)malloc((size_t)(Depth * (long)sizeof(double)));
	if (Line == (double *)NULL) {
		printf("Row allocation failed\n");
		return(1);
	}
for (y = 0L; y < Height; y++) 
	for (x = 0L; x < Width; x++) {
		GetZLine(Image, x, y, Line, Width, Height, Depth);
		ConvertToInterpolationCoefficients(Line, Depth, Pole, 1, DBL_EPSILON);
		PutZLine(Image, x, y, Line, Width, Height, Depth);
	}
	free(Line);

if (Depth > 1) {
	Line = (double *)malloc((size_t)(Width * (long)sizeof(double)));
	if (Line == (double *)NULL) {
		printf("Row allocation failed\n");
		return(1);
	}
   for (z = 0L; z < Depth; z++) {
	for (y = 0L; y < Height; y++) {
		GetRow(Image, y, z, Line, Width, Height);
		ConvertToInterpolationCoefficients(Line, Width, Pole, 1, DBL_EPSILON);
		PutRow(Image, y, z, Line, Width, Height);
	}
	}
	free(Line);
}

	return(0);
} /* end SamplesToCoefficients */

extern int		SamplesToCoefficients2D
				(
					float	*Image,		/* in-place processing */
					long	Width,		/* width of the image */
					long	Height,		/* height of the image */
					long	SplineDegree/* degree of the spline model */
				)

{ /* begin SamplesToCoefficients */

	double	*Line;
	double	Pole[2];
	long	NbPoles;
	long	x, y;

	/* recover the poles from a lookup table */
	switch (SplineDegree) {
		case 2L:
			NbPoles = 1L;
			Pole[0] = sqrt(8.0) - 3.0;
			break;
		case 3L:
			NbPoles = 1L;
			Pole[0] = sqrt(3.0) - 2.0;
			break;
		case 4L:
			NbPoles = 2L;
			Pole[0] = sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0;
			Pole[1] = sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0;
			break;
		case 5L:
			NbPoles = 2L;
			Pole[0] = sqrt(135.0 / 2.0 - sqrt(17745.0 / 4.0)) + sqrt(105.0 / 4.0)
				- 13.0 / 2.0;
			Pole[1] = sqrt(135.0 / 2.0 + sqrt(17745.0 / 4.0)) - sqrt(105.0 / 4.0)
				- 13.0 / 2.0;
			break;
		default:
			printf("Invalid spline degree\n");
			return(1);
	}

	/* convert the image samples into interpolation coefficients */
	/* in-place separable process, along x */
	Line = (double *)malloc((size_t)(Width * (long)sizeof(double)));
	if (Line == (double *)NULL) {
		printf("Row allocation failed\n");
		return(1);
	}
	for (y = 0L; y < Height; y++) {
		GetRow(Image, y, 0, Line, Width, Height);
		ConvertToInterpolationCoefficients(Line, Width, Pole, NbPoles, DBL_EPSILON);
		PutRow(Image, y, 0, Line, Width, Height);
	}
	free(Line);

	/* in-place separable process, along y */
	Line = (double *)malloc((size_t)(Height * (long)sizeof(double)));
	if (Line == (double *)NULL) {
		printf("Column allocation failed\n");
		return(1);
	}
	for (x = 0L; x < Width; x++) {
		GetColumn(Image, x, 0, Line, Width, Height);
		ConvertToInterpolationCoefficients(Line, Height, Pole, NbPoles, DBL_EPSILON);
		PutColumn(Image, x, 0, Line, Width, Height);
	}
	free(Line);

	return(0);
} /* end SamplesToCoefficients */

extern double	InterpolatedValue
				(
					float	*Bcoeff,	/* input B-spline array of coefficients */
					long	Width,		/* width of the image */
					long	Height,		/* height of the image */
					long	Depth,		/* depth of the image */
					double	x,		/* x coordinate where to interpolate */
					double	y,		/* y coordinate where to interpolate */
					double	z		/* z coordinate where to interpolate */
				)

{ /* begin InterpolatedValue */

	float	*p;
	double	xWeight[6], yWeight[6], zWeight[6];
	double	interpolated;
	double	w, wx, wy, w2, w4, t, t0, t1;
	long	xIndex[6], yIndex[6], zIndex[6];
	long	Width2 = 2L * Width - 2L, Height2 = 2L * Height - 2L, Depth2 = 2L * Depth - 2L;
	long	i, j, k, l;

	/* compute the interpolation indexes */
		i = (long)floor(x) - 1;
		j = (long)floor(y) - 1;
		k = (long)floor(z) - 1;
		for (l = 0L; l <= 3; l++) {
			xIndex[l] = i++;
			yIndex[l] = j++;
			zIndex[l] = k++;
		}


	/* compute the interpolation weights */
			/* x */
			w = x - (double)xIndex[1];
			xWeight[3] = (1.0 / 6.0) * w * w * w;
			xWeight[0] = (1.0 / 6.0) + (1.0 / 2.0) * w * (w - 1.0) - xWeight[3];
			xWeight[2] = w + xWeight[0] - 2.0 * xWeight[3];
			xWeight[1] = 1.0 - xWeight[0] - xWeight[2] - xWeight[3];
			/* y */
			w = y - (double)yIndex[1];
			yWeight[3] = (1.0 / 6.0) * w * w * w;
			yWeight[0] = (1.0 / 6.0) + (1.0 / 2.0) * w * (w - 1.0) - yWeight[3];
			yWeight[2] = w + yWeight[0] - 2.0 * yWeight[3];
			yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];
			/* z */
			w = z - (double)zIndex[1];
			zWeight[3] = (1.0 / 6.0) * w * w * w;
			zWeight[0] = (1.0 / 6.0) + (1.0 / 2.0) * w * (w - 1.0) - zWeight[3];
			zWeight[2] = w + zWeight[0] - 2.0 * zWeight[3];
			zWeight[1] = 1.0 - zWeight[0] - zWeight[2] - zWeight[3];


	/* apply the mirror boundary conditions */
	for (l = 0L; l <= 3; l++) {
		xIndex[l] = (Width == 1L) ? (0L) : ((xIndex[l] < 0L) ?
			(-xIndex[l] - Width2 * ((-xIndex[l]) / Width2))
			: (xIndex[l] - Width2 * (xIndex[l] / Width2)));
		if (Width <= xIndex[l]) {
			xIndex[l] = Width2 - xIndex[l];
		}
		yIndex[l] = (Height == 1L) ? (0L) : ((yIndex[l] < 0L) ?
			(-yIndex[l] - Height2 * ((-yIndex[l]) / Height2))
			: (yIndex[l] - Height2 * (yIndex[l] / Height2)));
		if (Height <= yIndex[l]) {
			yIndex[l] = Height2 - yIndex[l];
		}
		zIndex[l] = (Depth == 1L) ? (0L) : ((zIndex[l] < 0L) ?
			(-zIndex[l] - Depth2 * ((-zIndex[l]) / Depth2))
			: (zIndex[l] - Depth2 * (zIndex[l] / Depth2)));
		if (Depth <= zIndex[l]) {
			zIndex[l] = Depth2 - zIndex[l];
		}
	}

	/* perform interpolation */
	interpolated = 0.0;
for (k = 0L; k <= 3; k++) {
/*	p = Bcoeff + (ptrdiff_t)(zIndex[k] * Width * Height); */
	wy = 0.0;
	for (j = 0L; j <= 3; j++) {
		p = Bcoeff + (ptrdiff_t)(zIndex[k] * Width * Height) + (ptrdiff_t)(yIndex[j] * Width);
		wx = 0.0;
		for (i = 0L; i <= 3; i++) {
			wx += xWeight[i] * p[xIndex[i]];
		}
		wy += yWeight[j] * wx;
	}
	interpolated += zWeight[k] * wy;
}

	return(interpolated);
} /* end InterpolatedValue */


extern double	InterpolatedValue2D
				(
					float	*Bcoeff,	/* input B-spline array of coefficients */
					long	Width,		/* width of the image */
					long	Height,		/* height of the image */
					double	x,		/* x coordinate where to interpolate */
					double	y		/* y coordinate where to interpolate */
				)

{ /* begin InterpolatedValue */

	float	*p;
	double	xWeight[6], yWeight[6];
	double	interpolated;
	double	w, w2, w4, t, t0, t1;
	long	xIndex[6], yIndex[6];
	long	Width2 = 2L * Width - 2L, Height2 = 2L * Height - 2L;
	long	i, j, k;

	/* compute the interpolation indexes */
		i = (long)floor(x) - 1;
		j = (long)floor(y) - 1;
		for (k = 0L; k <= 3; k++) {
			xIndex[k] = i++;
			yIndex[k] = j++;
		}


	/* compute the interpolation weights */
			/* x */
			w = x - (double)xIndex[1];
			xWeight[3] = (1.0 / 6.0) * w * w * w;
			xWeight[0] = (1.0 / 6.0) + (1.0 / 2.0) * w * (w - 1.0) - xWeight[3];
			xWeight[2] = w + xWeight[0] - 2.0 * xWeight[3];
			xWeight[1] = 1.0 - xWeight[0] - xWeight[2] - xWeight[3];
			/* y */
			w = y - (double)yIndex[1];
			yWeight[3] = (1.0 / 6.0) * w * w * w;
			yWeight[0] = (1.0 / 6.0) + (1.0 / 2.0) * w * (w - 1.0) - yWeight[3];
			yWeight[2] = w + yWeight[0] - 2.0 * yWeight[3];
			yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];


	/* apply the mirror boundary conditions */
	for (k = 0L; k <= 3; k++) {
		xIndex[k] = (Width == 1L) ? (0L) : ((xIndex[k] < 0L) ?
			(-xIndex[k] - Width2 * ((-xIndex[k]) / Width2))
			: (xIndex[k] - Width2 * (xIndex[k] / Width2)));
		if (Width <= xIndex[k]) {
			xIndex[k] = Width2 - xIndex[k];
		}
		yIndex[k] = (Height == 1L) ? (0L) : ((yIndex[k] < 0L) ?
			(-yIndex[k] - Height2 * ((-yIndex[k]) / Height2))
			: (yIndex[k] - Height2 * (yIndex[k] / Height2)));
		if (Height <= yIndex[k]) {
			yIndex[k] = Height2 - yIndex[k];
		}
	}

	/* perform interpolation */
	interpolated = 0.0;
	for (j = 0L; j <= 3; j++) {
		p = Bcoeff + (ptrdiff_t)(yIndex[j] * Width);
		w = 0.0;
		for (i = 0L; i <= 3; i++) {
			w += xWeight[i] * p[xIndex[i]];
		}
		interpolated += yWeight[j] * w;
	}

	return(interpolated);
} /* end InterpolatedValue2D */
