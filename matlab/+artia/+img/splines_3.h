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

/*--------------------------------------------------------------------------*/
extern int		SamplesToCoefficients
				(
					float	*Image,		/* in-place processing */
					long	Width,		/* width of the image */
					long	Height,		/* height of the image */
					long	Depth		/* depth  of the image */
				);


/*--------------------------------------------------------------------------*/
extern int		SamplesToCoefficients2D
				(
					float	*Image,		/* in-place processing */
					long	Width,		/* width of the image */
					long	Height,		/* height of the image */
					long	SplineDegree/* degree of the spline model */
				);


/*--------------------------------------------------------------------------*/
extern double	InterpolatedValue
				(
					float	*Bcoeff,	/* input B-spline array of coefficients */
					long	Width,		/* width of the image */
					long	Height,		/* height of the image */
					long	Depth,		/* depth of the image */
					double	x,		/* x coordinate where to interpolate */
					double	y,		/* y coordinate where to interpolate */
					double	z		/* z coordinate where to interpolate */
				);

extern double	InterpolatedValue2D
				(
					float	*Bcoeff,	/* input B-spline array of coefficients */
					long	Width,		/* width of the image */
					long	Height,		/* height of the image */
					double	x,			/* x coordinate where to interpolate */
					double	y			/* y coordinate where to interpolate */
				);
