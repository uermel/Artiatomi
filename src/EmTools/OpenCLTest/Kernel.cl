//
//__kernel void VectorAdd(__global const float* a, __global const float* b, __global float* c, int iNumElements)
//{
//	// get index into global data array
//	int iGID = get_global_id(0);
//
//	// bound check (equivalent to the limit on a 'for' loop for standard/serial C code
//	if (iGID >= iNumElements)
//	{
//		return;
//	}
//
//	// add the vector elements
//	c[iGID] = a[iGID] + b[iGID];
//}

__kernel void Test2D(__local float* locBuffer, __global const float* a, __global const float* b, __global float* c, int dimX, int dimY)
{
	// get index into global data array
	int x = get_global_id(0);
	int y = get_global_id(1); 

	// bound check (equivalent to the limit on a 'for' loop for standard/serial C code
	if (x >= dimX || y >= dimY)
	{
		return;
	}

	int idx = y * dimX + x;

	// add the vector elements
	c[idx] = a[idx] + b[idx];
}