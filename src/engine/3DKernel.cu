extern "C"
__global__ void applyRotationMatrix(int n, double *matrix, double **pointArray, double **out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		double *point = pointArray[i];
		double x = matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2];
		double y = matrix[3] * point[0] + matrix[4] * point[1] + matrix[5] * point[2];
		double z = matrix[6] * point[0] + matrix[7] * point[1] + matrix[8] * point[2];
		double *output = out[i];
		output[0] = x;
		output[1] = y;
		output[2] = z;
	}
}

__global__ void rasterize(int numPoints, int screenWidth, int screenHeight, double fov, double *camPos, double *matrix, double **pointArray, double *out) {
	//Calculate thread index
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n) 
	{
		double *point = pointArray[i];
		double *modifiedPoint = new double[3];
		for(int i = 0; i < 3; i++) 
		{
			modifiedPoint[i] = point[i] - camPos[i];
		}
		for(int i = 0; i < 3; i++) 
		{
			modifiedPoint[i] = matrix[i] * modifiedPoint[0] + matrix[i + 1] * modifiedPoint[1] + matrix[i + 2] * modifiedPoint[2];
		}
	}
}
