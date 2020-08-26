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

__global__ void rasterize(int numPoints, int screenWidth, int screenHeight, double fovDistance, double *camPos, double *matrix, double **pointArray, double *out) {
	//Calculate thread index
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n) 
	{
		double *point = pointArray[i];
		double *modifiedPoint = new double[3];
		//Center point
		for(int i = 0; i < 3; i++) 
		{
			modifiedPoint[i] = point[i] - camPos[i];
		}
		//Apply Rotation Matrix
		for(int i = 0; i < 3; i++) 
		{
			modifiedPoint[i] = matrix[i * 3] * modifiedPoint[0] + matrix[i * 3 + 1] * modifiedPoint[1] + matrix[i * 3 + 2] * modifiedPoint[2];
		}
		double screenX = modifiedPoint[0] / modifiedPoint[2] * fovDistance * screenWidth / 2;
		double screenY = modifiedPoint[1] / modifiedPoint[2] * fovDistance * screenHeight / 2;
	}
}
