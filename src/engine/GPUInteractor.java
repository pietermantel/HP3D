package engine;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuMemFree;

import engine.math.Point3D;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class GPUInteractor {

	private CUdevice device;
	private CUcontext context;
	private CUmodule module;

	public GPUInteractor() {
		initialize();
	}
	
	public static void test() {
		GPUInteractor gpu = new GPUInteractor();
		int numPoints = 100000;
		Point3D[] points = new Point3D[numPoints];
		for (int i = 0; i < numPoints; i++) points[i] = new Point3D(Math.random(), Math.random(), Math.random());
		long timeBefore = System.nanoTime();
		gpu.applyRotationMatrices(new double[] { 2, -3, 1, 0.5, 3, -1, -4, 2, 1.5 }, points);
		System.out.println("Time elapsed: " + (System.nanoTime() - timeBefore) / 1000000);
	}

//	public static void main(String... args) {
//		test();
//	}

	public void initialize() {
		// Enable exceptions
		JCudaDriver.setExceptionsEnabled(true);
		String ptxFileName = "src\\engine\\3DKernel.ptx";

		// Initialize cuda driver
		JCudaDriver.cuInit(0);
		device = new CUdevice();
		JCudaDriver.cuDeviceGet(device, 0);
		context = new CUcontext();
		JCudaDriver.cuCtxCreate(context, 0, device);

		// Load the module
		module = new CUmodule();
		JCudaDriver.cuModuleLoad(module, ptxFileName);
	}

	public void applyRotationMatrices(double[] rotationMatrix, Point3D[] points) {
		// Load the applyRotationMatrices function
		CUfunction function = new CUfunction();
		cuModuleGetFunction(function, module, "applyRotationMatrix");

		int numElements = points.length;

		int blockSizeX = 256;
		int gridSizeX = (int) Math.ceil((double) numElements / blockSizeX);
		System.out.println(gridSizeX);

		// Allocate and copy rotation matrix to device
		long rotationMatrixSize = rotationMatrix.length * Sizeof.DOUBLE;
		CUdeviceptr deviceRotationMatrix = new CUdeviceptr();
		cuMemAlloc(deviceRotationMatrix, rotationMatrixSize);
		cuMemcpyHtoD(deviceRotationMatrix, Pointer.to(rotationMatrix), rotationMatrixSize);

		// Allocate pointer to points array
		long devicePointsPointersSize = points.length * Sizeof.POINTER;
		CUdeviceptr devicePointsPtr = new CUdeviceptr();
		cuMemAlloc(devicePointsPtr, devicePointsPointersSize);

		// Allocate and create pointers to all points
		long pointSize = 3 * Sizeof.DOUBLE;
		CUdeviceptr[] pointsDevicePointers = new CUdeviceptr[points.length];
		for (int i = 0; i < points.length; i++) {
			// Allocate and copy a point to device
			Point3D currentPoint = points[i];
			double[] values = currentPoint.getValues();
			CUdeviceptr currentPtr = new CUdeviceptr();
			cuMemAlloc(currentPtr, pointSize);
			cuMemcpyHtoD(currentPtr, Pointer.to(values), pointSize);

			pointsDevicePointers[i] = currentPtr;
		}
		
		cuMemcpyHtoD(devicePointsPtr, Pointer.to(pointsDevicePointers), devicePointsPointersSize);

		// Allocate output
		Point3D[] outputPoints = new Point3D[points.length];

		long deviceOutputPointersSize = points.length * Sizeof.POINTER;
		CUdeviceptr deviceOutput = new CUdeviceptr();
		cuMemAlloc(deviceOutput, deviceOutputPointersSize);

		// Allocate and create pointers to all output points
		CUdeviceptr[] outputDevicePointers = new CUdeviceptr[points.length];
		for (int i = 0; i < points.length; i++) {
			// Allocate and copy a point to device
			double[] values = new double[] { 0, 0, 0 };
			Point3D currentPoint = new Point3D(values);
			CUdeviceptr currentPtr = new CUdeviceptr();
			cuMemAlloc(currentPtr, pointSize);

			outputDevicePointers[i] = currentPtr;
		}
		
		cuMemcpyHtoD(deviceOutput, Pointer.to(outputDevicePointers), deviceOutputPointersSize);

		Pointer kernelParameters = Pointer.to(Pointer.to(new int[] { numElements }), Pointer.to(deviceRotationMatrix),
				Pointer.to(devicePointsPtr), Pointer.to(deviceOutput));
		
		cuLaunchKernel(function,
				gridSizeX, 1, 1,
				blockSizeX, 1, 1,
				0, null, 
				kernelParameters, null);
		cuCtxSynchronize();
		
		//Retrieve output
		for (int i = 0; i < points.length; i++) {
			double[] currentValues = points[i].getValues();
			CUdeviceptr currentPtr = outputDevicePointers[i];
			cuMemcpyDtoH(Pointer.to(currentValues), currentPtr, pointSize);
			cuMemFree(currentPtr);
			cuMemFree(pointsDevicePointers[i]);
//			for(int j = 0; j < currentValues.length; j++) {
//				System.out.println(currentValues[j]);
//			}
		}
	}

}
