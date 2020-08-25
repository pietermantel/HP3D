package engine.math;

public class Point3D {
	
	private double[] values;

	public Point3D(double x, double y, double z) {
		values = new double[] {x, y, z};
	}
	
	public Point3D(double[] values) {
		if(values.length != 3) throw new Error("Point3D values.length != 3");
		this.values = values;
	}
	
	public double[] getValues() {
		return values;
	}
	
	public double getX() {
		return values[0];
	}
	
	public void setX(double x) {
		values[0] = x;
	}
	
	public double getY() {
		return values[1];
	}
	
	public void setY(double y) {
		values[1] = y;
	}
	
	public double getZ() {
		return values[2];
	}
	
	public void setZ(double z) {
		values[2] = z;
	}
	

}
