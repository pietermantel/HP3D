package engine.math;

public class Point2D {
	private double[] values;

	public Point2D(double x, double y) {
		values = new double[] {x, y};
	}

	public Point2D(double[] values) {
		if (values.length != 2)
			throw new Error("Point2D values.length != 2");
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
}
