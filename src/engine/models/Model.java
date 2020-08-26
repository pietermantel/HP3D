package engine.models;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

import engine.math.Point2D;
import engine.math.Point3D;

public class Model {
	private Point3D[] vertices;
	private Polygon[] polygons;

	public Model(String objPath, String pngPath) {
		// Read file
		try {
			InputStream in = getClass().getResourceAsStream("/res/" + objPath + ".obj");
			BufferedReader reader = new BufferedReader(new InputStreamReader(in));

			// Load data
			String line;
			ArrayList<Point3D> vertexCoords = new ArrayList<Point3D>();
			ArrayList<Point2D> textureCoords = new ArrayList<Point2D>();
			ArrayList<Point3D> vertexNormals = new ArrayList<Point3D>();
			ArrayList<String[]> faces = new ArrayList<String[]>();
			while ((line = reader.readLine()) != null) {
				String[] data = line.split(" ");
				if (line.startsWith("v ")) {
					// Vertex
					double[] values = new double[3];
					for (int i = 0; i < 3; i++)
						values[i] = Double.parseDouble(data[i + 1]);
					vertexCoords.add(new Point3D(values));
				} else if (line.startsWith("vt ")) {
					// Texture Coord
					double[] values = new double[2];
					for (int i = 0; i < 2; i++)
						values[i] = Double.parseDouble(data[i + 1]);
					textureCoords.add(new Point2D(values));
				} else if (line.startsWith("vn ")) {
					// Vertex Normal
					double[] values = new double[3];
					for (int i = 0; i < 3; i++)
						values[i] = Double.parseDouble(data[i + 1]);
					vertexNormals.add(new Point3D(values));
				} else if (line.startsWith("f ")) {
					// Face
					String[] face = new String[3];
					for (int i = 0; i < 3; i++)
						face[i] = data[i + 1];
					faces.add(face);
				}
			}
			reader.close();
			in.close();

			// Add vertices
			vertices = new Point3D[vertexCoords.size()];
			for (int i = 0; i < vertices.length; i++)
				vertices[i] = vertexCoords.get(i);
			
			// Generate polygons
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
