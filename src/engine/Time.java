package engine;

public class Time implements Runnable {
	public static double SPT = 1d / 60d;
	public static double SPF = 1d / 60d;
	private Main main;
	
	public Time(Main main) {
		this.main = main;
	}
	
	private long lastTick;
	private long lastRender;
	private long lastSecond;
	private int tickAmount = 0;
	private int renderAmount = 0;
	private int TPS = 0, FPS = 0;
	private boolean running;
	
	@Override
	public void run() {
		lastTick = System.nanoTime();
		lastRender = System.nanoTime();
		lastSecond = System.nanoTime();
		while (running) {
			if (System.nanoTime() - SPT * 1000000000 >= lastTick) {
				main.tick();
				lastTick = System.nanoTime();
				tickAmount++;
			}
			if (System.nanoTime() - SPF * 1000000000 >= lastRender) {
				main.render();
				lastRender = System.nanoTime();
				renderAmount++;
			}
			if (System.nanoTime() - 1000000000 > lastSecond) {
				
			}
		}
	}
	
}
