package perceptron;

public class SquaredLoss implements LossFunction {
    @Override
    public double eval(double[] y, double[] d) {
        double loss = 0;
        int N = d.length;
        for (int n = 0; n < N; ++n) {
            loss += (y[n] - d[n]) * (y[n] - d[n]);
        }
        return 1 / (2.0 * N) * loss;
    }

    @Override
    public double[] grad(double[] y, double[] d) {
        int N = d.length;
        double[] g = new double[N];
        for (int n = 0; n < N; ++n) {
            g[n] = 1.0 / N * (y[n] - d[n]);
        }
        return g;
    }
}
