package perceptron;

import java.util.Arrays;

public class LinearActivation implements ActivationFunction {

    @Override
    public double[] eval(double[] a) {
        return Arrays.copyOf(a, a.length);
    }

    @Override
    public double[] grad(double[] a) {
        double[] g = new double[a.length];
        Arrays.fill(g, 1);
        return g;
    }
}
