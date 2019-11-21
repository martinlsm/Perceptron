package perceptron;

import java.util.Arrays;
import java.util.Random;

public class Perceptron {
    private ActivationFunction activationFunction;
    private LossFunction lossFunction;
    private double[] w;
    private double b;

    // parameters from the most recent forward call
    double[][] x;
    double[] a;
    double[] y;

    public Perceptron(ActivationFunction activationFunction, LossFunction lossFunction, int inputDim) {
        w = new double[inputDim];
        this.lossFunction = lossFunction;
        this.activationFunction = activationFunction;
        randomize();
    }

    private void randomize() {
        Random rand = new Random();
        for (int i = 0; i < w.length; ++i) {
            w[i] = 2 * rand.nextDouble() - 1;
        }
        b = 2 * rand.nextDouble() - 1;
    }

    public double[] forward(double[][] x) {
        int N = x.length;
        double[] a = new double[N];
        double[] y;

        Arrays.fill(a, b);
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < w.length; ++k) {
                a[n] += w[k] * x[n][k];
            }
        }
        y = activationFunction.eval(a);
        this.x = x;
        this.a = a;
        this.y = y;
        return y;
    }

    /**
     * Perform backward propagation from the last forward call
     *
     * returns the value returned by the loss function
     */
    public double backward(double[] d, double learningRate) {
        final int N = d.length;
        double[] delta = lossFunction.grad(y, d);
        double[] dyda = activationFunction.grad(a);
        for (int k = 0; k < w.length; ++k) {
            double dLdwk = 0;
            for (int n = 0; n < N; ++n) {
                dLdwk += delta[n] * dyda[n] * x[n][k];
            }
            w[k] -= learningRate * dLdwk;
        }
        double dLdb = 0;
        for (int n = 0; n < N; ++n) {
            dLdb += delta[n] * dyda[n];
        }
        b -= learningRate * dLdb;
        return lossFunction.eval(y, d);
    }

    public double[] train(double[][] x, double[] d, double learningRate, int epochs) {
        double[] loss = new double[epochs];
        for (int e = 0; e < epochs; ++e) {
            double[] y = forward(x);
            loss[e] = backward(d, learningRate);
        }
        return loss;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Perceptron:\n");
        sb.append("\tw = {");
        for (int i = 0; i < w.length; ++i) {
            if (i == w.length - 1) {
                sb.append(String.format("%.3f", w[i]));
            } else {
                sb.append(String.format("%.3f", w[i])).append(", ");
            }
        }
        sb.append("}\n\tb = ").append(String.format("%.3f", b)).append("\n");
        return sb.toString();
    }
}
