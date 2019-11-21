package perceptron;

import java.util.Arrays;
import java.util.Random;

public class Perceptron {
    private ActivationFunction activationFunction;
    private LossFunction lossFunction;
    private double learningRate;
    private double[] w;
    private double b;

    // parameters stored from the most recent forward call
    private double[] x;
    private double a;
    private double y;

    public Perceptron(ActivationFunction activationFunction, LossFunction lossFunction,
                      double learningRate, int inputDim) {
        w = new double[inputDim];
        this.lossFunction = lossFunction;
        this.activationFunction = activationFunction;
        this.learningRate = learningRate;
        randomize();
    }

    private void randomize() {
        Random rand = new Random();
        for (int i = 0; i < w.length; ++i) {
            w[i] = 2 * rand.nextDouble() - 1;
        }
        b = 2 * rand.nextDouble() - 1;
    }

    public double forward(double[] x) {
        double a, y;
        a = b;
        for (int i = 0; i < w.length; ++i) {
            a += w[i] * x[i];
        }
        y = activationFunction.eval(a);
        this.x = Arrays.copyOf(x, x.length);
        this.a = a;
        this.y = y;
        return y;
    }

    /**
     * Perform backward propagation from the last forward call
     *
     * returns the value returned by the loss function
     */
    public double backward(double d) {
        double delta = lossFunction.grad(y, d);
        double dyda = activationFunction.grad(a);
        for (int i = 0; i < w.length; ++i) {
            w[i] -= learningRate * delta * dyda * x[i];
        }
        b -= learningRate * delta * dyda;
        return lossFunction.eval(y, d);
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
