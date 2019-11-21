package perceptron;

public interface ActivationFunction {
    double[] eval(double[] a);
    double[] grad(double[] a);
}
