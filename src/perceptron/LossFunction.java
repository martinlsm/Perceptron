package perceptron;

public interface LossFunction {
    double eval(double[] y, double[] d);
    double[] grad(double[] y, double[] d);
}
