package perceptron;

public class LinearActivation implements ActivationFunction {

    @Override
    public double eval(double a) {
        return a;
    }

    @Override
    public double grad(double a) {
        return 1;
    }
}
