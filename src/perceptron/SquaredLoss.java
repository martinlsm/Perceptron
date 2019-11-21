package perceptron;

public class SquaredLoss implements LossFunction {
    @Override
    public double eval(double y, double d) {
        return 1/2.0 * (y - d) * (y - d);
    }

    @Override
    public double grad(double y, double d) {
        return y - d;
    }
}
