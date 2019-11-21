import perceptron.LinearActivation;
import perceptron.Perceptron;
import perceptron.SquaredLoss;

import java.util.Random;

public class TestLinearRegression {

    static private Random rand = new Random();

    private static double[][] generateRandomX(int n, int k) {
        double[][] x = new double[n][k];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                x[i][j] = rand.nextDouble();
            }
        }
        return x;
    }

    private static double[] generateTargets(double[][] x) {
        double[] d = new double[x.length];
        for (int n = 0; n < x.length; ++n) {
            d[n] = generateTarget(x[n]);
        }
        return d;
    }

    private static double generateTarget(double[] x) {
        double d = 1;
        for (int i = 0; i < x.length; ++i) {
            d += (i + 1) * x[i];
        }
        d += 0.1 * rand.nextGaussian();
        return d;
    }

    private static void testOnSet(Perceptron perceptron, double x[][], double[] d) {
        double[] y = perceptron.forward(x);
        for (int n = 0; n < y.length; ++n) {
            System.out.println("target: " + String.format("%.2f", d[n]) + "\tprediction: " + String.format("%.2f", y[n]));
        }
    }

    public static void main(String[] args) {
        int samples = 100;
        int features = 5;
        double learningRate = 0.01;
        Perceptron perceptron = new Perceptron(new LinearActivation(), new SquaredLoss(), features);
        double[][] x = generateRandomX(samples, features);
        double[] d = generateTargets(x);

        System.out.println(perceptron);
        perceptron.train(x, d, learningRate, 100000);
        System.out.println(perceptron);

        double[][] testx = generateRandomX(10, features);
        double[] testd = generateTargets(testx);
        testOnSet(perceptron, testx, testd);
    }
}
