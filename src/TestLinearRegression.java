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
        for (int n = 0; n < d.length; ++n) {
            double y = perceptron.forward(x[n]);
            System.out.println("target: " + String.format("%.3f", d[n]) + "\tprediction: " + String.format("%.3f", y));
        }
    }

    public static void main(String[] args) {
        int samples = 100;
        int features = 5;
        Perceptron perceptron = new Perceptron(new LinearActivation(),
                new SquaredLoss(), 0.01, features);
        double[][] x = generateRandomX(samples, features);
        double[] d = generateTargets(x);
        System.out.println(perceptron);
        for (int i = 0; i < 1000; ++i) {
            double loss = 0;
            for (int n = 0; n < samples; ++n) {
                perceptron.forward(x[n]);
                loss += perceptron.backward(d[n]);
            }
            loss /= samples;
            System.out.println("loss: " + String.format("%.3f", loss) + " (iteration " + i + ")");
        }
        System.out.println(perceptron);

        double[][] testx = generateRandomX(10, features);
        double[] testd = generateTargets(testx);
        testOnSet(perceptron, testx, testd);
    }
}
