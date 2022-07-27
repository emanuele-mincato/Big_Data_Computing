import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class Project_2 {

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Input reading methods
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }


    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // WRITE CLUSTERING METHODS
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    // Weighted variant of kcenterOUT
    public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, Integer k, Integer z,float alpha) {
        // we consider k <= P.size()
        // we consider z <= P.size()

        // P and W have the same length and the weight at position i in W corresponds to the weight of the point
        // at position i in P

        int points_number = P.size();

        // Compute full distance matrix between points in the pointset P
        double[][] distances = new double[points_number][points_number];
        for (int i = 0; i < points_number; i++) {
            for (int j = i; j < points_number; j++) {
                distances[i][j] = Math.sqrt(Vectors.sqdist(P.get(i), P.get(j)));
                distances[j][i] = distances[i][j];
            }
        }

        // calculate r = r_min, the initial guess for ball radius
        double r = Double.MAX_VALUE;
        for ( int i = 0; i < (k + z + 1); i++) {
            for ( int j = i+1; j < (k + z + 1); j++) {
                // start from i+1 to avoid the zero distance between a point and itself
                if (distances[i][j] < r) { r = distances[i][j];}
            }
        }
        r = r / 2;

        // print first radius guess
        System.out.println("Initial guess = " + r);

        // guess counter
        int numberOfGuesses = 1;

        // start of the algorithm
        while(true) {
            // our representation of Z (vector containing the uncovered point) is a simple vector of integers
            // of P-length, with every entry set to 0 at the start of the algorithm. The entry at position i
            // represents the state (uncovered = 0, covered = 1) of the i-th point in the vector P.
            // in this way we don't have to manually remove entries from P and move them to another vector
            // but we can simply check when looking at the vector P if a point is covered or uncovered with
            // an if condition on the same index in vector Z.
            ArrayList<Integer> Z = new ArrayList<Integer>(Collections.nCopies(points_number, 0));
            ArrayList<Vector> S = new ArrayList<Vector>();
            // calculate the initial full weight of the uncovered point by summing all weights
            long Wz = W.stream().mapToLong(Long::longValue).sum();
            while ( (S.size() < k) && (Wz > 0) ) {
                double max = 0;
                int new_center = 0;
                for (int i = 0; i < points_number; i++) {
                    // calculate Z-ball (uncovered points) weight for point x
                    double ball_weight = 0;
                    for (int j = 0; j < points_number; j++) {
                        if ((Z.get(j) == 0) && (distances[i][j] <= (1+2*alpha)*r)) {
                            ball_weight = ball_weight + W.get(j);
                        }
                    }
                    // if the ball for x has max weight between scouted points then select x as new center
                    if (ball_weight > max ) {
                        max = ball_weight;
                        new_center = i;
                    }
                }
                // add the new center point to S
                S.add(P.get(new_center));
                // now we remove the covered point by the new center found and respective weights
                for (int i = 0; i < points_number; i++) {
                    if ((Z.get(i) == 0) && (distances[new_center][i] <= (3+4*alpha)*r)) {
                        // set the point as covered in Z (remove it from P in practice by setting the value to 1 in Z)
                        Z.set(i, 1);
                        // remove it's weight from total weight of uncovered points
                        Wz = Wz - W.get(i);

                    }
                }

            }

            // check if we can complete the algorithm for the case when the uncovered points have combined weights < z
            // condition always met at some point assuring the end of the algorithm in inputs are correct
            if (Wz <= z) {

                // print final radius guess
                System.out.println("Final guess = " + r);
                // print number of guesses
                System.out.println("Number of guesses = " + numberOfGuesses);

                return S;
            }
            else {
                // try a new guess of radius
                r = r * 2;

                // update number of guesses
                numberOfGuesses = numberOfGuesses +1;
            }

        }

    }

    // Method for computing the objective function given the centers (not considering weights)
    public static double ComputeObjective(ArrayList<Vector> P, ArrayList<Vector> S, Integer z){
        ArrayList<Double> distances = new ArrayList<Double>();
        int c = 0;
        // compute nearest distances for all k centers in S with all points in P (included itself since distance will be 0)
        for (int i = 0; i < P.size(); i++) {
            // for each point we compute the distance from each center, saving it in a vector "nearest"
            ArrayList<Double> nearest = new ArrayList<Double>();
            for (int j = 0; j < S.size(); j++) {
                Double d = Math.sqrt(Vectors.sqdist(S.get(j), P.get(i)));
                nearest.add(d);
            }
            // select the closest center to the point i and add it's distance from it as a candidate for objective value
            double closestCenterDistance = Collections.min(nearest);
            distances.add(closestCenterDistance);
        }
        // sort the arraylist in descending order to find longest distance from a center
        Collections.sort(distances, Collections.reverseOrder());

        // remove z largest distances to not consider outliers
        for (int i = 0; i < z; i++) {
            // remove first z elements of the arraylist, remove shift the array to the left so index to remove is always 0
            distances.remove(0);
        }

        // return objective value (first entry of sorted array)
        return distances.get(0);
    }




    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: Path to file, Number of centers = K, Number of outliers = z
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING FOR MAIN
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read input file
        String path = args[0];

        // Read number of centers
        int K = Integer.parseInt(args[1]);

        // Read number of allowed outliers
        int Z = Integer.parseInt(args[2]);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Reading file in path to produce ArrayList of vectors (points of n dimensions)
        ArrayList<Vector>  inputPoints = readVectorsSeq(path);

        // Create ArrayList of weights (all initialized with ones for this homework)
        ArrayList<Long> weights = new ArrayList<Long>(Collections.nCopies(inputPoints.size(), 1L));

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // EXECUTION AND OUTPUTS
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // input size
        System.out.println("input size n = " + inputPoints.size());

        // number of centers k
        System.out.println("Number of centers k = " + K);

        // number of outliers z
        System.out.println("Number of outliers z = " + Z);

        // run SeqWeightedOutliers, it prints initial r guess, final r guess and number of r guesses. Returns centers.
        // also count the time to run
        long startTime = System.currentTimeMillis();
        ArrayList<Vector> solution = SeqWeightedOutliers(inputPoints, weights, K, Z, 0);
        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;

        // run ComputeObjective to get objective function value
        double objective = ComputeObjective(inputPoints, solution, Z);
        System.out.println("Objective function = " + objective);

        // print execution time of SeqWeightedOutliers
        System.out.println("Time of SeqWeightedOutliers = " + elapsedTime);




    }


}