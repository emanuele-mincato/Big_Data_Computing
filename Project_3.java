import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Project_3
{

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// MAIN PROGRAM
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: filepath k z L");
        }

        // ----- Initialize variables
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);
        long start, end; // variables for time measurements

        // ----- Set Spark Configuration
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("MR k-center with outliers");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // ----- Read points from file
        start = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(args[0], L)
                .map(x-> strToVector(x))
                .repartition(L)
                .cache();
        long N = inputPoints.count();
        end = System.currentTimeMillis();

        // ----- Pring input parameters
        System.out.println("File : " + filename);
        System.out.println("Number of points N = " + N);
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        System.out.println("Number of partitions L = " + L);
        System.out.println("Time to read from file: " + (end-start) + " ms");

        // ---- Solve the problem
        ArrayList<Vector> solution = MR_kCenterOutliers(inputPoints, k, z, L);

        // ---- Compute the value of the objective function
        start = System.currentTimeMillis();
        double objective = computeObjective(inputPoints, solution, z);
        end = System.currentTimeMillis();
        System.out.println("Objective function = " + objective);
        System.out.println("Time to compute objective function: " + (end-start) + " ms");

    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// AUXILIARY METHODS
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method strToVector: input reading
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method euclidean: distance function
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method MR_kCenterOutliers: MR algorithm for k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> MR_kCenterOutliers (JavaRDD<Vector> points, int k, int z, int L)
    {

        //------------- ROUND 1 ---------------------------
        // time measurament start for round 1
        long startOne = System.currentTimeMillis();
        JavaRDD<Tuple2<Vector,Long>> coreset = points.mapPartitions(x ->
        {
            ArrayList<Vector> partition = new ArrayList<>();
            while (x.hasNext()) partition.add(x.next());
            ArrayList<Vector> centers = kCenterFFT(partition, k+z+1);
            ArrayList<Long> weights = computeWeights(partition, centers);
            ArrayList<Tuple2<Vector,Long>> c_w = new ArrayList<>();
            for(int i =0; i < centers.size(); ++i)
            {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weights.get(i));
                c_w.add(i,entry);
            }
            return c_w.iterator();
        }); // END OF ROUND 1

        //------------- ROUND 2 ---------------------------

        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>((k+z)*L);
        elems.addAll(coreset.collect());
        // considering the lazy evalutation of Spark, we include in the time measurement for the Round 1 the inexpensive
        // operation coreset.collect() from the start of round 2, since here will also be executed all the operations
        // from the mapPartitions call in the first round. In this way we will get a slightly lower time measurement
        // for round 2 but a more accurate measurement for round 1.
        long endOne = System.currentTimeMillis(); // lazy evaluation
        // time measurement end for round 1

        // time measurement start for round 2
        // measuring the time for the sequential clustering algorithm on the weighted coreset only.
        long startTwo = System.currentTimeMillis();
        // extract the points and weights in 2 separate lists to get inputs for SeqWeightedOutliers()
        ArrayList<Vector> pointsT = new ArrayList<>();
        ArrayList<Long> weightsT = new ArrayList<>();
        for (Tuple2<Vector, Long> elem : elems) {
            pointsT.add(elem._1());
            weightsT.add(elem._2());
        }
        ArrayList<Vector> solution = SeqWeightedOutliers(pointsT, weightsT, k, z, 2);
        // END OF ROUND 2
        long endTwo = System.currentTimeMillis();
        // time measurement end for round 2

        System.out.println("Time Round 1: " + (endOne-startOne) + " ms");
        System.out.println("Time Round 2: " + (endTwo-startTwo) + " ms");

        return solution;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method kCenterFFT: Farthest-First Traversal
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> kCenterFFT (ArrayList<Vector> points, int k) {

        final int n = points.size();
        double[] minDistances = new double[n];
        Arrays.fill(minDistances, Double.POSITIVE_INFINITY);

        ArrayList<Vector> centers = new ArrayList<>(k);

        Vector lastCenter = points.get(0);
        centers.add(lastCenter);
        double radius =0;

        for (int iter=1; iter<k; iter++) {
            int maxIdx = 0;
            double maxDist = 0;

            for (int i = 0; i < n; i++) {
                double d = euclidean(points.get(i), lastCenter);
                if (d < minDistances[i]) {
                    minDistances[i] = d;
                }

                if (minDistances[i] > maxDist) {
                    maxDist = minDistances[i];
                    maxIdx = i;
                }
            }

            lastCenter = points.get(maxIdx);
            centers.add(lastCenter);
        }
        return centers;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeWeights: compute weights of coreset points
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Long> computeWeights(ArrayList<Vector> points, ArrayList<Vector> centers)
    {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for(int i = 0; i < points.size(); ++i)
        {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for(int j = 1; j < centers.size(); ++j)
            {
                if(euclidean(points.get(i),centers.get(j)) < tmp)
                {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            // System.out.println("Point = " + points.get(i) + " Center = " + centers.get(mycenter));
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method SeqWeightedOutliers: sequential k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

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
            // our rapresentation of Z (vector containing the uncovered point) is a simple vector of integers
            // of P-lenght, with every entry set to 0 at the start of the algorithm. The entry at position i
            // represents the state (uncovered = 0, covered = 1) of the i-th point in the vector P.
            // in this way we don't have to phisically remove entries from P and move them to another vector
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


// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeObjective: computes objective function
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double computeObjective (JavaRDD<Vector> points, ArrayList<Vector> centers, int z)
    {
        // MapReduce implementation:
        // FIRST ROUND:  mapPartitions: map from vector (single point) to it's minimum computed distance from the centers
        // SECOND ROUND: map empty -> descending sorting -> take z (outliers) + 1 (real objective) values

        // Round 1
        JavaRDD<Double> distances = points.mapPartitions(x -> {
            ArrayList<Double> values = new ArrayList<>();
            while(x.hasNext()) {
                double dist = Double.POSITIVE_INFINITY;
                Vector current = x.next();
                for (Vector center : centers) {
                    double distCenter = euclidean(current, center);
                    if (distCenter < dist) dist = distCenter;
                }
                values.add(dist);
            }
            return values.iterator();
        });

        // Round 2
        // retrieving the rdd partitioning (distances.partitions().size()) for the rdd sorting.
        ArrayList<Double> candidates = new ArrayList<>(distances.sortBy(x -> x, false, distances.partitions().size()).take(z+1));

        // return the z+1th element from the arraylist (objective function value)
        return candidates.get(z);

    }

}