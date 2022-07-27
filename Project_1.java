import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;

public class Project_1 {


    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: num_partitions = K, n_popularity = H, Country = S, <path_to_file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("G065HW");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");


        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read number of products with highest Popularity to print
        int H = Integer.parseInt(args[1]);

        // Read the Country parameter
        String S = new String(args[2]);

        // POINT 1
        // Read input file and subdivide it into K random partitions
        JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        Random randomGenerator = new Random();

        // global lists declarations for output
        List<Tuple2<Integer, String>> topHPopularity = null;
        List<Tuple2<String, Integer>> popularity1List = null;
        List<Tuple2<String, Integer>> popularity2List = null;


        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // RDD COMPUTATION
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // POINT 2
        JavaPairRDD<String, Integer> productCustomer;
        productCustomer = rawData
                .flatMapToPair((element) -> {
                    // MAP PHASE (R1) -> maps the strings to pairs where a tuple (productid, CustomerID)
                    // is the key and 1 the fixed value, filtering per input requisites, so strings entry that doesn't
                    // conform to them won't result in a tuple (pair).
                    String[] tokens = element.split(",");
                    int k = 1; // single element
                    ArrayList<Tuple2<Tuple2<String, Integer>, Integer>> pairs = new ArrayList<>();
                    // token 7 is country, token 1 is product id, token 6 is CustomerID, token 3 is quantity
                    if (S.equals("all") && Integer.parseInt(tokens[3]) > 0) {
                        pairs.add(new Tuple2<>(new Tuple2<>(tokens[1], Integer.parseInt(tokens[6])), k));
                    } else if (S.equals(tokens[7]) && Integer.parseInt(tokens[3]) > 0) {
                        pairs.add(new Tuple2<>(new Tuple2<>(tokens[1], Integer.parseInt(tokens[6])), k));
                    }
                    return pairs.iterator();
                })
                // By grouping by key we are able to delete all duplicates with same customerid and productids
                .groupByKey()
                // REDUCE PHASE (R1) -> removes 1s and just keeps the CustomerID as value
                .flatMapToPair((element) -> {
                    ArrayList<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    pairs.add(new Tuple2<String, Integer>(element._1()._1(), element._1()._2()));
                    return pairs.iterator();
                });

        /*
        // check if step 2 is correct with sample_50
        System.out.println("Output of point 2");
        productCustomer.foreach(data -> {
            System.out.println("ProductID="+data._1() + " CustomerID=" + data._2());
        });
        */

        // POINT 3
        JavaPairRDD<String, Integer> productPopularity1;
        productPopularity1 = productCustomer
                // MAP PHASE (R1) -> empty since we already have pairs (done in point 2 with flatmaptopair), we will partition them in the first reduce step
                .mapPartitionsToPair((element) -> {
                    // REDUCE PHASE (R1) -> partition of tuples ProductID CustomerID (using default K parameter defined at the first RDD),
                    // we count the number of productids keys inside each partition using an hashmap.
                    HashMap<String, Integer> counts = new HashMap<>();
                    while (element.hasNext()) {
                        Tuple2<String, Integer> tuple = element.next();
                        counts.put(tuple._1(), 1 + counts.getOrDefault(tuple._1(), 0)); // we don't care about customer ids, just summing 1s
                    }
                    ArrayList<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Integer> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue().intValue()));
                    }
                    return pairs.iterator();
                })
                // MAP PHASE (R2) -> empty
                // By grouping by key we are able to unite all partial popularity values calculated in different partitions for the same keys
                .groupByKey()
                // REDUCE PHASE (R2) -> sum all the integer values associated with a ProductID key to get final popularity values
                .mapValues((it) -> {
                    int sum = 0;
                    for (int c : it) {
                        sum += c;
                    }
                    return sum;
                });

        /*
        // check if step 3 is correct with sample_50
        System.out.println("Output of point 3");
        productPopularity1.foreach(data -> {
            System.out.println("ProductID="+data._1() + " Popularity=" + data._2());
        });
        */

        // POINT 4
        JavaPairRDD<String, Integer> productPopularity2;
        productPopularity2 = productCustomer
                // MAP PHASE (R1) since we don't have partitions we do it in one round
                // by first mapping each (productid, customerid) to (productid, 1)
                .mapToPair((element) -> {
                    String token = element._1();
                    return new Tuple2<>(token, 1);
                })
                // REDUCE PHASE (R1) then we simply calculate the popularity value for each unique key
                .reduceByKey((x, y) -> x + y);

        /*
        // check if step 4 is correct with sample_50
        System.out.println("Output of point 4");
        productPopularity2.foreach(data -> {
            System.out.println("ProductID="+data._1() + " Popularity=" + data._2());
        });
         */

        // POINT 5 AND 6
        if (H > 0) {
            // In a new rdd we map the values of the pairs of productPopularity2 in a way to have the popularity value as
            // key of the pairs, using sortByKey to efficiently sort the pairs in a descending order
            JavaPairRDD<Integer, String> swappedPairs;
            swappedPairs = productPopularity2
                    .mapToPair((document) -> {
                        Tuple2<String, Integer> token = document;
                        return new Tuple2<>(token._2(), token._1());
                    }).sortByKey(false);
            // Saving the top H popular products in a list
            topHPopularity = swappedPairs.take(H);
        }
        else if (H == 0)  {
            // collecting in 2 lists the ordered by ProductID values of the 2 popularity RDDs
            popularity1List = productPopularity1.sortByKey(true).collect();
            popularity2List = productPopularity2.sortByKey(true).collect();
        }
        else {
            System.out.println("H must be an Integer >= 0");
        }


        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // OUTPUTS
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Printing number of rows in the original dataset RDD
        System.out.println("Number of rows = " + rawData.count());

        // Printing the number of unique product customer pairs from the original dataset
        System.out.println("Product-Customer Pairs = " + productCustomer.count());

        // Desired outputs based on the value of the H input parameter:
        if (H > 0) {
            System.out.println("Top " + H + " Products and their Popularities");
            topHPopularity.forEach( (c) -> System.out.printf("Product " + c._2() + " Popularity " + c._1() + "; "));
            System.out.println("");
        }
        else if (H == 0)  {
            System.out.println("productPopularity1:");
            popularity1List.forEach( (c) -> System.out.printf("Product: " + c._1() + " Popularity: " + c._2() + "; "));
            System.out.println("");
            System.out.println("productPopularity2:");
            popularity2List.forEach( (c) -> System.out.printf("Product: " + c._1() + " Popularity: " + c._2() + "; "));
            System.out.println("");
        }
        else {
            // no print if H < 0 here, message already printed before
        }

    }
}

