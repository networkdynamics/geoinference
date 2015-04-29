/*
 * Copyright (c) 2015, David Jurgens
 *
 * All rights reserved. See LICENSE file for details
 */

package ca.mcgill.networkdynamics.geoinference.evaluation;

import java.io.*;
import java.util.*;
import java.util.zip.*;
import gnu.trove.map.*;
import gnu.trove.map.hash.*;
import gnu.trove.list.*;
import gnu.trove.list.array.*;
import gnu.trove.set.*;
import gnu.trove.set.hash.*;
import org.gavaghan.geodesy.*;
import org.apache.commons.math3.analysis.*;
import org.apache.commons.math3.analysis.integration.*;

import ca.mcgill.networkdynamics.geoinference.util.Files;
import ca.mcgill.networkdynamics.geoinference.util.Geometry;


/**
 * The implementation of the scoring program when evaluating the
 * cross-validation results of {@link
 * ca.mcgill.networkdynamics.geoinference.App}
 */
public class CrossValidationScorer {

    /**
     * Furthest distance on earth (km)
     */
    private static final int MAX_KM = 20039; 

    public static void main(String[] args) throws Exception {        

        if (args.length != 4) {
            System.out.println("java CVS predictions-dir/ " +
                               "cv-gold-dir/ results.txt error-sample.tsv");
            return;
        }

        File predDir = new File(args[0]);        
        File cvDir = new File(args[1]);

        TDoubleList errors = new TDoubleArrayList(10_000_000);
        TLongSet locatedUsers = new TLongHashSet(10_000_000);
        TLongSet allUsers = new TLongHashSet(10_000_000);
        TLongObjectMap<TDoubleList> userToErrors = 
            new TLongObjectHashMap<TDoubleList>();       

        TLongDoubleMap tweetIdToError = new TLongDoubleHashMap(10_000_000);        
        TLongObjectMap<double[]> idToPredLoc 
            = new TLongObjectHashMap<double[]>();


        int tweetsSeen = 0;
        int tweetsLocated = 0;

        BufferedReader cvBr = new BufferedReader(
            new FileReader(new File(cvDir, "folds.info.tsv")));
        for (String foldLine = null; (foldLine = cvBr.readLine()) != null; ) {
            String[] cols = foldLine.split("\t");
            String foldName = cols[0];

            System.out.printf("Scoring results for fold %s%n", foldName);
                      
            File foldPredictionsFile = 
                new File(predDir, foldName + ".results.tsv.gz");
                        
            File goldLocFile = new File(cvDir,
                foldName + ".gold-locations.tsv");
                       

            if (foldPredictionsFile.exists()) {
                BufferedReader br = Files.openGz(foldPredictionsFile);
                for (String line = null ; (line = br.readLine()) != null; ) {
                    String[] arr = line.split("\t");
                    long id = Long.parseLong(arr[0]);
                    idToPredLoc.put(id, new double[] { 
                            Double.parseDouble(arr[1]), 
                            Double.parseDouble(arr[2]) });
                }
                br.close();
            }

            System.out.printf("loaded predictions for %d tweets; " +
                              "scoring predictions%n", idToPredLoc.size());

            
            BufferedReader br = 
                new BufferedReader(new FileReader(goldLocFile));
            for (String line = null ; (line = br.readLine()) != null; ) {
                String[] arr = line.split("\t");
                long id = Long.parseLong(arr[0]);
                long userId = Long.parseLong(arr[1]);
                
                allUsers.add(userId);
                tweetsSeen++;
                
                double[] predLoc = idToPredLoc.get(id);
                if (predLoc == null)
                    continue;


                tweetsLocated++;
                locatedUsers.add(userId);
                
                double[] goldLoc = new double[] { 
                    Double.parseDouble(arr[2]), 
                    Double.parseDouble(arr[3]) };
                
                double dist = Geometry.getDistance(predLoc, goldLoc);
                errors.add(dist);
                tweetIdToError.put(id, dist);                

                TDoubleList userErrors = userToErrors.get(userId);
                if (userErrors == null) {
                    userErrors = new TDoubleArrayList();
                    userToErrors.put(userId, userErrors);
                }
                userErrors.add(dist);
                
            }
            br.close();            
        }
        
        errors.sort();
        System.out.println("Num errors to score: " + errors.size());
        
        double auc = 0;
        double userCoverage = 0;
        double tweetCoverage = tweetsLocated / (double)tweetsSeen;
        double medianMaxUserError = Double.NaN;
        double medianMedianUserError = Double.NaN;
        
        if (errors.size() > 0) {
            auc = computeAuc(errors);
            userCoverage = locatedUsers.size() / ((double)allUsers.size());
            TDoubleList maxUserErrors = new TDoubleArrayList(locatedUsers.size());
            TDoubleList medianUserErrors = new TDoubleArrayList(locatedUsers.size());
            for (TDoubleList userErrors : userToErrors.valueCollection()) {
                userErrors.sort();
                maxUserErrors.add(userErrors.get(userErrors.size() -1));
                medianUserErrors.add(userErrors.get(userErrors.size() / 2));
            }
            
            maxUserErrors.sort();
            medianMaxUserError = 
                maxUserErrors.get(maxUserErrors.size() / 2);
            
            medianUserErrors.sort();
            medianMedianUserError = 
                medianUserErrors.get(medianUserErrors.size() / 2);
            
            // Compute CDF
            int[] errorsPerKm = new int[MAX_KM];
            for (int i = 0; i < errors.size(); ++i) {
                int error = (int)(Math.round(errors.get(i)));
                errorsPerKm[error]++;
            }
            
            // The accumulated sum of errors per km
            int[] errorsBelowEachKm = new int[errorsPerKm.length];
            for (int i = 0; i < errorsBelowEachKm.length; ++i) {
                errorsBelowEachKm[i] = errorsPerKm[i];
                if (i > 0)
                    errorsBelowEachKm[i] += errorsBelowEachKm[i-1];
            }
            
            final double[] cdf = new double[errorsBelowEachKm.length];
            double dSize = errors.size(); // to avoid casting all the time
            for (int i = 0; i < cdf.length; ++i)
                cdf[i] = errorsBelowEachKm[i] / dSize;
        }
        
        PrintWriter pw = new PrintWriter(new File(args[2]));
        pw.println("AUC\t" + auc);
        pw.println("user coverage\t" + userCoverage);
        pw.println("tweet coverage\t" + tweetCoverage);
        pw.println("median-max error\t" + medianMaxUserError);
        pw.close();

        // Choose a random sampling of 10K tweets to pass on to the authors
        // here.        
        PrintWriter errorsPw = new PrintWriter(args[3]);
        TLongList idsWithErrors = new TLongArrayList(tweetIdToError.keySet());
        idsWithErrors.shuffle(new Random());
        // Choose the first 10K
        for (int i = 0, chosen = 0; i < idsWithErrors.size() && chosen < 10_000;
                ++i) {
                        
            long id = idsWithErrors.get(i);
            double[] prediction = idToPredLoc.get(id);
            double error = tweetIdToError.get(id);
            errorsPw.println(id + "\t" + error + "\t" + prediction[0]
                             + "\t" + prediction[1]);
            ++chosen;
        }
        errorsPw.close();
    }

    public static double computeAuc(TDoubleList errors) {

        double[] normalizedErrors = new double[errors.size()];

        int[] errorsPerKm = new int[MAX_KM];

        for (int i = 0; i < errors.size(); ++i) {
            int error = (int)(Math.round(errors.get(i)));
            errorsPerKm[error]++;
        }

        // The accumulated sum of errors per km
        int[] errorsBelowEachKm = new int[errorsPerKm.length];
        for (int i = 0; i < errorsBelowEachKm.length; ++i) {
            errorsBelowEachKm[i] = errorsPerKm[i];
            if (i > 0)
                errorsBelowEachKm[i] += errorsBelowEachKm[i-1];
        }

        final double[] cdf = new double[errorsBelowEachKm.length];
        double dSize = errors.size(); // to avoid casting all the time
        for (int i = 0; i < cdf.length; ++i)
            cdf[i] = errorsBelowEachKm[i] / dSize;

        final double maxLogKm = Math.log10(MAX_KM - 1);

        // At this point, the CDF is between [0, 20038], so we first need
        // log-scale the x-values and then to normalize it into [0, 1]
        UnivariateFunction logNormalizedScaledCdf = new UnivariateFunction() {
                public double value(double x) {
                    // First, unscale by the log(MAX_DIST) so the valus is just
                    // Math.log10(x)
                    double unscaled = x * maxLogKm;

                    // Second, invert the log transformation
                    double errorInKm = Math.pow(10, unscaled);

                    // Get the probability of having an error less than this
                    // amount
                    double prob = cdf[(int)(Math.round(errorInKm))];

                    // Now look up the CDF value for that error
                    return prob;
                }
            };
            
        TrapezoidIntegrator ti = new TrapezoidIntegrator();
        double auc = ti.integrate(10_000_000, logNormalizedScaledCdf, 0, 1);
        return auc;
    }
}
