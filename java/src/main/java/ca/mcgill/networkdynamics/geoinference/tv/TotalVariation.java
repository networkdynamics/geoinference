/*
 * Copyright (c) 2015, David Jurgens
 *
 * All rights reserved. See LICENSE file for details
 */

package ca.mcgill.networkdynamics.geoinference.tv;

import ca.mcgill.networkdynamics.geoinference.Dataset;
import ca.mcgill.networkdynamics.geoinference.GeoinferenceMethod;
import ca.mcgill.networkdynamics.geoinference.GeoinferenceModel;
import ca.mcgill.networkdynamics.geoinference.LatLon;

import ca.mcgill.networkdynamics.geoinference.util.Geometry;
import ca.mcgill.networkdynamics.geoinference.util.StaticGeoinferenceModel;

import gnu.trove.iterator.TLongIterator;
import gnu.trove.iterator.TLongIntIterator;
import gnu.trove.iterator.TLongObjectIterator;

import gnu.trove.list.TIntList;
import gnu.trove.list.TLongList;

import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.array.TLongArrayList;

import gnu.trove.map.TLongIntMap;
import gnu.trove.map.TLongObjectMap;

import gnu.trove.map.hash.TLongIntHashMap;
import gnu.trove.map.hash.TLongObjectHashMap;

import gnu.trove.set.TLongSet;

import gnu.trove.set.hash.TLongHashSet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOError;
import java.io.IOException;
import java.io.PrintWriter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import java.util.concurrent.atomic.AtomicLong;

import org.json.JSONObject;

import org.jgrapht.Graph;
import org.jgrapht.Graphs;
import org.jgrapht.WeightedGraph;

import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.DefaultWeightedEdge;


/**
 * An implementation of the Total Variation algorithm of Compton, Jurgens, and
 * Allan (IEEE BigData, 2014).  
 */
public class TotalVariation implements GeoinferenceMethod {

    public static final double DEFAULT_MAX_LOCATION_DISPERSION_IN_KM = 100;

    public GeoinferenceModel load(JSONObject settings, File modelDir) {
        File userLocFile = new File(modelDir, "user-to-location.tsv");
        if (!userLocFile.exists()) {
            throw new IllegalStateException(
                "Expected to load saved TV model from " + userLocFile
                + " but file did not exist in " + modelDir);
        }
        try {
            TLongObjectMap<LatLon> userToLoc = new TLongObjectHashMap<LatLon>();
            BufferedReader br = new BufferedReader(new FileReader(userLocFile));
            for (String line = null; (line = br.readLine()) != null; ) {
                String[] arr = line.split("\t");
                long userId = Long.parseLong(arr[0]);
                double lat = Double.parseDouble(arr[1]);
                double lon = Double.parseDouble(arr[2]);
                userToLoc.put(userId, new LatLon(lat, lon));
            }
            br.close();
            return new StaticGeoinferenceModel(userToLoc);
        } catch (IOException ioe) {
            throw new IllegalStateException(
                "An error occurred when trying to read " +
                "in a saved TV model", ioe);
        }
    }

    public GeoinferenceModel train(JSONObject settings, Dataset dataset,
                                   File modelDir) {

        final int numIterations = settings.optInt("num_iterations", 4);
        final int numThreads = settings.optInt("num_threads", 10);
        final double maxDispersion = settings.optDouble("max_dispersion",
            DEFAULT_MAX_LOCATION_DISPERSION_IN_KM);
       
        // Load the locations that are already known for some users in the
        // dataset.  These serve has the seeds for the spatial label propagation.
        final TLongObjectMap<LatLon> userToLocation = 
            new TLongObjectHashMap<LatLon>();
        for (Map.Entry<Long,LatLon> e : dataset.getUserLocations())
            userToLocation.put(e.getKey(), e.getValue());

        // This map is where we keep all the new locations for the users.  For
        // users with gold standard locations, the estimated locations are
        // always the same as the gold standard locations.
        final TLongObjectMap<LatLon> userToEstimatedLocation = 
            new TLongObjectHashMap<LatLon>(userToLocation);

        // Get the weighted version of the network
        Graph<Long,DefaultEdge> socialGraph = 
            (Graph<Long,DefaultEdge>)dataset.getMentionNetwork(false, true);

        // Ain't no one got time to use JGraphT.  Dump the contents to faster
        // primitive iteration.
        final TLongObjectMap<TLongIntMap> weightedSocialNetwork =
            new TLongObjectHashMap<TLongIntMap>(socialGraph.vertexSet().size());
        for (Long userId : socialGraph.vertexSet()) {
            long id = userId;
            List<Long> incident = Graphs.neighborListOf(socialGraph, userId);
            TLongIntMap neighbors = new TLongIntHashMap(incident.size());
            for (long n : incident) {
                neighbors.put(n, (int)socialGraph.getEdgeWeight(
                                  socialGraph.getEdge(id, n)));
            }
            weightedSocialNetwork.put(id, neighbors);
        }

        final TLongSet users = weightedSocialNetwork.keySet(); // for convenience
        System.out.println("Generating user splits");

        final TLongList[] splits = new TLongList[numThreads];
        for (int i = 0; i < splits.length; ++i) {
            splits[i] = new TLongArrayList(users.size() / splits.length);
        }
        TLongIterator userIter = users.iterator();
        for (int i = 0; userIter.hasNext(); ++i) {
            splits[i % numThreads].add(userIter.next());
        }
        System.out.println("Finished generating user splits");
        
        for (int iteration = 0; iteration < numIterations; ++iteration) {
            final int iterNum_ = iteration;
            System.out.println("Beginning iteration " + iteration);

            // This map holds the *next* iteration's estimated locations so that
            // we're not both updating and estimating at the same time
            final TLongObjectMap<LatLon> userToNextEstimatedLocation = 
                new TLongObjectHashMap<LatLon>(users.size());
            final AtomicLong ctr_ = new AtomicLong();

            List<Thread> threads = new ArrayList<Thread>();
            for (int it = 0; it < numThreads; ++it) {
                final int id = it;
                Thread t = new Thread() {
                        public void run() {

                            TLongObjectMap<LatLon> localUser2estLoc = 
                                new TLongObjectHashMap<LatLon>(
                                    userToLocation.size());

                            TLongList split = splits[id];
                            TLongIterator userIter = split.iterator();
                            while (userIter.hasNext()) {
                                long userId = userIter.next();
                                
                                // Short-circuit if we already know where this
                                // user is located so that we always preserve
                                // the "hint" going forward
                                if (userToLocation.containsKey(userId)) {
                                    localUser2estLoc.put(
                                        userId, userToLocation.get(userId));
                                    continue;
                                }
                                
                                long ctr = ctr_.incrementAndGet();
                                if (ctr % 500000 == 0) {
                                    System.out.printf(
                                        "Thread %d, Iter %d, " +
                                        "processed %d/%d users, " +
                                        "%d have locations%n", id, iterNum_,
                                        ctr, users.size(),
                                        localUser2estLoc.size());
                                }
                            
                                LatLon userLoc = getUserLocation(
                                    userId, weightedSocialNetwork,
                                    userToEstimatedLocation,
                                    maxDispersion);
                                if (userLoc != null) {
                                    localUser2estLoc.put(userId, userLoc);
                                }
                            }
                            synchronized(userToNextEstimatedLocation) {
                                userToNextEstimatedLocation
                                    .putAll(localUser2estLoc);
                            }
                        }                    
                    
                    };
                threads.add(t);
                t.start();
            }

            try {
                for (Thread t : threads)
                    t.join();
            } catch (InterruptedException ie) {
                throw new IllegalStateException(
                    "An exception occurred while waiting for the SLP " +
                    "processing threads to finish", ie);
            }

            System.out.printf("END TV-Iter %d, processed %d/%d users, " +
                              "%d have locations%n", iteration,
                              ctr_.get(), users.size(),
                              userToNextEstimatedLocation.size());


            // Replace all the old location estimates with what we estimated
            // from this round.
            userToEstimatedLocation.putAll(userToNextEstimatedLocation);
        }

        
        if (modelDir != null) {
            File userLocFile = new File(modelDir, "user-to-location.tsv");
            try {
                PrintWriter pw = new PrintWriter(userLocFile);
                TLongObjectIterator<LatLon> iter = 
                    userToEstimatedLocation.iterator();
                while (iter.hasNext()) {
                    iter.advance();
                    long userId = iter.key();
                    LatLon loc = iter.value();
                    pw.println(userId + "\t" + loc.lat + "\t" + loc.lon);
                }
                pw.close();
            }
            catch (IOException ioe) {
                throw new IllegalStateException(
                    "Unable to save fully trained model file in " 
                    + userLocFile, ioe);
            }
        }

        return new StaticGeoinferenceModel(userToEstimatedLocation);
    }

    private static LatLon getUserLocation(long userId,
            TLongObjectMap<TLongIntMap> weightedSocialNetwork,
            TLongObjectMap<LatLon> userToEstimatedLocation,
            double maxDispersion) {
    
        // Get the user's social network
        TLongIntMap neighborToWeights = weightedSocialNetwork.get(userId);
                
        // For each of the users in the network, get their estimated
        // location, if any
        List<LatLon> locations = new ArrayList<LatLon>();
        TIntList weights = new TIntArrayList();
        TLongIntIterator neighborIter = neighborToWeights.iterator();
        while (neighborIter.hasNext()) {
            neighborIter.advance();
            long neighborId = neighborIter.key();
            LatLon estimated = userToEstimatedLocation.get(neighborId);
            if (estimated != null) {
                locations.add(estimated);
                weights.add(neighborIter.value());
            }
        }
        
        return (locations.isEmpty())
            ? null
            : getGeometricMedianWithDispersionContraints(
                  locations, weights, maxDispersion);
    }

    /**
     * Returns the geometric median of the list of locations using the maximum
     * dispersion to constrain the output to only those locations with less
     * than the specified dispersion.
     *
     * @return the geometric median of the coordinates <i>or<i> {@code null} if
     *         the median absolute deviation of the coordinates is greater than
     *         {@code maxDispersion}
     */
    private static LatLon getGeometricMedianWithDispersionContraints(
            List<LatLon> coordinates, TIntList weights, 
            double maxDispersion) {
        // The geometric median is a degenerate case for one and two
        // coordinates, so we return an arbitrary coordinate
        if (coordinates.size() < 3) {
            // In the event that there are two locations, check that they aren't
            // too dispersed.
            if (coordinates.size() == 2) {
                double dist = 
                    Geometry.getDistance(coordinates.get(0), coordinates.get(1));
                if (dist > maxDispersion)
                    return null;
            }
            return coordinates.get((int)(Math.random() * coordinates.size()));
        }

        double argMinDistance = Double.MAX_VALUE;
        LatLon argMin = null;

        double argMaxDistance = 0;

        int n = coordinates.size();        
        double[] distancesFromMedian = null;


        next_point:
        for (int i = 0; i < n; ++i) {
            LatLon p1 = coordinates.get(i);
            double distSum = 0;
            double[] distances = new double[n];
            for (int j = 0; j < n; ++j) {
                LatLon p2 = coordinates.get(j);
                // Skip self-comparison
                if (i == j)
                    continue;
                double dist = Geometry.getDistance(p1, p2);
                distSum += weights.get(j) * dist;
                distances[j] = dist;
            }

            // distances[i] = distSum;
            if (distSum < argMinDistance) {
                argMinDistance = distSum;
                argMin = p1;
                distancesFromMedian = distances;
            }
        }

        // Get the median distance as a measure of geographic dispersion
        Arrays.sort(distancesFromMedian);
        double medianAbsoluteDeviation =
            distancesFromMedian[distancesFromMedian.length/2];
        
        // Don't return a location for this user if the dispersion is above the
        // threshold
        if (medianAbsoluteDeviation > maxDispersion)
            return null;

        return argMin;
    }

}
