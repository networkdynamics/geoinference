package ca.mcgill.networkdynamics.geoinference.slp;

import ca.mcgill.networkdynamics.geoinference.Dataset;
import ca.mcgill.networkdynamics.geoinference.GeoinferenceMethod;
import ca.mcgill.networkdynamics.geoinference.GeoinferenceModel;
import ca.mcgill.networkdynamics.geoinference.LatLon;

import ca.mcgill.networkdynamics.geoinference.util.Geometry;
import ca.mcgill.networkdynamics.geoinference.util.StaticGeoinferenceModel;

import gnu.trove.iterator.TLongIterator;
import gnu.trove.iterator.TLongObjectIterator;

import gnu.trove.list.TLongList;

import gnu.trove.list.array.TLongArrayList;

import gnu.trove.map.TLongObjectMap;

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
import java.util.List;
import java.util.Map;

import java.util.concurrent.atomic.AtomicLong;

import org.json.JSONObject;

import org.jgrapht.Graph;
import org.jgrapht.Graphs;

import org.jgrapht.graph.DefaultEdge;



public class SpatialLabelPropagation implements GeoinferenceMethod {

    public SpatialLabelPropagation() {

    }

    public GeoinferenceModel load(JSONObject settings, File modelDir) {
        File userLocFile = new File(modelDir, "user-to-location.tsv");
        if (userLocFile.exists()) {
            throw new IllegalStateException(
                "Expected to load saved SLP model from " + userLocFile
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
                "in a saved SLP model", ioe);
        }
    }

    public GeoinferenceModel train(JSONObject settings, Dataset dataset,
                                   File modelDir) {

        final int numIterations = settings.optInt("num_iterations", 4);
        final int numThreads = settings.optInt("num_threads", 10);
       
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


        Graph<Long,? extends DefaultEdge> socialGraph = 
            dataset.getMentionNetwork(false, false);

        // Ain't no one got time to use JGraphT.  Dump the contents to faster
        // primitive iteration.
        TLongObjectMap<long[]> userToNeighbors = 
            new TLongObjectHashMap<long[]>(socialGraph.vertexSet().size());
        for (Long userId : socialGraph.vertexSet()) {
            long id = userId;
            List<Long> incident = Graphs.neighborListOf(socialGraph, userId);
            long[] neighbors = new long[incident.size()];
            int i = 0;
            for (long n : incident)
                neighbors[i++] = n;
            userToNeighbors.put(id, neighbors);
        }

        final TLongSet users = userToNeighbors.keySet(); // for convenience

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
                                    userId, userToNeighbors,
                                    userToEstimatedLocation);
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

            System.out.printf("END Iter %d, processed %d/%d users, " +
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
            TLongObjectMap<long[]> userToNeighbors,
            TLongObjectMap<LatLon> userToEstimatedLocation) {
    
        // Get the user's social network
        long[] neighbors = userToNeighbors.get(userId);
                
        // For each of the users in the network, get their estimated
        // location, if any
        List<LatLon> locations = new ArrayList<LatLon>();
        for (long neighborId : neighbors) {
            LatLon estimated = userToEstimatedLocation.get(neighborId);
            if (estimated != null)
                locations.add(estimated);
        }
        
        return (locations.isEmpty())
            ? null
            : Geometry.getGeometricMedian(locations);
    }
}
