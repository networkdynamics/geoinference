/*
 * Copyright (c) 2015, David Jurgens
 *
 * All rights reserved. See LICENSE file for details
 */

package ca.mcgill.networkdynamics.geoinference;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;
import java.io.IOError;
import java.io.IOException;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

import org.jgrapht.Graph;
import org.jgrapht.WeightedGraph;

import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleGraph;
import org.jgrapht.graph.SimpleDirectedGraph;
import org.jgrapht.graph.SimpleDirectedWeightedGraph;
import org.jgrapht.graph.SimpleWeightedGraph;

import gnu.trove.set.TLongSet;

import gnu.trove.set.hash.TLongHashSet;

import ca.mcgill.networkdynamics.geoinference.util.Files;


/**
 * The abstract base class for datasets that handles most of the common
 * functions so that subclasses only need to specify how each user's tweets are
 * loaded.
 */
public abstract class AbstractDataset implements Dataset {

    public static final String DEFAULT_LOCATION_TYPE = "geo-median";

    protected final File datasetDir;

    protected final TLongSet usersWithExcludedLocations;

    protected final String locationType;

    public AbstractDataset(File datasetDir) {
        this(datasetDir, DEFAULT_LOCATION_TYPE, new TLongHashSet());
    }


    public AbstractDataset(File datasetDir, String locationType) {
        this(datasetDir, locationType, new TLongHashSet());
    }


    public AbstractDataset(File datasetDir,
                           TLongSet usersWithExcludedLocations) {
        this(datasetDir, DEFAULT_LOCATION_TYPE, usersWithExcludedLocations);
    }

    public AbstractDataset(File datasetDir, 
                           String locationType,
                           TLongSet usersWithExcludedLocations) {
        if (datasetDir == null)
            throw new NullPointerException("datasetDir cannot be null");
        if (locationType == null)
            throw new NullPointerException("locationType cannot be null");
        if (usersWithExcludedLocations == null) {
            throw new NullPointerException(
                "usersWithExcludedLocations cannot be null");
        }

        this.datasetDir = datasetDir;
        this.locationType = locationType;
        this.usersWithExcludedLocations = usersWithExcludedLocations;
    }

    /**
     * {@inheritDoc}
     */
    public Iterable<JSONObject> getPosts() {
        return new Posts();
    }

    /**
     * {@inheritDoc}
     */
    public Iterable<Map.Entry<Long,LatLon>> getUserLocations() {
        return new UserLocations(locationType);
    }

    /**
     * {@inheritDoc}
     */
    public Graph<Long,? extends DefaultEdge> getMentionNetwork(
        boolean isDirected, boolean isWeighted) {

        // Split the load behavior at the weighted condition, since we need to
        // use a different method to add edge weights regardless of directed or not
        if (isWeighted) {
            WeightedGraph<Long,DefaultWeightedEdge> g = (isDirected)
                ? new SimpleDirectedWeightedGraph<Long,DefaultWeightedEdge>(
                          DefaultWeightedEdge.class)
                : new SimpleWeightedGraph<Long,DefaultWeightedEdge>(
                          DefaultWeightedEdge.class);

            try {
                BufferedReader br = (isDirected)
                    ? new BufferedReader(new FileReader(
                          new File(datasetDir, 
                                 "bi_mention_network.directed.weighted.elist")))
                    : new BufferedReader(new FileReader(
                          new File(datasetDir, 
                                   "bi_mention_network.weighted.elist")));
                for (String line = null; (line = br.readLine()) != null; ) {
                    String[] arr = line.split("\\s+");
                    if (arr.length != 3)
                        continue;
                    Long fromUserId = Long.valueOf(arr[0]);
                    Long toUserId = Long.valueOf(arr[1]);
                    double weight = Double.parseDouble(arr[2]);
                    g.addVertex(fromUserId);
                    g.addVertex(toUserId);
                    DefaultWeightedEdge e = g.addEdge(fromUserId, toUserId);
                    g.setEdgeWeight(e, weight);
                }              
                br.close();
            } catch (IOException ioe) {
                throw new IOError(ioe);
            }        
            return g;
        }
        else {
            Graph<Long,? extends DefaultEdge> g = (isDirected)
                ? new SimpleDirectedGraph<Long,DefaultEdge>(DefaultEdge.class)
                : new SimpleGraph<Long,DefaultEdge>(DefaultEdge.class);

            try {
                BufferedReader br = (isDirected)
                    ? new BufferedReader(new FileReader(
                          new File(datasetDir, 
                                   "bi_mention_network.directed.elist")))
                    : new BufferedReader(new FileReader(
                          new File(datasetDir, "bi_mention_network.elist")));
                for (String line = null; (line = br.readLine()) != null; ) {
                    String[] arr = line.split("\\s+");
                    if (arr.length != 2)
                        continue;
                    Long fromUserId = Long.valueOf(arr[0]);
                    Long toUserId = Long.valueOf(arr[1]);
                    g.addVertex(fromUserId);
                    g.addVertex(toUserId);
                    g.addEdge(fromUserId, toUserId);
                }
                br.close();                
            } catch (IOException ioe) {
                throw new IOError(ioe);
            }
            return g;
        }
    }

    /**
     * An {@code Iterable} class that wraps the per-user iterator and simply
     * returns a user's posts one at a time.
     */
    class Posts implements Iterable<JSONObject> {
        
        public Posts() {

        }

        public Iterator<JSONObject> iterator() {
            return new PostIterator();
        }

        class PostIterator implements Iterator<JSONObject> {

            private Map.Entry<Long,List<JSONObject>> nextEntry;
            private int nextIndex = 0;

            private JSONObject next;

            private Iterator<Map.Entry<Long,List<JSONObject>>> backingIter;

            public PostIterator() {
                backingIter = getPostsByUser().iterator();
                advance();
            }

            private void advance() {
                next = null;
                if (nextIndex < 0)
                    return;
                while (nextEntry == null 
                           || nextEntry.getValue().size() <= nextIndex) {
                    if (backingIter.hasNext()) {
                        nextEntry = backingIter.next();
                        nextIndex = 0;
                    }
                    else {
                        nextEntry = null;
                        nextIndex = -1;
                        return;
                    }
                }
                
                next = nextEntry.getValue().get(nextIndex);
                nextIndex++;
            }

            public boolean hasNext() {
                return next != null;
            }

            public JSONObject next() {
                if (next == null)
                    throw new NoSuchElementException();
                JSONObject n = next;
                advance();
                return n;
            }
            
            public void remove() {
                throw new UnsupportedOperationException();
            }
        }
    }

    /**
     * An iterator over the user's precomputed locations.
     */
    class UserLocations implements Iterable<Map.Entry<Long,LatLon>> {

        private final String locationType;

        public UserLocations(String locationType) { 
            this.locationType = locationType;
        }
            
        public Iterator<Map.Entry<Long,LatLon>> iterator() {
            return new UserLocationsIterator();
        }

        class UserLocationsIterator implements Iterator<Map.Entry<Long,LatLon>> {

            private BufferedReader userLocFileReader;

            private Map.Entry<Long,LatLon> next;

            public UserLocationsIterator() {
                try {
                    userLocFileReader = 
                        Files.openGz(new File(datasetDir,
                          "users.home-locations." + locationType + ".tsv.gz"));
                } catch (IOException ioe) {
                    throw new IOError(ioe);
                }
                advance();
            }
        
            private void advance() {
                next = null;
                String line = null;
                try {
                    while (true) {
                        line = userLocFileReader.readLine();
                        if (line == null) {
                            userLocFileReader.close();
                            return;
                        }
                        String[] arr = line.split("\t");
                        Long userId = Long.valueOf(arr[0]);
                        if (usersWithExcludedLocations.contains(userId))
                            continue;
                        double lat = Double.parseDouble(arr[1]);
                        double lon = Double.parseDouble(arr[2]);
                        next = new AbstractMap.SimpleImmutableEntry<Long,LatLon>(
                            userId, new LatLon(lat, lon));                
                        break;
                    }
                } catch (IOException ioe) {
                    throw new IOError(ioe);
                }
            }

            public boolean hasNext() {
                return next != null;
            }

            public Map.Entry<Long,LatLon> next() {
                if (next == null)
                    throw new NoSuchElementException();
                Map.Entry<Long,LatLon> n = next;
                advance();
                return n;
            }

            public void remove() {
                throw new UnsupportedOperationException();
            }          
        }
    }   
}
