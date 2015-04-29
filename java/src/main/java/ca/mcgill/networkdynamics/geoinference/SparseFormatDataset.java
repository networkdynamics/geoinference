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
 * The class for accessing Datasets stored in sparse TSV format.
 */
public class SparseFormatDataset extends AbstractDataset {

    public SparseFormatDataset(File datasetDir) {
        this(datasetDir, DEFAULT_LOCATION_TYPE, new TLongHashSet());
    }


    public SparseFormatDataset(File datasetDir, String locationType) {
        this(datasetDir, locationType, new TLongHashSet());
    }


    public SparseFormatDataset(File datasetDir,
                               TLongSet usersWithExcludedLocations) {
        this(datasetDir, DEFAULT_LOCATION_TYPE, usersWithExcludedLocations);
    }

    public SparseFormatDataset(File datasetDir, 
                               String locationType,
                               TLongSet usersWithExcludedLocations) {
        super(datasetDir, locationType, usersWithExcludedLocations);
    }

    /**
     * {@inheritDoc}
     */
    public Iterable<Map.Entry<Long,List<JSONObject>>> getPostsByUser() {
        return new UserPosts();
    }

    /**
     * {@inheritDoc}
     */
    public Iterable<Map.Entry<Long,LatLon>> getUserLocations() {
        return new UserLocations(locationType);
    }

    /**
     * The {@code Iterable} class that returns each user's posts.
     */
    class UserPosts implements Iterable<Map.Entry<Long,List<JSONObject>>> {


        public UserPosts() { }
            
        public Iterator<Map.Entry<Long,List<JSONObject>>> iterator() {
            return new UserPostsIterator();
        }

        class UserPostsIterator implements Iterator<Map.Entry<Long,List<JSONObject>>> {

            private BufferedReader usersFileReader;

            private Map.Entry<Long,List<JSONObject>> next;

            public UserPostsIterator() {
                try {
                    usersFileReader = 
                        Files.openGz(new File(datasetDir, "users.tsv.gz"));
                } catch (IOException ioe) {
                    throw new IOError(ioe);
                }
                advance();
            }

            private void advance() {
                next = null;
                String line = null;
                try {
                    line = usersFileReader.readLine();
                    if (line == null) {
                        usersFileReader.close();
                        return;
                    }

                    String[] arr = line.split("\t");
                    long userId = Long.parseLong(arr[0]);
                    JSONObject[] posts = new JSONObject[(arr.length-1) / 8];
                    

                    for (int i = 1; i < arr.length; i += 8) {
                        if (i + 7 >= arr.length) {
                            throw new IllegalStateException(
                                "Sparse-formatted user.tsv line has too " +
                                "few columns");
                        }
                        String text = arr[i];
                        long tweetId = Long.parseLong(arr[i+1]);
                        String selfReportedLoc = arr[i+2];
                        String geoStr = arr[i+3];
                        String mentionsStr = arr[i+4];
                        String hashtagsStr = arr[i+5];
                        boolean isRetweet = Boolean.parseBoolean(arr[i+6]);
                        String placeStr = arr[i+7];

                        // Reconstruct the post in its original JSON format from
                        // the reduced data
                        JSONObject post = new JSONObject();
                        posts[(i-1)/8] = post;
                        post.put("id", tweetId);
                        post.put("id_str", Long.toString(tweetId));
                        post.put("text", text);
                        if (isRetweet)
                            post.put("retweeted_status", new JSONObject());
                        
                        JSONObject entities = new JSONObject();
                        post.put("entities", entities);

                        JSONArray userMentions = new JSONArray();
                        entities.put("user_mentions", userMentions);
                        for (String um : mentionsStr.split(" ")) {
                            JSONObject jo = new JSONObject();
                            jo.put("id", Long.parseLong(um));
                            jo.put("id_str", um);
                            userMentions.put(jo);
                        }

                        JSONArray hashtags = new JSONArray();
                        entities.put("hashtags", hashtags);
                        for (String ht : hashtagsStr.split(" ")) {
                            JSONObject jo = new JSONObject();
                            jo.put("text", ht);
                            hashtags.put(jo);
                        }

                        if (geoStr.length() > 0
                              && !usersWithExcludedLocations.contains(userId)) {
                            JSONObject geo = new JSONObject();
                            JSONArray coordinates = new JSONArray();
                            for (String coord : geoStr.split(" "))
                                coordinates.put(Double.parseDouble(coord));
                            geo.put("coordinates", coordinates);
                            post.put("geo", geo);
                        }

                        if (placeStr.length() > 0
                              && !usersWithExcludedLocations.contains(userId)) {
                            JSONObject place = new JSONObject(placeStr);
                            post.put("place", place);
                        }
                    }
                    
                    next = new AbstractMap
                        .SimpleImmutableEntry<Long,List<JSONObject>>(
                            userId, Arrays.asList(posts));
                } 
                catch (JSONException je) {
                    throw new IOError(je);                
                } catch (IOException ioe) {
                    throw new IOError(ioe);
                }
            }

            public boolean hasNext() {
                return next != null;
            }

            public Map.Entry<Long,List<JSONObject>> next() {
                if (next == null)
                    throw new NoSuchElementException();
                Map.Entry<Long,List<JSONObject>> n = next;
                advance();
                return n;
            }

            public void remove() {
                throw new UnsupportedOperationException();
            }
        }
    }
}
