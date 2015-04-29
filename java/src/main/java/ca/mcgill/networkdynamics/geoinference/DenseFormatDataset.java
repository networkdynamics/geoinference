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
 * The implementation of {@link Dataset} for data that is stored in full JSON
 * format.  See the wiki for details.
 */
public class DenseFormatDataset extends AbstractDataset {

    private static final String DEFAULT_USERS_FILE = "users.json.gz";

    private final String usersFileName;

    public DenseFormatDataset(File datasetDir) {
        this(datasetDir, DEFAULT_LOCATION_TYPE, new TLongHashSet(),
             DEFAULT_USERS_FILE);
    }


    public DenseFormatDataset(File datasetDir, String locationType) {
        this(datasetDir, locationType, new TLongHashSet(),
             DEFAULT_USERS_FILE);
    }


    public DenseFormatDataset(File datasetDir,
                              TLongSet usersWithExcludedLocations) {
        this(datasetDir, DEFAULT_LOCATION_TYPE, usersWithExcludedLocations,
             DEFAULT_USERS_FILE);
    }

    public DenseFormatDataset(File datasetDir, 
                              String locationType,
                              TLongSet usersWithExcludedLocations) {
        this(datasetDir, locationType, usersWithExcludedLocations,
              DEFAULT_USERS_FILE);
    }

    public DenseFormatDataset(File datasetDir, 
                              String locationType,
                              TLongSet usersWithExcludedLocations,
                              String usersFileName) {
        super(datasetDir, locationType, usersWithExcludedLocations);
        this.usersFileName = usersFileName;
    }

    /**
     * {@inheritDoc}
     */
    public Iterable<Map.Entry<Long,List<JSONObject>>> getPostsByUser() {
        return new UserPosts();
    }

    /**
     * The class for iterating over a user's posts.
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
                        Files.openGz(new File(datasetDir, usersFileName));
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

                    JSONObject jo = new JSONObject(line);
                    Long userId = Long.valueOf(jo.getString("user_id"));
                    JSONArray postsArr = jo.getJSONArray("posts");
                    
                    JSONObject[] posts = new JSONObject[postsArr.length()];
                    for (int i = 0; i < posts.length; ++i) {
                        posts[i] = postsArr.getJSONObject(i);
                        // Strip out the location data if necessary
                        if (usersWithExcludedLocations.contains(userId)) {
                            posts[i].remove("geo");
                            posts[i].remove("place");
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
