/*
 * Copyright (c) 2015, David Jurgens
 *
 * All rights reserved. See LICENSE file for details
 */

package ca.mcgill.networkdynamics.geoinference;

import org.json.JSONObject;

import java.util.List;
import java.util.Map;

import org.jgrapht.Graph;

import org.jgrapht.graph.DefaultEdge;


/**
 * The interface by which the underlying Twitter datasets are accessed.
 * Instances of this class are allowed to provide a partial view of the data for
 * the sake of efficiency by omitting some information from the post content.
 */
public interface Dataset {

    /**
     * Returns an iterator over all posts in this dataset.  Posts are specified
     * in the Twitter <a
     * href="https://dev.twitter.com/overview/api/tweets">format</a>.
     */
    Iterable<JSONObject> getPosts();

    /**
     * Returns an iterator over each tuple consisting of a user and all their
     * all posts in this dataset.  Posts are specified in the Twitter <a
     * href="https://dev.twitter.com/overview/api/tweets">format</a>.
     */
    Iterable<Map.Entry<Long,List<JSONObject>>> getPostsByUser();

    /**
     * For those users whose location is known ahead of time, returns an
     * iterator of tuples consisting of such a user and their location.  This
     * method is based only on information in the dataset and is provided as a
     * convenience for applications who do would otherwise not need to process
     * the post content to extract GPS-tagged posts or the content of the
     * self-reported location field.  The source of where these location comes
     * from is ultimately up to the dataset implementation, but may be
     * configuable.
     */
    Iterable<Map.Entry<Long,LatLon>> getUserLocations();

    /**
     * Returns the social network build around mentions between users found
     * within the posts of this dataset.
     *
     * @param isDirected {@code true} if the edges should be directed
     * @param isWeighted {@code true} if the edges should be weighted by the
     *        number of times a user was mentioned
     */
    Graph<Long,? extends DefaultEdge> getMentionNetwork(
         boolean isDirected, boolean isWeighted);
}
