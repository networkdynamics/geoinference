/*
 * Copyright (c) 2015, David Jurgens
 *
 * All rights reserved. See LICENSE file for details
 */
package ca.mcgill.networkdynamics.geoinference;

import org.json.JSONObject;

import java.util.List;


/**
 * The interface for interacting with a trained geoinference model.  Models are
 * first produced through a training procedure with a {@link
 * GeoinferenceMethod}.  After training has been completed, the model can be
 * used to infer locations.
 */
public interface GeoinferenceModel {

    /**
     * Infers the location of the post or returns {@code null} if no location can be inferred.
     *
     * @param post a JSON post in the Twitter <a
     * href="https://dev.twitter.com/overview/api/tweets">format</a>
     */
    LatLon inferLocation(JSONObject post);
    
    /**
     * Infers the location of all posts belonging to this user and returns a
     * list of where each post was located.  Location inferences may be {@code
     * null} if no location can be inferred but the returned {@code List} should
     * not be {@code null}.
     *
     * @param id the ID of the user for which the posts are being inferred
     * @param posts a list of JSON posts in the Twitter <a
     * href="https://dev.twitter.com/overview/api/tweets">format</a>
     */
    List<LatLon> inferPostLocationsByUser(long id, List<JSONObject> posts);

}
