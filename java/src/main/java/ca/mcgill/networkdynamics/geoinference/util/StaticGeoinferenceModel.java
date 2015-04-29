/*
 * Copyright (c) 2015, David Jurgens
 *
 * All rights reserved. See LICENSE file for details
 */

package ca.mcgill.networkdynamics.geoinference.util;

import ca.mcgill.networkdynamics.geoinference.GeoinferenceModel;
import ca.mcgill.networkdynamics.geoinference.LatLon;

import gnu.trove.map.TLongObjectMap;

import gnu.trove.map.hash.TLongObjectHashMap;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * An implementation of {@link GeoinferenceModel} where all predictions are made
 * from a fixed mapping from user to their location.  This model is incapabile
 * of making predictions for users not present in the mapping.
 */
public class StaticGeoinferenceModel implements GeoinferenceModel {

    private final TLongObjectMap<LatLon> userIdToLocation;

    public StaticGeoinferenceModel(TLongObjectMap<LatLon> userIdToLocation) {
        this.userIdToLocation = userIdToLocation;    
    }

    public StaticGeoinferenceModel(Map<Long,LatLon> userIdToLocation) {
        this.userIdToLocation = 
            new TLongObjectHashMap<LatLon>(userIdToLocation.size());
        this.userIdToLocation.putAll(userIdToLocation);
    }   

    public LatLon inferLocation(JSONObject post) {
        try {
            JSONObject user = post.getJSONObject("user");
            if (user == null)
                return null;
            return userIdToLocation.get(user.getLong("id"));
        } catch (JSONException je) {
            throw new IllegalStateException("Invalid JSON while inferring", je);
        }
    }

    public List<LatLon> inferPostLocationsByUser(long id, List<JSONObject> posts) {
        LatLon loc = userIdToLocation.get(id);
        return Collections.nCopies(posts.size(), loc);
    }

}
