/*
 * Copyright (c) 2015, David Jurgens
 *
 * All rights reserved. See LICENSE file for details
 */

package ca.mcgill.networkdynamics.geoinference;


/**
 * Utility structure for holding a latitude and longitude.
 */
public class LatLon {

    public final double lat;

    public final double lon; 

    public LatLon(double lat, double lon) {
        this.lat = lat;
        this.lon = lon;
    }

    public boolean equals(Object o) {
        if (o instanceof LatLon) {
            LatLon loc = (LatLon)o;
                return loc.lat == lat && loc.lon == lon;
        }
        return false;
    }

    public int hashCode() {
        return (int)lat + (int)lon;
    }

    public String toString() {
        return "(" + lat + ", " + lon + ")";
    }
}
