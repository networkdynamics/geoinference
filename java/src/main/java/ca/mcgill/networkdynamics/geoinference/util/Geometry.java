/*
 * Copyright (c) 2015, David Jurgens
 *
 * All rights reserved. See LICENSE file for details
 */

package ca.mcgill.networkdynamics.geoinference.util;

import ca.mcgill.networkdynamics.geoinference.LatLon;

import org.gavaghan.geodesy.Ellipsoid;
import org.gavaghan.geodesy.GeodeticCalculator;
import org.gavaghan.geodesy.GeodeticCurve;
import org.gavaghan.geodesy.GlobalCoordinates;

import java.util.List;


/**
 * A utility class with geometric functions.
 */
public class Geometry {


    /**
     *
     */
    private static final GeodeticCalculator geoCalc = new GeodeticCalculator();

    /**
     *
     */
    private static final Ellipsoid reference = Ellipsoid.WGS84;


    /**
     * Returns the geometric median of the list of locations.
     */
    public static LatLon getGeometricMedian(List<LatLon> coordinates) {
        // The geometric median is a degenerate case for one and two
        // coordinates, so we return an arbitrary coordinate
        if (coordinates.size() < 3)
            return coordinates.get((int)(Math.random() * coordinates.size()));

        double argMinDistance = Double.MAX_VALUE;
        LatLon argMin = null;

        int n = coordinates.size();
        next_point:
        for (int i = 0; i < n; ++i) {
            LatLon p1 = coordinates.get(i);
            double distSum = 0;
            for (int j = 0; j < n; ++j) {
                LatLon p2 = coordinates.get(j);
                // Skip self-comparison
                if (i == j)
                    continue;
                double dist = getDistance(p1, p2);
                distSum += dist;
                // Break early if this point can't be the min
                if (distSum > argMinDistance)
                    break next_point;
            }
            if (distSum < argMinDistance) {
                argMinDistance = distSum;
                argMin = p1;
            }
        }
        return argMin;
    }

    /**
     * Computes the distance between the two points using Vincenty's Formulae.
     * This method wraps the Geodesy library.
     */
    public static double getDistance(LatLon p1, LatLon p2) {
        // Convert our data structures into Geodesy's 
        GlobalCoordinates gc1 = new GlobalCoordinates(p1.lat, p1.lon);
        GlobalCoordinates gc2 = new GlobalCoordinates(p2.lat, p2.lon);

        // Calculate the curve and distance between the two points
        GeodeticCurve geoCurve = geoCalc.calculateGeodeticCurve(reference, gc1, gc2);
        double ellipseKilometers = geoCurve.getEllipsoidalDistance() / 1000.0;

        return ellipseKilometers;
    }

    /**
     * Computes the distance between the two points using Vincenty's Formulae.
     * This method wraps the Geodesy library.
     */
    public static double getDistance(double[] p1, double[] p2) {
        // Convert our data structures into Geodesy's 
        GlobalCoordinates gc1 = new GlobalCoordinates(p1[0], p1[1]);
        GlobalCoordinates gc2 = new GlobalCoordinates(p2[0], p2[1]);

        // Calculate the curve and distance between the two points
        GeodeticCurve geoCurve = geoCalc.calculateGeodeticCurve(reference, gc1, gc2);
        double ellipseKilometers = geoCurve.getEllipsoidalDistance() / 1000.0;

        return ellipseKilometers;
    }    
}
