##
#  Copyright (c) 2015, Tyler Finethy, David Jurgens
#
#  All rights reserved. See LICENSE file for details
##

"""
Geocoder and Reverse-Geocoder to be used by the Geolocation Project
Allows for multiple dataset inputs
"""

import os, os.path
import csv
import re
from collections import defaultdict
from geopy.distance import vincenty
import collections
import logging
import gzip

LOGGER = logging.getLogger(os.path.basename(__file__))

class Geocoder(object):

    """
    Geocoder to be used on the Geolocation Inference Project.
    """
    def __init__(self,dataset="geonames"):
        """
        Initializes the "reverse_geocoder" and "geocoder" dictionaries,
        based on the dataset selected. By default "geonames" is selected,
        """
        self.reverse_geocoder = defaultdict(list)
        self.geocoder = {}
        self.abbv_to_state = state_abbv_data()
        self.state_abbv_regex = re.compile(r'(\b' + (r'\b|\b'.join(self.abbv_to_state.keys())) + r'\b)')
        self.all_city_names = set()

        LOGGER.debug("Geocoder loading city-location mapping from %s" % (dataset))

        # If the user specifies GeoLite data or if they are using GPS data, for
        # which GeoList is the default gazetteer
        if dataset == "geolite" or dataset == "geo-median":
            data = geolite_data()

            city_to_latlon = {}
            city_name_counts = collections.Counter()

            for line in data[2:]:
                country_name = line[2].lower()
                region_name = line[3].lower()
                city_name = line[4].lower()

                if not city_name:
                    continue

                lat = float(line[0])
                lon = float(line[1])

                # Keep track of how many times city names occur
                city_to_latlon[city_name] = (lat,lon)
                city_name_counts[city_name] += 1


                #sets bins of 0.01 accuracy of lat/lon for reverse_geocoding
                rounded_lat = round(lat,2)
                rounded_lon = round(lon,2)

                #builds the geocoder dictionary based on a city\tregion\tcountry format
                if city_name and region_name and country_name:
                    city_region_country = city_name+"\t"+region_name+"\t"+country_name
                    city_region = city_name+"\t"+region_name
                    city_country = city_name+"\t"+country_name
                    self.geocoder[city_region_country] = (lat,lon)
                    self.geocoder[city_region] = (lat,lon)
                    self.geocoder[city_country] = (lat,lon)
                    self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,city_region_country))
                    self.all_city_names.add(city_region_country)
                elif city_name and region_name:
                    city_region = city_name+"\t"+region_name
                    self.geocoder[city_region] = (lat,lon)
                    self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,city_region))

                elif city_name and country_name:
                    if city_name == country_name:
                        self.geocoder[city_name] = (lat,lon)
                        self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,city_name))
                        self.all_city_names.add(city_name)
                    else:
                        city_country = city_name+"\t"+country_name
                        self.geocoder[city_country] = (lat,lon)
                        self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,city_country))
                        self.all_city_names.add(city_country)

            # If there was only ever one city with this name, allow it to be an
            # unabiguosus lookup with just the city name
            unambiguous_cities = 0
            for city_name, (lat,lon) in city_to_latlon.iteritems():
                if city_name_counts[city_name] == 1:
                    self.geocoder[city_name] = (lat,lon)
                    unambiguous_cities += 1
            #print "Saw %d unambiguous cities in %s" % (unambiguous_cities, dataset)


        elif dataset == "google":
            data = google_data()
            
            city_to_latlon = {}
            city_name_counts = collections.Counter()

            for line in data[1:]:
                #TODO this city name should be formatted the same as incoming tweets
                city_name = line[6].lower()
                if not city_name:
                    continue

                country_name = line[2].lower()
                region_name = line[3].lower()
                lat = float(line[0])
                lon = float(line[1])
                rounded_lat = round(lat,2)
                rounded_lon = round(lon,2)
                #self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,country_name,region_name,city_name))

                # Keep track of how many times city names occur
                city_to_latlon[city_name] = (lat,lon)
                city_name_counts[city_name] += 1

                if city_name and region_name and country_name:
                    city_region_country = city_name+"\t"+region_name+"\t"+country_name
                    city_region = city_name+"\t"+region_name
                    city_country = city_name+"\t"+country_name
                    self.geocoder[city_region_country] = (lat,lon)
                    self.geocoder[city_region] = (lat,lon)
                    self.geocoder[city_country] = (lat,lon)
                    self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,city_region_country))
                    self.all_city_names.add(city_region_country)

                elif city_name and region_name:
                    city_region = city_name+"\t"+region_name
                    self.geocoder[city_region] = (lat,lon)
                    self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,city_region))
                    self.all_city_names.add(city_region)

                elif city_name and country_name:
                    if city_name == country_name:
                        self.geocoder[city_name] = (lat,lon)
                        self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,city_name))
                        self.all_city_names.add(city_name)
                    else:
                        city_country = city_name+"\t"+country_name
                        self.geocoder[city_country] = (lat,lon)
                        self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,city_country))
                    self.all_city_names.add(city_country)

            # If there was only ever one city with this name, allow it to be an
            # unabiguosus lookup with just the city name
            unambiguous_cities = 0
            for city_name, (lat,lon) in city_to_latlon.iteritems():
                if city_name_counts[city_name] == 1:
                    self.geocoder[city_name] = (lat,lon)
                    unambiguous_cities += 1
            #print "Saw %d unambiguous cities in %s" % (unambiguous_cities, dataset)

        elif dataset == "dbpedia":
            data = dbpedia_data()
            
            city_to_latlon = {}
            city_name_counts = collections.Counter()
            already_entered = set()
            line_no = 0

            for cols in data[1:]:

                line_no += 1
                if line_no % 1000000 == 0:
                    LOGGER.debug("currently read %d locations from %s" %
                                 (line_no, dataset))


                lat = cols[3]
                lon = cols[4]

                # Guard against weirdness
                if lat == 'NAN' or lon == 'NAN':
                    continue
                try:
                    lat = float(lat)
                    lon = float(lon)
                except ValueError:
                    continue

                # Ensure we can use this location if we're not allowing duplicates
                lat_lon = (lat, lon)
                already_entered.add(lat_lon)
                
                city = cols[0].lower()
                country = cols[2].lower()
                states = cols[1].lower().split('|')
                
                city_to_latlon[city] = (lat,lon)
                city_name_counts[city] += 1

                self.__add_name(city + "\t" + country, lat_lon)
                if city == country:
                    self.__add_name(city, lat_lon)
                for state in states:
                    self.__add_name(city + "\t" + state + "\t" + country, lat_lon)

            unambiguous_cities = 0
            for city_name, (lat,lon) in city_to_latlon.iteritems():
                if city_name_counts[city_name] == 1:
                    self.geocoder[city_name] = (lat,lon)
                    unambiguous_cities += 1

        elif dataset == "geonames":
            data = geonames_data()
            
            city_to_latlon = {}
            city_name_counts = collections.Counter()

            line_no = 0
            for line in data[1:]:
                #TODO this city name should be formatted the same as incoming tweets
                city_name = line[0].lower()
                if not city_name:
                    continue

                line_no += 1
                if line_no % 1000000 == 0:
                    LOGGER.debug("currently read %d locations from %s" %
                                 (line_no, dataset))

                country_name = line[2].lower()
                region_name = line[1].lower()
                lat = float(line[3])
                lon = float(line[4])
                rounded_lat = round(lat,2)
                rounded_lon = round(lon,2)
                
                # Keep track of how many times city names occur
                city_to_latlon[city_name] = (lat,lon)
                city_name_counts[city_name] += 1

                if city_name and region_name and country_name:
                    city_region_country = city_name+"\t"+region_name+"\t"+country_name
                    city_region = city_name+"\t"+region_name
                    city_country = city_name+"\t"+country_name
                    self.geocoder[city_region_country] = (lat,lon)
                    self.geocoder[city_region] = (lat,lon)
                    self.geocoder[city_country] = (lat,lon)
                    self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,city_region_country))
                    self.all_city_names.add(city_region_country)

                elif city_name and region_name:
                    city_region = city_name+"\t"+region_name
                    self.geocoder[city_region] = (lat,lon)
                    self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,city_region))
                    self.all_city_names.add(city_region)

                elif city_name and country_name:
                    if city_name == country_name:
                        self.geocoder[city_name] = (lat,lon)
                        self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,city_name))
                        self.all_city_names.add(city_name)
                    else:
                        city_country = city_name+"\t"+country_name
                        self.geocoder[city_country] = (lat,lon)
                        self.reverse_geocoder[(rounded_lat,rounded_lon)].append((lat,lon,city_country))
                    self.all_city_names.add(city_country)

            # If there was only ever one city with this name, allow it to be an
            # unabiguosus lookup with just the city name
            unambiguous_cities = 0
            for city_name, (lat,lon) in city_to_latlon.iteritems():
                if city_name_counts[city_name] == 1:
                    self.geocoder[city_name] = (lat,lon)
                    unambiguous_cities += 1
            #print "Saw %d unambiguous cities in %s" % (unambiguous_cities, dataset)


        else:
            raise NotImplementedError(dataset)


        # create a lower-case dictionary for noisy lookups
        self.lc_name_to_location = {}
        for name, (lat, lon) in self.geocoder.iteritems():
            self.lc_name_to_location[name.lower()] = (lat, lon)

        LOGGER.debug("Geocoder loaded %d locations from %s" %
                     (len(self.geocoder), dataset))

    def __add_name(self, name, lat_lon):
        lat = lat_lon[0]
        lon = lat_lon[1]
        rounded_lat = round(lat, 2)
        rounded_lon = round(lon, 2)

        self.geocoder[name] = lat_lon
        self.reverse_geocoder[(rounded_lat, rounded_lon)].append((lat, lon, name))
        self.all_city_names.add(name)


    def canonicalize(self, lat, lon):
        """
        Identifies the settlement in which this (lat, lon) pair occurs and
        returns the canonical (lat, lon) pair for that settlement.  If the
        provided (lat, lon) pair does not occur in any settlement, the original
        pair is returned.  This method provides a way for coarsening nearby
        lat/lon values into a single point, analogous to a city name that
        represents all of those points
        """
        location_name = self.reverse_geocode(lat, lon)
        
        # Check whether we can associate this lat/lon with a city and if not
        # just return the input
        if location_name is None:
            return (lat, lon)
        # Otherwise, grab the city and return it.
        else:
            return self.geocode(location_name)
        

    def geocode(self, location_name):
        """
        Returns the latitude and lonitude (tuple) of a city name if found
        """
        try:
            return self.geocoder[location_name.lower()]
        except KeyError:
            return None

    def geocode_noisy(self, location_name):
        """
        Returns the latitude and lonitude (tuple) of a noisy location name
        (e.g., the location field of a social media user's profile).  If your
        input isn't cleaned, you probably want this method instead of geocode().
        """

        usaRegex = re.compile("\\bUSA\\b")
        usRegex = re.compile("\\bUS\\b")
        ukRegex = re.compile("\\bUK\\b")
        
        name = location_name
        name = name.strip()

        # Correct for a few common noisy prefices
        if name.startswith("the city of "):
            name = name[12:] #.substring("the city of ".length())
        if name.startswith("downtown "):
            name = name[9:] #.substring("downtown ".length())

        # Swap out the three common contry abbrevations
        name = re.sub(usaRegex, "United States", name)
        name = re.sub(usRegex, "United States", name)
        name = re.sub(ukRegex, "United Kingdom", name)

        # Substitute out state names from the US
        matches = re.search(self.state_abbv_regex, name)
        if not matches is None:
            abbv = matches.group(0)
            expanded = name[:matches.start(0)] + self.abbv_to_state[abbv] + name[matches.end(0):]
            #print "%s:: %s -> %s" % (abbv, name, expanded)
            name = expanded

        # Once we've matched abbreivations, lower case for all further
        # comparisons
        name = name.lower();

        if name == "nyc":
            name = "new york, new york"

        # Strip off all the cruft on either side
        name = re.sub(ur'^[\W+]+', " ", name);
        name = re.sub(ur'[\W+]+$', " ", name);
        name = name.strip();

        # Rename the dict for brevity since we're going to referencing it a lot
        # in the next section
        locs = self.lc_name_to_location
        lat_lon = None

#        print "SEACHING %s..." % (name)

        # Look for some name delimeters in the name to try matching on
        # city/state, etc.
        if name.find(',') >= 0 or name.find('-') >= 0 or name.find('|') >= 0:
            parts = re.split(r'[,\-|]+', name)

            if len(parts) == 2:
                p1 = parts[0].strip()
                p2 = parts[1].strip()
                # print "CASE1: (%s) (%s)" % (p1, p2)
                if p1 + '\t' + p2 in locs:
                    lat_lon = locs[p1 + '\t' + p2]
                elif p2 + '\t' + p1 in locs:
                    lat_lon = locs[p2 + '\t' + p1]
                elif p1 in locs:
                    lat_lon = locs[p1]

                if lat_lon is None and p1.find("st.") >= 0:
                    p1 = re.sub("st.", "saint", p1)
                    if p1 + '\t' + p2 in locs:
                        lat_lon = locs[p1 + '\t' + p2]
                    elif p2 + '\t' + p1 in locs:
                        lat_lon = locs[p2 + '\t' + p1]
                    elif p1 in locs:
                        lat_lon = locs[p1]

                elif lat_lon is None and p1.find("saint") >= 0:
                    p1 = re.sub("saint", "st.", p1)
                    if p1 + '\t' + p2 in locs:
                        lat_lon = locs[p1 + '\t' + p2]
                    elif p2 + '\t' + p1 in locs:
                        lat_lon = locs[p2 + '\t' + p1]
                    elif p1 in locs:
                        lat_lon = locs[p1]

            elif len(parts) == 3:
                p1 = parts[0].strip()
                p2 = parts[1].strip()
                p3 = parts[2].strip()
                # print "CASE2: (%s) (%s) (%s)" % (p1, p2, p3)
                if p1 + '\t' + p2 in locs:
                    lat_lon = locs[p1 + '\t' + p2]
                elif p1 + '\t' + p3 in locs:
                    lat_lon = locs[p1 + '\t' + p3]
                elif p1 in locs:
                    lat_lon = locs[p1]

                if lat_lon is None and p1.find("st.") >= 0:
                    p1 = re.sub("st.", "saint", p1)
                    if p1 + '\t' + p2 in locs:
                        lat_lon = locs[p1 + '\t' + p2]
                    elif p1 + '\t' + p3 in locs:
                        lat_lon = locs[p1 + '\t' + p3]
                    elif p1 in locs:
                        lat_lon = locs[p1]
                if lat_lon is None and p1.find("saint") >= 0:
                    p1 = re.sub("saint", "st.", p1)
                    if p1 + '\t' + p2 in locs:
                        lat_lon = locs[p1 + '\t' + p2]
                    elif p1 + '\t' + p3 in locs:
                        lat_lon = locs[p1 + '\t' + p3]
                    elif p1 in locs:
                        lat_lon = locs[p1]

            else:
                pass #print "CASE5: %s" % (parts)            

        # Otherwise no delimeters so we're left to guess at where the name
        # breaks
        else:
            parts = re.split(r'[ \t\n\r]+', name)
            if len(parts) == 2:
                p1 = parts[0]
                p2 = parts[1]
                #print "CASE3: (%s) (%s)" % (p1, p2)
                if p1 + '\t' + p2 in locs:
                    lat_lon = locs[p1 + '\t' + p2]
                elif p2 + '\t' + p1 in locs:
                    lat_lon = locs[p2 + '\t' + p1]
                elif p1 in locs:
                    lat_lon = locs[p1]
                
                if lat_lon is None and p1.find("st.") >= 0:
                    p1 = re.sub("st.", "saint", p1)
                    if p1 + '\t' + p2 in locs:
                        lat_lon = locs[p1 + '\t' + p2]
                    elif p2 + '\t' + p1 in locs:
                        lat_lon = locs[p2 + '\t' + p1]
                    elif p1 in locs:
                        lat_lon = locs[p1]

                elif lat_lon is None and p1.find("saint") >= 0:
                    p1 = re.sub("saint", "st.", p1)
                    if p1 + '\t' + p2 in locs:
                        lat_lon = locs[p1 + '\t' + p2]
                    elif p2 + '\t' + p1 in locs:
                        lat_lon = locs[p2 + '\t' + p1]
                    elif p1 in locs:
                        lat_lon = locs[p1]


            elif len(parts) > 2:
                # Guess that the last name is a country/state and try
                # city/<whatever>
                #print "CASE4: %s" % (parts)                
                last = parts[-1]
                city = ' '.join(parts[:-1])
            else:
                pass #print "CASE6: %s" % (parts)

        # Last ditch effort: just try matching the whole name and hope it's
        # a single unambiguous city match
        if lat_lon is None and name in locs:
            lat_lon = locs[name]                              

        #print "FOUND? %s ('%s') -> %s" % (location_name, name, lat_lon)

            

        return lat_lon



    def reverse_geocode(self, lat, lon):
        """
        Returns the closest city name given a latitude and lonitude.
        """
        cities = self.reverse_geocoder[(round(lat,2),round(lon,2))]
        closest_location = float("inf")
        best_location = None
        for city in cities:
            lat2 = city[0]
            lon2 = city[1]
            distance = vincenty((lat,lon),(lat2,lon2)).km
            if distance < closest_location:
                closest_location = distance
                best_location = city[2]

        #if the direct search fails to find a location within 20km of the requested location,
        #the lat/lon is assumed to be either not available or a fringe case. In the latter
        #we check all the surrounding boxes in the reverse geocoder.
        if closest_location > 20:
            cities = self.reverse_geocoder[(round(lat+0.01,2),round(lon+0.01,2))] \
                 + self.reverse_geocoder[(round(lat+0.01,2),round(lon-0.01,2))] + self.reverse_geocoder[(round(lat-0.01,2),round(lon+0.01,2))] \
                 + self.reverse_geocoder[(round(lat-0.01,2),round(lon-0.01,2))] + self.reverse_geocoder[(round(lat,2),round(lon+0.01,2))] \
                 + self.reverse_geocoder[(round(lat,2),round(lon-0.01,2))] + self.reverse_geocoder[(round(lat+0.01,2),round(lon,2))] \
                 + self.reverse_geocoder[(round(lat-0.01,2),round(lon,2))]
            for city in cities:
                lat2 = city[0]
                lon2 = city[1]
                distance = vincenty((lat,lon),(lat2,lon2)).km
                if distance < closest_location:
                    closest_location = distance
                    best_location = city[2]

        return best_location


    def get_cities(self):
        """
        Return the set of cities that have any latitude and longitude
        coordinates in this instance
        """
        return self.all_city_names

def geolite_data():
    """
    Returns the file contents of the geolite dataset.
    """
    file_contents = []
    file_name = os.path.join("geolocate/resources","geolite.csv")
    with open(file_name, 'rb') as csv_file:
        for line in csv.reader(csv_file):
            file_contents.append(line)
    return file_contents

def geonames_data():
    """
    Returns the file contents of the geolite dataset.
    """
    file_contents = []
    file_name = os.path.join("geolocate/resources","geonames.exanded-cities.tsv.gz")
    f = gzip.open(file_name, 'rb') 
    for line in f:
        file_contents.append(line.strip().split('\t'))
    f.close()
    return file_contents

def google_data():
    """
    Returns the file contents of the google dataset.
    """
    file_name = os.path.join("geolocate/resources","google-formatted.tsv")
    file_contents = []
    with open(file_name, 'r') as tsv_file:
        for line in csv.reader(tsv_file, dialect="excel-tab"):
            file_contents.append(line)
    return file_contents

def dbpedia_data():
    """
    Returns the file contents of the google dataset.
    """
    file_name = os.path.join("geolocate/resources","dbpedia.settlement-to-locations.localized.cleaned.uniq.tsv")
    file_contents = []
    with open(file_name, 'r') as tsv_file:
        for line in csv.reader(tsv_file, dialect="excel-tab"):
            file_contents.append(line)
    return file_contents


def state_abbv_data():
    """
    Returns a dict containing state abbreviations
    """
    file_name = os.path.join("geolocate/resources","state_table.csv")
    abbv_to_state = {}
    with open(file_name, 'r') as csv_file:
        line_no = 0
        for line in csv.reader(csv_file, delimiter=',', quotechar='"'):
            line_no += 1
            if line_no == 1:
                continue
            name = line[1]
            abbv = line[2]
            if len(abbv) > 0:
                # print "%s -> %s" % (abbv, name)
                abbv_to_state[abbv] = name
    return abbv_to_state






