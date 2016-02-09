__author__ = 'mcardle'

import random as rnd

import pickle

import numpy as np

import ajmAxa.tripComparitor.tripCompare as tc
import ajmAxa.utility as util
from ajmAxa.drivers import tripFilesConverter as mtf
import Trip


class Driver(object):
    """ Base class for driver trip data.

        Instances should be acquired from DriverSource:
           driverSrc = new DriverSource()
           driver = driverSrc.get_driver(x)

        A Driver object is initialized with the driver Id and the list of trip Ids.
        Actual trip data is only loaded (cached) with a call to getDataForAllTrips()

        Object is indexed.  So you can use: trip = driver[n] to obtain nth trip data.
        Note: driver[n] will always rely on loading data from a single file unless trip data was previously cached.
        If looping through all trip data, use getDataForAllTrips() first to cahce all trips,
        then driver[n] will return the nth trip data from local cache

    """
    driverSource = None

    def __init__(self, id, driverSource):
        self.Id = int(id)
        self.label = str(self.Id)
        self.trip_data = None
        self.trip_ids = None
        self.driverSource = driverSource
        self.matching_trips = None
        self.trip_object_list = None

    def __eq__(self, other):
        return self.Id == other.id()

    def __getitem__(self, key):
        if key in self.trip_ids:
            if not self.trip_data:
                return Trip(key, self.driverSource.get_data_for_trip( self.Id , key ))
            else:
                return Trip(key, self.trip_data[key-1])
        else:
            return None

    def set_driver_source(self, driver_source):
        self.driverSource = driver_source

    def get_matching_trips(self):
        match_trip_list = self._matching_trips_list()
        return [trip for trip in self.trip_object_list if trip.Id in match_trip_list]

    def set_matching_trips(self, value):
        self.matching_trips = value

    def get_trip_list(self):
        return self.trip_ids

    def set_trip_list(self, trip_list):
        self.trip_ids = trip_list

    def get_all_trips(self, find_matches = True):
        if self.trip_data is None:
            self.trip_data = self.driverSource.get_data_for_multiple_trips( self.Id, self.trip_ids )

        if find_matches:
            if self.matching_trips is None:
                self.build_matching_trip_set()

        self.trip_object_list = [Trip(id, data) for id, data in zip(self.trip_ids, self.trip_data)]
        return self.trip_object_list

    def get_random_trip_id(self):
        return rnd.sample(self.trip_ids, 1)

    def get_data_random_trip(self):
        random_trip_data = None

        random_trip_num = rnd.sample(self.trip_ids, 1)[0]
        if self.trip_data is not None:
            random_trip_data = self.trip_data[int(random_trip_num)-1]
        else:
            random_trip_data = self.driverSource.get_data_for_trip( self.Id , int(random_trip_num) )[0]
        rand_trip = Trip(random_trip_num, random_trip_data)
        return rand_trip

    def as_string(self):
        return self.label

    def id(self):
        return self.Id

    def trip_count(self):
        return len(self.get_trip_list())

    def _matching_trips_list(self):
        matched = []
        if self.matching_trips is not None:
            for set_ in self.matching_trips:
                for item in set_:
                    matched.append(item)

        return matched

    def build_matching_trip_set(self):
        if self.trip_data is None:
            return

        cmp_obj = tc.TripCompare()
        match_list = []

        for idx, t in zip(self.trip_ids, self.trip_data):

            for idx2, t2 in zip(self.trip_ids, self.trip_data):

                if idx2 == idx:
                    continue

                if (idx, idx2) in match_list or (idx2,idx) in match_list:
                    continue

                t1 = cmp_obj.remove_rotation(t)
                t2 = cmp_obj.remove_rotation(t2)

                min_x = min(max(np.abs(t1[:, 0])), max(np.abs(t2[:, 0])))

                # small trips should be much closer together
                score_factor = 0.5
                if min_x < 2500.0:
                    score_factor = 0.8

                test_criteria = np.shape(t1)[0] * score_factor

                score = cmp_obj.raw_compare(t1,t2)
                match_found = False
                if score > (test_criteria):
                       match_found = True
                else:
                    t2 = cmp_obj.mirror_trip(t2)
                    score = cmp_obj.raw_compare(t1,t2)
                    if score > (test_criteria):
                        match_found = True
                    else:
                        t2 = cmp_obj.reverse_trip(t2)
                        score = cmp_obj.raw_compare(t1,t2)
                        if score > (test_criteria):
                            match_found = True

                if match_found:
                    match_list.append((idx, idx2))

        self.matching_trips = util.make_match_set(match_list)


class DriverSource(object):
    """ Source for driver data.  All driver records should obtained from this object and all drivers are
        cahced by default. Thsi could create memory probems so if loaded all drivers, you can call 'cacheOff()'
        to disable caching.

        Also, acts as an iterator for drivers.

        Every Driver instance has a property pointing to this object that will allow for lazy loading of data
    """
    file_source = None

    def __init__(self):
        self.last_driver_idx = 0
        self.driver_cache = []
        self.using_cache = True
        self.file_source = FileSource()
        self.drivers = self.file_source.get_set_of_drivers()

    def __getitem__(self, key):
        return self.get_driver(key)

    def __iter__(self):
        self.last_driver_idx = 0
        return self

    def next(self):
        if self.last_driver_idx == len(self.drivers) -1:
            raise StopIteration()
        else:
            self.last_driver_idx += 1
        return self.get_driver(self.last_driver_idx)

    def get_driver(self, id = None ):
        if id is None:
            id = rnd.sample(self.drivers,1)
        Id = int(id)
        driver = self.retrieve_driver_from_cache(Id)
        if driver is None:
            driver = Driver(Id, self)
            trips = self.file_source.get_trips_for_driver(Id)
            driver.set_trip_list( trips )
            self.driver_cache.append(driver)

        return driver

    def get_any_driver(self):
        return self.get_driver()

    def get_data_for_trip(self, driverId , tripId ):
        trip_data_array = []
        trip_data_array.append( self.file_source.load_raw_trip_data(driverId, tripId))
        return trip_data_array

    def get_data_for_multiple_trips(self, driverId , tripList ):
        trip_data_array = []
        for tripId in tripList:
            trip_data_array.append(self.file_source.load_raw_trip_data(driverId, tripId))
        return trip_data_array

    def get_rand_drivers(self, sampleSize):
        random_arr = []
        idx_array = rnd.sample(self.drivers,sampleSize)
        for idx in idx_array:
            d = self.get_driver(idx)
            random_arr.append(d)
        return random_arr

    def clear_driver_cache(self):
        del self.driver_cache[:]

    def cache_off(self):
        self.clear_driver_cache()
        self.using_cache = False

    def retrieve_driver_from_cache(self, driverId):
        cached_driver = None
        if self.using_cache:
            if len(self.driver_cache) > 0:
                for driver in self.driver_cache:
                    if driver.id() == driverId:
                        cached_driver = driver
                        break

        return cached_driver

    def persist_drivers(self):
        self.file_source.persist_drivers(self.driver_cache)

    def reload_drivers(self):
        self.clear_driver_cache()
        Driver.driverSource = self
        pkl_files = self.file_source.drivers_load()
        for file in pkl_files:
            driver = pickle.load(open(file, 'rb'))
            self.driver_cache.append(driver)



class FileSource(object):
    """ Encapsulates the file functions from module tripFilesConverter
        Should not be called directly, used by DriverSource
    """
    drivers_trip_Ids = None

    def __init__(self):
        self.drivers_trip_Ids = mtf.load_driver_trips_file()

    def get_trips_for_driver(self, driverId):
        return mtf.get_trips_for_driver(self.drivers_trip_Ids, driverId)

    def get_set_of_drivers(self):
        return mtf.get_driver_set(self.drivers_trip_Ids)

    def load_raw_trip_data(self, driver, trip ):
        return mtf.load_trip_data( driver, trip )

    def persist_drivers(self, cache):
        mtf.pickle_drivers(cache)

    def driver_persist_files(self):
        return mtf.get_driver_cache_persist_files()

    def drivers_load(self):
        return mtf.get_driver_cache_persist_files()



