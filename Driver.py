__author__ = 'mcardle'

import os
import numpy as np
import logging
import pickle
import random as rnd
import Trip as tripSrc

root_folder = ""
driver_folders = root_folder + "/drivers/"
drivers_trip_id_file = root_folder + '/drivertrips.csv'


class Driver(object):
    """ Base class for driver trip data.

        Instances should be acquired from DriverSource:
           driverSrc = new DriverSource()
           driver = driverSrc.get_driver(x)

        A Driver object is initialized with the driver Id, the list of trip Ids and trip array.

        Object is indexed.  So you can use: trip = driver[n] to obtain nth trip data.
    """

    def __init__(self, id, driverSource):
        self.Id = int(id)
        self.label = str(self.Id)
        self.trip_data = None
        self.trip_ids = None
        self.trip_object_list = []

    def __eq__(self, other):
        return self.Id == other.id()

    def __getitem__(self, key):
        if key in self.trip_ids:
            idx = key - 1
            return self.trip_object_list[idx]
        else:
            return None

    def get_trips(self):
        return self.trip_object_list

    def get_trip_list(self):
        return self.trip_ids

    def set_trip_list(self, trip_list):
        self.trip_ids = trip_list

    def set_trip_data(self, trip_data):
        self.trip_data = trip_data
        self.trip_object_list = [tripSrc.Trip(id, data) for id, data in zip(self.trip_ids, self.trip_data)]

    def get_random_trip_id(self):
        return rnd.sample(self.trip_ids, 1)

    def get_data_random_trip(self):
        return rnd.sample(self.trip_object_list, 1)[0]

    def as_string(self):
        return self.label

    def id(self):
        return self.Id

    def trip_count(self):
        return len(self.get_trip_list())


class DriverSource(object):
    """ Source for driver data.  All driver records should obtained from this object and all drivers are
        cahced by default. This could create memory probems so if loaded all drivers, you can call 'cacheOff()'
        to disable caching.

        Also, acts as an iterator for drivers.

    """

    def __init__(self):
        self.driver_trip_array = self.load_driver_trips_file()
        self.drivers = np.unique( self.driver_trip_array[:,0] )
        self.driver_cache = []
        self.using_cache = True

    def __getitem__(self, key):
        return self.get_driver(key)

    def get_driver(self, id = None ):
        if id is None:
            id = rnd.sample(self.drivers,1)[0]
        Id = int(id)
        driver = self.retrieve_driver_from_cache(Id)
        if driver is None:
            driver = Driver(Id, self)
            trip_ids = np.arange(1,201)
            trip_list = self.get_driver_trip_list(driver.id())
            driver.set_trip_list(trip_ids)
            driver.set_trip_data(trip_list)
            self.driver_cache.append(driver)

        return driver

    def get_rand_drivers(self, rand_count ):
        rand_drivers = []
        for i in range(rand_count):
            rand_driver = self.get_driver()
            while rand_driver in rand_drivers:
                rand_driver = self.get_driver()
            rand_drivers.append(rand_driver)

        return rand_drivers


    def get_driver_trip_list(self, driverId):
        return np.load(driver_folders + str(driverId) + '/' + 'trips.npy')

    def get_any_driver(self):
        return self.get_driver()

    def load_driver_trips_file(self):
        return np.loadtxt(drivers_trip_id_file, delimiter=',', skiprows=1)

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
