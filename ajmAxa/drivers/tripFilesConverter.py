__author__ = 'mcardle'

"""

 Routines to convert the raw CSV files into python (numpy) binary files
 for faster processing.
 File had already been processed by low pass filter

"""

# WARNING: need to adapt to make cross platform for file folders, etc

import os
import numpy as np
import logging
import pickle
import os

root_folder = "..."
driver_folders = root_folder + os.pathsep + "drivers" + os.pathsep
smoothed_trips_folder = root_folder + os.pathsep + "smoothtrips" + os.pathsep
trip_binaries_folder = root_folder + os.pathsep + "bintrips" + os.pathsep
drivers_trip_id_file = root_folder + os.pathsep + 'drivertrips.csv'
driver_pickle_folder = root_folder + os.pathsep+ 'driver_pkl' + os.pathsep

logging.basicConfig(filename='maktripfiles.log',level=logging.DEBUG)

def get_driver_set( driver_trips ):
    return np.unique( driver_trips[:,0] )

def get_trips_for_driver( driver_trips , driverId ):
    return driver_trips[ driver_trips[:,0] == driverId ][:,1]

def get_csv_file_name(driverId,tripId):
    return smoothed_trips_folder + str(int(driverId)) + os.pathsep + str(int(tripId)) + '.csv'

def get_csv_file_name_orig(driverId,tripId):
    return driver_folders + str(int(driverId)) + os.pathsep + str(int(tripId)) + '.csv'

def load_csv_trip_file( file_name):
    return np.loadtxt( file_name, delimiter=',' , skiprows=1)

def convert_array_to_ints( nparr ):
    return np.rint(nparr)

def create_numpy_file_name( driverId, tripId , orig_data=False):
    suffix = '.npy'
    new_folder = trip_binaries_folder + str(int(driverId))

    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    return new_folder + os.pathsep + str(int(tripId)) + suffix

def save_trip_binary_file( binFile , nparray ):
    np.save( binFile , nparray )

def load_trip_binary( binFile ):
    return np.load(binFile)

def load_trip_data( driverId, tripId):
    return np.load( create_numpy_file_name( driverId, tripId ) )

def load_trip_data_orig( driverId, tripId ):
    new_folder = driver_folders + str(int(driverId)) + os.pathsep + str(int(tripId)) + '.csv'
    return np.loadtxt( new_folder, delimiter=',', skiprows=1 )

def load_driver_trips_file():
    return np.loadtxt(drivers_trip_id_file, delimiter=',', skiprows=1)

def pickle_drivers(driver_cache):
    for driver in driver_cache:
        file_name = driver_pickle_folder + str(driver.id()) + '.p'
        pickle.dump(driver, open(file_name, 'wb'))

def get_driver_cache_persist_files():
    files = []
    file_names = os.listdir(driver_pickle_folder)
    for fname in file_names:
        drv_file_name = driver_pickle_folder + fname
        files.append(drv_file_name)
    return files





def wee_test():
    driver_trip_ids = load_driver_trips_file()
    drivers = get_driver_set(driver_trip_ids)

    for driver in drivers[0:2]:
        trips = get_trips_for_driver(driver_trip_ids, driver)
        for trip in trips:
            tripRawData = load_csv_trip_file( get_csv_file_name( driver, trip ) )
            tripData = convert_array_to_ints( tripRawData )
            save_trip_binary_file( create_numpy_file_name( driver, trip ), tripData )

    for driver in drivers[0:2]:
        trips = get_trips_for_driver(driver_trip_ids, driver)
        for trip in trips:
            tripRawData = load_csv_trip_file( get_csv_file_name( driver, trip ) )
            tripData = convert_array_to_ints( tripRawData )

            tripBinData = load_trip_binary( create_numpy_file_name( driver, trip ) )

            print np.array_equal(tripData,tripBinData)


def run_conversion():
    driver_trip_ids = load_driver_trips_file()
    drivers = get_driver_set(driver_trip_ids)
    N = len(drivers)
    count = 0
    for driver in drivers:
        trips = get_trips_for_driver(driver_trip_ids, driver)
        count += 1
        for trip in trips:
            trip_raw_data = load_csv_trip_file( get_csv_file_name_orig( driver, trip ) )
            #trip_data = convert_array_to_ints( trip_raw_data )
            save_trip_binary_file( create_numpy_file_name( driver, trip ), trip_data )

        if count % 100 == 0:
            logging.info('Driver: ' + str(driver) + ' of: ' + str(N) )

    print "File conversion complete."


def run_conversion_2():
    driver_trip_ids = load_driver_trips_file()
    drivers = get_driver_set(driver_trip_ids)
    N = len(drivers)
    count = 0
    for driver in drivers:
        trips = get_trips_for_driver(driver_trip_ids, driver)
        count += 1
        trip_list = []
        for trip in trips:
            trip_raw_data = load_csv_trip_file( get_csv_file_name_orig( driver, trip ) )
            trip_list.append(trip_raw_data)

        file_name = driver_folders + str(int(driver)) + os.pathsep + 'trips.npy'
        save_trip_binary_file( file_name , trip_list )

        if count % 100 == 0:
            logging.info('Driver: ' + str(driver) + ' of: ' + str(N) )

    print "File conversion complete."

