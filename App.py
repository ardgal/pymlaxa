__author__ = 'mcardle'

import random as rnd
import multiprocessing as mp

import numpy as np
import Trip
import Driver


from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt




def plotRoutes(trip1, trip2, score, tripId):
    #plt.subplot(211)

    plt.scatter( trip1[:,0],trip1[:,1] ,c='red')
    plt.scatter( trip2[:,0],trip2[:,1] ,c='blue')
    plt.xlabel(str(score))
    plt.ylabel(str(tripId))

    plt.show()

def plotTripData(trip1, trip2):
    #plt.subplot(211)
    X = range(len(trip1))
    X2 = range(len(trip2))
    plt.plot( X,trip1,c='red')
    plt.plot( X2,trip2,c='blue')
    #plt.bar( X,trip1)
    plt.show()

class App(object):

    def run_test(self):
        driver_source = Driver.DriverSource()
        main_driver = driver_source.get_driver(1081)

        for n in range(5):
            trip = main_driver.get_data_random_trip()
            plotTripData(trip.tang_accel_mag, trip.speed)




    def run_one(self):
        """

        Use drive id: ####
        get all his matching trips (~60).  Split 40/20 train/test
        Get 60 random driver, and for each select a random trip
        and gain split 40/20

        for each trip, extract:
            mean_speed_to_stop; mean_speed_from_stop, mean_accel_inout_bends, mean_trip_speed

        Run SVM with RBF kernal and use 4 fold cross val to obtain gamma and C.

        See what we get with model in Test.  Exceed 90 - go for it with add drivers

        :return:
        """

        # minimum number of trips to process
        MIN_MATCH = 100

        # get a particular driver, obtain his/her trips then find the ones that match
        driver_source = Driver.DriverSource()
        main_driver = driver_source.get_driver(2053)

        trip_list = main_driver.get_all_trips()
        trips_to_process = main_driver.get_matching_trips()

        # fill up the trip collection with other trips, using random trips from
        # other drivers if necessary
        record_count = len(trips_to_process)
        print 'Matching Records:', record_count
        if record_count < MIN_MATCH:
            trip_ids_now = [trip.Id for trip in trips_to_process]
            other_trips = [trip for trip in trip_list if trip.Id not in trip_ids_now]
            extra_trips = rnd.sample(other_trips, MIN_MATCH - record_count)
            trips_to_process.extend(extra_trips)

        record_count = len(trips_to_process)

        # get randoom other driver but make sure it is not this one
        rand_drivers = driver_source.get_rand_drivers(1)
        while rand_drivers[0].id() == main_driver.id():
            rand_drivers = driver_source.get_rand_drivers(1)

        # add in all the trips for this random driver
        rand_trips = rand_drivers[0].get_all_trips()
        trips_to_process.extend( rnd.sample(rand_trips,record_count) )

        # classify the trips (set Y)
        targ = np.full((len(trips_to_process),1),False,dtype=bool)
        targ[0:record_count]=True

        target = targ.flatten()

        # create a vector to hold features that will be calculated
        trips_features=np.zeros(shape=[len(trips_to_process),11], dtype=float)

        # calc and add in the features...
        for idx, trip in enumerate(trips_to_process):
            mean_speed, mean_accel, mean_decel = trip.mean_speed_accel_decel()
            trips_features[idx,0] = mean_speed
            trips_features[idx,1] = mean_accel
            trips_features[idx,2] = mean_decel

            std_speed, std_accel, st_decel = trip.std_speed_accel_decel()
            trips_features[idx,3] = std_speed
            trips_features[idx,4] = std_accel
            trips_features[idx,5] = st_decel

            m_ta_l, s_ta_l = trip.mean_std_tang_accel_left()

            trips_features[idx,6] = m_ta_l
            trips_features[idx,7] = s_ta_l

            acpm = trip.accel_change_per_meter()
            trips_features[idx,8] = acpm

            m_ac, s_ac = trip.mean_std_accel_changes()
            trips_features[idx,9] = m_ac
            trips_features[idx,10] = s_ac

        # set up our split for train cross validation and test
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            trips_features, target, test_size=0.4)

        # set up our grid to find optimal parameters
        param_grid = [{'learning_rate': [0.01, 0.1, 1], 'n_estimators': [100,200], 'max_depth': [1,2,3]}]

        top_score = 0.0
        best_params = None
        scores = ['precision', 'recall']
        for score in scores:
            clf = GridSearchCV(GradientBoostingClassifier(), param_grid, scoring=score)

            clf.fit(X_train, y_train)

            print("Best parameters set found on training set:")
            print()
            print(clf.best_estimator_)
            print()
            for params, mean_score, scores in clf.grid_scores_:
                if mean_score > top_score:
                    top_score = mean_score
                    best_params = params
                print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()

            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
            print
        print best_params





    def run_two(self):
        """

        Use drive 1081
        get all his matching trips (~60).  Split 40/20 train/test
        Get 60 random driver, and for each select a random trip
        and gain split 40/20

        for each trip, extract:
            mean_speed_to_stop; mean_speed_from_stop, mean_accel_inout_bends, mean_trip_speed

        Run SVM with RBF kernal and use 4 fold crodd val to obtain gamma and C.

        See what we get with model in Test.  Exceed 90 - go for it with add drivers

        :return:
        """
        MIN_MATCH = 100
        driver_source = ds.DriverSource()
        #SVM:1081(.8) 100(.75)  2053 (.6)  15(.6) 22(.6)
        main_driver = driver_source.get_driver(1081)

        trip_list = main_driver.get_all_trips()
        trips_to_process = main_driver.get_matching_trips()

        record_count = len(trips_to_process)
        print 'Matching Records:', record_count
        if record_count < MIN_MATCH:
            extra_count = MIN_MATCH - record_count
            trip_ids_now = [trip.Id for trip in trips_to_process]
            while extra_count > 0:
                trip_rnd = main_driver.get_data_random_trip()
                while trip_rnd.Id in trip_ids_now:
                    trip_rnd = main_driver.get_data_random_trip()
                trips_to_process.append(trip_rnd)
                extra_count -= 1

        record_count = len(trips_to_process)

        rand_drivers = driver_source.get_rand_drivers(1)
        while rand_drivers[0].id() == main_driver.id():
            rand_drivers = driver_source.get_rand_drivers(1)

        rand_trips = rand_drivers[0].get_all_trips()
        trips_to_process.extend( rnd.sample(rand_trips,record_count) )

        targ = np.full((len(trips_to_process),1),False,dtype=bool)
        targ[0:record_count]=True

        target = targ.flatten()

        trips_features=np.zeros(shape=[len(trips_to_process),11], dtype=float)

        for idx, trip in enumerate(trips_to_process):
            mean_speed, mean_accel, mean_decel = trip.mean_speed_accel_decel()
            trips_features[idx,0] = mean_speed
            trips_features[idx,1] = mean_accel
            trips_features[idx,2] = mean_decel

            std_speed, std_accel, st_decel = trip.std_speed_accel_decel()
            trips_features[idx,3] = std_speed
            trips_features[idx,4] = std_accel
            trips_features[idx,5] = st_decel

            m_ta_l, s_ta_l = trip.mean_std_tang_accel_left()

            trips_features[idx,6] = m_ta_l
            trips_features[idx,7] = s_ta_l


            acpm = trip.accel_change_per_meter()
            trips_features[idx,8] = acpm

            m_ac, s_ac = trip.mean_std_accel_changes()
            trips_features[idx,9] = m_ac
            trips_features[idx,10] = s_ac

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            trips_features, target, test_size=0.4)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0005, 0.0001], 'kernel': ['rbf']},]

        scores = ['precision', 'recall']
        for score in scores:
            clf = GridSearchCV(SVC(C=1), param_grid, scoring=score)
            #clf = GridSearchCV(SVC(C=1), param_grid)
            clf.fit(X_train_scaled, y_train)

            print("Best parameters set found on training set:")
            print()
            print(clf.best_estimator_)
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()

            y_true, y_pred = y_test, clf.predict(X_test_scaled)
            print(classification_report(y_true, y_pred))
            print()



def run_process(driver_list, fname):

    FEATURE_COUNT = 20
    RAND_DRIVER_COUNT = 5

    driver_source = Driver.DriverSource()

    for the_driver in driver_list:
        main_driver = driver_source.get_driver(the_driver)
        trip_list = main_driver.get_trips()

        record_count = len(trip_list)

        # set up the feature and target arrays and set the first 200 targets to True
        trips_features=np.zeros(shape=[2 * record_count, FEATURE_COUNT], dtype=float)
        targ = np.zeros(shape=[2 * record_count,1],dtype=int)
        targ[0:record_count]=1

        target = targ.flatten()

        # fill out the feature array for the main driver...
        for idx, trip in enumerate(trip_list):
            p_accel = trip.percentiles_accel()
            p_decel = trip.percentiles_decel()
            p_speed = trip.percentiles_speed()
            p_delta = trip.percentiles_delta_accel()
            p_tang = trip.percentiles_tang_accel()

            trips_features[idx,0] = p_accel[0]
            trips_features[idx,1] = p_accel[1]
            trips_features[idx,2] = p_accel[2]
            trips_features[idx,3] = p_accel[3]

            trips_features[idx,4] = p_decel[0]
            trips_features[idx,5] = p_decel[1]
            trips_features[idx,6] = p_decel[2]
            trips_features[idx,7] = p_decel[3]

            trips_features[idx,8] = p_speed[0]
            trips_features[idx,9] = p_speed[1]
            trips_features[idx,10] = p_speed[2]
            trips_features[idx,11] = p_speed[3]

            trips_features[idx,12] = p_delta[0]
            trips_features[idx,13] = p_delta[1]
            trips_features[idx,14] = p_delta[2]
            trips_features[idx,15] = p_delta[3]

            trips_features[idx,16] = p_tang[0]
            trips_features[idx,17] = p_tang[1]
            trips_features[idx,18] = p_tang[2]
            trips_features[idx,19] = p_tang[3]

        # now get bunch random drivers...
        rand_drivers = driver_source.get_rand_drivers(RAND_DRIVER_COUNT)
        while main_driver in rand_drivers:
            rand_drivers = driver_source.get_rand_drivers(RAND_DRIVER_COUNT)

        # for each of the rand drivers, complete the feature array
        # and train the model on the features.
        prob_array = np.zeros(shape=[record_count, RAND_DRIVER_COUNT], dtype=float)
        col_idx = -1
        for drv_rand in rand_drivers:

            for i, trip in enumerate(trip_list):
                p_accel = trip.percentiles_accel()
                p_decel = trip.percentiles_decel()
                p_speed = trip.percentiles_speed()
                p_delta = trip.percentiles_delta_accel()
                p_tang = trip.percentiles_tang_accel()

                idx = record_count + i
                trips_features[idx,0] = p_accel[0]
                trips_features[idx,1] = p_accel[1]
                trips_features[idx,2] = p_accel[2]
                trips_features[idx,3] = p_accel[3]

                trips_features[idx,4] = p_decel[0]
                trips_features[idx,5] = p_decel[1]
                trips_features[idx,6] = p_decel[2]
                trips_features[idx,7] = p_decel[3]

                trips_features[idx,8] = p_speed[0]
                trips_features[idx,9] = p_speed[1]
                trips_features[idx,10] = p_speed[2]
                trips_features[idx,11] = p_speed[3]

                trips_features[idx,12] = p_delta[0]
                trips_features[idx,13] = p_delta[1]
                trips_features[idx,14] = p_delta[2]
                trips_features[idx,15] = p_delta[3]

                trips_features[idx,16] = p_tang[0]
                trips_features[idx,17] = p_tang[1]
                trips_features[idx,18] = p_tang[2]
                trips_features[idx,19] = p_tang[3]

            learning_rate   = 0.1
            n_estimators    = 200
            max_depth       = 2

            scaler = preprocessing.StandardScaler().fit(trips_features)
            X_train_scaled = scaler.transform(trips_features)
            #clf = RandomForestClassifier()
            clf = SVC(C=1, probability=True, kernel='rbf',gamma=0.001)
            clf.fit(X_train_scaled, target)

            X_pred_scaled = scaler.transform(trips_features[0:record_count,:])
            probs = clf.predict_proba(X_pred_scaled)

            class_label_index = 0
            if clf.classes_[1] == 1:
                class_label_index = 1

            col_idx += 1
            prob_array[:,col_idx][:] = probs[:,class_label_index]

        trip_probabilities = np.mean(prob_array,axis=1)

        print str(int(main_driver.id())) + ' - output to file....'
        with open(fname,'a') as writer:
            for trip, prob in zip(trip_list, trip_probabilities):
                writer.write( str(int(main_driver.id())) + '_' + str(int(trip.Id)) + ', ' + str(prob) + '\n')





def merge_files():
    with open('sub.csv', 'w') as f:
        with open('first.csv','r') as f1:
            for l in f1.readlines():
                f.write(l)
        with open('second.csv','r') as f2:
            for l in f2.readlines():
               f.write(l)
        with open('third.csv','r') as f3:
            for l in f3.readlines():
               f.write(l)
        with open('fouorth.csv','r') as f4:
            for l in f4.readlines():
                f.write(l)
        with open('fifth.csv','r') as f5:
            for l in f5.readlines():
                f.write(l)
        with open('six.csv','r') as f6:
            for l in f6.readlines():
                f.write(l)
        with open('seven.csv','r') as f7:
            for l in f7.readlines():
                f.write(l)


def main_proc():
    run_process([1081, 100, 2053, 15, 22], 'first.csv')
    """
    Run as parallel processes (if on multicore hardware)


    driver_source = ds.DriverSource()
    drivers = driver_source.drivers
    counter = len(drivers)/7

    part_1 = drivers[0:counter]
    part_2 = drivers[counter: 2 * counter]
    part_3 = drivers[2*counter: 3 * counter]
    part_4 = drivers[3*counter: 4 * counter]
    part_5 = drivers[4*counter: 5 * counter]
    part_6 = drivers[5*counter: 6 * counter]
    part_7 = drivers[6*counter:]

    processes = [mp.Process(target=run_process_2 , args=(part_1, 'first.csv')),
                 mp.Process(target=run_process_2 , args=(part_2, 'second.csv')),
                 mp.Process(target=run_process_2 , args=(part_3, 'third.csv')),
                 mp.Process(target=run_process_2 , args=(part_4, 'fouorth.csv')),
                 mp.Process(target=run_process_2 , args=(part_5, 'fifth.csv')),
                 mp.Process(target=run_process_2 , args=(part_6, 'six.csv')),
                 mp.Process(target=run_process_2 , args=(part_7, 'seven.csv'))]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    merge_files()
    """

if __name__ == "__main__":
    app = App()
    main_proc()

