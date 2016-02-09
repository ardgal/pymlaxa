__author__ = 'ajm'

import numpy as np


class TripAnalyzer(object):
    """ Object to analyze driver trips
    """
    @staticmethod
    def convert_2_velocity(trip):
        return TripAnalyzer._get_diff(trip)

    @staticmethod
    def convert_to_speeds(pos_vectors):
        vel = TripAnalyzer.convert_2_velocity(pos_vectors)
        return TripAnalyzer.normalize(vel)

    @staticmethod
    def convert_to_speed_changes(vel):
        accel = TripAnalyzer._get_diff(vel)
        return TripAnalyzer.normalize(accel)

    @staticmethod
    def _get_diff(vectors, n=1):
        return np.diff(vectors, n, axis=0)

    @staticmethod
    def convert_to_accel(vel_vectors):
        return TripAnalyzer._get_diff(vel_vectors)

    @staticmethod
    def normalize(vectors):
        #...why slower than np.linalg.normalize???
        return np.sqrt(np.sum(vectors**2, axis=1))

    @staticmethod
    def find_zeros(speeds, tolerance = 0.1):
        """ Find indices of consecutive zeros
        :param speeds: array of speed values
        :param tolerance: value below that considered not moving
        :return: array of indices
        """
        return np.where(speeds < tolerance )[0]

    @staticmethod
    def remove_consec_zeros(value_array):
        """ Provide a copy of the original value array with the consecutive zero values removed
        """
        result_list = []
        indices = TripAnalyzer.find_zeros(value_array)
        last_val = -1
        for idx in indices:
            if idx == last_val + 1:
                result_list.append(idx)
            last_val = idx
        return np.delete(value_array, result_list)

    @staticmethod
    def calc_cos_anges(vel_vecs):
        a = vel_vecs[0:-1,:]
        b = vel_vecs[1:,:]
        a_dot_b = np.sum(a * b, axis=1)
        norm_ab = TripAnalyzer.normalize(a) * TripAnalyzer.normalize(b)
        # deal with divide by zero nonsense
        norm_ab[norm_ab < 0.001 ] = 0.001
        result = a_dot_b / np.nan_to_num(norm_ab)
        return result

    @staticmethod
    def find_trip_bends(velocities, cos_theta=0.5):
        cos_angles = np.abs(TripAnalyzer.calc_cos_anges(velocities))
        return np.where(cos_angles[cos_angles < cos_theta])[0]

    @staticmethod
    def calc_mean_accel_inout_bends(trip):
        """
        Need to improve this - everything
        :param trip:
        :return:
        """
        vel = TripAnalyzer.convert_2_velocity(trip)
        accel = TripAnalyzer.convert_to_accel(vel)
        bend_indices = TripAnalyzer.find_trip_bends(vel)
        if len(bend_indices) > 0:
            bend_accel = TripAnalyzer.normalize(accel[bend_indices])
        else:
            bend_accel = []

        if len(bend_accel) > 0:
            result = np.mean(bend_accel)
        else:
            result = 0.0
        return result

    @staticmethod
    def mean_speed_tofrom_start(trip, num_seconds=3):
        """ Calculate the mean speed n seconds after moving to and from stop
        :param trip:
        :return: tuple of mean value before and mean value after
        """
        adj_trip = TripAnalyzer.remove_consec_zeros(TripAnalyzer.convert_to_speeds(trip))
        trip_len = len(adj_trip)-1
        stops = np.where(adj_trip == 0)[0]
        before_stops = np.subtract(stops, [num_seconds])
        after_stops = np.add(stops, [num_seconds])

        # make sure indices are valid
        before_stops[before_stops < 0] = 0
        after_stops[after_stops > trip_len] = trip_len

        bfs = adj_trip[before_stops]
        afs = adj_trip[after_stops]

        if len(bfs) > 0:
            before_res = np.mean(bfs)
        else:
            before_res = 0
        if len(afs) > 0:
            after_res = np.mean(afs)
        else:
            after_res = 0

        return before_res, after_res


    @staticmethod
    def mean_speed_when_moving(trip):
        """ Calculate the mean speed after removing consecutive zeros
        :param trip:
        :return: float value
        """
        adj_trip = TripAnalyzer.remove_consec_zeros(TripAnalyzer.convert_to_speeds(trip))
        if len(adj_trip) > 0:
            return np.mean(adj_trip)
        else:
            return 0.0