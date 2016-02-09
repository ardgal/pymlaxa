__author__ = 'mcardle'

import numpy as np


class Trip(object):

    def __init__(self, id, trip_data):
        self.Id = id
        self.trip_data = trip_data

        self.velocity_vectors   = np.diff(trip_data, axis=0)
        self.tang_accel_vectors = np.diff(self.velocity_vectors, axis=0)

        self.speed = Trip.normalize(self.velocity_vectors)

        self.tang_accel_mag = Trip.normalize(self.tang_accel_vectors)

        self.accel_mag = np.diff(self.speed, axis=0)
        self.delta_accel_mag = np.diff(self.accel_mag, axis=0)

        self.accel_mag_pos = self.accel_mag[self.accel_mag > 0]
        self.accel_mag_neg = np.fabs(self.accel_mag[self.accel_mag < 0])

        self.trip_len_meters = np.sum(self.speed, axis=0)

    def percentiles_accel(self):
        if len(self.accel_mag_pos) > 0:
            return np.percentile(self.accel_mag_pos, [25.0,50.0,75.0,100.0])
        else:
            return [0.0,0.0,0.0,0.0]

    def percentiles_decel(self):
        if len(self.accel_mag_neg) > 0:
            return np.percentile(self.accel_mag_neg, [25.0,50.0,75.0,100.0])
        else:
            return [0.0,0.0,0.0,0.0]

    def percentiles_tang_accel(self):
        if len(self.tang_accel_mag) > 0:
            return np.percentile(self.tang_accel_mag, [25.0,50.0,75.0,100.0])
        else:
            return [0.0,0.0,0.0,0.0]

    def percentiles_delta_accel(self):
        if len(self.delta_accel_mag) > 0:
            return np.percentile(self.delta_accel_mag, [25.0,50.0,75.0,100.0])
        else:
            return [0.0,0.0,0.0,0.0]

    def percentiles_speed(self):
        if len(self.speed) > 0:
            return np.percentile(self.speed, [25.0,50.0,75.0,100.0])
        else:
            return [0.0,0.0,0.0,0.0]


    @staticmethod
    def _mean(value_array):
        if len(value_array)> 0:
            return np.mean(value_array)
        else:
            return 0

    @staticmethod
    def _std(value_array):
        if len(value_array)> 0:
            return np.std(value_array)
        else:
            return 0

    @staticmethod
    def indices_consec_zeros(value_array):
        """ Provide indices of subsequent consecutive zeros
        """
        indicies_list = []
        indices = Trip.find_zeros(value_array)
        last_val = -100
        for idx in indices:
            if idx == last_val + 1:
                indicies_list.append(idx)
            last_val = idx
        return indicies_list

    @staticmethod
    def find_zeros(speeds, tolerance = 0.2):
        """ Find indices of  zeros
        :param speeds: array of speed values
        :param tolerance: value below that considered not moving
        :return: array of indices
        """
        return np.where(speeds < tolerance )[0]

    @staticmethod
    def normalize(vectors):
        #...why faster than np.linalg.normalize???
        return np.sqrt(np.sum(vectors**2, axis=1))