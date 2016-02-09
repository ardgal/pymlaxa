__author__ = 'ajm'

import numpy as np
import math as m

class TripCompare(object):
    """ Class to encapsulate methods of comparing trips
    """

    @staticmethod
    def raw_compare(trip1, trip2, proximity_factor=20., common_factor = 0.75):
        """ Performs a compare of two trips.  Trips must be rotated to same basis.
            This is a 'spacial compare'.  Each point in trip2 must be within a proximity factor
            of the corresponding point in trip1.

            :param trip1 first trip of position vectors (x.y)
            :param trip2 second trip of position vectors
            :param proximity factor

            :return Score > 0. Higher score reflects greater similarity.
            Returns -1 if trips are not within 75% of each other in length

        """
        # we'll treat x and y separately and tweak proximity where range of x and y are small
        proximity_factor_x = proximity_factor
        proximity_factor_y = proximity_factor

        # only compare first n points that they have in common
        point_count = min(len(trip1), len(trip2))
        max_points = max(len(trip1), len(trip2))
        if point_count < (common_factor * max_points):
            return -1
        else:
            max_y = max(max(np.abs(trip1[:, 1])), max(np.abs(trip2[:, 1])))
            max_x = max(max(np.abs(trip1[:, 0])), max(np.abs(trip2[:, 0])))

            delta = trip1[0:point_count,:] - trip2[0:point_count,:]
            delta_norm = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)

            trip1_diff_vec = np.diff(trip1[0:point_count,:], axis=0)
            trip1_diff_vec_len = np.sqrt(trip1_diff_vec[:, 0] ** 2 + trip1_diff_vec[:, 1] ** 2)

            # adjust proximity for trips where the y displacement is small
            y_comp = max_y / 2.0
            while proximity_factor_y > y_comp:
                proximity_factor_y /= 2
            # ...same for x
            x_comp = max_x / 2.0
            while proximity_factor_x > x_comp:
                proximity_factor_x /= 2

            new_proximity_factor = m.sqrt(proximity_factor_x ** 2 + proximity_factor_y ** 2)

            trip1_tolerance = trip1_diff_vec_len * new_proximity_factor

            result_array = (trip1_tolerance >= delta_norm[0:-1])

            return np.sum(result_array)

    @staticmethod
    def mirror_trip(trip):
        mirroredTrip = trip.copy()
        mirroredTrip[:, 1] = -mirroredTrip[:, 1]
        return mirroredTrip

    @staticmethod
    def reverse_trip(trip):
        return trip[::-1, :]

    @staticmethod
    def get_speed_from_pos_vectors(trip):
        """ Calculate the norm of the difference of array of position vectors.
            If vectors are equally spaced in time, this is the dist per unit time (ie speed).
            :param trip - array of (x,y) position vectors
            :return array of 'speed' values
        """
        data_vectors = np.diff(trip, axis=0)
        #size_array = np.linalg.norm(data_vectors, axis=1) - why is raw calc faster???
        size_array = np.sqrt(data_vectors[0]**2 + data_vectors[1]**2)
        return size_array

    @staticmethod
    def remove_rotation(XY):
        """ change of basis matrix so that the horizontal (x) axis is the vector between the first
            and last point

            Param: XY must be an N x 2 numpy array
            Return: Nx2 array of vectors in new basis

            Assumes all XY vectors start at origin (obvious from fn name)
        """
        # calc the unit vectors of the new basis
        x_dash = XY[-1]

        y_dash = np.array([-x_dash[1], x_dash[0]])

        norm_x_dash = np.linalg.norm(x_dash)
        norm_y_dash = np.linalg.norm(y_dash)

        # adapt for round trip!!!
        if norm_x_dash > 0:
            u = x_dash / norm_x_dash
        else:
            u = np.array([1, 0])
        if norm_y_dash > 0:
            v = y_dash / norm_y_dash
        else:
            v = np.array([0, 1])

        # change of basis 'matrix' - (x',y') = M(inv)(x,y)
        # Minv is just transpose of the new basis matrix M since rotn about origin
        M_dash = np.array([[u[0], u[1]], [v[0], v[1]]])

        # now transform aall the points t the new basis
        # Mdash * XY -> 2x2 x (2xN) hence transpose
        XY_new = np.dot(M_dash, XY.T)

        # return it back as Nx2
        return XY_new.T
