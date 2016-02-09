""" Utility functions for this module
"""

__author__ = 'ajm'


def make_match_set(match_list):
    """ Takes a set of tuples (p, q) corresponding to matching pairs and returns a list of sets
        where each set corresponds to all matching combinations of p, q
    """
    matching_set_list = []

    for x, y in match_list:
        s = get_parent_set(matching_set_list,set([x,y]))
        if s:
            s.add(x)
            s.add(y)
        else:
            s_new = set()
            s_new.add(x)
            s_new.add(y)
            matching_set_list.append(s_new)

    set_list = []

    while len(matching_set_list) > 0:
        s1 = matching_set_list.pop()
        not_matched = []
        set_list.append(s1)
        while len(matching_set_list) > 0:
            s = matching_set_list.pop()
            if s1.isdisjoint(s):
                not_matched.append(s)
            else:
                s1.union(s)

        matching_set_list = not_matched

    return set_list

def get_parent_set(set_list, set_cmp):
    for s in set_list:
        intersect = (s & set_cmp)
        if len(intersect) != 0:
            return s
    return None