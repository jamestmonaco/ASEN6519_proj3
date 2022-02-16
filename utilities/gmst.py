'''
gmst.py

Routine for converting time to GMST in hours

@author Brian Breitsch
@email brian.breitsch@colorado.edu
'''

from datetime import datetime, timezone
from numpy import asarray, ndarray

def time2gmst(time):
    '''Converts some time format into GMST (Greenwich Mean Standard Time) in units of hours.
    `time` could be of forms:
        ndarray -- each entry is interpreted as time in GPS seconds
        single float -- interpreted as GPS seconds
        datetime -- interpreted as UTC datetime
        list of datetime objects -- interpreted as UTC datetime
    '''
    if isinstance(time, ndarray) or isinstance(time, float):  # assume GPS seconds vector
        delta_days = (time - 630763213.0) / (3600 * 24) 
    elif isinstance(time, datetime):
        delta_days = (time - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds() / (3600 * 24) 
    elif isinstance(time, list):
        delta_days = asarray([(t - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds() / (3600 * 24) for t in time])
    return 18.697374558 + 24.065709824419 * delta_days
