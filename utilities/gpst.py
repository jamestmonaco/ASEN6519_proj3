'''
gpst.py

Routines for converting time to/from GPS seconds

@author Brian Breitsch
@email brian.breitsch@colorado.edu
'''

from numpy import modf
from datetime import datetime, timedelta, timezone
from .leap_seconds import utc_tai_offset

GPS_EPOCH = datetime(year=1980, month=1, day=6, hour=0, minute=0, second=0, tzinfo=timezone.utc)
GPS_TAI_OFFSET = utc_tai_offset(GPS_EPOCH)
SECONDS_IN_WEEK = 3600 * 24 * 7 


def dt2gpst(dt):
    ''' 
    Computes GPST from `datetime` object.

    Generally GPS time is specified any of the following:
     - total seconds (without leap) since GPS epoch
     - week number and day of week
     - week number and seconds into week

    This function returns the total number of seconds since the GPST epoch.

    Input
    -----
    time: datetime
        the time to convert to GPS time
    '''
    if not hasattr(dt, 'tzinfo') or dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    time_gps_offset = utc_tai_offset(dt) - GPS_TAI_OFFSET
    timedelta = dt - GPS_EPOCH + time_gps_offset
    return timedelta.total_seconds()

def gpst2dt(seconds, week_no=None, rollover=0):
    ''' 
    Returns a UTC datetime object given the GPS week
    number and/or number of seconds.

    input
    -----
    seconds: float
        if `seconds` is the only argument (i.e. `week_no` is
        `None`) then it is taken to be the total number of
        seconds since the GPS epoch on 6 January 1980.
        Otherwise, it is the number of seconds plus 
        fractional seconds since the last GPS week epoch
    week_no: int
        GPS week number
    rollover: int
        Number of GPS week rollovers to include (i.e. multiples of 1024 added to GPS week)
        Defaults to zero.

    output
    ------
        the UTC datetime object corresponding to the GPS
        time input
    '''
    seconds, microseconds = int(seconds), int((seconds % 1) * 1e6)
    total_seconds = seconds + (week_no + 1024 * rollover) * SECONDS_IN_WEEK if week_no else seconds
    ## TODO this is a tricky function
    time = GPS_EPOCH + timedelta(seconds=total_seconds, microseconds=microseconds)
    return time + GPS_TAI_OFFSET - utc_tai_offset(time)




def gpst_seconds(week_no, tow, rollover=0):
    '''Returns the GPS time in seconds since GPS epoch (Jan 6, 1980) given
    the week number and time of week in seconds. Also accepts
    `rollover` argument (default 0).
    '''
    return (week_no + rollover * 1024) * SECONDS_IN_WEEK + tow

def gpst_week_and_fow(gpst_seconds):
    'Returns GPS week number and fraction of the week'
    frac_week, week = modf(gpst_seconds / SECONDS_IN_WEEK)
    return int(week), frac_week

def gpst_week_number(gpst_seconds):
    'Returns GPS week number (see e.g. `gpst_week_and_fow`)'
    return gpst_week_and_fow(gpst_seconds)[0]

def gpst_week_seconds(gpst_seconds):
    'Returns GPS week seconds'
    return gpst_week_and_fow(gpst_seconds)[1] * SECONDS_IN_WEEK

def gpst_week_day(gpst_seconds):
    'Returns GPS day of week'
    return gpst_week_and_fow(gpst_seconds)[1] * 7

