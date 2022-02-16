'''
leap_seconds.py

Routines for computing the number of elapsed leap seconds

Note:
Loading this file will attempt to download the latest set of leap seconds from the internet.
It will store them in the file `leap_seconds.txt` inside of this directory.

@author Brian Breitsch
@email brian.breitsch@colorado.edu
'''

from datetime import datetime, timedelta, timezone
try:
    from urllib.request import urlretrieve  # python 3
except ImportError:
    from urllib2 import urlretrieve  # python 2
from os.path import isfile, dirname, join
from collections import namedtuple

OffsetEpoch = namedtuple('OffsetEpoch', ['epoch', 'offset'])

NTP_EPOCH = datetime(year=1900, month=1, day=1, hour=0, minute=0, second=0, tzinfo=timezone.utc)
LEAP_SECOND_EPOCHS = []

def download_tai_leap_seconds(filepath):
    leap_seconds_list_url = 'http://www.ietf.org/timezones/data/leap-seconds.list'
    leap_seconds_data = urlretrieve(leap_seconds_list_url, filepath)

def parse_tai_leap_seconds(filepath):
    leap_seconds_epochs = []
    with open(filepath, 'r') as leap_seconds_data:
        for line in leap_seconds_data.readlines():
            line = line.decode('utf-8') if isinstance(line, bytes) else line
            if line.startswith('#$'):
                file_update_ntp = int(line.split()[1])
            elif line.startswith('#@'):
                file_expiration_ntp = int(line.split()[1])
            elif line.startswith('#') or line == '': 
                # if line is comment or blank, ignore
                continue
            else:
                ntp_timestamp = int(line.split()[0])
                offset = int(line.split()[1])
                epoch = NTP_EPOCH + timedelta(seconds=ntp_timestamp)
                leap_seconds_epochs.append(OffsetEpoch(epoch, offset))
    return leap_seconds_epochs

_leap_seconds_file = join(dirname(__file__), './leap_second_epochs.txt')
def update_leap_seconds_data():
    download_tai_leap_seconds(_leap_seconds_file)
    LEAP_SECOND_EPOCHS = parse_tai_leap_seconds(_leap_seconds_file)

# need to download/parse leap second epochs if not already done
if not LEAP_SECOND_EPOCHS:
    if not isfile(_leap_seconds_file):
        download_tai_leap_seconds(_leap_seconds_file)
        print('downloaded leap seconds file')
    LEAP_SECOND_EPOCHS = parse_tai_leap_seconds(_leap_seconds_file)
    # print('there are {0} leap second epochs'.format(len(LEAP_SECOND_EPOCHS)))

def utc_tai_offset(time):
    '''
    Calculates the offset (number of leap seconds) between a
    given time and TAI. If `time` is before the first leap
    seconds were introduced in 1972, returns 10--which is the
    original offset introduced in 1972. Otherwise, returns 
    the offset corresponding to the last offset before
    `time`.

    Note: `time` must be timezone aware.
    
    input
    -----
    time: datetime
        the time for which to find leap seconds
    
    output
    ------
    offset: timedelta
        the total leap second offset
    '''
    for i in range(len(LEAP_SECOND_EPOCHS)):
        if LEAP_SECOND_EPOCHS[i].epoch > time:
            offset = LEAP_SECOND_EPOCHS[i-1].offset if i > 0 else LEAP_SECOND_EPOCHS[0].offset
            return timedelta(seconds=offset)
    return timedelta(seconds=LEAP_SECOND_EPOCHS[-1].offset)
