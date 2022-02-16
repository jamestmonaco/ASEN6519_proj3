'''
sp3.py

Utilities for SP3 data file download and parsing

@author Brian Breitsch
@email brian.breitsch@colorado.edu
'''

import os
from functools import lru_cache

import numpy
from numpy import searchsorted, alltrue, polyfit, polyval, nanmean, nan, zeros, diff, isnan, nanmax
from numpy import nan, zeros, argsort, alltrue, concatenate, diff

import scipy.interpolate

from datetime import datetime, timezone, timedelta

from .data_utils import cddis_download, format_filepath, decompress
from .gpst import gpst2dt, dt2gpst, gpst_week_and_fow

from functools import lru_cache

import warnings

GPS_EPOCH = datetime(year=1980, month=1, day=6, hour=0, minute=0, second=0, tzinfo=timezone.utc)
ONE_HOUR = timedelta(hours=1)
ONE_DAY = timedelta(days=1)


def parse_sp3_header(header_lines):

    header = {}

    line = header_lines[0]
    header['version'] = line[:2]
    header['position_velocity_flag'] = line[2]
    header['start_time'] = _sp3_strptime(line[3:31])
    header['number_of_epochs'] = int(line[32:39])
    header['data_used'] = line[40:45].strip()
    header['coordinate_sys'] = line[46:51]
    header['orbit_type'] = line[52:55]
    header['agency'] = line[56:60].strip()

    line = header_lines[1]
    header['gps_week'] = int(line[3:7])
    header['seconds_of_week'] = float(line[8:23])
    header['epoch_interval'] = float(line[24:38])
    header['mod_jul_day_start'] = int(line[39:44])
    header['fractional_day'] = float(line[45:60])

    return header


def _sp3_test_nan(x, nan_value=999999.999999, eps=1e-3):
    '''Tests if `x` is NaN according to SP3 spec, i.e. is within `eps` of `nan_value`'''
    return abs(x - nan_value) < eps 


def _sp3_strptime(timestr):
    ''' 
    Takes SP3 file epoch date string format and returns GPS seconds.
    Because datatime only tracks up to microsecond accuracy, we cannot use 
    the last 2 digits in the seconds decimal.  We will throw an error if the
    last two digits are not 0.  Also, the times in SP3 files are given in GPS time, even
    thought the format is YYYY MM DD HH MM SS.  This means that if we subtract
    the GPS epoch using two UTC datetimes, we'll get the correct time in GPS
    seconds (note, datetime is not leap-second aware, which is why this works).
    '''
    if int(timestr[26:]) != 0:
        raise Exception('`datetime` cannot handle sub-microsecond precision, but epoch in file appears to specify this level of precision.')
    time = datetime.strptime(timestr[:26], '%Y %m %d %H %M %S.%f').replace(tzinfo=timezone.utc)
    return (time - GPS_EPOCH).total_seconds()  # GPS time


def _sp3_parse_position_and_clock(line):
    '''
    Returns <vehicle id>, <x-coordinate>, <y-coordinate>, <z-coordinate>, <clock>
    x, y, z coordinates are in units of km and clock offset is in units of microseconds
    '''
    veh_id, x, y, z, c = line[1:4], float(line[4:18]), float(line[18:32]), float(line[32:46]), float(line[46:60])
    x = nan if _sp3_test_nan(x) else x * 1e3
    y = nan if _sp3_test_nan(y) else y * 1e3
    z = nan if _sp3_test_nan(z) else z * 1e3  # convert from km to m
    clock = nan if _sp3_test_nan(c) else c
    return veh_id, x, y, z, clock

def _sp3_parse_velocity_and_clock(line):
    '''
    Returns <vehicle id>, <x-velocity>, <y-velocity>, <z-velocity>, <clock-rate-change>
    x, y, z velocities are in units of dm/s and clock rate is in units of s/s
    '''
    return _sp3_parse_position_and_clock(line)


def parse_sp3_records(record_lines, parse_position=True, parse_velocity=False):
    epochs = []
    records = []
    for line in record_lines:
        if line.startswith('*'):
            epochs.append(_sp3_strptime(line[2:].strip()))
            records.append({})
        elif line.startswith('P') and parse_position:
            veh_id, x, y, z, c = _sp3_parse_position_and_clock(line)
            records[-1][veh_id] = (x, y, z, c)
        elif line.startswith('V') and parse_velocity:
            veh_id, x, y, z, c = _sp3_parse_velocity_and_clock(line)
            records[-1][veh_id] = (x, y, z, c)
    return epochs, records


@lru_cache(maxsize=10)
def parse_sp3_file(filepath):
    '''
    Parse a single SP3 file.
    
    Returns `header, epochs, records`
    '''
    lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if not lines:
        raise Exception('File was empty')
    header_lines = lines[:22]  # there should always be 22 header lines
    record_lines = lines[22:]
    header = parse_sp3_header(header_lines)
    epochs, records = parse_sp3_records(record_lines)
    return header, epochs, records


def parse_sp3_data(filepaths):
    '''
    Parse and merge multiple SP3 files.

    Returns:
    `all_epochs, data`

        `all_epochs` -- GPST epochs of SP3 records
        `data` -- dict with vehicle IDs as keys and (N, 4) ndarray of records
    '''
    # First, parse everything
    headers, epochs_list, records_list = [], [], []
    veh_ids = set()
    for filepath in filepaths:
        header, epochs, records = parse_sp3_file(filepath)
        headers.append(header)
        epochs_list.append(epochs)
        records_list.append(records)

        for record in records:
            veh_ids = veh_ids | set(record.keys())

    veh_ids = sorted(list(veh_ids))

    all_epochs = numpy.sort(numpy.unique(numpy.concatenate(epochs_list)))
    N = len(all_epochs)
    data = {veh_id: nan * zeros((N, 4)) for veh_id in veh_ids}
    for epochs, records in zip(epochs_list, records_list):
        indices = numpy.searchsorted(all_epochs, epochs)
        for index, record in zip(indices, records):
            for veh_id in record.keys():
                data[veh_id][index, :] = record[veh_id]

    return all_epochs, data


def download_and_decompress_sp3_file(dt, data_dir, overwrite=False):
    '''
    Automatically downloads MGEX SP3 file for datetime `dt`
    Returns the filepath to the downloaded and decompressed SP3 file.
    '''
    gpst = dt2gpst(dt)
    week_no, week_day = gpst_week_and_fow(gpst)
    if week_no >= 1962:
        # if greater than or equal to GPS week 1962, then use new file format
        filepath_template = 'pub/gps/products/mgex/{wwww}/COD0MGXFIN_{yyyy}{ddd}0000_01D_05M_ORB.SP3.gz'
        decompressed_filepath_template = 'pub/gps/products/mgex/{wwww}/COD0MGXFIN_{yyyy}{ddd}0000_01D_05M_ORB.SP3'
    else:
        # else use old file format
        filepath_template = 'pub/gps/products/mgex/{wwww}/com{wwww}{d}.sp3.Z'
        decompressed_filepath_template = 'pub/gps/products/mgex/{wwww}/com{wwww}{d}.sp3'
    filepath = format_filepath(filepath_template, dt)
    decompressed_filepath = format_filepath(decompressed_filepath_template, dt)
    
    output_dir = os.path.join(data_dir, 'igs/')
    output_filepath = os.path.join(output_dir, filepath)
    decompressed_filepath = os.path.join(output_dir, decompressed_filepath)
    if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath))
    downloaded = decompressed = False
    if overwrite or not (os.path.exists(output_filepath) or os.path.exists(decompressed_filepath)):
        downloaded = cddis_download(filepath, output_filepath)
    if os.path.exists(output_filepath):
        decompressed = decompress(output_filepath, decompressed_filepath)
    
    if os.path.exists(decompressed_filepath):
        return decompressed_filepath
    elif os.path.exists(output_filepath):
        return output_filepath
    else:
        return False


def download_and_parse_sp3_data(start_time_gpst, end_time_gpst, data_dir):
    '''
    `start_time_gpst` -- earliest time at which SP3 data is required
    `end_time_gpst` -- latest time at which SP3 data is required
    `data_dir` -- root directory from which SP3 data is stored
    `sat_ids` -- the satellite IDs corresponding to the satellites for which splines should
        be computed.  If `None` (default), computes for all satellites in the SP3 files
    '''
    day_start = gpst2dt(start_time_gpst).replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = gpst2dt(end_time_gpst).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=24)

    sp3_filepaths = []
    for i in range(-3, 3):
        day = day_start + timedelta(days=i)
        try:
            sp3_filepath = download_and_decompress_sp3_file(day, data_dir)
            sp3_filepaths.append(sp3_filepath)
        except Exception as e:
            print('Failed to download/decompress SP3 filepath for day: {0}'.format(day.strftime('%Y%m%d')))
            raise e
    
    epochs, records = parse_sp3_data(sp3_filepaths)
    
    return epochs, records


def compute_sp3_splines(epochs, records, order=5):
    '''
    For each SP3 record, computes a spline
    '''
    splines = {}
    for sat_id, rec in records.items():
        # Compute univariate spline for each record
        splines[sat_id] = [scipy.interpolate.UnivariateSpline(epochs, rec[:, i], k=order) for i in range(4)]
    return splines


def compute_sp3_positions(times, sp3_splines):
    '''
    Given the splines computed for a set of SP3 records, evaluates each spline at `times`
    '''
    interpolated = {}
    for sat_id, splines in sp3_splines.items():
        interpolated[sat_id] = numpy.stack([spline(times) for spline in splines[:3]]).T
    return interpolated


def compute_sp3_velocities(times, sp3_splines):
    '''
    Given the splines computed for a set of SP3 records, obtains the derivative spline and then evaluates it at `times`
    '''
    interpolated = {}
    for sat_id, splines in sp3_splines.items():
        interpolated[sat_id] = numpy.stack([spline.derivative()(times) for spline in splines[:3]]).T
    return interpolated


def compute_satellite_ecf_positions(times, data_dir, order=5, sat_ids=None):
    '''
    `times` -- times (GPS seconds) for which to compute satellite positions
    `data_dir` -- the root directory for storing SP3 data
    '''
    start_time_gpst, end_time_gpst = times[[0, -1]]
    epochs, records = download_and_parse_sp3_data(start_time_gpst, end_time_gpst, data_dir)
    
    if sat_ids is None:
        sat_ids = records.keys()
    else:
        records = {k:records[k] for k in sat_ids if k in records.keys()}
    
    sp3_splines = compute_sp3_splines(epochs, records, order)
    sp3_positions = compute_sp3_positions(times, sp3_splines)
    
    interpolated = {}
    for sat_id in sat_ids:
        if sat_id not in sp3_splines:
            continue
        ecf_pos = sp3_positions[sat_id]
        if numpy.all(numpy.isnan(ecf_pos)):
            continue
        interpolated[sat_id] = ecf_pos
    
    return interpolated