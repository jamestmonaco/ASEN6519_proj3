

import os
import numpy

def get_numpy_dtype(bit_depth, is_signed, is_integer, is_complex):
    '''
    Determines and returns the appropriate numpy datatype according to the input parameters.  If the datatype is
    unsupported in numpy. then returns None.
    --------------------------------------------------------------------------------------------------------------------
    
    Inputs:
        `bit_depth` -- the bits per value.
        `is_signed` -- whether or not the type is signed
        `is_integer` -- whether the datatype is integer- versus float-valued
        `is_complex` -- whether the datatype is complex-valued versus real-valued
    
    Note: numpy does not have native complex integer datatypes, so this only works for real integer datatypes or complex
    float types
    
    Outputs:
        numpy dtype object or None
    '''
    if is_complex:
        if is_signed and not is_integer:
            signed_complex_types = {32: numpy.complex64, 64: numpy.complex128, 128: numpy.complex256}
            return signed_complex_types.get(bit_depth, None)
        else:
            return None
    else:
        # is real
        if is_integer:
            signed_integer_types = {8: numpy.int8, 16: numpy.int16, 32: numpy.int32, 64: numpy.int64}
            unsigned_integer_types = {8: numpy.uint8, 16: numpy.uint16, 32: numpy.uint32, 64: numpy.uint64}
            if is_signed:
                return signed_integer_types.get(bit_depth, None)
            else:
                return unsigned_integer_types.get(bit_depth, None)
        elif is_signed:
            # is real, signed, float-valued
            signed_float_types = {16: numpy.float16, 32: numpy.float32, 64: numpy.float64, 128: numpy.float128}
            return signed_float_types.get(bit_depth, None)
    return None

def convert_bytes_as_4bit_signed_real(byte_buffer, i_lsb=True):
    byte_buffer = numpy.frombuffer(byte_buffer, dtype=numpy.byte)
    msn = byte_buffer.astype(numpy.int8) >> 4
    lsn = ((byte_buffer & 0xf) << 4).astype(numpy.int8) >> 4
    return numpy.vstack((lsn, msn) if i_lsb else (msn, lsn)).reshape((-1,), order='F')

def convert_bytes_as_4bit_unsigned_real(byte_buffer, i_lsb=True):
    byte_buffer = numpy.frombuffer(byte_buffer, dtype=numpy.byte)
    msn = byte_buffer.astype(numpy.uint8) >> 4
    lsn = ((byte_buffer & 0xf) << 4).astype(numpy.uint8) >> 4
    return numpy.vstack((lsn, msn) if i_lsb else (msn, lsn)).reshape((-1,), order='F')

def convert_bytes_as_4bit_signed_complex(byte_buffer, i_lsb=True):
    byte_buffer = numpy.frombuffer(byte_buffer, dtype=numpy.byte)
    msn = byte_buffer.astype(numpy.int8) >> 4
    lsn = ((byte_buffer & 0xf) << 4).astype(numpy.int8) >> 4
    return 1j * msn + lsn if i_lsb else msn + 1j * lsn

def convert_bytes_as_4bit_unsigned_complex(byte_buffer, i_lsb=True):
    byte_buffer = numpy.frombuffer(byte_buffer, dtype=numpy.byte)
    msn = byte_buffer.astype(numpy.uint8) >> 4
    lsn = ((byte_buffer & 0xf) << 4).astype(numpy.uint8) >> 4
    return 1j * msn + lsn if i_lsb else msn + 1j * lsn

def convert_bytes_as_complex_dtype(byte_buffer, dtype):
    # parse integer real type and conver to complex
    buffer = numpy.frombuffer(byte_buffer, dtype=dtype)
    N = (len(buffer) // 2) * 2
    return buffer[:N].astype(numpy.float32).view(numpy.complex64)  # <- mildly faster

def get_bytes_to_samples_converter(bit_depth, is_signed, is_integer, is_complex, i_lsb):
    '''
    Returns a function that converts a bytebuffer to a 1D sample array according to the datatype specified by the input
    arguments.
    --------------------------------------------------------------------------------------------------------------------
    Inputs:
        `bit_depth` -- the number of consecutive bits used to define one sample component.  For real samples this is
            the size of each sample in bits, whereas complex samples occupy twice the space.
        `is_signed` -- whether or not the sample components are signed types
        `is_integer` -- whether or not the sample components are integer types
        `is_complex` -- whether the samples are complex-valued versus real.  Complex samples occupy twice the space specified
            by `bit_depth`
        `i_lsb` -- whether or not complex samples have the I component in the least-significant or most significant
            bytes
    
    Outputs:
        function that converts an array of bytes to an array of samples
    '''
    # check if datatype is valid numpy dtype; if so, just use `numpy.frombuffer`
    dtype = get_numpy_dtype(bit_depth, is_signed, is_integer, is_complex)
    if dtype is not None:
#         print('Using numpy dtype: {0}'.format(dtype))
        return lambda byte_buffer: numpy.frombuffer(byte_buffer, dtype=dtype)
    # otherwise, handle special cases
    elif bit_depth == 4:
        # this is a good candidate for using "match" syntax once we upgrade python versions
        if is_complex:
            if is_signed:
                return lambda byte_buffer: convert_bytes_as_4bit_signed_complex(byte_buffer, i_lsb)
            else:
                return lambda byte_buffer: convert_bytes_as_4bit_unsigned_complex(byte_buffer, i_lsb)
        else:
            if is_signed:
                return lambda byte_buffer: convert_bytes_as_4bit_signed_real(byte_buffer, i_lsb)
            else:
                return lambda byte_buffer: convert_bytes_as_4bit_unsigned_real(byte_buffer, i_lsb)
    elif is_complex:
        component_dtype = get_numpy_dtype(bit_depth, is_signed, is_integer, False)
        if component_dtype is not None:
            return lambda byte_buffer: convert_bytes_as_complex_dtype(byte_buffer, component_dtype)
    raise Exception('Unsupported sample parameters: {0}-bit {1}signed {2}'.format(
        bit_depth, '' if is_signed else 'un', 'complex' if is_complex else 'real'))


def compute_bytes_per_sample(bit_depth, is_complex):
    '''
    Given the bit_depth and whether samples are real or complex-valued, determines the number of bytes per sample.
    --------------------------------------------------------------------------------------------------------------------
    '''
    bytes_per_sample = (bit_depth * (2 if is_complex else 1)) // 8
    return bytes_per_sample

def generate_sample_block(f, start_sample, block_size, bytes_per_sample, convert_bytes_to_samples_function):
    '''
    Given the a file stream handle, returns a block of samples from the file that starts at `block_start_time` and lasts
    `block_duration` seconds.
    --------------------------------------------------------------------------------------------------------------------
    '''
    block_start_byte = start_sample * bytes_per_sample
    block_byte_buffer = bytearray(block_size * bytes_per_sample)
    f.seek(block_start_byte)
    f.readinto(block_byte_buffer)
    return convert_bytes_to_samples_function(block_byte_buffer)

class SampleLoader:
    
    def __init__(self, samp_rate, bit_depth, is_signed=True, is_integer=True, is_complex=False, i_lsn=True):
        '''
        Input: specify
            `samp_rate' -- float, the uniform rate of samples
            'bit_depth' -- int, number of bits per numeric value (so an entire complex sample require `2*bit_depth` bits)
            'is_signed' -- bool (default True), whether the numeric types are signed (versus unsigned)
            'is_integer' -- bool (default True), whether the numeric types are integer (versus float)
            'is_complex' -- bool (default False), whether the samples are complex-valued (versus real-valued)
            'i_lsn' -- bool (default True), whether (for 4-bit samples) the I component is in the least significant nibble
        '''
        self.samp_rate = samp_rate
        self.bit_depth = bit_depth
        self.is_signed = is_signed
        self.is_integer = is_integer
        self.is_complex = is_complex
        self.i_lsn = i_lsn
        self.bytes_per_sample = compute_bytes_per_sample(bit_depth, is_complex)
        self.convert_bytes_to_samples = get_bytes_to_samples_converter(bit_depth, is_signed, is_integer, is_complex, i_lsn)
        self.generate_sample_block = lambda f, start_sample, block_size: \
            generate_sample_block(f, start_sample, block_size, self.bytes_per_sample, self.convert_bytes_to_samples)

def get_file_source_info(**kwargs):
    '''
    Input: specify
   
        `samp_rate' -- float, the uniform rate of samples
        'bit_depth' -- int, number of bits per numeric value (so an entire complex sample require `2*bit_depth` bits)
        'is_signed' -- bool (default True), whether the numeric types are signed (versus unsigned)
        'is_integer' -- bool (default True), whether the numeric types are integer (versus float)
        'is_complex' -- bool (default False), whether the samples are complex-valued (versus real-valued)
        'i_lsn' -- bool (default True), whether (for 4-bit samples) the I component is in the least significant nibble
        'start_time' -- float (default 0), the time of the first sample in the file    
    --------------------------------------------------------------------------------------------------------------------
    Returns: `file_source_info` dict containing all the parameters in `file_source_params` along with:
        `file_size` -- the size of the file in bytes
        `bytes_per_sample` -- the number of bytes per sample in the file
        `file_length` -- the size of the file in samples
        `file_duration` -- the duration of the file in seconds
    '''
    # Required parameters
    filepath, samp_rate, bit_depth = (kwargs[k] for k in ['filepath', 'samp_rate', 'bit_depth',])
    
    # Optional parameters
    is_signed = kwargs.get('is_signed', True)
    is_integer = kwargs.get('is_integer', True)
    is_complex = kwargs.get('is_complex', False)
    i_lsn = kwargs.get('i_lsn', True)
    start_time = kwargs.get('start_time', 0.)
    
    file_size = os.path.getsize(filepath)
    bytes_per_sample = compute_bytes_per_sample(bit_depth, is_complex)
    file_length = int(file_size / bytes_per_sample)
    file_duration = file_length / samp_rate
    
    params = {
        'bit_depth': bit_depth,
        'is_signed': is_signed,
        'is_integer': is_integer,
        'is_complex': is_complex,
        'i_lsn': i_lsn,
        'start_time': start_time,
        'file_size': file_size,
        'bytes_per_sample': bytes_per_sample,
        'file_length': file_length,
        'file_duration': file_duration,
    }
    params.update(kwargs)
    return params