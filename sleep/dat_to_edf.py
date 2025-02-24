import wfdb
import pyedflib
import numpy as np
from datetime import datetime, date
from wfdb.io.convert.edf import read_edf


def convert_ecg_wfdb_to_edf(dat_file_path, output_edf_path, ecg_channel_name='II', block_size=1000):
    """
    Convert specifically the ECG channel from a WFDB file to EDF format.

    Parameters:
    dat_file_path (str): Path to the .dat file (without extension)
    output_edf_path (str): Path where the EDF file should be saved
    ecg_channel_name (str): Name of the ECG channel to extract (default: 'II')
    block_size (int): Number of samples to write at once (default: 1000)

    Returns:
    bool: True if conversion was successful, False otherwise
    """
    # Read the WFDB record
    record = wfdb.rdrecord(dat_file_path)

    # Find the ECG channel
    try:
        ecg_index = record.sig_name.index(ecg_channel_name)
    except ValueError:
        print(f"ECG channel '{ecg_channel_name}' not found. Available channels: {record.sig_name}")
        return False

    # Extract ECG data
    ecg_data = record.p_signal[:, ecg_index].astype(np.float64)
    data_length = len(ecg_data)

    # Create EDF file with single channel
    edf_writer = pyedflib.EdfWriter(
        output_edf_path,
        n_channels=1,
        file_type=pyedflib.FILETYPE_EDFPLUS
    )

    # Prepare channel information for ECG
    ch_dict = {
        'label': 'ECG',
        'dimension': 'mV',
        'sample_frequency': record.fs,
        'physical_max': np.max(ecg_data),
        'physical_min': np.min(ecg_data),
        'digital_max': 32767,
        'digital_min': -32768,
        'transducer': '',
        'prefilter': ''
    }

    # Get record header
    record_header = record.__dict__

    # Set header information

    header = {
        'technician': '',
        'recording_additional': '',
        'patientname': record_header.get('record_name', ''),
        'patient_additional': '',
        'patientcode': '',
        'equipment': 'WFDB',
        'admincode': '',
        'sex': "",
        'startdate': datetime.combine(date(2000, 1, 1), record.base_time),
        'birthdate': ''
    }

    # Set all header properties
    edf_writer.setHeader(header)
    edf_writer.setSignalHeader(0, ch_dict)

    # Write the ECG data
    try:
        # pyedflib expects a list of signals, so we need to pass the data as a list with one signal
        edf_writer.writeSamples([ecg_data])

    except Exception as e:
        print(f"Error writing data: {str(e)}")
        edf_writer.close()
        return False

    edf_writer.close()
    return True


def another_conversion_try(path):
    read_edf(path, pn_dir=None, header_only=False, verbose=False, rdedfann_flag=False,
                                 encoding='iso8859-1')


def print_record_info(dat_file_path):
    """
    Print information about the WFDB record to help with debugging
    """
    try:
        record = wfdb.rdrecord(dat_file_path)
        print("\nRecord Information:")
        print(f"Number of signals: {record.n_sig}")
        print(f"Signal names: {record.sig_name}")
        print(f"Sample frequency: {record.fs}")
        print(f"Number of samples: {record.sig_len}")
        print(f"Units: {record.units}")
        return record
    except Exception as e:
        print(f"Error reading record: {str(e)}")
        return None


# Example usage
if __name__ == "__main__":

    # Example for converting only ECG channel
    #num = 5
    #path = f"data/test/3544749_000{num}"
    #out_path = f"data/test/000{num}.edf"
    num = 14
    path = f"data/test/3278512_00{num}"
    out_path = f"data/test/000{num}.edf"

    print_record_info(path)
    # another_conversion_try(path)
    convert_ecg_wfdb_to_edf(path, out_path, ecg_channel_name='I')