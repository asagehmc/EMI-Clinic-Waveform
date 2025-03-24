import wfdb
import pyedflib
import numpy as np
from datetime import datetime, date


def convert_ecg_wfdb_to_edf(dat_file_path, output_edf_path, ecg_channel_input_name='II',
                            ecg_channel_output_name='ECG', block_size=1000):
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
    try:
        record = wfdb.rdrecord(dat_file_path)
    except Exception as e:
        print(f"Error reading WFDB record: {str(e)}")
        return False

    # Find the ECG channel
    try:
        ecg_index = record.sig_name.index(ecg_channel_input_name)
    except ValueError:
        print(f"ECG channel '{ecg_channel_input_name}' not found. Available channels: {record.sig_name}")
        return False

    # Extract ECG data and ensure it's properly scaled
    ecg_data = record.p_signal[:, ecg_index]

    # get an array of indices that are NaN
    empty_indexes = np.array(np.where(np.isnan(ecg_data)))[0]
    i = 0

    while i < len(empty_indexes):
        current_index = empty_indexes[i]
        lerp_size = 0
        while i < len(empty_indexes) and empty_indexes[i] == current_index + lerp_size:
            i += 1
            lerp_size += 1
        start = ecg_data[current_index - 1] if current_index > 0 else 0
        end = ecg_data[current_index + lerp_size] if current_index + lerp_size < len(ecg_data) else 0
        for lerp_num in range(lerp_size):
            ecg_data[current_index + lerp_num] = start + (end - start) * (lerp_num + 1) / (lerp_size + 1)

    # Calculate actual physical min/max from the data
    physical_max = np.max(ecg_data)
    physical_min = np.min(ecg_data)
    # print(np.where(np.isnan(ecg_data)))

    # Add small padding to avoid exact boundary issues
    physical_max += abs(physical_max) * 0.01
    physical_min -= abs(physical_min) * 0.01

    # Create EDF file
    try:
        edf_writer = pyedflib.EdfWriter(
            output_edf_path,
            n_channels=1,
            file_type=pyedflib.FILETYPE_EDFPLUS
        )

        # Prepare channel information
        ch_dict = {
            'label': ecg_channel_output_name,
            'dimension': record.units[ecg_index] if record.units else 'mV',
            'sample_frequency': record.fs,
            'physical_max': physical_max,
            'physical_min': physical_min,
            'digital_max': 32767,
            'digital_min': -32768,
            'transducer': 'ECG electrode',
            'prefilter': 'None'
        }

        # Set header information
        header = {
            'technician': 'Unknown',
            'recording_additional': '',
            'patientname': record.record_name,
            'patient_additional': '',
            'patientcode': '',
            'equipment': 'WFDB',
            'admincode': '',
            'sex': 0,  # Unknown
            'startdate': datetime.combine(date.today(),
                                          datetime.now().time()) if not record.base_time else datetime.combine(
                date.today(), record.base_time),
            'birthdate': date(1900, 1, 1)  # Default date
        }

        # Set headers
        edf_writer.setHeader(header)
        edf_writer.setSignalHeader(0, ch_dict)

        # Write data in chunks to avoid memory issues
        chunk_size = min(block_size, len(ecg_data))
        for i in range(0, len(ecg_data), chunk_size):
            chunk = ecg_data[i:i + chunk_size]
            if len(chunk) > 0:  # Ensure we're not writing empty chunks
                edf_writer.writeSamples([chunk])
        edf_writer.close()
        print(f"Successfully wrote EDF file to {output_edf_path}")
        return True

    except Exception as e:
        print(f"Error during EDF conversion: {str(e)}")
        if 'edf_writer' in locals():
            edf_writer.close()
        return False


if __name__ == "__main__":
    num = 5
    path = f"data/test/3544749_000{num}"
    out_path = f"data/test/000{num}.edf"

    # Print information about the record
    regenerate = True
    if regenerate:
        # Convert the file
        success = convert_ecg_wfdb_to_edf(path, out_path, ecg_channel_input_name='II')

        if success:
            # Verify the conversion by reading back the EDF file
            try:
                f = pyedflib.EdfReader(out_path)
                print(f"Number of signals: {f.signals_in_file}")
                print(f"Signal labels: {f.getSignalLabels()}")
                print(f"Sample frequency: {f.getSampleFrequency(0)}")
                n_samples = f.getNSamples()[0]
                print(f"Number of samples: {n_samples}")
                if n_samples > 0:
                    signal = f.readSignal(0)
                    print(f"Signal statistics:")
                    print(f"Mean: {np.mean(signal):.3f}")
                    print(f"Min: {np.min(signal):.3f}")
                    print(f"Max: {np.max(signal):.3f}")
                f.close()
            except Exception as e:
                print(f"Error verifying EDF file: {str(e)}")