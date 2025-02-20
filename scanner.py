import json
from traceback import print_exc

import wfdb
import requests
import glob
import os

PHYSIONET = "https://physionet.org/files/"
MIMIC = "mimic3wdb-matched/1.0/"
# Manual Download: https://physionet.org/files/mimic3wdb-matched/1.0/


sample_threshold = 1024
required_signals = ["ABP", "I"]

if __name__ == "__main__":
    # a dictionary of the valid (large enough, have the right data lines) segments of data, and their sizes
    usable_records = {}

    # get the list of patients from the db
    record_list = wfdb.get_record_list(MIMIC)

    # for showing percentages while running
    num_records_handled = 0
    num_queries_made = 0
    for patient in record_list:
        try:
            # get the list of records for an individual patient
            patient_records = wfdb.get_record_list(f"{MIMIC}{patient}")
            # get a list of the master headers:
            # NOTE: This is suuper ugly but to this point I don't know a better way to sort TODO figure out a better way
            master_header_names = [f for f in patient_records if f.startswith('p') and not f.endswith('n')]

            for header_name in master_header_names:
                # convert the patient master header into a MultiRecord
                header = wfdb.rdheader(f"{header_name}", rd_segments=True, pn_dir=f"{MIMIC}{patient}")

                try:
                    for segment in header.segments:
                        if segment is not None and "layout" in segment.record_name:
                            # we can skip all segment queries if the layout doesn't even have the right signals
                            has_required_signals = not (False in [(x in header.sig_name) for x in required_signals])
                            if not has_required_signals:
                                break
                            print("HAS REQUIRED SIGNALS!")
                        elif segment is not None:
                            # get only segments that aren't generated from the layout header, and are long enough
                            has_required_signals = not (False in [(x in segment.sig_name) for x in required_signals])
                            if has_required_signals and segment.sig_len > sample_threshold:

                                # make filepath and get file size in bytes
                                data_url = f"{PHYSIONET}{MIMIC}{patient}{segment.record_name}.dat"
                                data_response = requests.head(data_url)
                                # query file size
                                size = int(data_response.headers.get('content-length', 0)) if data_response.ok else -1

                                # add to usable_records the current subsection of data
                                if patient not in usable_records:
                                    usable_records[patient] = []
                                usable_records[patient].append({
                                    "id": segment.record_name,
                                    "size": size,
                                })
                                print("FOUND:", segment.record_name)
                                with open(f"filtered_{'_'.join(required_signals)}.txt", "w") as f:
                                    json.dump(usable_records, f, indent=2)
                            elif has_required_signals:
                                print(f"record {segment.record_name} has the required signals but is too short")

                except Exception as e:
                    print(f"Problem with [patient: {patient}, in subsegment]")
                    print(print_exc(e))

            # reporting
            print(patient)


        except Exception as e:
            print(f"Problem with [patient: {patient}]")

