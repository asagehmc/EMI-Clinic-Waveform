from datetime import datetime
from edfio import read_edf
import sleepecg
from matplotlib import pyplot as plt


def graph_sleep_stages(path):
    edf = read_edf(path)

    # crop dataset (we only want data for the sleep duration)
    # start = datetime(2023, 3, 1, 23, 0, 0)
    # stop = datetime(2023, 3, 2, 6, 0, 0)
    rec_start = datetime.combine(edf.startdate, edf.starttime)
    # edf.slice_between_seconds((start - rec_start).seconds, (stop - rec_start).seconds)

    # get ECG time series and sampling frequency
    ecg = edf.get_signal("ECG").data
    fs = edf.get_signal("ECG").sampling_frequency

    # detect heartbeats
    beats = sleepecg.detect_heartbeats(ecg, fs)
    sleepecg.plot_ecg(ecg, fs, beats=beats)

    # load SleepECG classifier (requires tensorflow)
    clf = sleepecg.load_classifier("wrn-gru-mesa-weighted", "SleepECG")

    # predict sleep stages
    record = sleepecg.SleepRecord(
        sleep_stage_duration=30,
        recording_start_time=rec_start,
        heartbeat_times=beats / fs,
    )

    stages = sleepecg.stage(clf, record, return_mode="prob")

    sleepecg.plot_hypnogram(
        record,
        stages,
        stages_mode=clf.stages_mode,
        merge_annotations=True,
    )
    plt.show()


if __name__ == "__main__":
    # path = "data/p000020/out/0005.edf"
    path = "data/test_sleep.edf"
    graph_sleep_stages(path)
