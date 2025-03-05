import h5py

if __name__ == "__main__":
    # fp = "ecg-sleep-staging-main/your_own_data/primary_model/results.h5"
    fp = "local_data/out/p000801_3054941_0029.h5"
    with h5py.File(fp, 'r') as f:
        print("Keys: %s" % f.keys())

        data = f['predictions'][:]
        confusions = f['confusions'][:]

        print(data.tolist())
        print(confusions.tolist())





