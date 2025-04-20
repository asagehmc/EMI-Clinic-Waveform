In order to run the mesa pipeline (mesa_pipeline.py), a few files need to added that are too large to store in the
repo on their own.

To run the code, please download the following files:
    ApproximateNetwork.h5 https://drive.google.com/file/d/1R0t3VxPLBpQmIKKyH9ulDdr9irwsyOpD/view?usp=sharing
    RefinementNetwork.h5 https://drive.google.com/file/d/1qc97paeXHDWrOEsPR1_2tl7WtMcJxQbF/view?usp=sharing
Place these files in the directory PPGtoBP/PPG_model/bloodPressureModel/models

You will also need a token.txt file to run the mesa download. This file should be a 1-line file containing only the NSRR
token which can be accessed by logging in at
    https://sleepdata.org/token
Place this file in the PPGtoBP directory (same level as this README)