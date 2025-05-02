# Waveform Repo EMI Clinic 24-25

## About

This repo contains the final code related to the major adverse cardiovascular event (MACE) risk classification 
model (RCMs) pipeline created and implemented by the 2024-2025 Harvey Mudd College clinic team sponsored by the Ellison 
Medical Institute. This repo contains the sleep stage estimation from MIMIC III's ECG data, the blood pressure 
estimation from MESA's PPG data, and the necessary functions to build different kinds of RCMs from aligned blood 
pressure and sleep stage data.

## File Structure
ECGtoSS contains the code for sleep stage estimation from MIMIC III's ECG data as well as the code for training and 
testing RCMs in the risk_classification folder.

PPGtoBP contains the code for blood pressure estimation from MESA's PPG data.

``` bash
├── ECGtoSS
│   ├── __init__.py
│   ├── data
│   │   ├── errors.txt
│   │   ├── mesa_aligned_data
│   │   │   ├── patient_aligned_data  (HOW TO INDICATE MULTIPLE OF THESE)
│   │   └── mimic_aligned_data
│   │       ├── patient_id_subsets (HOW TO INDICATE MULTIPLE OF THESE)
│   │       │   ├── patient_ids  (HOW TO INDICATE MULTIPLE OF THESE)
│   │       │   │   ├── patient_aligned_data  (HOW TO INDICATE MULTIPLE OF THESE)
│   ├── ecg_to_ss_full.py
│   ├── risk_classification
│   │   ├── __init__.py
│   │   ├── accuracy_results
│   │   │   ├── saved_results_data (HOW TO INDICATE MULTIPLE OF THESE)
│   │   ├── example_script.py
│   │   ├── mimic_data
│   │   │   ├── ADMISSIONS.csv
│   │   │   ├── DIAGNOSES_ICD.csv
│   │   │   ├── D_ICD_DIAGNOSES.csv
│   │   │   ├── PATIENTS.csv
│   │   │   ├── filtered_ABP_II.txt
│   │   │   └── icd9toicd10cmgem.csv
│   │   ├── mimic_diagnoses.py
│   │   ├── models.py
│   │   ├── preprocessing.py
│   │   ├── rfc_smote_vs_borderline_smote_comparisons.py
│   │   ├── smote_model_comparisons.py
│   │   └── visualizations.py
│   └── time2feat
│       ├── README.md
│       ├── __init__.py
│       ├── demo.py
│       ├── main.py
│       ├── requirements.txt
│       └── t2f
│           ├── __init__.py
│           ├── data
│           │   ├── __init__.py
│           │   ├── dataset.py
│           │   └── reader.py
│           ├── extraction
│           │   ├── __init__.py
│           │   ├── extractor.py
│           │   ├── extractor_pair.py
│           │   └── extractor_single.py
│           ├── model
│           │   ├── __init__.py
│           │   ├── clustering.py
│           │   └── preprocessing.py
│           ├── ranking
│           │   ├── __init__.py
│           │   ├── algorithms.py
│           │   ├── baseline.py
│           │   ├── ensemble.py
│           │   └── wrapper.py
│           ├── selection
│           │   ├── PFA.py
│           │   ├── __init__.py
│           │   ├── search.py
│           │   └── selection.py
│           └── utils
│               ├── __init__.py
│               └── importance_old.py
├── PPGtoBP
│   ├── PPG_model
│   │   ├── EMI-Clinic-Model-Research.ipynb
│   │   ├── MIMIC-III-Database-Management
│   │   │   ├── data_pipeline.py
│   │   │   ├── records-waveforms.txt
│   │   │   └── records.txt
│   │   ├── bloodPressureModel
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── codes
│   │   │   │   ├── PPG2ABP.ipynb
│   │   │   │   ├── README.md
│   │   │   │   ├── data
│   │   │   │   │   └── meta9.p
│   │   │   │   ├── data_handling.py
│   │   │   │   ├── data_processing.py
│   │   │   │   ├── evaluate.py
│   │   │   │   ├── helper_functions.py
│   │   │   │   ├── metrics.py
│   │   │   │   ├── models.py
│   │   │   │   ├── predict_test.py
│   │   │   │   └── train_models.py
│   │   │   ├── models
│   │   │   │   ├── ApproximateNetwork.h5
│   │   │   │   └── RefinementNetwork.h5
│   │   │   └── requirements.txt
│   │   ├── calibration.py
│   │   ├── error.py
│   │   ├── evaluation_functions.py
│   │   ├── helper_functions_bp_model.py
│   │   └── validation_pipeline.py
│   ├── README.txt
│   ├── download_nsrr.py
│   ├── mesa-sleep-dataset-0.7.0.csv
│   └── mesa_pipeline.py
└── README.md
```

## Usage

### Sleep Stage Estimation
To run the SleepECG weighted model to estimate sleep stages and store the probabilities in time aligned JSON files,
make the following commands.
```bash
cd ECGtoSS
python ecg_to_ss_full.py
```

### Blood Pressure Estimation
To run the blood pressure estimation model and store aligned blood pressure and sleep stages from MESA, refer to the
README.txt file in the PPGtoBP folder.

### Risk Classification Models
Once aligned blood pressure and sleep stages data is stored, either from MIMIC III or from MESA, place the MIMIC data
in a folder named 'mimic_aligned_data' in the data folder in ECGtoSS and/or place the MESA data in a folder named
'mesa_aligned_data' in the data folder in ECGtoSS. Then, to build RCMs, navigate to the risk_classification folder in 
the ECGtoSS folder.
```bash
cd ECGtoSS/risk_classification
```
Then, example_script.py contains examples of how to build RCMs from the aligned data that is stored from MIMIC III
and MESA. To get fixed block instances of six hours from MIMIC III, run lines 28-75 of example_script.py, and for MESA,
run lines 113-140.


## Contributors
Thank you to the Harvey Mudd Clinic Program and Ellison Medical Institute for giving us the ability to work on an 
amazing project that has the ability to help people. 

### Project Contributors
| Role               | Name                 | 
|--------------------|----------------------|
| Faculty Advisor    | Prof. Jamie Haddock  |
| Project Liaison    | Xingyao Chen         |
| Team Lead (Fall)   | Shreya Balaji        |
| Team Lead (Spring) | Lydia Stone          |
| Team Member        | Adam Sage            | 
| Team Member        | Luis Mendoza Ramirez | 
| Team Member        | Channing Christian   | 