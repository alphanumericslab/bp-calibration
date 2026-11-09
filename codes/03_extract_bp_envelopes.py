import os

import numpy as np
import utils
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import env_detector

fs = 125                # sampling frequency
finalPressure = 20      # final cuff pressure
winLen = 10             # Moving average window length
def_rate = 2            # deflation rate

data_dir = '../../../../../../data/physionet.org/files/mimicdb/1.0.0'
record_names_path = os.path.join(data_dir, 'RECORDS')
env_dir = '../results/envelopes'
os.makedirs(env_dir, exist_ok=True)

with open(record_names_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
subjects = [s.rstrip('/\n') for s in lines]
no_BP_subjects = ['208', '209', '210', '450']
subjects = [subject for subject in subjects if subject not in no_BP_subjects]
highBPSubjects = [
    '211', '220', '221', '224', '225',
    '226', '253', '284', '409', '411',
    '430', '438', '442', '466', '471'
]
nSubject = len(subjects)


#%% extract SBP and DBP, upper and lower envelopes with their respective PSDs
def process_subject(i):
    subject = subjects[i]
    initPressure = 230 if subject in highBPSubjects else 179
    segLen = (initPressure - finalPressure) / def_rate
    segments = utils.extract_segment(subject, initPressure, segLen=segLen, fs=fs)
    if not segments:
        raise ValueError(f'Subject {subject}: no valid ABP segments')
    # Detect within contiguous segments so joins cannot create false beats.
    envelopes = [env_detector(segment, winLen=winLen, fs=fs)
                 for segment in segments]
    env_u = np.concatenate([upper for upper, _ in envelopes])
    env_d = np.concatenate([lower for _, lower in envelopes])
    valid = np.isfinite(env_u) & np.isfinite(env_d)
    if not np.any(valid):
        raise ValueError(f'Subject {subject}: no reliable envelope intervals')
    print(f'Subject {subject}: {valid.mean():.1%} envelope coverage')
    path = f'{env_dir}/{subject}.npz'
    np.savez(path, env_u=env_u, env_d=env_d, valid=valid,
             segment_lengths=np.array([len(segment) for segment in segments]))
    print(f'Subject: {subject} envelopes are saved!')


# Use joblib.Parallel and joblib.delayed for parallel execution
result = Parallel(n_jobs=-1)(delayed(process_subject)(i) for i in tqdm(range(nSubject)))
