import os

import numpy as np
import utils
from joblib import Parallel, delayed
from tqdm import tqdm

fs = 125                                    # sampling frequency
finalPressure = 20                          # final cuff pressure
def_rates = np.arange(0.5, 10.01, 0.5)      # deflation rates to sweep

data_dir = '../../../../../../data/physionet.org/files/mimicdb/1.0.0'
record_names_path = f'{data_dir}/RECORDS'
BP_dir = '../results/simulated_BP'

with open(record_names_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
subjects = [s.rstrip('/\n') for s in lines]
no_BP_subjects = ['208', '209', '210', '450']
subjects = [subject for subject in subjects if subject not in no_BP_subjects]
highBPSubjects = [
    '211', '220', '221', '224', '225',
    '226', '253', '284', '409', '411',
    '430', '438', '442', '466','471'
]
nSubject = len(subjects)

#%% sweep deflation rates
def process_subject(i, def_rate):
    subject = subjects[i]
    initPressure = 230 if subject in highBPSubjects else 179
    segLen = (initPressure - finalPressure) / def_rate
    N = int(np.ceil(segLen * fs))
    t = np.arange(N) / fs
    cuffPressure = initPressure - def_rate * t
    segments = utils.extract_segment(subject, initPressure, segLen=segLen, fs=fs)
    BP, err = utils.estimate_SDBP(segments, cuffPressure, decompose_error=True)

    # save
    def_rate_name = f'{def_rate:.1f}'.replace('.', 'p')
    def_rate_dir = f'{BP_dir}/{def_rate_name}'
    os.makedirs(def_rate_dir, exist_ok=True)
    path = f'{def_rate_dir}/{subject}.npz'
    np.savez(path, BP=BP, err=err)
    print(f'Subject: {subject}, Deflation Rate: {def_rate:.1f} is saved!')

tasks = [
    (i, def_rate)
    for def_rate in def_rates
    for i in range(nSubject)
]

results = Parallel(n_jobs=-1)(
    delayed(process_subject)(i, def_rate)
    for i, def_rate in tqdm(tasks)
)
