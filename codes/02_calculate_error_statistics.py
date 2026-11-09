import os
import numpy as np

def_rates = np.arange(0.5, 10.01, 0.5)
for def_rate in def_rates:
    print(f'Processing deflation rate: {def_rate:.1f} mmHg/s ...')
    def_rate_dir = f'{def_rate:.1f}'.replace('.', 'p')
    BP_dir = f'../results/simulated_BP/{def_rate_dir}'
    files = sorted(os.listdir(BP_dir))
    files = [file for file in files if file.endswith('.npz') and file != 'mean_cov.npz']

    # Store each subject's error samples
    subject_errors = []
    for file in files:
        path = os.path.join(BP_dir, file)
        data = np.load(path)
        SBP = np.asarray(data['BP'][0]).ravel()
        DBP = np.asarray(data['BP'][1]).ravel()
        SBP_est = np.asarray(data['BP'][2]).ravel()
        DBP_est = np.asarray(data['BP'][3]).ravel()
        err_SBP = SBP_est - SBP
        err_DBP = DBP_est - DBP
        errors = np.column_stack((err_SBP, err_DBP))
        subject_errors.append(errors)

    # LOOCV estimates
    mu_w, C_w = [], []
    for test_idx in range(len(files)):
        # Combine errors from every subject except the held-out subject
        training_errors = np.vstack([subject_errors[i] for i in range(len(files)) if i != test_idx])
        mu_w_fold = np.mean(training_errors, axis=0)
        C_w_fold = np.cov(training_errors, rowvar=False, ddof=1)
        mu_w.append(mu_w_fold)
        C_w.append(C_w_fold)

    mu_w = np.asarray(mu_w)  # Shape: (number of subjects, 2)
    C_w = np.asarray(C_w)    # Shape: (number of subjects, 2, 2)

    save_path = f'{BP_dir}/mean_cov.npz'
    np.savez(save_path, mu_w=mu_w, C_w=C_w)

#%%
