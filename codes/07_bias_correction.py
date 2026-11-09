import os

import numpy as np
from tqdm import tqdm
from utils import bias_correction_Bayes, bias_correction_LSE, bias_correction_ML

N = 3                                       # Number of observations
L_prior = 5                                 # Number of previous observations used for prior estimation
cov_regularization = 1e-5                   # Regularization term for covariance matrix to avoid singularity
np.random.seed(0)                           # Set random seed for reproducibility
def_rates = np.arange(0.5, 10.01, 0.5)      # Deflation rates from 0.5 to 10 with a step of 0.5

BP_dir = '../results/simulated_BP'
result_path = f'../results/bias_corrected_BP_N_{N}'

#%%
for def_rate in tqdm(def_rates):
    def_rate_dir = f'{def_rate:.1f}'.replace('.', 'p')
    path = f'{BP_dir}/{def_rate_dir}/mean_cov.npz'
    data = np.load(path)
    mu_w_all = data['mu_w']
    C_w_all = data['C_w']

    files = sorted(os.listdir(f'{BP_dir}/{def_rate_dir}'))
    files = [file for file in files if file.endswith('.npz') and file != 'mean_cov.npz']
    for k, file in enumerate(files):
        # Load data
        path = f'{BP_dir}/{def_rate_dir}/{file}'
        data = np.load(path)
        BP_tru = data['BP'][:2, :]
        BP_est = data['BP'][2:, :]
        N_samples = BP_tru.shape[1]

        # Load LOOCV mean and covariance of measurement error
        mu_w = mu_w_all[k]
        C_w = C_w_all[k]

        # bias correction loop
        sbp_sing_list, sbp_ls_list, sbp_ml_list, sbp_bayes_list, sbp_tru_sing_list, sbp_tru_mult_list = [], [], [], [], [], []
        dbp_sing_list, dbp_ls_list, dbp_ml_list, dbp_bayes_list, dbp_tru_sing_list, dbp_tru_mult_list = [], [], [], [], [], []
        map_sing_list, map_ls_list, map_ml_list, map_bayes_list, map_tru_sing_list, map_tru_mult_list = [], [], [], [], [], []
        pup_sing_list, pup_ls_list, pup_ml_list, pup_bayes_list, pup_tru_sing_list, pup_tru_mult_list = [], [], [], [], [], []

        for i in range(L_prior, N_samples - N + 1):
            # Prioirs
            prior_indices = np.arange(i - L_prior, i)
            BP_prior = BP_tru[:, prior_indices]
            mu_theta = np.mean(BP_prior, axis=1)
            C_theta = np.cov(BP_prior) + cov_regularization * np.eye(2)

            # Ground truth
            indices = np.arange(i, i + N)
            BP_tru_sing = BP_tru[:, indices[0]]
            BP_tru_mult = BP_tru[:, indices].mean(axis=1)
            sbp_tru_sing, dbp_tru_sing = BP_tru_sing
            sbp_tru_mult, dbp_tru_mult = BP_tru_mult
            map_tru_sing = (sbp_tru_sing + 2 * dbp_tru_sing) / 3
            map_tru_mult = (sbp_tru_mult + 2 * dbp_tru_mult) / 3
            pup_tru_sing = sbp_tru_sing - dbp_tru_sing
            pup_tru_mult = sbp_tru_mult - dbp_tru_mult

            # Measurements
            BP_est_sing = BP_est[:, indices[0]]
            BP_est_mult = BP_est[:, indices]
            sbp_sing, dbp_sing = BP_est_sing
            map_sing = (sbp_sing + 2 * dbp_sing) / 3
            pup_sing = sbp_sing - dbp_sing

            # Bias correction
            BP_ls = bias_correction_LSE(BP_est_mult).flatten()
            BP_ml = bias_correction_ML(BP_est_mult, mu_w).flatten()
            BP_bayes = bias_correction_Bayes(BP_est_mult, mu_w, C_w, mu_theta, C_theta).flatten()

            sbp_ls, dbp_ls = BP_ls
            sbp_ml, dbp_ml = BP_ml
            sbp_bayes, dbp_bayes = BP_bayes
            map_ls = (sbp_ls + 2 * dbp_ls) / 3
            map_ml = (sbp_ml + 2 * dbp_ml) / 3
            map_bayes = (sbp_bayes + 2 * dbp_bayes) / 3
            pp_ls = sbp_ls - dbp_ls
            pp_ml = sbp_ml - dbp_ml
            pp_bayes = sbp_bayes - dbp_bayes

            # Store measurement and bias-corrected values
            sbp_sing_list.append(sbp_sing)
            sbp_ls_list.append(sbp_ls)
            sbp_ml_list.append(sbp_ml)
            sbp_bayes_list.append(sbp_bayes)
            dbp_sing_list.append(dbp_sing)
            dbp_ls_list.append(dbp_ls)
            dbp_ml_list.append(dbp_ml)
            dbp_bayes_list.append(dbp_bayes)
            map_sing_list.append(map_sing)
            map_ls_list.append(map_ls)
            map_ml_list.append(map_ml)
            map_bayes_list.append(map_bayes)
            pup_sing_list.append(pup_sing)
            pup_ls_list.append(pp_ls)
            pup_ml_list.append(pp_ml)
            pup_bayes_list.append(pp_bayes)

            # Store Ground truth values
            sbp_tru_sing_list.append(sbp_tru_sing)
            sbp_tru_mult_list.append(sbp_tru_mult)
            dbp_tru_sing_list.append(dbp_tru_sing)
            dbp_tru_mult_list.append(dbp_tru_mult)
            map_tru_sing_list.append(map_tru_sing)
            map_tru_mult_list.append(map_tru_mult)
            pup_tru_sing_list.append(pup_tru_sing)
            pup_tru_mult_list.append(pup_tru_mult)


        # Convert to numpy arrays
        sbp_sing_list, sbp_ls_list, sbp_ml_list, sbp_bayes_list, sbp_tru_sing_list, sbp_tru_mult_list = map(np.array, [sbp_sing_list, sbp_ls_list, sbp_ml_list, sbp_bayes_list, sbp_tru_sing_list, sbp_tru_mult_list])
        dbp_sing_list, dbp_ls_list, dbp_ml_list, dbp_bayes_list, dbp_tru_sing_list, dbp_tru_mult_list = map(np.array, [dbp_sing_list, dbp_ls_list, dbp_ml_list, dbp_bayes_list, dbp_tru_sing_list, dbp_tru_mult_list])
        map_sing_list, map_ls_list, map_ml_list, map_bayes_list, map_tru_sing_list, map_tru_mult_list = map(np.array, [map_sing_list, map_ls_list, map_ml_list, map_bayes_list, map_tru_sing_list, map_tru_mult_list])
        pup_sing_list, pup_ls_list, pup_ml_list, pup_bayes_list, pup_tru_sing_list, pup_tru_mult_list = map(np.array, [pup_sing_list, pup_ls_list, pup_ml_list, pup_bayes_list, pup_tru_sing_list, pup_tru_mult_list])

        result_dir = f'{result_path}/{def_rate_dir}'
        os.makedirs(result_dir, exist_ok=True)
        path = f'{result_dir}/{file}'
        np.savez(
            path,
            sbp_sing=sbp_sing_list, sbp_ls=sbp_ls_list, sbp_ml=sbp_ml_list, sbp_bayes=sbp_bayes_list, sbp_tru_sing=sbp_tru_sing_list, sbp_tru_mult=sbp_tru_mult_list,
            dbp_sing=dbp_sing_list, dbp_ls=dbp_ls_list, dbp_ml=dbp_ml_list, dbp_bayes=dbp_bayes_list, dbp_tru_sing=dbp_tru_sing_list, dbp_tru_mult=dbp_tru_mult_list,
            map_sing=map_sing_list, map_ls=map_ls_list, map_ml=map_ml_list, map_bayes=map_bayes_list, map_tru_sing=map_tru_sing_list, map_tru_mult=map_tru_mult_list,
            pup_sing=pup_sing_list, pup_ls=pup_ls_list, pup_ml=pup_ml_list, pup_bayes=pup_bayes_list, pup_tru_sing=pup_tru_sing_list, pup_tru_mult=pup_tru_mult_list
        )
