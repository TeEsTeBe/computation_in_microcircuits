import numpy as np
from scipy.optimize import nnls
# from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


class RateTaskEvaluator:

    def __init__(self, state_matrix_dict, task_targets, train_trials, test_trials):
        self.state_matrix_dict = state_matrix_dict
        self.task_targets = task_targets
        self.train_trials = train_trials
        self.test_trials = test_trials

        self.readouts = []

    def evaluate_all_tasks(self):
        print('\n## Results')
        results_dict = {}
        for statemat_name, statemat in self.state_matrix_dict.items():
            print(f'\n### Readout population: {statemat_name}')
            for task_name, task_target in self.task_targets.items():
                print(f'\n#### {task_name}')
                if task_name not in results_dict.keys():
                    results_dict[task_name] = {}
                for readout_type in ['nnls', 'linreg']:

                    train_targets_nonans = task_target[:self.train_trials]
                    train_mask = np.isfinite(train_targets_nonans)
                    train_targets_nonans = train_targets_nonans[train_mask]
                    train_states_nonans = statemat[:self.train_trials][train_mask]
                    train_trials_nonans = train_targets_nonans.size

                    test_targets_nonans = task_target[-self.test_trials:]
                    test_mask = np.isfinite(test_targets_nonans)
                    test_targets_nonans = test_targets_nonans[test_mask]
                    test_trials_nonans = test_targets_nonans.size

                    statemat_nonans = statemat[np.isfinite(task_target)]

                    if readout_type == 'nnls':
                        trained_x, rnorm = nnls(train_states_nonans, train_targets_nonans)
                        prediction = statemat_nonans @ trained_x

                    elif readout_type == 'linreg':
                        reg = LinearRegression().fit(train_states_nonans, train_targets_nonans)
                        prediction = reg.predict(statemat_nonans)

                    cc_train = 'failed'
                    cc_test = 'failed'
                    try:
                        cc_train = pearsonr(prediction[:train_trials_nonans], train_targets_nonans)[0]
                        cc_test = pearsonr(prediction[-test_trials_nonans:], test_targets_nonans)[0]
                    except:
                        print('Correlation coefficent failed')

                    readout_name = f'{statemat_name}_{readout_type}'
                    results_dict[task_name][readout_name] = {
                        'prediction_raw': prediction,
                        'cc_train': cc_train,
                        'cc_test': cc_test,
                    }
                    print(f'- {readout_name}')
                    print(f'\t- correlation coefficient (train): {cc_train}')
                    print(f'\t- correlation coefficient (test): {cc_test}')

        return results_dict


