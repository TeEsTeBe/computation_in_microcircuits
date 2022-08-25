import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.linear_model import LinearRegression


class ClassificationTaskEvaluator:
    """ Class that implements the functionality to evaluate spike pattern classification tasks """

    def __init__(self, state_matrix_dict, task_targets, train_trials, test_trials):
        self.state_matrix_dict = state_matrix_dict
        self.task_targets = task_targets
        self.train_trials = train_trials
        self.test_trials = test_trials

        self.readouts = []

    def evaluate_all_tasks(self):
        """ Evaluates the classification tasks

        Returns
        -------
        dict
            dictionary with all the evaluated metrics for all tasks

        """

        print('\n## Results')
        results_dict = {}
        for statemat_name, statemat in self.state_matrix_dict.items():
            print(f'\n### Readout population: {statemat_name}')
            for task_name, task_target in self.task_targets.items():
                print(f'\n#### {task_name}')
                if task_name not in results_dict.keys():
                    results_dict[task_name] = {}
                for readout_type in ['nnls', 'linreg']:

                    if readout_type == 'nnls':
                        trained_x, rnorm = nnls(statemat[:self.train_trials], task_target[:self.train_trials])
                        prediction = statemat @ trained_x

                    elif readout_type == 'linreg':
                        reg = LinearRegression().fit(statemat[:self.train_trials], task_target[:self.train_trials])
                        prediction = reg.predict(statemat)

                    binary_prediction = np.zeros_like(prediction)
                    binary_prediction[prediction>0.5] = 1

                    kappa_train = 'failed'
                    kappa_test = 'failed'
                    try:
                        kappa_train = cohen_kappa_score(binary_prediction[:self.train_trials], task_target[:self.train_trials])
                        kappa_test = cohen_kappa_score(binary_prediction[-self.test_trials:], task_target[-self.test_trials:])
                    except:
                        print('Kappa coefficent failed')

                    accuracy_train = accuracy_score(binary_prediction[:self.train_trials], task_target[:self.train_trials])
                    accuracy_test = accuracy_score(binary_prediction[-self.test_trials:], task_target[-self.test_trials:])

                    readout_name = f'{statemat_name}_{readout_type}'
                    results_dict[task_name][readout_name] = {
                        'prediction_raw': prediction,
                        'prediction_binary': binary_prediction,
                        'kappa_train': kappa_train,
                        'kappa_test': kappa_test,
                        'accuracy_train': accuracy_train,
                        'accuracy_test': accuracy_test,
                        'error_train': 1. - accuracy_train,
                        'error_test': 1. - accuracy_test,
                    }
                    print(f'- {readout_name}')
                    print(f'\t- kappa (train): {kappa_train}')
                    print(f'\t- kappa (test): {kappa_test}')
                    print(f'\t- accuracy (train): {accuracy_train}')
                    print(f'\t- accuracy (test): {accuracy_test}')

        return results_dict


