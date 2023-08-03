import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import copy

class Evaluation:

    def __init__(self, name_dataset, name_embeddings, moreTargets = False):

        self.experiment_counter = -1
        self.experiment_list = []
        self.experiment_statistics = {}
        self.name_dataset = name_dataset
        self.name_embeddings = name_embeddings
        self.moreTargets = moreTargets

    def _hungarian_match_adjusted(self, flat_preds, flat_targets, preds_k, targets_k):
        # Based on implementation from IIC
        num_samples = len(flat_targets)
        num_correct = np.zeros((preds_k, targets_k))

        for c1 in range(preds_k):
            for c2 in range(targets_k):
                # elementwise, so each sample contributes once
                votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
                num_correct[c1, c2] = votes
        print(sum(num_correct[0,:]))
        print(sum(num_correct[:, 0]))
        match = [(pred, np.argmax(num_correct[pred,:])) for pred in range(preds_k)]
        res = []
        for out_c, gt_c in match:
            res.append((out_c, gt_c))

        return res

    def _hungarian_match(self, flat_preds, flat_targets, preds_k, targets_k):
        # Based on implementation from IIC
        num_samples = len(flat_targets)

        assert (preds_k == targets_k)  # one to one
        num_k = preds_k
        num_correct = np.zeros((num_k, num_k))

        for c1 in range(num_k):
            for c2 in range(num_k):
                # elementwise, so each sample contributes once
                votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
                num_correct[c1, c2] = votes
        match = linear_sum_assignment(num_samples - num_correct)
        match = np.array(list(zip(*match)))
        res = []
        for out_c, gt_c in match:
            res.append((out_c, gt_c))

        return res

    def hungarian_evaluate(self, targets, predictions):
        '''
        targets = tatsächliche klasse
        predictions = predictions direkt von docscan (d.h clusternummer)
        reodered_predicitons = predictions nach hungarian matching (d.h clusternummer entspricht optimalem target)
        '''
#
        num_classes = len(np.unique(targets))
        num_preds = len(np.unique(predictions))
        #num_predictions = len(np.unique(predictions))
        num_elems = len(targets)
        if self.moreTargets:
            match = self._hungarian_match_adjusted(predictions, targets, preds_k=num_preds, targets_k=num_classes)
            print(match)
        else:
            match = self._hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
        #match_niklas = self._hungarian_match_niklas(predictions, targets, preds_k=num_predictions, targets_k=num_classes)
        reordered_preds = np.zeros(num_elems, dtype=predictions.dtype)
        for pred_i, target_i in match:
            reordered_preds[predictions == int(pred_i)] = int(target_i)

        # Gather performance metrics
        if num_elems != 0:
            acc = int((reordered_preds == targets).sum()) / float(num_elems)
        else:
            acc = 0

        full_statistics = {}
        full_statistics['class_recall'] = {}
        full_statistics['class_precition'] = {}
        full_statistics['class_f1'] = {}
        full_statistics['score'] = {}
        full_statistics['number_predictions'] = {}
        full_statistics['relative_score'] = {}
        full_statistics['number_classes'] = num_classes



        for target in np.unique(targets):
            full_statistics['score'][target] = list(targets).count(target)
            full_statistics['number_predictions'][target] = list(reordered_preds).count(target)
            if full_statistics['score'][target] != 0:
                full_statistics['class_recall'][target] = int(sum([1 for i, prediction in enumerate(reordered_preds) if (prediction == target and targets[i] == target)])) / full_statistics['score'][target]
            else:
                full_statistics['class_recall'][target] = 0
            if sum([1 for preds in reordered_preds if preds == target]) != 0:
                full_statistics['class_precition'][target] = int(
                    sum([1 for i, prediction in enumerate(reordered_preds) if (prediction == target and targets[i] == target)])) / sum([1 for preds in reordered_preds if preds == target])
            else:
                full_statistics['class_precition'][target] = 0
            if full_statistics['class_recall'][target] +full_statistics['class_precition'][target] != 0:
                full_statistics['class_f1'][target] = 2*(full_statistics['class_recall'][target] *
                                                   full_statistics['class_precition'][target]) / (
                                                              full_statistics['class_recall'][target] +
                                                              full_statistics['class_precition'][target])
            else:
                full_statistics['class_f1'][target] = 0

            full_statistics['relative_score'][target] = full_statistics['score'][target] / len(targets)

        full_statistics['macro_f1'] = sum([full_statistics['class_f1'][target] for target in np.unique(targets)]) / (
            len(np.unique(targets)))
        full_statistics['macro_avg_recall'] = sum(
            [full_statistics['class_recall'][target] for target in np.unique(targets)]) / (len(np.unique(targets)))
        full_statistics['macro_avg_precition'] = sum(
            [full_statistics['class_precition'][target] for target in np.unique(targets)]) / (len(np.unique(targets)))

        full_statistics['weighted_average_f1'] = sum(
            [(full_statistics['score'][target] / len(targets)) * full_statistics['class_f1'][target] for target in
             np.unique(targets)])
        full_statistics['weighted_average_recall'] = sum(
            [(full_statistics['score'][target] / len(targets)) * full_statistics['class_recall'][target] for target in
             np.unique(targets)])
        full_statistics['weighted_average_precition'] = sum(
            [(full_statistics['score'][target] / len(targets)) * full_statistics['class_precition'][target] for target in
             np.unique(targets)])

        full_statistics['accuracy'] = acc
        full_statistics['full_score'] = len(targets)

        nmi = metrics.normalized_mutual_info_score(targets, predictions)
        ari = metrics.adjusted_rand_score(targets, predictions)

        full_statistics['normalised_mutual_information'] = nmi

        classification_report = metrics.classification_report(targets, reordered_preds)
        cm = metrics.confusion_matrix(targets, reordered_preds)

        return {'full_statistics': full_statistics, 'ACC': acc, 'ARI': ari, 'NMI': nmi, 'hungarian_match': match,
                "classification_report": classification_report, "confusion matrix": cm,
                "reordered_preds": reordered_preds}

    def evaluate(self,targets, predictions, addToStatistics: bool = True, doPrint: bool = False):

        targets = np.array(targets)
        predictions = np.array(predictions)
        hungarian_match_metrics = self.hungarian_evaluate(targets, predictions)
        if addToStatistics:
            self.add_evaluation(hungarian_match_metrics['full_statistics'])

        if doPrint:
            self.print_statistic_of_latest_experiment(hungarian_match_metrics['full_statistics'])


        return hungarian_match_metrics

    def add_evaluation(self,full_statistics):

        self.experiment_counter += 1
        self.experiment_statistics[int(self.experiment_counter)] = full_statistics
        self.experiment_list.append(int(self.experiment_counter))

    def print_statistic_of_latest_experiment(self, experiment = None):

        if experiment is None:
            experiment = self.experiment_statistics[self.experiment_counter]

        print(f'DocSCAN Experiment with {self.name_dataset} using {self.name_embeddings} number {int(self.experiment_counter)+1}:')
        print('\n')
        print(f'Corpus Statistics')
        print(f'Corpus Name: {self.name_dataset}, Number Classes: {experiment["number_classes"]}, Number Texts: {experiment["full_score"]}')
        print('\n')
        print('Class Statistics:')
        for target in experiment['class_recall'].keys():
            print(f'Class: {target}')
            print(f'Class Recall: {experiment["class_recall"][target]}, Class Precision: {experiment["class_precition"][target]}, Class F1-Score: {experiment["class_f1"][target]}, Class Score: {experiment["score"][target]}, Class Relative Score: {experiment["relative_score"][target]}, Total Number of Predictions: {experiment["number_predictions"][target]}')
        print('\n')
        print('Macro Averages:')
        print(f'Macro Average F1: {experiment["macro_f1"]}, Macro Average Recall: {experiment["macro_avg_recall"]}, Macro Average Precision: {experiment["macro_avg_precition"]}, ')
        print('\n')
        print('Weighted Averages:')
        print(f'Weighted Average F1: {experiment["weighted_average_f1"]}, Weighted Average Recall: {experiment["weighted_average_recall"]}, Weighted Average Precision: {experiment["weighted_average_precition"]}, ')
        print('\n')
        print('Other statistics:')
        print(f'Accuracy: {experiment["accuracy"]}, Normalised Mutual Information: {experiment["normalised_mutual_information"]}')
        print('\n')

    #Returns mean and standartdeviation of a result indicator calculated from all experiments
    def return_median_and_std(self, experiments, variable):

        values = np.array([experiments[i][variable] for i in self.experiment_list])

        return f'{np.mean(values).round(3)} ({np.std(values).round(3)})'

    def return_median_accuracy(self, variable = "accuracy"):

        experiments = copy.deepcopy(self.experiment_statistics)
        values = np.array([experiments[i][variable] for i in self.experiment_list])

        return np.mean(values).round(3)

    # Returns mean and standartdeviation of a result indicator calculated from all experiments if the indicator depends on the class
    def return_median_and_std_classwise(self, experiments, variable, target):


        values = np.array([experiments[i][variable][target] for i in self.experiment_list if target in experiments[i][variable].keys()])

        try:
            return f'{np.mean(values).round(3)} ({np.std(values).round(3)})'
        except:
            return '0.000 (0.000)'

    def print_full_statistics(self):


       experiments = self.experiment_statistics

       if len(self.experiment_list) == 0:
           print('ERROR PRINTING FULL STATISTICS! NO EXPERIMENT EXECUTED')
           return None

       print(
           f'Average outcomes of DocSCAN Experiment with {self.name_dataset} using {self.name_embeddings} evaluating {int(self.experiment_counter) + 1} Experiments:')
       print('\n')
       print(f'Corpus Statistics')
       print(
           f'Corpus Name: {self.name_dataset}, Number Classes: {experiments[self.experiment_list[0]]["number_classes"]}, Number Texts: {experiments[self.experiment_list[0]]["full_score"]}')
       print('\n')
       print('Class Statistics:')
       for target in experiments[self.experiment_list[0]]['class_recall'].keys():
           print(f'Class: {target}')
           print(
               f'Class Recall: {self.return_median_and_std_classwise(experiments,"class_recall",target)}, Class Precision: {self.return_median_and_std_classwise(experiments,"class_precition",target)}, Class F1-Score: {self.return_median_and_std_classwise(experiments,"class_f1",target)}')
       print('\n')
       print('Macro Averages:')
       print(
           f'Macro Average F1: {self.return_median_and_std(experiments,"macro_f1")}, Macro Average Recall: {self.return_median_and_std(experiments,"macro_avg_recall")}, Macro Average Precision: {self.return_median_and_std(experiments,"macro_avg_precition")}, ')
       print('\n')
       print('Weighted Averages:')
       print(
           f'Weighted Average F1: {self.return_median_and_std(experiments,"weighted_average_f1")}, Weighted Average Recall: {self.return_median_and_std(experiments,"weighted_average_recall")}, Weighted Average Precision: {self.return_median_and_std(experiments,"weighted_average_precition")}, ')
       print('\n')
       print('Other statistics:')
       print(
           f'Accuracy: {self.return_median_and_std(experiments,"accuracy")}, Normalised Mutual Information: {self.return_median_and_std(experiments,"normalised_mutual_information")}')
       print('\n')


