import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

class Evaluation:

    def __init__(self, name_dataset, name_embeddings):

        self.experiment_counter = -1
        self.experiment_list = []
        self.experiment_statistics = {}
        self.name_dataset = name_dataset
        self.name_embeddings = name_embeddings

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
        # print (num_correct)
        # print (num_samples - num_correct)
        # num_correct is small
        match = linear_sum_assignment(num_samples - num_correct)
        # print (match)
        match = np.array(list(zip(*match)))
        # print (match)
        # return as list of tuples, out_c to gt_c
        res = []
        for out_c, gt_c in match:
            res.append((out_c, gt_c))
        return res

    def hungarian_evaluate(self, targets, predictions, class_names=None,
                           compute_purity=True, compute_confusion_matrix=True,
                           confusion_matrix_file=None):
        # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
        # This is computed only for the passed subhead index.

        # Hungarian matching

        num_classes = len(np.unique(targets))
        num_elems = len(targets)
        match = self._hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
        reordered_preds = np.zeros(num_elems, dtype=predictions.dtype)
        for pred_i, target_i in match:
            reordered_preds[predictions == int(pred_i)] = int(target_i)

        # Gather performance metrics
        acc = int((reordered_preds == targets).sum()) / float(num_elems)

        full_statistics = {}
        full_statistics['class_recall'] = {}
        full_statistics['class_precition'] = {}
        full_statistics['class_f1'] = {}
        full_statistics['score'] = {}
        full_statistics['relative_score'] = {}
        full_statistics['number_classes'] = num_classes

        for target in np.unique(targets):
            full_statistics['class_recall'][target] = int(sum([1 for i, predictions in enumerate(reordered_preds) if
                                                               predictions == target and targets[
                                                                   i] == target])) / targets.count(target)
            full_statistics['class_precition'][target] = int(
                sum([1 for i, preds in enumerate(reordered_preds) if preds == target and targets[i] == target])) / sum(
                [1 for preds in reordered_preds if preds == target])
            full_statistics['class_f1'][target] = (full_statistics['class_recall'][target] *
                                                   full_statistics['class_precition'][target]) / (
                                                              full_statistics['class_recall'][target] +
                                                              full_statistics['class_precition'][target])
            full_statistics['score'][target] = targets.count(target)
            full_statistics['relative_score'][target] = targets.count(target) / len(targets)

        full_statistics['macro_f1'] = sum([full_statistics['class_f1'][target] for target in np.unique(targets)]) / (
            len(np.unique(targets)))
        full_statistics['macro_avg_recall'] = sum(
            [full_statistics['class_recall'][target] for target in np.unique(targets)]) / (len(np.unique(targets)))
        full_statistics['macro_avg_precition'] = sum(
            [full_statistics['class_precition'][target] for target in np.unique(targets)]) / (len(np.unique(targets)))

        full_statistics['weighted_average_f1'] = sum(
            [(targets.count(target) / len(targets)) * full_statistics['class_f1'][target] for target in
             np.unique(targets)])
        full_statistics['weighted_average_recall'] = sum(
            [(targets.count(target) / len(targets)) * full_statistics['class_recall'][target] for target in
             np.unique(targets)])
        full_statistics['weighted_average_precition'] = sum(
            [(targets.count(target) / len(targets)) * full_statistics['class_precition'][target] for target in
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

    def evaluate(self,targets, predictions):

        hungarian_match_metrics = self.hungarian_evaluate(targets, predictions)
        self.add_evaluation(hungarian_match_metrics['full_statistics'])

        return hungarian_match_metrics

    def add_evaluation(self,full_statistics):

        self.experiment_counter += 1
        self.experiment_statistics[self.experiment_counter] = full_statistics
        self.experiment_list.append(self.experiment_counter)

    def print_statistic_of_latest_experiment(self):

        experiment = self.experiment_statistics[self.experiment_counter]

        print(f'DocSCAN Experiment with {self.name_dataset} using {self.name_embeddings} number {int(self.experiment_counter)+1}:')
        print('\n')
        print('\n')
        print(f'Corpus Statistics')
        print('\n')
        print(f'Corpus Name: {self.name_dataset}, Number Classes: {experiment["number_classes"]}, Number Texts: {experiment["full_score"]}')
        print('\n')
        print('\n')
        print('Class Statistics:')
        print('\n')
        for target in experiment['class_recall'].keys():
            print(f'Class: {target}')
            print(f'Class Recall: {experiment["class_recall"][target]}, Class Precition: {experiment["class_precition"][target]}, Class F1-Score: {experiment["class_f1"][target]}, Class Score: {experiment["score"][target]}, Class Relative Score: {experiment["relative_score"][target]} ')
            print('\n')
        print('\n')
        print('Macro Averages:')
        print('\n')
        print(f'Macro Average F1: {experiment["macro_f1"]}, Macro Average Recall: {experiment["macro_avg_recall"]}, Macro Average Precition: {experiment["macro_avg_precition"]}, ')
        print('\n')
        print('\n')
        print('Weighted Averages:')
        print('\n')
        print(f'Weighted Average F1: {experiment["weighted_average_f1"]}, Weighted Average Recall: {experiment["weighted_average_recall"]}, Weighted Average Precition: {experiment["weighted_average_precition"]}, ')
        print('\n')
        print('\n')
        print('Other statistics:')
        print('\n')
        print(f'Accuracy: {experiment["accuracy"]}, Normalised Mutual Information: {experiment["normalised_mutual_information"]}')
        print('\n')
        print('\n')

    #Returns mean and standartdeviation of a result indicator calculated from all experiments
    def return_median_and_std(self, experiments, variable):

        values = np.array([experiments[i][variable] for i in range(self.experiment_counter)])

        return f'{np.mean(values).round(3)}, ({np.std(values).round(3)})'

    # Returns mean and standartdeviation of a result indicator calculated from all experiments if the indicator depends on the class
    def return_median_and_std_classwise(self, experiments, variable, target):

        values = np.array([experiments[i][target][variable] for i in range(self.experiment_counter)])

        return f'{np.mean(values).round(3)}, ({np.std(values).round(3)})'

    def print_full_statistics(self):

       experiments = self.experiment_statistics

       print(
           f'Average outcomes of DocSCAN Experiment with {self.name_dataset} using {self.name_embeddings} evaluating {int(self.experiment_counter) + 1} Experiments:')
       print('\n')
       print('\n')
       print(f'Corpus Statistics')
       print('\n')
       print(
           f'Corpus Name: {self.name_dataset}, Number Classes: {experiments[0]["number_classes"]}, Number Texts: {experiments[0]["full_score"]}')
       print('\n')
       print('\n')
       print('Class Statistics:')
       print('\n')
       for target in experiments[0]['class_recall'].keys():
           print(f'Class: {target}')
           print(
               f'Class Recall: {self.return_median_and_std_classwise(experiments,"class_recall",target)}, Class Precition: {eself.return_median_and_std_classwise(experiments,"class_precition",target)}, Class F1-Score: {self.return_median_and_std_classwise(experiments,"class_f1",target)}, Class Score: {experiments[0]["score"][target]}, Class Relative Score: {experiments[0]["relative_score"][target]} ')
           print('\n')
       print('\n')
       print('Macro Averages:')
       print('\n')
       print(
           f'Macro Average F1: {self.return_median_and_std(experiments,"macro_f1")}, Macro Average Recall: {self.return_median_and_std(experiments,"macro_avg_recall")}, Macro Average Precition: {self.return_median_and_std(experiments,"macro_avg_precition")}, ')
       print('\n')
       print('\n')
       print('Weighted Averages:')
       print('\n')
       print(
           f'Weighted Average F1: {self.return_median_and_std(experiments,"weighted_average_f1")}, Weighted Average Recall: {self.return_median_and_std(experiments,"weighted_average_recall")}, Weighted Average Precition: {self.return_median_and_std(experiments,"weighted_average_precition")}, ')
       print('\n')
       print('\n')
       print('Other statistics:')
       print('\n')
       print(
           f'Accuracy: {self.return_median_and_std(experiments,"accuracy")}, Normalised Mutual Information: {self.return_median_and_std(experiments,"normalised_mutual_information")}')
       print('\n')
       print('\n')

