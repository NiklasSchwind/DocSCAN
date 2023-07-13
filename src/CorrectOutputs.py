
import numpy as np



def correct_accuracy_calculation_selflabeling(selflabelingfile):
    i = 0
    file = open(selflabelingfile, 'r')
    lines_file = file.readlines()
    accuracies_beforeSL = []
    accuracies_afterSL = []
    out = open(selflabelingfile.replace('.txt', '_edit.txt'), 'w')
    for j, line in enumerate(lines_file):
        if line[0:3] == '[0.':
            accuracies_beforeSL.append(float(line.split(', ')[0][1:]))
            accuracies_afterSL.append(float(line.split(', ')[-1][:-2]))
            out.write(line)
        elif line[0:9] == 'Accuracy:' and i == 0 and line.split(' ')[2][0] == '(':
            out.write(f'Accuracy: {np.mean(accuracies_beforeSL).round(3)} ({np.std(accuracies_beforeSL).round(3)}) \n')
            i += 1
        elif line[0:9] == 'Accuracy:' and i == 1 and line.split(' ')[2][0] == '(':
            out.write(f'Accuracy: {np.mean(accuracies_afterSL).round(3)} ({np.std(accuracies_afterSL).round(3)})\n')
            i += 1
        else:
            out.write(line)
    out.close()
    file.close()


file = open('/vol/fob-vol7/mi19/schwindn/DocSCAN/DeletionRatioExperimentsNew/Dataset_TREC-6_Em_IS_clustering_method_EntropyLoss_model_method_DocSCAN_finetuning_multi_epochs_5_indicativesentence__^mask?._entropy_weight_3.0_threshold_0.99_augmentation_method_Deletion_ratio_0_new.txt', 'r')
correct_accuracy_calculation_selflabeling('/vol/fob-vol7/mi19/schwindn/DocSCAN/DeletionRatioExperimentsNew/Dataset_TREC-6_Em_IS_clustering_method_EntropyLoss_model_method_DocSCAN_finetuning_multi_epochs_5_indicativesentence__^mask?._entropy_weight_3.0_threshold_0.99_augmentation_method_Deletion_ratio_0_new.txt')

