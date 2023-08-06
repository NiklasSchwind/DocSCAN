
import numpy as np
from os import listdir
from os.path import isfile, join



def correct_accuracy_calculation_selflabeling(in_file, out_file=None):
    i = 0
    file = open(in_file, 'r')
    lines_file = file.readlines()
    accuracies_beforeSL = []
    weighted_accuracy_afterSL = []
    accuracies_afterSL = []
    #out = open(out_file, 'w')
    for j, line in enumerate(lines_file):
        if line[0:3] == '[0.':
            accuracies_beforeSL.append(float(line.split(', ')[0][1:]))
            accuracies_afterSL.append(float(line.split(', ')[-1][:-2]))
            weighted_accuracy_afterSL.append(float(lines_file[j-11].split(', ')[3]))
            #out.write(line)
        elif line[0:16] == 'Macro Average F1' and i == 1:
            print(weighted_accuracy_afterSL)
            print(line)
            # out.write(f'Accuracy: {np.mean(weighted_accuracy_afterSL).round(3)} ({np.std(weighted_accuracy_afterSL).round(3)}) \n')
        elif line[0:9] == 'Accuracy:' and i == 0 and line.split(' ')[2][0] == '(':
            #out.write(f'Accuracy: {np.mean(accuracies_beforeSL).round(3)} ({np.std(accuracies_beforeSL).round(3)}) \n')
            i += 1
        elif line[0:9] == 'Accuracy:' and i == 1 and line.split(' ')[2][0] == '(':
            #out.write(f'Accuracy: {np.mean(accuracies_afterSL).round(3)} ({np.std(accuracies_afterSL).round(3)})\n')
            i += 1
        else:
            #out.write(line)
    #out.close()
    file.close()

def correct_folder(inpath, outpath):
    onlyfiles = [str(f) for f in listdir(inpath) if isfile(join(inpath, f))]
    for file in onlyfiles:
        in_file = '/'.join([inpath,file])
        out_file = '/'.join([outpath, file])
        print(in_file)
        print(out_file)
        correct_accuracy_calculation_selflabeling(in_file, out_file)


correct_accuracy_calculation_selflabeling('/vol/fob-vol7/mi19/schwindn/DocSCAN/DeletionRatioExperimentsNew/Dataset_TREC-6_Em_IS_clustering_method_EntropyLoss_model_method_DocSCAN_finetuning_multi_epochs_5_indicativesentence__^mask?._entropy_weight_3.0_threshold_0.99_augmentation_method_Deletion_ratio_0_new.txt')

#correct_folder('/vol/fob-vol7/mi19/schwindn/DocSCAN/CroppingExperiments', '/vol/fob-vol7/mi19/schwindn/DocSCAN/CroppingExperimentsDoubleCorrected')
