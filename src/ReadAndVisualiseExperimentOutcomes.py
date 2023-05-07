
import pandas as pd
from os import listdir
from os.path import isfile, join





def return_accuracy_values_and_difference(selflabelingfile):
    i = 0
    lines_file = selflabelingfile.readlines()
    for line in reversed(lines_file):
        if i == 0:
            print(line[0:8])
            i+=1
        if line[0:8] == 'Accuracy:' and i == 0:
            print(line.split(' ')[2][0])
            if line.split(' ')[2][0] == '(':
                after_selflabeling = float(line.split(' ')[1])
            i += 1
        elif line[0:8] == 'Accuracy:' and i == 1:
            if line.split(' ')[2][0] == '(':
                before_selflabeling = float(line.split(' ')[1])
            i += 1
    try:
        difference = after_selflabeling - before_selflabeling
    except:
        difference = 'Experiment not finished'
        after_selflabeling = 'Experiment not finished'
        before_selflabeling = 'Experiment not finished'
    return before_selflabeling, after_selflabeling, difference


def return_list_of_accuracies_selflabeling(path):

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    columns = []
    for file in onlyfiles:
        file = open(file, 'r')
        before_selflabeling, after_selflabeling, difference = return_accuracy_values_and_difference(file)
        columns.append([file.name, before_selflabeling, after_selflabeling, difference])


    return pd.DataFrame(columns,
                      columns=['ExperimentName', 'Before Selflabeling', 'After Selflabeling', 'Difference'],
                      )







mypath = '/vol/fob-vol7/mi19/schwindn/DocSCAN/TrueSelfLabelingLogs'
print(return_list_of_accuracies_selflabeling(mypath))