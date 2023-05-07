
import pandas as pd
from os import listdir
from os.path import isfile, join





def return_accuracy_values_and_difference(selflabelingfile):
    i = 0
    lines_file = selflabelingfile.readlines()
    for line in reversed(lines_file):
        if line[0:9] == 'Accuracy:' and i == 0:
            if line.split(' ')[2][0] == '(':
                after_selflabeling = float(line.split(' ')[1])*100
            i += 1
        elif line[0:9] == 'Accuracy:' and i == 1:
            if line.split(' ')[2][0] == '(':
                before_selflabeling = float(line.split(' ')[1])*100
            i += 1
    try:
        difference = after_selflabeling - before_selflabeling
    except:
        difference = 'Experiment not finished'
        after_selflabeling = 'Experiment not finished'
        before_selflabeling = 'Experiment not finished'
    return before_selflabeling, after_selflabeling, difference

def return_next_in_list(previous,list,add):
    return list[list.index(previous)+add]


def return_list_of_accuracies_selflabeling(path):

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    columns = []
    for file in onlyfiles:
        file = open(file, 'r')
        before_selflabeling, after_selflabeling, difference = return_accuracy_values_and_difference(file)
        columns.append([return_next_in_list('Dataset',file.name.split('_'),1),
                        return_next_in_list('Embedding', file.name.split('_'),1),
                        return_next_in_list('clustering', file.name.split('_'),2),
                        return_next_in_list('epochs', file.name.split('_'), 1),
                        return_next_in_list('threshold', file.name.split('_'), 1),
                        return_next_in_list('threshold', file.name.split('_'), 2),
                        before_selflabeling,
                        after_selflabeling,
                        difference])


    return pd.DataFrame(columns,
                      columns=['Dataset','Embedding','Clustering Method', 'Epochs', 'Threshold', 'Augmentation Method', 'Before Selflabeling', 'After Selflabeling', 'Difference'],
                      )







mypath = '/vol/fob-vol7/mi19/schwindn/DocSCAN/TrueSelfLabelingLogs'
frame= return_list_of_accuracies_selflabeling(mypath)

frame = frame[frame.Difference != 'Experiment not finished'].sort_values('Difference')

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):

    print(frame)