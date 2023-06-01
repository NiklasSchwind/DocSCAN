
import pandas as pd
from os import listdir
from os.path import isfile, join
import json
import numpy as np
import random
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


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

def return_accuracy_values(file):
    i = 0
    lines_file = file.readlines()
    after_selflabeling = 'Experiment not finished'
    try:
        for line in reversed(lines_file):
            if line[0:9] == 'Accuracy:' and i == 0:
                if line.split(' ')[2][0] == '(':
                    after_selflabeling = float(line.split(' ')[1])*100
    except:
        after_selflabeling = 'Experiment not finished'




    return after_selflabeling

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

def return_list_of_accuracies_entropy_weight(path):

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    columns = []
    for file in onlyfiles:
        file = open(file, 'r')
        after_selflabeling = return_accuracy_values(file)
        columns.append([return_next_in_list('Dataset',file.name.split('_'),1),
                        return_next_in_list('Embedding', file.name.split('_'),1),
                        return_next_in_list('clustering', file.name.split('_'),2),
                        return_next_in_list('epochs', file.name.split('_'), 1),
                        float(return_next_in_list('entropy', file.name.split('_'), 2).replace('.txt', '')),
                        after_selflabeling,
                        ])


    return pd.DataFrame(columns,
                      columns=['Dataset','Embedding','Clustering Method', 'Epochs', 'Entropy Weight', 'Accuracy'],
                      )

def return_list_of_accuracies_ratio(path):

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
                        return_next_in_list('ratio', file.name.split('_'), 1).replace('.txt',''),
                        before_selflabeling,
                        after_selflabeling,
                        difference])


    return pd.DataFrame(columns,
                      columns=['Dataset','Embedding','Clustering Method', 'Epochs', 'Threshold', 'Augmentation Method', 'Ratio', 'Before Selflabeling', 'After Selflabeling', 'Difference'],
                      )




def display_selflabeling_experiments():

    mypath = '/vol/fob-vol7/mi19/schwindn/DocSCAN/TrueSelfLabelingLogs'
    frame= return_list_of_accuracies_selflabeling(mypath)

    frame = frame[frame.Difference != 'Experiment not finished'].sort_values('Difference').sort_values(['Augmentation Method','Dataset', 'Embedding', 'Clustering Method', 'Difference'])

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                            'display.width', 1000,
                           'display.precision', 10,
                           ):

        print(frame[frame['Augmentation Method'] == 'Cropping.txt'])
    for j in frame['Augmentation Method'].unique():
        for i in frame['Dataset'].unique():
            average_score = frame.result = frame.loc[(frame['Dataset'] == i) & (frame['Augmentation Method'] == j), 'Difference'].mean()# .loc[frame['Dataset'].str.contains(i), 'Difference'].mean()
            print(f'Average Score for {i} using {j} is {average_score}')



def display_experiments(mode: Literal['ratio', 'entropy'], mypath):

    if mode == 'entropy':
        frame = return_list_of_accuracies_entropy_weight(mypath)

        frame = frame[frame.Accuracy != 'Experiment not finished'].sort_values('Entropy Weight')
        frame = frame[frame.Dataset == '20newsgroup'].sort_values('Entropy Weight')
        frame = frame[frame['Clustering Method'] == 'EntropyLoss'].sort_values('Entropy Weight')

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 1000,
                           'display.precision', 3,
                           ):
        print(frame)
    if mode == 'entropy':
        print(frame['Entropy Weight'].to_list())
        print(frame['Accuracy'].to_list())
    elif mode == 'ratio':
        print(frame['Before Selflabeling'].to_list())
        print(frame['Ratio'].to_list())
        print(frame['After Selflabeling'].to_list())
        print(frame['Difference'].to_list())



#display_experiments(mode = 'entropy', mypath = '/vol/fob-vol7/mi19/schwindn/DocSCAN/EntropyWeightExperiments')

def load_data(filename):
    sentences, labels = [], []
    with open(filename) as f:
        for line in f:
            line = json.loads(line)
            sentences.append(line["text"])
            labels.append(line["label"])
    df = pd.DataFrame(list(zip(sentences, labels)), columns=["sentence", "label"])
    return df

def write_in_jsonl(list, outfile):
    with open(outfile, 'w') as f:
        for entry in list:
            json.dump(entry, f)
            f.write('\n')

def get_random_data_in_same_ratio(train_data,  amount ):
    share = {}
    sentence= []
    labels = []
    for label in set(list(train_data["label"])):
        share[label] = int((len(list(train_data.loc[train_data['label'] == label]))/len(list(train_data["label"])))*amount)
        print(share[label])
        df = train_data.loc[train_data['label'] == label].sample(n = share[label])
        sentence.extend(list(df["sentence"]))
        labels.extend(list(df["label"]))
    dictlist = []
    print(labels)
    combined_lists = list(zip(sentence, labels))
    random.shuffle(combined_lists)
    for sentence, label in combined_lists:
        dictlist.append({'text': sentence  ,'label': label })
    print(len(dictlist))
    return dictlist




train_data = load_data('/vol/fob-vol7/mi19/schwindn/DocSCAN/DBPedia/train.jsonl')
dictlist = get_random_data_in_same_ratio(train_data,  10000)
write_in_jsonl(dictlist, '/vol/fob-vol7/mi19/schwindn/DocSCAN/DBPedia_smaller/train.jsonl')



'''
import matplotlib.pyplot as plt

# Some sample data
Before_Selflabeling = [55.1, 56.10000000000001, 56.10000000000001, 55.00000000000001, 55.900000000000006, 56.10000000000001, 57.199999999999996, 55.900000000000006, 57.8, 54.900000000000006, 56.49999999999999, 58.5, 56.39999999999999, 57.49999999999999, 55.800000000000004]
After_Selflabeling = [59.5, 58.599999999999994, 59.099999999999994, 59.4, 59.3, 60.5, 60.699999999999996, 59.4, 61.3, 58.8, 59.0, 60.099999999999994, 58.9, 60.8, 58.3]
Ratio = ['0.01', '0.02', '0.03', '0.05', '0.07', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55']
Ratio = [float(i) for i in Ratio]
Difference = [4.399999999999999, 2.499999999999986, 2.999999999999986, 4.3999999999999915, 3.3999999999999915, 4.3999999999999915, 3.5, 3.499999999999993, 3.5, 3.8999999999999915, 2.500000000000007, 1.5999999999999943, 2.500000000000007, 3.3000000000000043, 2.499999999999993]


# Create a scatter plot with x_values on the x-axis and y_values on the y-axis

plt.scatter([r*100 for r in Ratio], Before_Selflabeling)
plt.scatter([r*100 for r in Ratio], After_Selflabeling)
plt.errorbar([r*100 for r in Ratio], [(Before_Selflabeling[i] + After_Selflabeling[i])/2 for i in range(len(Before_Selflabeling))], yerr=[i/2 for i in Difference], fmt='none', ecolor='black', capsize=3)

# Set the x-axis to be scaled proportionally to the numerical values of x_values
plt.autoscale(enable=True, axis='x', tight=False)
plt.title('Effectiveness of Data Augmentation by Random Deletion \n for Selflabeling under different Deletion Rates')
plt.xlabel('Probability for Token Deletion [%]')
plt.ylabel('Accuracy [%]')
plt.legend()
# Show the plot
plt.show()
'''