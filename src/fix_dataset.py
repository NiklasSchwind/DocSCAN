import json
import csv
from os import listdir
from os.path import isfile, join
import pandas as pd

'''
filename_in = '//vol/fob-vol7/mi19/schwindn/DocSCAN/yahoo_answers_topic/yahoo_answers_csv/train.csv'
filename_out = 'train.jsonl'
split = 'test'


def read_csv(infile):
    dictlist = []
    with open(infile, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            dictlist.append({'text': row[3]  ,'label': int(row[0]) - 1 })


                #"topic": int(row[0]) - 1,
                #"question_title": row[1],
                #"question_content": row[2],
                #"best_answer": row[3],



    return dictlist



def write_in_jsonl(list, outfile):
    with open(outfile, 'w') as f:
        for entry in list:
            json.dump(entry, f)
            f.write('\n')


dictlist = read_csv(filename_in)
write_in_jsonl(dictlist,filename_out)

'''


def load_data(filename):
    sentences, labels = [], []
    with open(filename) as f:
        for line in f:
            line = json.loads(line)
            sentences.append(line["text"])
            labels.append(line["label"])
    df = pd.DataFrame(list(zip(sentences, labels)), columns=["sentence", "label"])
    return df


def count_labels(data):
    print(len(data))
    # Group the data by 'label' and count the occurrences of each label
    label_counts = data['label'].value_counts()

    # Print the label counts
    print("Label Counts:")
    for label, count in label_counts.items():
        print(f"Label: {label}, Count: {count}")


dataset_train = load_data('/vol/fob-vol7/mi19/schwindn/DocSCAN/RNC/train.jsonl')
dataset_test = load_data('/vol/fob-vol7/mi19/schwindn/DocSCAN/RNC/test.jsonl')
count_labels(dataset_train)
print('########################')
count_labels(data_test)