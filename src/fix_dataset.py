import json
import csv
from os import listdir
from os.path import isfile, join


#filename_in = 'dbpedia_csv/test.csv'
filename_out = 'train.jsonl'
split = 'test'


def read_csv(infile):
    rename = {0:"positiv", 1:"negativ",}
    dictlist = []
    onlyfiles_neg = [f for f in listdir('/vol/fob-vol7/mi19/schwindn/DocSCAN/IMDB/aclImdb/train/neg') if isfile(join('/vol/fob-vol7/mi19/schwindn/DocSCAN/IMDB/aclImdb/train/neg', f))]
    onlyfiles_pos = [f for f in listdir('/vol/fob-vol7/mi19/schwindn/DocSCAN/IMDB/aclImdb/train/pos') if
                     isfile(join('/vol/fob-vol7/mi19/schwindn/DocSCAN/IMDB/aclImdb/train/pos', f))]

    for f in onlyfiles_pos:
        with open(join('/vol/fob-vol7/mi19/schwindn/DocSCAN/IMDB/aclImdb/train/pos', f), encoding='utf-8') as text:
            dictlist.append({"text": str(text), "label": "positiv"})
    for f in onlyfiles_neg:
        with open(join('/vol/fob-vol7/mi19/schwindn/DocSCAN/IMDB/aclImdb/train/neg', f), encoding='utf-8') as text:
            dictlist.append({"text": str(text), "label": "negativ"})

    return dictlist



def write_in_jsonl(list, outfile):
    with open(outfile, 'w') as f:
        for entry in list:
            json.dump(entry, f)
            f.write('\n')


dictlist = read_csv()#filename_in)
write_in_jsonl(dictlist,filename_out)