import json
import csv
from os import listdir
from os.path import isfile, join


#filename_in = 'dbpedia_csv/test.csv'
filename_out = 'test.jsonl'
split = 'test'


def read_csv(infile):
    rename = {0:"positiv", 1:"negativ",}
    dictlist = []
    onlyfiles_neg = [f for f in listdir('/vol/fob-vol7/mi19/schwindn/DocSCAN/IMDB/aclImdb/test/neg') if isfile(join('/vol/fob-vol7/mi19/schwindn/DocSCAN/IMDB/aclImdb/test/neg', f))]
    onlyfiles_pos = [f for f in listdir('/vol/fob-vol7/mi19/schwindn/DocSCAN/IMDB/aclImdb/test/pos') if
                     isfile(join('/vol/fob-vol7/mi19/schwindn/DocSCAN/IMDB/aclImdb/test/pos', f))]

    for f in onlyfiles_pos:
        with open(join('/vol/fob-vol7/mi19/schwindn/DocSCAN/IMDB/aclImdb/test/pos', f), encoding='utf-8') as text:
            lines = text.readlines()
            dictlist.append({"text": str(lines), "label": "positiv"})
    for f in onlyfiles_neg:
        with open(join('/vol/fob-vol7/mi19/schwindn/DocSCAN/IMDB/aclImdb/test/neg', f), encoding='utf-8') as text:
            lines = text.readlines()
            dictlist.append({"text": str(lines), "label": "negativ"})

    return dictlist



def write_in_jsonl(list, outfile):
    with open(outfile, 'w') as f:
        for entry in list:
            json.dump(entry, f)
            f.write('\n')


dictlist = read_csv(0)
write_in_jsonl(dictlist,filename_out)