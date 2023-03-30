import json
import csv
from os import listdir
from os.path import isfile, join


filename_in = '/vol/fob-vol7/mi19/schwindn/DocSCAN/TREC-50/raw_train.label'
filename_out = 'train.jsonl'
split = 'test'


def read_csv(infile):

    with open(infile) as f:
        lines = (line.decode("utf-8") for line in f)
        rows = csv.reader(lines)
        for i, row in enumerate(rows):
            yield i, {
                "id": i,
                "topic": int(row[0]) - 1,
                "question_title": row[1],
                "question_content": row[2],
                "best_answer": row[3],
            }
        break


    return dictlist



def write_in_jsonl(list, outfile):
    with open(outfile, 'w') as f:
        for entry in list:
            json.dump(entry, f)
            f.write('\n')


dictlist = read_csv(filename_in)
write_in_jsonl(dictlist,filename_out)