import json
import csv
from os import listdir
from os.path import isfile, join


filename_in = '/vol/fob-vol7/mi19/schwindn/DocSCAN/TREC-50/raw_train.label'
filename_out = 'train.jsonl'
split = 'test'


def read_csv(infile):
    #rename = {0:"positiv", 1:"negativ",}
    label_name_map_6 = {'ENTY': 'question about entity',
                        'DESC': 'question about description',
                        'ABBR': 'question about abbreviation',
                        'HUM': 'question about person',
                        'NUM': 'question about number',
                        'LOC': 'question about location'
                        }

    label_name_map_50 = {'ENTY:cremat': 'question about creative material',
                         'HUM:title': 'question about a title of a person',
                         'NUM:code': 'question about postcodes or other codes',
                         'ENTY:color': 'question about colors',
                         'ENTY:substance': 'question about elements and substances',
                         'ENTY:termeq': 'question about equivalent terms',
                         'NUM:speed': 'question about speed',
                         'NUM:dist': 'question about linear measures',
                         'ENTY:plant': 'question about plants',
                         'NUM:money': 'question about prices',
                         'ENTY:food': 'question about food',
                         'NUM:temp': 'question about temperature',
                         'ENTY:veh': 'question about vehicles',
                         'ENTY:symbol': 'question about symbols and signs',
                         'ENTY:techmeth': 'question about techniques and methods',
                         'LOC:mount': 'question about mountains',
                         'HUM:desc': 'question about the description of a person',
                         'ENTY:dismed': 'question about diseases and medicine',
                         'DESC:reason': 'question about reasons',
                         'LOC:country': 'question about countries',
                         'NUM:volsize': 'question about size, area and volume',
                         'ENTY:sport': 'question about sports',
                         'ENTY:word': 'question about words with a special property',
                         'NUM:ord': 'question about ranks',
                         'HUM:gr': 'question about a group or organization of persons',
                         'ENTY:other': 'question about other entities',
                         'DESC:desc': 'question about the description of something',
                         'ABBR:abb': 'question about abbreviation',
                         'ENTY:animal': 'question about animals',
                         'ENTY:religion': 'question about religions',
                         'LOC:city': 'question about cities',
                         'ENTY:body': 'question about organs of body',
                         'LOC:state': 'question about states',
                         'NUM:date': 'question about dates',
                         'LOC:other': 'question about other locations',
                         'ABBR:exp': 'question about expressions abbreviated',
                         'ENTY:event': 'question about events',
                         'ENTY:lang': 'question about languages',
                         'ENTY:currency': 'question about currency names',
                         'ENTY:letter': 'question about letters like a-z',
                         'DESC:def': 'question about the definition of something',
                         'NUM:count': 'question about the number of something',
                         'NUM:other': 'question about other numbers',
                         'DESC:manner': 'question about the manner of an action',
                         'NUM:period': 'question about the lasting time of something',
                         'ENTY:product': 'question about products',
                         'NUM:perc': 'question about fractions',
                         'NUM:weight': 'question about weight',
                         'HUM:ind': 'question about an individual',
                         'ENTY:instru': 'question about a musical instrument'
                         }
    dictlist = []
    with open(infile, "rb") as f:
        for id_, row in enumerate(f):
            # One non-ASCII byte: sisterBADBYTEcity. We replace it with a space
            fine_label, _, text = row.replace(b"\xf0", b" ").strip().decode().partition(" ")
            coarse_label = fine_label.split(":")[0]
            dictlist.append({"text": str(text), "label": label_name_map_50[coarse_label]})


    return dictlist



def write_in_jsonl(list, outfile):
    with open(outfile, 'w') as f:
        for entry in list:
            json.dump(entry, f)
            f.write('\n')


dictlist = read_csv(filename_in)
write_in_jsonl(dictlist,filename_out)