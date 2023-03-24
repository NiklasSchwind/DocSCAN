import json
import csv


filename_in = 'dbpedia_csv/test.csv'
filename_out = 'test.jsonl'



def read_csv(infile):
    rename = {0:"Company",
                        1:"EducationalInstitution",
                        2:"Artist",
                        3:"Athlete",
                        4:"OfficeHolder",
                        5:"MeanOfTransportation",
                        6:"Building",
                        7:"NaturalPlace",
                        8:"Village",
                        9:"Animal",
                        10:"Plant",
                        11:"Album",
                        12:"Film",
                        13:"WrittenWork"}
    dictlist = []
    with open(infile) as csv_file:  #, encoding="utf-8"
        csv_reader = csv.reader(
            csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
        )
        for id_, row in enumerate(csv_reader):
            label, title, description = row
            # Original labels are [1, 2, 3, 4] ->
            #                   ['World', 'Sports', 'Business', 'Sci/Tech']
            # Re-map to [0, 1, 2, 3].
            label = int(label) -1
            text = " ".join((title, description))
            dictlist.append({"text": text, "label": rename[int(label)]})
    print(len(dictlist))
    return dictlist



def write_in_jsonl(list, outfile):
    with open(outfile, 'w') as f:
        for entry in list:
            json.dump(entry, f)
            f.write('\n')


dictlist = read_csv(filename_in)
write_in_jsonl(dictlist,filename_out)