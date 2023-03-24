import json
import csv


filename_in = 'dbpedia/test.csv'
filename_out = 'test.jsonl'



def read_csv(infile):
    rename = {1:"Company",
                        2:"EducationalInstitution",
                        3:"Artist",
                        4:"Athlete",
                        5:"OfficeHolder",
                        6:"MeanOfTransportation",
                        7:"Building",
                        8:"NaturalPlace",
                        9:"Village",
                        10:"Animal",
                        11:"Plant",
                        12:"Album",
                        13:"Film",
                        14:"WrittenWork"}
    dictlist = []
    with open(infile, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(
            csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
        )
        for id_, row in enumerate(csv_reader):
            label, title, description = row
            # Original labels are [1, 2, 3, 4] ->
            #                   ['World', 'Sports', 'Business', 'Sci/Tech']
            # Re-map to [0, 1, 2, 3].
            label = int(label) - 1
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