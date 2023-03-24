import json
import csv


filename_in = 'test.csv'
filename_out = 'test.jsonl'



def read_csv(infile):
    rename = {0:'World', 1:'Sports', 2:'Business', 3:'Sci/Tech'}
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


dictlist = read_csv(filename)
write_in_jsonl(dictlist,filename)