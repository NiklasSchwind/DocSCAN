import json

filename = 'train.jsonl'



def read_jsonl(infile):
    dictlist = []
    rename = {'0': 'World', '1': 'Sports' ,'2': 'Business', '3': 'Science and Technology' }
    with open(filename) as f:
        for line in f:
            lines = json.loads(line)
            print(lines)
            for lin in lines["rows"]:
                print(lin)
                dictlist.append({"text": lin["row"]["text"], "label": rename[str(lin["row"]["label"])]})
    return dictlist



def write_in_jsonl(list, outfile):
    with open(outfile, 'w') as f:
        for entry in list:
            json.dump(entry, f)
            f.write('\n')


dictlist = read_jsonl(filename)
write_in_jsonl(dictlist,filename)