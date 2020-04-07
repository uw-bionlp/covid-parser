import os
import json
from pathLib import Path

TAB = '\t'
NL  = '\n'



class BratRow():
    def __init__(self, data, cnt):
        self.entity_id = f'T{cnt}'
        self.anno_type = data['semanticLabel']
        self.begin_idx = data['beginDocumentCharIndex']
        self.end_idx   = data['endDocumentCharIndex']
        self.text      = data['sourcePhrase'].replace(NL, '  ')

    def write(self):
        return f'{self.entity_id}{TAB}{self.anno_type} {self.begin_idx} {self.end_idx}{TAB}{self.text}'

def json_to_brat(json):
    i = 1
    text = json['text']
    rows = []
    for sent in json['sentences']:
        cons = sorted(sent['concepts'], key=lambda x: x['beginDocumentCharIndex'])
        for con in cons:
            if 'sosy' not in con['semanticLabel']:
                continue
            rows.append(BratRow(con, i))
            i += 1

    return text, rows

def main():
    cwd = os.getcwd()
    Path(f'{cwd}/brat').mkdir(parents=True, exist_ok=True)

    for fl in os.listdir(cwd):
        stem = Path(f'{cwd}/{fl}').stem
        with open(f'{cwd}/{fl}', 'r') as f:
            data = json.loads(f.read())
        text, rows = json_to_brat(data)

        with open(f'{cwd}/{fl}.txt', 'w+') as f:
            f.write(text)
        with open(f'{cwd}/{fl}.ann', 'w+') as f:
            for row in rows:
                f.write(f'{row.write()}{NL}')

main()



    