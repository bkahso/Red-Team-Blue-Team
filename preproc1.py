import sys
import argparse
import os
import json
import html
import re
import spacy

TAG_MAP = [
    ".",
    ",",
    "-LRB-",
    "-RRB-",
    "``",
    "\"\"",
    "''",
    ",",
    "$",
    "#",
    "AFX",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "HYPH",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NIL",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
    "ADD",
    "NFP",
    "GW",
    "XX",
    "BES",
    "HVS",
    "_SP",
]

nlp = spacy.load('en', disable=['parser', 'ner'])

with open('StopWords') as file:
    data = file.read()
    sw = data.rstrip().split('\n')

indir = 'data';


def splitPos(comment):
    pattern = '(\S+)/(\S+)'
    token = re.sub(pattern, r'\1', str(comment))
    pos = re.sub(pattern, r'\2', str(comment))
    return (token, pos)

def splitPos2(comment):
    pattern = '(\S+)/(.+)'
    token = re.sub(pattern, r'\1', str(comment))
    pos = re.sub(pattern, r'\2', str(comment))
    return (token, pos)


def preproc1(comment, steps=range(1, 11)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''

    modComm = ''
    if 1 in steps:
        modComm = comment.replace('\n', '')
    if 2 in steps:
        modComm = html.unescape(modComm)
    if 3 in steps:
        modComm = re.sub('^http\S+|^www\S+', '', modComm)
    if 4 in steps:
        # problem with starts with punctuation
        # more punctuation
        pattern = '(\d.|\w.)(\?+|!+|\.+|,+|:+|;+)'
        modComm = re.sub(pattern, r'\1 \2', modComm)
        pattern2 = '(\?+|!+|\.+|,+|:+|;+)(\d.|\w.)'
        modComm = re.sub(pattern2, r'\1 \2', modComm)
    if 5 in steps:
        pattern = "(\d.|\w.)('\S*)"
        modComm = re.sub(pattern, r'\1 \2', modComm)
    if 6 in steps:
        temp = ''
        utt = nlp(modComm)
        for token in utt:
            temp += str(token) + "/" + str(token.tag_) + " "
        modComm = temp.rstrip()
    if 7 in steps:
        # split tokens and pos tags into two groups
        token, pos = splitPos(modComm)
        j, temp2 = 0, ""
        # get individual words
        tupList = token.split(" ")
        posList = pos.split(" ")
        # check for stop words and update modComm accordingly
        while len(tupList) > j:
            # RUINS FIRST 3 FEATURES
            if tupList[j].lower() not in sw:
                temp2 += tupList[j] + "/" + posList[j] + " "
            j += 1
        # remove rightmost whitespace
        modComm = temp2.rstrip()
    if 8 in steps:
        utt = nlp(modComm)
        new = ''
        for word in utt:
            if word.text == "./.":
                new += "./." + " "
            if "/" in word.text or word.text in TAG_MAP:
                continue
            elif word.text[0].isnumeric():
                new += word.text + " "
            elif word.lemma_[0] != "-":
                new += str(word.lemma_) + "/" + word.tag_ + " "
            else:
                new += str(word.text) + "/" + word.tag_ + " "
        modComm = new.rstrip()
    if 9 in steps:
        pattern = '(.*\.) (.*)'
        modComm = re.sub(pattern, r'\1\n\2', modComm)
        while re.match(pattern, modComm) is not None:
            modComm = re.sub(pattern, r'\1\n\2', modComm)
        patt2 = '(.*)(\\n)(.*)'
        modComm = re.sub(patt2, r'\1 \2 \3', modComm)
    if 10 in steps:
        modComm = modComm.lower()
    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            lines = int(args.max)
            # TODO: read those lines with something like `j = json.loads(line)`
            startval = 1001677820 % len(data)
            for i in range(startval, startval + lines):
                dic = {}
                j = json.loads(data[i])
                # TODO: choose to retain fields from those lines that are relevant to you
                idd = j['id']
                body = j['body']
                # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
                cat = str(file)
                # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                dic['body'] = preproc1(body)
                dic['ID'] = idd
                dic['Category'] = cat
                # TODO: replace the 'body' field with the processed text
                # TODO: append the result to 'allOutput'
                allOutput.append(dic)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    #FIX NEW LINE
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)
