import numpy as np
import re
import sys
import argparse
import os
import json
import pandas as pd

fp = ['i', 'me', 'mine', 'we', 'us', 'our', 'ours']
sp = ['you', 'yours', 'u', 'urs', 'ur']
tp = ['he', 'him', 'she', 'her', 'his', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']
slang = ['smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff', 'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw',
'bw', 'imho', 'fyi', 'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys', 'afn', 'bbs', 'cya', 'ez', 'f2f',
'gtr', 'ic', 'jk', 'k', 'ly', 'ya', 'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml']
futur = ['will', 'gon']
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
bg = pd.read_csv('BristolNorms+GilhoolyLogie.csv')
w = pd.read_csv('Ratings_Warriner_et_al.csv')

posTags = {TAG_MAP[i].lower(): [] for i in range(len(TAG_MAP))}

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    tup = pos(comment)
    r = tup[0]
    l = tup[1]
    s = tup[2][0]
    n = tup[2][1]
    v = np.zeros(174)
    prp = r['prp']
    prp.extend(r['prp$'])
    fp, sp, tp = 0, 0, 0
    for pronoun in prp:
        if pronoun in fp:
            fp += 1
        elif pronoun in sp:
            sp += 1
        else:
            tp += 1
    v[0], v[1], v[2] = fp, sp, tp
    v[3] = len(r['cc'])
    v[4] = len(r['vbn']) + len(r['vbd']) #past participle?
    v[5] = l[0]
    v[6] = r['.'].count(',')
    v[7] = len([x for x in r['.'] if len(x) > 1])
    v[8] = len(r['nn']) + len(r['nns'])
    v[9] = len(r['nnp']) + len(r['nnps'])
    v[10] = len(r['rb']) + len(r['rbr']) + len(r['rbs'])
    v[11] = len(r['wdt']) + len(r['wp']) + len(r['wp$']) + len(r['wrb'])
    v[12] = l[1]
    v[13] = l[2]
    v[14] = s[0]
    v[15] = s[1]
    v[16] = s[2]
    for i in range(len(n)):
        v[17 + i] = n[i]
    return v


def splitPos(comment):
    pattern = '(\S+)/(\S+)'
    token = re.sub(pattern, r'\1', str(comment))
    pos = re.sub(pattern, r'\2', str(comment))
    return (token, pos)


def pos(comment):
    copy = comment[:]
    sent = sentences(copy)
    comment = re.sub(' ', '/', comment)
    comment = re.split("/", comment)
    num_slang, num_futur, num_caps, num_sentences = 0, 0, 0, 0
    (token, pos) = splitPos(comment)
    fin = posTags.copy()

    for j in range(len(token)):
        if token[j] in slang:
            num_slang += 1
        if token[j] in futur:
            num_futur += 1
        if token[j].isupper() and len(token[j]) >= 3:
            num_caps += 1
        if pos[j] not in fin.keys():
            fin[pos[j]] = [token[j]]
        else:
            fin[pos[j]].append(token[j])

    rList = [num_futur, num_slang, num_caps]
    return (fin, rList, sent)

def norms(sentences):
    pattern = '(\S+)/(\S+)'
    token = re.sub(pattern, r'\1', str(sentences))
    bg_indices = []
    w_indices = []
    words = token.split(" ")
    for word in words:
        bg_index = bg.index[bg['WORD'] == word].tolist()
        w_index = w.index[w['Word'] == word].tolist()
        if len(bg_index) > 0:
            bg_indices.append(bg_index[0])
        if len(w_index) > 0:
            w_indices.append(w_index[0])
    aoa_list = [bg['AoA (100-700)'][index] for index in bg_indices]
    img_list = [bg['IMG'][index] for index in bg_indices]
    fam_list = [bg['FAM'][index] for index in bg_indices]
    v_list = [w['V.Mean.Sum'][index] for index in w_indices]
    a_list = [w['A.Mean.Sum'][index] for index in w_indices]
    d_list = [w['D.Mean.Sum'][index] for index in w_indices]

    (aoa_mu, aoa_sd) = get_stats(aoa_list) if len(aoa_list) > 0 else (0, 0)
    (img_mu, img_sd) = get_stats(img_list) if len(img_list) > 0 else (0, 0)
    (fam_mu, fam_sd) = get_stats(fam_list) if len(fam_list) > 0 else (0, 0)
    (v_mu, v_sd) = get_stats(v_list) if len(v_list) > 0 else (0, 0)
    (a_mu, a_sd) = get_stats(a_list) if len(a_list) > 0 else (0, 0)
    (d_mu, d_sd) = get_stats(d_list) if len(d_list) > 0 else (0, 0)
    return [aoa_mu, img_mu, fam_mu, aoa_sd, img_sd,
            fam_sd, v_mu, a_mu, d_mu, v_sd, a_sd, d_sd]


def get_stats(vals):
    mu = np.sum(vals) / len(vals)
    vector = np.array(vals)
    sd = np.sum((vector - mu) ** 2) / mu
    return (mu, sd)



def sentences(comment):
    copy = comment[:]
    sent = re.split('\n', copy)
    num_sentences = len(sent)
    norm = norms(copy)
    lst = [re.split(' ', x) for x in sent]
    tot = [len(sentence) for sentence in lst]
    avg_token = sum(tot) / num_sentences
    for sentence in lst:
        for word in sentence:
            if word[-2:] == "/.":
                sentence.remove(word)
    # CREATE NEW LIST HERE?
    tot2 = [len(sentence) for sentence in lst]
    avg_token_no_punct = sum(tot2) / num_sentences
    return ([avg_token, avg_token_no_punct, num_sentences], norm)



def receptiviti():
    direc = 'feats/'
    ids = "_IDs.txt"
    featFile = "_feats.dat.npy"
    categories = ['Left', 'Center', 'Right', 'Alt']
    dictionaries= []
    for i in range(len(categories)):
        cat = categories[i]
        print("Running " + cat)
        f = open(direc + cat + ids)
        lines = f.readlines()
        lst = [x.split() for x in lines]
        temp = [(lst[i], i) for i in range(len(lst))]
        dic = {temp[i][1] : temp[i][0] for i in range(len(temp))}
        feats = np.load(direc + cat + featFile, 'r')
        final = {dic[k][0]: feats[k] for k in range(len(temp))}
        dictionaries.append(final)
        f.close()
    return dictionaries


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))
    dicts = receptiviti()
    categories = ['Left', 'Center', 'Right', 'Alt']
    q = 0

    for dic in data:
        cat = categories.index(dic['Category'])
        extracted = extract1(dic['body'])
        idd = dic['ID']
        extracted[28:172] = dicts[cat][idd]
        extracted[173] = cat
        feats[q] = extracted
        q += 1

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    main(args)
