# Copyright (c) Microsoft. All rights reserved.
from random import shuffle
import json
import csv
import pandas as pd
from nltk.tokenize import word_tokenize

def load_scitail(file):
    """Loading data of scitail
    """
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            blocks = line.strip().split('\t')
            assert len(blocks) > 2
            if blocks[0] == '-': continue
            sample = {'uid': str(cnt), 'premise': blocks[0], 'hypothesis': blocks[1], 'label': blocks[2]}
            rows.append(sample)
            cnt += 1
    return rows

def load_snli(file, header=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 10
            if blocks[-1] == '-': continue
            lab = blocks[-1]
            if lab is None:
                import pdb; pdb.set_trace()
            sample = {'uid': blocks[0], 'premise': blocks[7], 'hypothesis': blocks[8], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_winogrande(file, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            blocks = json.loads(line.strip())
            option1 = blocks['option1']
            option2 = blocks['option2']
            sentence = blocks['sentence']
            
            conj = '_'

            premise = sentence.replace(conj, option1)
            hypothesis = sentence.replace(conj, option2)

            lab = 0
            if is_train:
                lab = int(blocks['answer']) - 1                
            
            if lab is None:
                import pdb; pdb.set_trace()

            sample = {'uid': blocks['qID'], 'premise': premise, 'hypothesis': hypothesis, 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_sickr(file, is_train=True):
    rows = []
    cnt = 0
    df = pd.read_csv(file, encoding='utf8', sep='\t', header=0)
    for i, row in df.iterrows():
        if is_train and row['SemEval_set'] != "TEST":
            sample = {'uid': row['pair_ID'], 'premise': row['sentence_A'], 'hypothesis': row['sentence_B'], 'label': row['relatedness_score']}
        elif not is_train and row['SemEval_set'] == "TEST":
            sample = {'uid': row['pair_ID'], 'premise': row['sentence_A'], 'hypothesis': row['sentence_B'], 'label': row['relatedness_score']}
        else:
            continue
        # print(sample)
        rows.append(sample)

    return rows

def load_sicke(file, is_train=True):
    rows = []
    cnt = 0
    df = pd.read_csv(file, encoding='utf8', sep='\t', header=0)
    for i, row in df.iterrows():
        
        premise = row['sentence_A'].replace("  ", " ")
        # premise = word_tokenize(premise)
        # premise = " ".join(premise).lower()

        hypothesis = row['sentence_B'].replace("  ", " ")
        # hypothesis = word_tokenize(hypothesis)
        # hypothesis = " ".join(hypothesis).lower()

        if is_train and row['SemEval_set'] != "TEST":
            sample = {'uid': row['pair_ID'], 'premise': premise, 'hypothesis': hypothesis, 'label': row['entailment_label'].lower()}
        elif not is_train and row['SemEval_set'] == "TEST":
            sample = {'uid': row['pair_ID'], 'premise': premise, 'hypothesis': hypothesis, 'label': row['entailment_label'].lower()}
        else:
            continue
        # print(sample)
        rows.append(sample)

    return rows

def load_wikiqa(file, is_train=True):
    rows = []
    cnt = 0
    df = pd.read_csv(file, encoding='utf8', sep='\t', header=0, quoting=csv.QUOTE_NONE)
    for i, row in df.iterrows():
        # print(row)
        premise = row['Question']
        hypothesis = row['Sentence']

        # premise = premise.replace("  ", " ")
        # premise = word_tokenize(premise)
        # premise = " ".join(premise).lower()

        # hypothesis = hypothesis.replace("  ", " ")
        # hypothesis = word_tokenize(hypothesis)
        # hypothesis = " ".join(hypothesis).lower()

        sample = {'uid': row['QuestionID'] + '-' + row['SentenceID'], 'premise': premise, 'hypothesis': hypothesis, 'label': row['Label']}
        # print(sample)
        rows.append(sample)

    return rows


def load_imdb(file, is_train=True):
    rows = []
    cnt = 0
    df = pd.read_csv(file, encoding='utf8', sep=',', header=0)
    to_label = {'neg':0,'pos':1}
    for i, row in df.iterrows():
        if row['label'] == 'unsup':
            continue
        # print(row)
        review = row['review'].replace('\t', " ").replace("<br />", "").replace("  ", " ")
        # review = word_tokenize(review)
        # review = " ".join(review).lower()
        # print(review)
        if is_train and row['type'] == "train":
            sample = {'uid': row['id'], 'premise': review, 'label': to_label[row['label']]}
        elif not is_train and row['type'] == "test":
            sample = {'uid': row['id'], 'premise': review, 'label': to_label[row['label']]}
        else:
            continue
        # print(sample)
        rows.append(sample)

    return rows

def load_mnli(file, header=True, multi_snli=False, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 9
            if blocks[-1] == '-': continue
            lab = "contradiction"
            if is_train:
                lab = blocks[-1]
            if lab is None:
                import pdb; pdb.set_trace()
            sample = {'uid': blocks[0], 'premise': blocks[8], 'hypothesis': blocks[9], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_ax(file, header=True, multi_snli=False, is_train=False):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) == 3
            if blocks[-1] == '-': continue
            lab = "contradiction"
            if is_train:
                lab = blocks[-1]
            if lab is None:
                import pdb; pdb.set_trace()
            sample = {'uid': blocks[0], 'premise': blocks[1], 'hypothesis': blocks[2], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_mrpc(file, header=True, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 4
            lab = 0
            if is_train:
                lab = int(blocks[0])
            sample = {'uid': cnt, 'premise': blocks[-2], 'hypothesis': blocks[-1], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_qnli(file, header=True, is_train=True):
    """QNLI for classification"""
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 2
            lab = "not_entailment"
            if is_train:
                lab = blocks[-1]
            if lab is None:
                import pdb; pdb.set_trace()
            sample = {'uid': blocks[0], 'premise': blocks[1], 'hypothesis': blocks[2], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_qqp(file, header=True, is_train=True):
    rows = []
    cnt = 0
    skipped = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            if is_train and len(blocks) < 6:
                skipped += 1
                continue
            if not is_train: assert len(blocks) == 3
            lab = 0
            if is_train:
                lab = int(blocks[-1])
                sample = {'uid': cnt, 'premise': blocks[-3], 'hypothesis': blocks[-2], 'label': lab}
            else:
                sample = {'uid': int(blocks[0]), 'premise': blocks[-2], 'hypothesis': blocks[-1], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_rte(file, header=True, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header =False
                continue
            blocks = line.strip().split('\t')
            if is_train and len(blocks) < 4: continue
            if not is_train: assert len(blocks) == 3
            lab = "not_entailment"
            if is_train:
                lab = blocks[-1]
                sample = {'uid': int(blocks[0]), 'premise': blocks[-3], 'hypothesis': blocks[-2], 'label': lab}
            else:
                sample = {'uid': int(blocks[0]), 'premise': blocks[-2], 'hypothesis': blocks[-1], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_wnli(file, header=True, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header =False
                continue
            blocks = line.strip().split('\t')
            if is_train and len(blocks) < 4: continue
            if not is_train: assert len(blocks) == 3
            lab = 0
            if is_train:
                lab = int(blocks[-1])
                sample = {'uid': cnt, 'premise': blocks[-3], 'hypothesis': blocks[-2], 'label': lab}
            else:
                sample = {'uid': cnt, 'premise': blocks[-2], 'hypothesis': blocks[-1], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_diag(file, header=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 3
            sample = {'uid': cnt, 'premise': blocks[-3], 'hypothesis': blocks[-2], 'label': blocks[-1]}
            rows.append(sample)
            cnt += 1
    return rows

def load_amazon(file, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        f_csv = csv.reader(f)
        for blocks in f_csv:
            lab = 0
            review = (blocks[1] + ' # ' + blocks[2]).replace("\t"," ").replace("  ", " ")
            # review = word_tokenize(review)
            # review = " ".join(review).lower()
            if is_train:
                lab = int(blocks[0])-1
                sample = {'uid': cnt, 'premise': review, 'label': lab}
            else:
                sample = {'uid': cnt, 'premise':  review, 'label': lab}

            cnt += 1
            rows.append(sample)
    return rows

def load_yelp(file, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        f_csv = csv.reader(f)
        for blocks in f_csv:
            lab = 0
            review = blocks[1].replace("\\n", " ").replace("\\\"", "\"").replace("\t", " ").replace("  ", " ")
            # review = word_tokenize(review)
            # review = " ".join(review).lower()
            if is_train:
                lab = int(blocks[0])-1
                sample = {'uid': cnt, 'premise': review, 'label': lab}
            else:
                sample = {'uid': cnt, 'premise': review, 'label': lab}

            cnt += 1
            rows.append(sample)
    return rows

def load_cr(file, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            # print(line)
            lab = int(line[0])
            review = line[2:].strip().replace("\\n", " ").replace("\\\"", "\"").replace("\t", " ").replace("  ", " ")
            # review = word_tokenize(review)
            # review = " ".join(review).lower()
            if is_train:
                
                sample = {'uid': cnt, 'premise': review, 'label': lab}
            else:
                sample = {'uid': cnt, 'premise': review, 'label': lab}

            cnt += 1
            rows.append(sample)
    return rows


def load_cola(file, header=True, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            if is_train and len(blocks) < 2: continue
            lab = 0
            if is_train:
                lab = int(blocks[1])
                sample = {'uid': cnt, 'premise': blocks[-1], 'label': lab}
            else:
                sample = {'uid': cnt, 'premise': blocks[-1], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_sts(file, header=True, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 8
            score = "0.0"
            if is_train:
                score = blocks[-1]
                sample = {'uid': cnt, 'premise': blocks[-3],'hypothesis': blocks[-2], 'label': score}
            else:
                sample = {'uid': cnt, 'premise': blocks[-2],'hypothesis': blocks[-1], 'label': score}
            rows.append(sample)
            cnt += 1
    return rows

def load_qnnli(file, header=True, is_train=True):
    """QNLI for ranking"""
    rows = []
    mis_matched_cnt = 0
    cnt = 0
    with open(file, encoding="utf8") as f:
        lines = f.readlines()
        if header: lines = lines[1:]

        assert len(lines) % 2 == 0
        for idx in range(0, len(lines), 2):
            block1 = lines[idx].strip().split('\t')
            block2 = lines[idx + 1].strip().split('\t')
            # train shuffle
            assert len(block1) > 2 and len(block2) > 2
            if is_train and block1[1] != block2[1]:
                mis_matched_cnt += 1
                continue
            assert block1[1] == block2[1]
            lab1, lab2 = "entailment", "entailment"
            if is_train:
                blocks = [block1, block2]
                shuffle(blocks)
                block1 = blocks[0]
                block2 = blocks[1]
                lab1 = block1[-1]
                lab2 = block2[-1]
                if lab1 == lab2:
                    mis_matched_cnt += 1
                    continue
            assert "," not in lab1
            assert "," not in lab2
            assert "," not in block1[0]
            assert "," not in block2[0]
            sample = {'uid': cnt, 'ruid': "%s,%s" % (block1[0], block2[0]), 'premise': block1[1], 'hypothesis': [block1[2], block2[2]],
                      'label': "%s,%s" % (lab1, lab2)}
            cnt += 1
            rows.append(sample)
    return rows


def submit(path, data, label_dict=None):
    header = 'index\tprediction'
    with open(path ,'w') as writer:
        predictions, uids = data['predictions'], data['uids']
        writer.write('{}\n'.format(header))
        assert len(predictions) == len(uids)
        # sort label
        paired = [(int(uid), predictions[idx]) for idx, uid in enumerate(uids)]
        paired = sorted(paired, key=lambda item: item[0])
        for uid, pred in paired:
            if label_dict is None:
                writer.write('{}\t{}\n'.format(uid, pred))
            else:
                assert type(pred) is int
                writer.write('{}\t{}\n'.format(uid, label_dict[pred]))

