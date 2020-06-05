#!/usr/bin/env python
# coding: utf-8


# In[1]:

import random
import codecs
import json
from os.path import join
import pickle
import os
import re

def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as f:
        return json.load(f)


def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as f:
        return pickle.load(f)


def dump_json(obj, wfpath, wfname, indent=None):
    with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as f:
        pickle.dump(obj, f)


class MetaPathGenerator:
    def __init__(self):
        self.paper_author = {}
        self.author_paper = {}
        self.paper_org    = {}
        self.org_paper    = {}
        self.paper_venue  = {}
        self.venue_paper  = {}

    def read_data(self, dirpath):
        with open(dirpath + "/paper_org.txt", encoding='utf-8') as f:
            for line in {v for v in f}:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    self.paper_org.setdefault(toks[0], []).append(toks[1])
                    self.org_paper.setdefault(toks[1], []).append(toks[0])

        with open(dirpath + "/paper_author.txt", encoding='utf-8') as f:
            for line in {v for v in f}:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    self.paper_author.setdefault(toks[0], []).append(toks[1])
                    self.author_paper.setdefault(toks[1], []).append(toks[0])

        with open(dirpath + "/paper_venue.txt", encoding='utf-8') as f:
            for line in {v for v in f}:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    self.paper_venue.setdefault(toks[0], []).append(toks[1])
                    self.venue_paper.setdefault(toks[1], []).append(toks[0])

        print ("#papers ", len(self.paper_venue))      
        print ("#authors", len(self.author_paper))
        print ("#orgs", len(self.org_paper))
        print ("#venues  ", len(self.venue_paper)) 
    
    def generate_WMRW(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')
        for paper0 in self.paper_venue: 
            for _ in range(0, numwalks): #wnum walks
                paper = paper0
                outline = ""
                for _ in range(walklength):
                    if paper in self.paper_author:
                        authors = self.paper_author[paper]
                        author_cnt = len(authors)
                        author = authors[random.randrange(author_cnt)]

                        papers = self.author_paper[author]
                        paper_cnt = len(papers)
                        if paper_cnt >1:
                            paper1 = papers[random.randrange(paper_cnt)]
                            while paper1 == paper:
                                paper1 = papers[random.randrange(paper_cnt)]
                            paper = paper1
                            outline += " " + paper

                    if paper in self.paper_org:
                        orgs = self.paper_org[paper]
                        org_cnt = len(orgs)
                        org = orgs[random.randrange(org_cnt)]

                        papers = self.org_paper[org]
                        paper_cnt = len(papers)
                        if paper_cnt >1:
                            paper1 = papers[random.randrange(paper_cnt)]
                            while paper1 == paper:
                                paper1 = papers[random.randrange(paper_cnt)]
                            paper = paper1
                            outline += " " + paper

                outfile.write(outline + "\n")
        outfile.close()


def tanimoto(p, q):
    c = p.intersection(q)
    return float(len(c) / (len(p) + len(q) - len(c)))


def generate_pair(pubs, outlier):
    dirpath = 'gene'
    
    paper_org = {}
    paper_venue = {}
    paper_author = {}
    paper_title = {}
    
    with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
        for line in {v for v in pafile}:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                paper_org.setdefault(toks[0], []).append(toks[1])

    with open(dirpath + "/paper_venue.txt", encoding='utf-8') as pafile:
        for line in {v for v in pafile}:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                paper_venue.setdefault(toks[0], []).append(toks[1])

    with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
        for line in {v for v in pafile}:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                paper_author.setdefault(toks[0], []).append(toks[1])

    with open(dirpath + "/paper_title.txt", encoding='utf-8') as pafile:
        for line in {v for v in pafile}:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                paper_title.setdefault(toks[0], []).append(toks[1])

    paper_paper = np.zeros((len(pubs), len(pubs)))
    for i, pid in enumerate(pubs):
        if i not in outlier:
            continue
        for j, pjd in enumerate(pubs):
            if j == i:
                continue

            author_cnt, venue_cnt, org_cnt, title_cnt = 0, 0, 0, 0

            if pid in paper_author and pjd in paper_author:
                author_cnt = len(set(paper_author[pid]) & set(paper_author[pjd])) * 1.5
            if pid in paper_venue and pjd in paper_venue and 'null' not in paper_venue[pid]:
                venue_cnt = tanimoto(set(paper_venue[pid]), set(paper_venue[pjd]))
            if pid in paper_org and pjd in paper_org:
                org_cnt = tanimoto(set(paper_org[pid]), set(paper_org[pjd]))
            if pid in paper_title and pjd in paper_title:
                title_cnt = len(set(paper_title[pid]) & set(paper_title[pjd])) / 3

            paper_paper[i][j] = author_cnt + venue_cnt + org_cnt + title_cnt

    return paper_paper

def pairwise_evaluate(correct_labels,pred_labels):
    TP, TP_FP, TP_FN = 0.0, 0.0, 0.0

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)
    return pairwise_precision, pairwise_recall, pairwise_f1

def formate_string(content, stopword, min_length=2):
    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'

    content = content.strip().lower()
    content = re.sub(r, ' ', content)
    content = re.sub(r'\s{2,}', ' ', content).strip().split(' ')
    return [word for word in content if len(word) >= min_length and word not in stopword]

def save_relation(file_name, name):
    name_pubs_raw = load_json('genename', file_name)
    model = word2vec.Word2Vec.load("word2vec/all_text.model")

    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
    stopword = ['at','based','in','of','for','on','and','to','an','using','with','the','by','we','be','is','are','can']
    stopword1 = ['university','univ','china','department','dept','laboratory','lab','school','al','et',
                 'institute','inst','college','chinese','beijing','journal','science','international']

    author_file = open('gene/paper_author.txt', 'w',encoding='utf-8')
    venue_file = open('gene/paper_venue.txt', 'w', encoding='utf-8')
    title_file = open('gene/paper_title.txt', 'w', encoding='utf-8')
    org_file = open('gene/paper_org.txt', 'w', encoding='utf-8')

    token = name.split("_")
    name = "".join(token)
    token.insert(0, token.pop())
    name_reverse = "".join(token)

    authorname_dict = {}
    paper_emb = {}

    tcp = set()
    for i, pid in enumerate(name_pubs_raw):
        pub = name_pubs_raw[pid]

        org = ""
        for author in pub["authors"]:
            authorname = re.sub(r,'', author["name"]).lower()
            token = authorname.split(" ")
            if len(token) > 1:
                authorname = "".join(token)
                authorname_reverse = "".join(list(reversed(token)))
            
                if authorname not in authorname_dict:
                    if authorname_reverse not in authorname_dict:
                        authorname_dict[authorname] = 1
                    else:
                        authorname = authorname_reverse 
            else:
                authorname = authorname.replace(" ","")
            
            if authorname != name and authorname != name_reverse:
                author_file.write(pid + '\t' + authorname + '\n')
            else:
                if "org" in author:
                    org = author["org"]

        orgs = set(formate_string(org, stopword + stopword1))
        for v in orgs:
            org_file.write(pid + '\t' + v + '\n')

        venues = formate_string(pub["venue"], stopword + stopword1)
        for v in venues:
            venue_file.write(pid + '\t' + v + '\n')
        if len(venues) == 0:
            venue_file.write(pid + '\t' + 'null' + '\n')

        titles = formate_string(pub["title"], stopword)
        for v in titles:
            title_file.write(pid + '\t' + v + '\n')

        keyword = " ".join(pub["keywords"]) if "keywords" in pub else ""
        pstr = keyword + " " + pub["title"] + " " + pub["venue"] + " " + org
        if "year" in pub:
              pstr = pstr +  " " + str(pub["year"])
        conts = formate_string(pstr, stopword + stopword1, 3)

        words_vec = [model[v] for v in conts if v in model]
        if len(words_vec) == 0:
            words_vec.append(np.zeros(100))
            tcp.add(i)

        paper_emb[pid] = np.mean(words_vec, 0)

    dump_data(paper_emb, 'gene', 'paper_emb.pkl')
    dump_data(tcp, 'gene', 'tcp.pkl')

    author_file.close()
    title_file.close()
    venue_file.close()
    org_file.close()


# In[2]:

pubs_raw = load_json("train","train_pub.json")
pubs_raw1 = load_json("sna_data","sna_valid_pub.json")
f1 = open('gene/all_text.txt', 'w', encoding='utf-8')

def clear_string(content):
    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
    content = content.strip().lower()
    content = re.sub(r,' ', content)
    return re.sub(r'\s{2,}', ' ', content).strip()

def encode_author(pub):
    for author in pub["authors"]:
        if "org" in author:
            f1.write(clear_string(author["org"]) + '\n')

    f1.write(clear_string(pub["title"]) + '\n')

    if "abstract" in pub and type(pub["abstract"]) is str:
        f1.write(clear_string(pub["abstract"]) + '\n')

    f1.write(clear_string(pub["venue"]) + '\n')

for paper_id, pub in pubs_raw1.items():
    encode_author(pub)

for paper_id, pub in pubs_raw.items():
    encode_author(pub)

f1.close()

# In[3]:

from gensim.models import word2vec

sentences = word2vec.Text8Corpus(r'gene/all_text.txt')
model = word2vec.Word2Vec(sentences, size=100, negative=5, min_count=2, window=5)
model.save('word2vec/all_text.model')


# In[4]:

import re
from gensim.models import word2vec
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def disambiguation(mode):
    is_train = mode == "train"

    pubs_raw = load_json("train","train_pub.json") if is_train else load_json("sna_data","sna_valid_pub.json")
    name_pubs = load_json("train","train_author.json") if is_train else load_json("sna_data","sna_valid_example_evaluation_scratch.json")

    train_res = []
    valid_res = {}
    for n, name in enumerate(name_pubs):
        pubs   = []
        labels = []

        if mode == "train":
            for i, iauthor_pubs in enumerate(name_pubs[name].values()):
                labels += [i] * len(iauthor_pubs)
                pubs.extend(iauthor_pubs)
        else:
            for cluster in name_pubs[name]:
                pubs.extend(cluster)

        print(n, name, len(pubs))

        if len(pubs) == 0:
            if is_train:
                train_res.append(0)
            else:
                valid_res[name] = []
            continue

        name_pubs_raw = {}
        for paper_id in pubs:
            name_pubs_raw[paper_id] = pubs_raw[paper_id]
            
        dump_json(name_pubs_raw, 'genename', name + '.json', indent=4)
        save_relation(name + '.json', name)  

        mpg = MetaPathGenerator()
        mpg.read_data("gene")

        walk_cnt = 3 if is_train else 10
        walk_sim = np.zeros((len(pubs), len(pubs)))
        cp = set()
        for _ in range(walk_cnt):
            mpg.generate_WMRW("gene/RW.txt", 5, 20)
            sentences = word2vec.Text8Corpus(r'gene/RW.txt')
            model = word2vec.Word2Vec(sentences, size=100, negative=25, min_count=1, window=10)
            embs = []
            for i, pid in enumerate(pubs):
                if pid in model.wv:
                    embs.append(model.wv[pid])
                else:
                    cp.add(i)
                    embs.append(np.zeros(100))
            walk_sim += pairwise_distances(embs, metric="cosine")
        walk_sim /= walk_cnt

        paper_emb = load_data('gene','paper_emb.pkl')
        tembs = [paper_emb[v] for v in pubs]

        sim = (np.array(walk_sim) + np.array(pairwise_distances(tembs, metric="cosine"))) / 2
        pre = np.array(DBSCAN(eps=0.2, min_samples=4, metric ="precomputed").fit_predict(sim))

        outlier = {v for v in cp}
        if is_train:
            tcp = load_data('gene','tcp.pkl')
            for i in tcp:
                outlier.add(i)

        for i, p in enumerate(pre):
            if p == -1:
                outlier.add(i)

        paper_pair = generate_pair(pubs, outlier)
        paper_pair1 = paper_pair.copy()
        idx = len(set(pre))
        for i in range(len(pre)):
            if i not in outlier:
                continue
            j = np.argmax(paper_pair[i])
            while j in outlier:
                paper_pair[i][j] = -1
                j = np.argmax(paper_pair[i])
            if paper_pair[i][j] >= 1.5:
                pre[i] = pre[j]
            else:
                pre[i] = idx
                idx = idx + 1

        for i, idx1 in enumerate(outlier):
            for j, idx2 in enumerate(outlier):
                if j > i and paper_pair1[idx1][idx2] >= 1.5:
                    pre[idx2] = pre[idx1]

        if is_train:
            labels = np.array(labels)
            pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(labels, pre)
            print(pairwise_precision, pairwise_recall, pairwise_f1)
            train_res.append(pairwise_f1)

            print('avg_f1:', np.mean(train_res))
        else:
            valid_res[name] = []
            for i in set(pre):
                oneauthor = []
                for idx,j in enumerate(pre):
                    if i == j:
                        oneauthor.append(pubs[idx])
                valid_res[name].append(oneauthor)

    if is_train:
        print('avg_f1:', np.mean(train_res))
    else:
        dump_json(valid_res, "genetest", "result_valid.json", indent=4)


# In[5]:
disambiguation("train")


# In[6]:
disambiguation("valid")


# In[ ]:
