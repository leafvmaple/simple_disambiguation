#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam

def l2Norm(x):
    return K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(_, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

class GlobalTripletModel:
    def __init__(self, dimension=100):
        self.model = None
        self.dimension = dimension
        self.fited = False
        self.create()

    def create(self):
        emb_anchor = Input(shape=(self.dimension, ), name='anchor_input')
        emb_pos = Input(shape=(self.dimension, ), name='pos_input')
        emb_neg = Input(shape=(self.dimension, ), name='neg_input')

        # shared layers
        layer1 = Dense(128, activation='relu', name='first_emb_layer')
        layer2 = Dense(64, activation='relu', name='last_emb_layer')
        norm_layer = Lambda(l2Norm, name='norm_layer', output_shape=[64])

        encoded_emb = norm_layer(layer2(layer1(emb_anchor)))
        encoded_emb_pos = norm_layer(layer2(layer1(emb_pos)))
        encoded_emb_neg = norm_layer(layer2(layer1(emb_neg)))

        pos_dist = Lambda(euclidean_distance, name='pos_dist')([encoded_emb, encoded_emb_pos])
        neg_dist = Lambda(euclidean_distance, name='neg_dist')([encoded_emb, encoded_emb_neg])

        def cal_output_shape(input_shape):
            shape = list(input_shape[0])
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists',
            output_shape=cal_output_shape
        )([pos_dist, neg_dist])

        self.model = Model([emb_anchor, emb_pos, emb_neg], stacked_dists, name='triple_siamese')
        self.model.compile(loss=triplet_loss, optimizer=Adam(lr=0.01), metrics=[accuracy])

        # inter_layer = Model(inputs=self.model.get_input_at(0), outputs=self.model.get_layer('norm_layer').get_output_at(0))

    def fit(self, data):
        self.fited = True
        triplets_count = data["anchor_input"].shape[0]
        self.model.fit(data, np.ones((triplets_count, 2)), batch_size=64, epochs=5, shuffle=True, validation_split=0.2)

    def get_inter(self, paper_embs):
        get_activations = K.function(self.model.inputs[:1] + [K.learning_phase()], [self.model.layers[5].get_output_at(0), ])
        activations = get_activations([paper_embs, 0])
        return activations[0]

def generate_triplet(paper_embs, author_data):
    anchor_data = []
    pos_data = []
    neg_data = []
    paper_ids = []

    for _, author in author_data.items():
        for _, papers in author.items():
            for paper_id in papers:
                paper_ids.append(paper_id)

    for _, author in author_data.items():
        for _, papers in author.items():
            for i, paper_id in enumerate(papers):
                if paper_id not in paper_embs:
                    continue
                sample_length = min(6, len(papers))
                idxs = random.sample(range(len(papers)), sample_length)
                for idx in idxs:
                    if idx == i:
                        continue
                    pos_paper = papers[idx]
                    if pos_paper not in paper_embs:
                        continue

                    neg_paper = paper_ids[random.randint(0, len(paper_ids) - 1)]
                    while neg_paper in papers or neg_paper not in paper_embs:
                        neg_paper = paper_ids[random.randint(0, len(paper_ids) - 1)]

                    anchor_data.append(paper_embs[paper_id])
                    pos_data.append(paper_embs[pos_paper])
                    neg_data.append(paper_embs[neg_paper])

    triplet_data = {}
    triplet_data["anchor_input"] = np.stack(anchor_data)
    triplet_data["pos_input"] = np.stack(pos_data)
    triplet_data["neg_input"] = np.stack(neg_data)

    return triplet_data

triplet = GlobalTripletModel()

# In[2]:


import codecs
import json
from os.path import join
import pickle
import os
import re

################# Load and Save Data ################

def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)


def dump_json(obj, wfpath, wfname, indent=None):
    with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)



def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)

    
################# Random Walk ################

import random
class MetaPathGenerator:
    def __init__(self):
        self.paper_author = dict()
        self.author_paper = dict()
        self.paper_org = dict()
        self.org_paper = dict()
        self.paper_conf = dict()
        self.conf_paper = dict()

    def read_data(self, dirpath):
        temp=set()

        with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)                       
        for line in temp: 
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, a = toks[0], toks[1]
                    if p not in self.paper_org:
                        self.paper_org[p] = []
                    self.paper_org[p].append(a)
                    if a not in self.org_paper:
                        self.org_paper[a] = []
                    self.org_paper[a].append(p)
        temp.clear()

              
        with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)                       
        for line in temp: 
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, a = toks[0], toks[1]
                    if p not in self.paper_author:
                        self.paper_author[p] = []
                    self.paper_author[p].append(a)
                    if a not in self.author_paper:
                        self.author_paper[a] = []
                    self.author_paper[a].append(p)
        temp.clear()
        
                
        with open(dirpath + "/paper_conf.txt", encoding='utf-8') as pcfile:
            for line in pcfile:
                temp.add(line)                       
        for line in temp: 
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, a = toks[0], toks[1]
                    if p not in self.paper_conf:
                        self.paper_conf[p] = []
                    self.paper_conf[p].append(a)
                    if a not in self.conf_paper:
                        self.conf_paper[a] = []
                    self.conf_paper[a].append(p)
        temp.clear()
                    
        print ("#papers ", len(self.paper_conf))      
        print ("#authors", len(self.author_paper))
        print ("#org_words", len(self.org_paper))
        print ("#confs  ", len(self.conf_paper)) 
    
    def generate_WMRW(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')
        for paper0 in self.paper_conf: 
            for j in range(0, numwalks): #wnum walks
                paper=paper0
                outline = ""
                i=0
                while(i<walklength):
                    i=i+1    
                    if paper in self.paper_author:
                        authors = self.paper_author[paper]
                        numa = len(authors)
                        authorid = random.randrange(numa)
                        author = authors[authorid]
                        
                        papers = self.author_paper[author]
                        nump = len(papers)
                        if nump >1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper           
                        
                    if paper in self.paper_org:
                        words = self.paper_org[paper]
                        numw = len(words)
                        wordid = random.randrange(numw) 
                        word = words[wordid]
                    
                        papers = self.org_paper[word]
                        nump = len(papers)
                        if nump >1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper  
                            
                outfile.write(outline + "\n")
        outfile.close()
        
        print ("walks done")
        
################# Compare Lists ################

def tanimoto(p,q):
    c = [v for v in p if v in q]
    return float(len(c) / (len(p) + len(q) - len(c)))



################# Paper similarity ################

def generate_pair(pubs,outlier): ##求匹配相似度
    dirpath = 'gene'
    
    paper_org = {}
    paper_conf = {}
    paper_author = {}
    paper_word = {}
    
    temp=set()
    with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)                       
    for line in temp: 
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_org:
                paper_org[p] = []
            paper_org[p].append(a)
    temp.clear()
    
    with open(dirpath + "/paper_conf.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)                       
    for line in temp: 
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_conf:
                paper_conf[p]=[]
            paper_conf[p]=a
    temp.clear()
    
    with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)                       
    for line in temp: 
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_author:
                paper_author[p] = []
            paper_author[p].append(a)
    temp.clear()
       
    with open(dirpath + "/paper_word.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)                       
    for line in temp: 
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_word:
                paper_word[p] = []
            paper_word[p].append(a)
    temp.clear()
    
    
    paper_paper = np.zeros((len(pubs),len(pubs)))
    for i,pid in enumerate(pubs):
        if i not in outlier:
            continue
        for j,pjd in enumerate(pubs):
            if j==i:
                continue
            ca=0
            cv=0
            co=0
            ct=0
          
            if pid in paper_author and pjd in paper_author:
                ca = len(set(paper_author[pid])&set(paper_author[pjd]))*1.5
            if pid in paper_conf and pjd in paper_conf and 'null' not in paper_conf[pid]:
                cv = tanimoto(set(paper_conf[pid]),set(paper_conf[pjd]))
            if pid in paper_org and pjd in paper_org:
                co = tanimoto(set(paper_org[pid]),set(paper_org[pjd]))
            if pid in paper_word and pjd in paper_word:
                ct = len(set(paper_word[pid])&set(paper_word[pjd]))/3
                    
            paper_paper[i][j] =ca+cv+co+ct
            
    return paper_paper

    
        
################# Evaluate ################
        
def pairwise_evaluate(correct_labels,pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

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


################# Save Paper Features ################

def save_relation(name_pubs_raw, name): # 保存论文的各种feature
    name_pubs_raw = load_json('genename', name_pubs_raw)
    ## trained by all text in the datasets. Training code is in the cells of "train word2vec"
    save_model_name = "word2vec/Aword2vec.model"
    model_w = word2vec.Word2Vec.load(save_model_name)
    
    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
    stopword = ['at','based','in','of','for','on','and','to','an','using','with','the','by','we','be','is','are','can']
    stopword1 = ['university','univ','china','department','dept','laboratory','lab','school','al','et',
                 'institute','inst','college','chinese','beijing','journal','science','international']
    
    f1 = open ('gene/paper_author.txt','w',encoding = 'utf-8')
    f2 = open ('gene/paper_conf.txt','w',encoding = 'utf-8')
    f3 = open ('gene/paper_word.txt','w',encoding = 'utf-8')
    f4 = open ('gene/paper_org.txt','w',encoding = 'utf-8')

    
    taken = name.split("_")
    name = taken[0] + taken[1]
    name_reverse = taken[1]  + taken[0]
    if len(taken)>2:
        name = taken[0] + taken[1] + taken[2]
        name_reverse = taken[2]  + taken[0] + taken[1]
    
    authorname_dict={}
    ptext_emb = {}  
    
    tcp=set()  
    for i,pid in enumerate(name_pubs_raw):
        
        pub = name_pubs_raw[pid]
        
        #save authors
        org=""
        for author in pub["authors"]:
            authorname = re.sub(r,'', author["name"]).lower()
            taken = authorname.split(" ")
            if len(taken)==2: ##检测目前作者名是否在作者词典中
                authorname = taken[0] + taken[1]
                authorname_reverse = taken[1]  + taken[0] 
            
                if authorname not in authorname_dict:
                    if authorname_reverse not in authorname_dict:
                        authorname_dict[authorname]=1
                    else:
                        authorname = authorname_reverse 
            else:
                authorname = authorname.replace(" ","")
            
            if authorname!=name and authorname!=name_reverse:
                f1.write(pid + '\t' + authorname + '\n')
        
            else:
                if "org" in author:
                    org = author["org"]
                    
                    
        #save org 待消歧作者的机构名
        pstr = org.strip()
        pstr = pstr.lower() #小写
        pstr = re.sub(r,' ', pstr) #去除符号
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip() #去除多余空格
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word)>1]
        pstr = [word for word in pstr if word not in stopword1]
        pstr = [word for word in pstr if word not in stopword]
        pstr=set(pstr)
        for word in pstr:
            f4.write(pid + '\t' + word + '\n')

        
        #save venue
        pstr = pub["venue"].strip()
        pstr = pstr.lower()
        pstr = re.sub(r,' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word)>1]
        pstr = [word for word in pstr if word not in stopword1]
        pstr = [word for word in pstr if word not in stopword]
        for word in pstr:
            f2.write(pid + '\t' + word + '\n')
        if len(pstr)==0:
            f2.write(pid + '\t' + 'null' + '\n')

            
        #save text
        pstr = ""    
        keyword=""
        if "keywords" in pub:
            for word in pub["keywords"]:
                keyword=keyword+word+" "
        pstr = pstr + pub["title"]
        pstr=pstr.strip()
        pstr = pstr.lower()
        pstr = re.sub(r,' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word)>1]
        pstr = [word for word in pstr if word not in stopword]
        for word in pstr:
            f3.write(pid + '\t' + word + '\n')
        
        #save all words' embedding
        pstr = keyword + " " + pub["title"] + " " + pub["venue"] + " " + org
        if "year" in pub:
              pstr = pstr +  " " + str(pub["year"])
        pstr=pstr.strip()
        pstr = pstr.lower()
        pstr = re.sub(r,' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word)>2]
        pstr = [word for word in pstr if word not in stopword]
        pstr = [word for word in pstr if word not in stopword1]

        words_vec=[]
        for word in pstr:
            if (word in model_w):
                words_vec.append(model_w[word])
        if len(words_vec)<1:
            words_vec.append(np.zeros(100))
            tcp.add(i)
            #print ('outlier:',pid,pstr)

        emb = np.mean(words_vec, 0)
        ptext_emb[pid] = triplet.get_inter(np.stack([emb]))[0] if triplet.fited else emb
        
    #  ptext_emb: key is paper id, and the value is the paper's text embedding
    dump_data(ptext_emb,'gene','ptext_emb.pkl')
    # the paper index that lack text information
    dump_data(tcp,'gene','tcp.pkl')
            
 
    f1.close()
    f2.close()
    f3.close()
    f4.close()

# # train word2vec

# In[3]:


## SAVE all text in the datasets

import codecs
import json
from os.path import join
import pickle
import os
import re


pubs_raw = load_json("train","train_pub.json")
pubs_raw1 = load_json("sna_data","sna_valid_pub.json")
r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
f1 = open ('gene/all_text.txt','w',encoding = 'utf-8')

for i,pid in enumerate(pubs_raw):
    pub = pubs_raw[pid]
    
    for author in pub["authors"]:
        if "org" in author:
                org = author["org"]
                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(r,' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                f1.write(pstr+'\n')
            
    title = pub["title"]
    pstr=title.strip()
    pstr = pstr.lower()
    pstr = re.sub(r,' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr+'\n')
    
    if "abstract" in pub and type(pub["abstract"]) is str:
        abstract = pub["abstract"]
        pstr=abstract.strip()
        pstr = pstr.lower()
        pstr = re.sub(r,' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        f1.write(pstr+'\n')
        
    venue = pub["venue"]
    pstr=venue.strip()
    pstr = pstr.lower()
    pstr = re.sub(r,' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr+'\n')
    
for i,pid in enumerate(pubs_raw1):
    pub = pubs_raw1[pid]
    
    for author in pub["authors"]:
        if "org" in author:
                org = author["org"]
                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(r,' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                f1.write(pstr+'\n')
            
    title = pub["title"]
    pstr=title.strip()
    pstr = pstr.lower()
    pstr = re.sub(r,' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr+'\n')
    
    if "abstract" in pub and type(pub["abstract"]) is str:
        abstract = pub["abstract"]
        pstr=abstract.strip()
        pstr = pstr.lower()
        pstr = re.sub(r,' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        f1.write(pstr+'\n')
        
    venue = pub["venue"]
    pstr=venue.strip()
    pstr = pstr.lower()
    pstr = re.sub(r,' ', pstr)
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
    f1.write(pstr+'\n')

        
        
f1.close()

# In[4]:


from gensim.models import word2vec

sentences = word2vec.Text8Corpus(r'gene/all_text.txt')
model = word2vec.Word2Vec(sentences, size=100,negative =5, min_count=2, window=5)
model.save('word2vec/Aword2vec.model')



# # name disambiguation (train)


# In[5]:


import re
from gensim.models import word2vec
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


pubs_raw = load_json("train","train_pub.json")
name_pubs = load_json("train","train_author.json")

result=[]
for n,name in enumerate(name_pubs):
    ilabel=0
    pubs=[] # all papers
    labels=[] # ground truth
    
    for author in name_pubs[name]:
        iauthor_pubs = name_pubs[name][author]
        for pub in iauthor_pubs:
            pubs.append(pub)
            labels.append(ilabel)
        ilabel += 1
        
    print (n,name,len(pubs))
    
    
    if len(pubs)==0:
        result.append(0)
        continue
    
    ##保存关系
    ###############################################################
    name_pubs_raw = {}
    for i,pid in enumerate(pubs):
        name_pubs_raw[pid] = pubs_raw[pid]
        
    dump_json(name_pubs_raw, 'genename', name+'.json', indent=4)
    save_relation(name+'.json', name)  
    ###############################################################
    
    
    
    ##元路径游走类
    ###############################################################r
    mpg = MetaPathGenerator()
    mpg.read_data("gene")
    ###############################################################
    
  
    
    ##论文关系表征向量
    ############################################################### 
    all_embs=[]
    rw_num =3
    cp=set()
    for k in range(rw_num):
        mpg.generate_WMRW("gene/RW.txt",5,20) #生成路径集
        sentences = word2vec.Text8Corpus(r'gene/RW.txt')
        model = word2vec.Word2Vec(sentences, size=100,negative =25, min_count=1, window=10)
        embs=[]
        for i,pid in enumerate(pubs):
            if pid in model:
                embs.append(model[pid])
            else:
                cp.add(i)
                embs.append(np.zeros(100))
        all_embs.append(embs)
    all_embs= np.array(all_embs)
    print ('relational outlier:',cp)
    ############################################################### 

    
    
    ##论文文本表征向量
    ###############################################################   
    ptext_emb=load_data('gene','ptext_emb.pkl')
    tcp=load_data('gene','tcp.pkl')
    print ('semantic outlier:',tcp)
    tembs=[]
    for i,pid in enumerate(pubs):
        tembs.append(ptext_emb[pid])
    ############################################################### 
    
    ##离散点
    outlier=set()
    for i in cp:
        outlier.add(i)
    for i in tcp:
        outlier.add(i)
    
    ##网络嵌入向量相似度
    sk_sim = np.zeros((len(pubs),len(pubs)))
    for k in range(rw_num):
        sk_sim = sk_sim + pairwise_distances(all_embs[k],metric="cosine")
    sk_sim =sk_sim/rw_num    
    
    ##文本相似度
    t_sim = pairwise_distances(tembs,metric="cosine")
    
    w=1
    sim = (np.array(sk_sim) + w*np.array(t_sim))/(1+w)
    
    
    
    ##evaluate
    ###############################################################
    pre = DBSCAN(eps = 0.2, min_samples = 4,metric ="precomputed").fit_predict(sim)
    
    
    for i in range(len(pre)):
        if pre[i]==-1:
            outlier.add(i)
    
    ## assign each outlier a label
    paper_pair = generate_pair(pubs,outlier)
    paper_pair1 = paper_pair.copy()
    K = len(set(pre))
    for i in range(len(pre)):
        if i not in outlier:
            continue
        j = np.argmax(paper_pair[i])
        while j in outlier:
            paper_pair[i][j]=-1
            j = np.argmax(paper_pair[i])
        if paper_pair[i][j]>=1.5:
            pre[i]=pre[j]
        else:
            pre[i]=K
            K=K+1
    
    ## find nodes in outlier is the same label or not
    for ii,i in enumerate(outlier):
        for jj,j in enumerate(outlier):
            if jj<=ii:
                continue
            else:
                if paper_pair1[i][j]>=1.5:
                    pre[j]=pre[i]
            
            
    
    labels = np.array(labels)
    pre = np.array(pre)
    print (labels,len(set(labels)))
    print (pre,len(set(pre)))
    pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(labels,pre)
    print (pairwise_precision, pairwise_recall, pairwise_f1)
    result.append(pairwise_f1)

    print ('avg_f1:', np.mean(result))

# train emb

ptext_emb = load_data('gene','ptext_emb.pkl')
triplet.fit(generate_triplet(ptext_emb, name_pubs))


# # name disambiguation (valid)

# In[7]:


import re
from gensim.models import word2vec
from sklearn.cluster import DBSCAN
import numpy as np

pubs_raw = load_json("sna_data","sna_valid_pub.json")
name_pubs1 = load_json("sna_data","sna_valid_example_evaluation_scratch.json")

result={}

for n,name in enumerate(name_pubs1):
    pubs=[]
    for cluster in name_pubs1[name]:
        pubs.extend(cluster)
    
    
    print (n,name,len(pubs))
    if len(pubs)==0:
        result[name]=[]
        continue
    
    
    ##保存关系
    ###############################################################
    name_pubs_raw = {}
    for i,pid in enumerate(pubs):
        name_pubs_raw[pid] = pubs_raw[pid]
        
    dump_json(name_pubs_raw, 'genename', name+'.json', indent=4)
    save_relation(name+'.json', name)  
    ###############################################################
    
    
    
    ##元路径游走类
    ###############################################################r
    mpg = MetaPathGenerator()
    mpg.read_data("gene")
    ###############################################################
    

    
    ##论文关系表征向量
    ############################################################### 
    all_embs=[]
    rw_num = 10
    cp=set()
    for k in range(rw_num):
        mpg.generate_WMRW("gene/RW.txt",5,20)
        sentences = word2vec.Text8Corpus(r'gene/RW.txt')
        model = word2vec.Word2Vec(sentences, size=100,negative =25, min_count=1, window=10)
        embs=[]
        for i,pid in enumerate(pubs):
            if pid in model:
                embs.append(model[pid])
            else:
                cp.add(i)
                embs.append(np.zeros(100))
        all_embs.append(embs)
    all_embs= np.array(all_embs)
    print (cp)    
    ############################################################### 
 


    ##论文语义表征向量
    ###############################################################  
    ptext_emb=load_data('gene','ptext_emb.pkl')
    tembs=[]
    for i,pid in enumerate(pubs):
        tembs.append(ptext_emb[pid])
    
    sk_sim = np.zeros((len(pubs),len(pubs)))
    for k in range(rw_num):
        sk_sim = sk_sim + pairwise_distances(all_embs[k],metric="cosine")
    sk_sim =sk_sim/rw_num    
    

    tembs = pairwise_distances(tembs,metric="cosine")
   
    w=1
    sim = (np.array(sk_sim) + w*np.array(tembs))/(1+w)
    ############################################################### 
    
    
  
    ##evaluate
    ###############################################################
    pre = DBSCAN(eps = 0.2, min_samples = 4,metric ="precomputed").fit_predict(sim)
    pre= np.array(pre)
    
    outlier=set()
    for i in range(len(pre)):
        if pre[i]==-1:
            outlier.add(i)
    for i in cp:
        outlier.add(i)
    for i in tcp:
        outlier.add(i)
            
        
    ##基于阈值的相似性匹配
    paper_pair = generate_pair(pubs,outlier)
    paper_pair1 = paper_pair.copy()
    K = len(set(pre))
    for i in range(len(pre)):
        if i not in outlier:
            continue
        j = np.argmax(paper_pair[i])
        while j in outlier:
            paper_pair[i][j]=-1
            j = np.argmax(paper_pair[i])
        if paper_pair[i][j]>=1.5:
            pre[i]=pre[j]
        else:
            pre[i]=K
            K=K+1
    
    for ii,i in enumerate(outlier):
        for jj,j in enumerate(outlier):
            if jj<=ii:
                continue
            else:
                if paper_pair1[i][j]>=1.5:
                    pre[j]=pre[i]
            
    

    print (pre,len(set(pre)))
    
    result[name]=[]
    for i in set(pre):
        oneauthor=[]
        for idx,j in enumerate(pre):
            if i == j:
                oneauthor.append(pubs[idx])
        result[name].append(oneauthor)
    

dump_json(result, "genetest", "result_valid.json",indent =4)

# In[ ]:



