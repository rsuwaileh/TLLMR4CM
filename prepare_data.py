import random 
import math
import os
import spacy
#!pip install spacy
#!python -m spacy download en

def read_annotated_data(path):
    tokens = []
    labels = []
    t = []
    l = []
    
    for token in open(path, encoding='utf-8').read().splitlines(): 
        if token == '':
            tokens.append(t)
            labels.append(l)
            t = []
            l = []
            continue
        splits = token.split()
        t.append(splits[0])
        l.append(splits[1])
        
    if len(t) > 0 and len(l) > 0:
        t.append(splits[0])
        l.append(splits[1])        
    return tokens, labels

def read_unannotated_data(path):
    tokens = []
    labels = []
    first = True
    for line in open(path, encoding='utf-8').read().splitlines(): 
        if first:
            first = False
            continue
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(line.split("\t")[2])
        t = []
        for token in doc:
            if not all(ord(c) < 128 for c in token.text):
                cleaned = "".join([c if ord(c) < 128 else c.encode().decode('ascii',errors='ignore') for c in token.text]) 
                cleaned = "".join(cleaned.split())
                if cleaned == "" or cleaned == " ":
                    continue
                t.append(cleaned)                    
            else:
                t.append(token.text)
                
        
        tokens.append(t)
        labels.append(["O"]*len(t))
       
    return tokens, labels

def text_data(path):
    tokens = []
    labels = []
    for line in open(path, encoding='utf-8').read().splitlines(): 
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(line)
        t = []
        for token in doc:
            if not all(ord(c) < 128 for c in token.text):
                cleaned = "".join([c if ord(c) < 128 else c.encode().decode('ascii',errors='ignore') for c in token.text]) 
                cleaned = "".join(cleaned.split())
                if cleaned == "" or cleaned == " ":
                    continue
                t.append(cleaned)                    
            else:
                t.append(token.text)
                
        
        tokens.append(t)
        labels.append(["O"]*len(t))

       
    return tokens, labels

def dump_biolike(out_path, tokens, labels):
    writer = open(out_path, 'w', encoding='utf-8', newline="")
    
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            writer.write(tokens[i][j] + " " + labels[i][j] + "\n")
        writer.write("\n")


def convert_txt2biolike(txt_path):
	tokens, labels = text_data(txt_path)
    dump_biolike(txt_path.replace(".txt", "-biolike.txt"), tokens, labels)
        

def convert_tsv2biolike(tsv_path):
	tokens, labels = read_unannotated_data(tsv_path)
    dump_biolike(json_path.replace(".tsv", "-biolike.txt"), tokens, labels)
     