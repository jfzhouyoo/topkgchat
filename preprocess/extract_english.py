import json
import networkx as nx
from tqdm import tqdm
import nltk
import math
import os

relation_mapping = {}

merge_relations = [
    'atlocation/locatednear',
    'capableof',
    'causes/causesdesire/*motivatedbygoal',
    'createdby',
    'desires',
    'antonym/distinctfrom',
    'hascontext',
    'hasproperty',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'isa/instanceof/definedas',
    'madeof',
    'notcapableof',
    'notdesires',
    'partof/*hasa',
    'relatedto/similarto/synonym',
    'usedfor',
    'receivesaction',
]

for line in merge_relations:
    ls = line.strip().split('/')
    rel = ls[0]
    for l in ls:
        if l.startswith("*"):
            relation_mapping[l[1:]] = "*" + rel
        else:
            relation_mapping[l] = rel

print(relation_mapping)

blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords += ["like", "gone", "did", "going", "would", "could", "get", "in", "up", "may", "wanter"]

def not_save(cpt):
    if cpt in blacklist:
        return True
    for t in cpt.split("_"):
        if t in nltk_stopwords:
            return True
    return False

def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s

def extract_english_and_save():
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    if os.path.exists('data/cpnet.graph'):
        print('data/cpnet.graph exist')
        return
    only_english = []
    with open('raw_files/conceptnet-assertions-5.7.0.csv', encoding="utf8") as f:
        for line in f:
            ls = line.rstrip('\n').split('\t')
            if ls[2].startswith('/c/en/') and ls[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = ls[1].split("/")[-1].lower()
                head = del_pos(ls[2]).split("/")[-1].lower()
                tail = del_pos(ls[3]).split("/")[-1].lower()

                if '_' in head or '_' in tail:
                    continue
                if not head.replace("_", "").replace("-", "").isalpha():
                    continue

                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue

                if rel not in relation_mapping:
                    continue
                rel = relation_mapping[rel]
                if rel.startswith("*"):
                    rel = rel[1:]
                    tmp = head
                    head = tail
                    tail = tmp

                data = json.loads(ls[4])

                only_english.append("\t".join([rel, head, tail, str(data["weight"])]))

    with open('data/conceptnet_en.txt', "w", encoding="utf8") as f:
        f.write("\n".join(only_english))

    # graph = nx.MultiGraph()
    graph = nx.Graph()

    for line in tqdm(only_english, desc="saving to graph"):
        ls = line.split('\t')
        rel = ls[0]
        subj = ls[1]
        obj = ls[2]
        weight = float(ls[3])
        # if ls[1] not in concept2id:
        #     continue
        # if ls[2] not in concept2id:
        #     continue
        if rel == "hascontext":
            continue
        if not_save(ls[1]) or not_save(ls[2]):
            continue
        if rel == "relatedto" or rel == "antonym":
            weight -= 0.3
            # continue
        if subj == obj: # delete loops
            continue
        weight = 1+float(math.exp(1-weight))
        graph.add_edge(subj, obj, rel=rel, weight=weight)
        # graph.add_edge(obj, subj, rel=rel+len(relation2id), weight=weight)

    # concept2id = { w:i for i, w in enumerate(graph.nodes) }

    # with open('conceptnet/c2id.json', 'w') as f:
    #     json.dump(concept2id, f,ensure_ascii=False)

    nx.write_gpickle(graph, 'conceptnet/cpnet.graph')

if __name__ == "__main__":
    extract_english_and_save()