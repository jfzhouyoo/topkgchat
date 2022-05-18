import json
import csv
import spacy
from tqdm import tqdm
import networkx as nx
import sys
sys.path.append('../')
import multiprocessing
from functools import partial


blacklist = set(["from", "as", "hey", "more", "either", "in", "and", "on", "an", "when", "too", "to", "i", "do", "can", "be", "that", "or", "the", "a", "of", "for", "is", "was", "the", "-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be","mine","us","em",
                 "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"])
   
def read_csv(data_path=None):
    # python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    concept_net = nx.read_gpickle('data/cpnet.graph')
    cpnet_vocab = list(concept_net.nodes)

    def hard_ground(sent):
        sent = sent.lower()
        doc = nlp(sent)
        res = set()
        for t in doc:
            if t.lemma_ in cpnet_vocab and t.lemma_ not in blacklist:
                if t.pos_ == "NOUN" or t.pos_ == "VERB":
                    res.add(t.lemma_)
        if len(res) == 0:
            for t in doc:
                if t.lemma_ in cpnet_vocab and t.lemma_ not in blacklist:
                    res.add(t.lemma_)
        return res
        
    source = []
    with open('raw_files/' + data_path + 'source.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            source.append((row[1], row[2]))
    target = []
    with open('raw_files/' + data_path + 'target.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            target.append(row[1])
    sents = [ [s[0], t, s[1]] for s, t in zip(source, target) ]
        
    res = []
    print("Begin matching concepts.")
    no_concepts_counter = 0
    for sid, sentence in tqdm(enumerate(sents), total=len(sents)):#, desc="grounding batch_id:%d"%batch_id):
        start_concepts = list(hard_ground(sentence[0]))
        bridge_concepts = list(hard_ground(sentence[1]))
        target_concepts = list(hard_ground(sentence[2]))
        if len(start_concepts) == 0 or len(bridge_concepts)  == 0 or len(target_concepts) == 0:
            no_concepts_counter += 1
            # print('no concepts ', sentence)
            
        res.append({
            "sid": sid,
            "dialog": sentence,
            "concepts": [start_concepts, bridge_concepts, target_concepts],
            "s_c": start_concepts,
            "b_c": bridge_concepts,
            "t_c": target_concepts,
        })
    print('no_concepts_counter ', no_concepts_counter)
    with open('data/' + data_path + "/concepts_nv.json".format(type), 'w') as f:
        for line in res:            
            json.dump(line, f)
            f.write('\n')

def handleConvAI(ns, start_idx):
    print("Begin matching concepts.")
    batch_data = ns.all_data[start_idx: start_idx + ns.batch_size]
    cpnet_vocab = ns.cpnet_vocab
    blacklist = ns.blacklist

    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    def hard_ground(sent):
        sent = sent.lower()
        doc = nlp(sent)
        res = set()
        for t in doc:
            if t.lemma_ in cpnet_vocab and t.lemma_ not in blacklist:
                if t.pos_ == "NOUN" or t.pos_ == "VERB":
                    res.add(t.lemma_)
        if len(res) == 0:
            for t in doc:
                if t.lemma_ in cpnet_vocab and t.lemma_ not in blacklist:
                    res.add(t.lemma_)
        return res

    res = []
    for dialog in tqdm(batch_data, total=len(batch_data), mininterval=30):
        no_concepts_counter = 0
        concepts = []
        for sen in dialog['session']:
            sen_concepts = list(hard_ground(sen))
            if len(sen_concepts) == 0:
                no_concepts_counter += 1
                # print('no concepts ', sen)
            concepts.append(sen_concepts)
        res.append({
            "sid": dialog['sid'],
            "session": dialog['session'],
            "concepts": concepts,
        })
    print('no_concepts_counter ', no_concepts_counter)
    return res

def read_convai(filename):
    dialogs = []
    with open('raw_files/' + filename + '.txt') as f:
        dialog_id = 0
        session = []
        for line in f:
            turn_num = int(line[:2])
            if turn_num == 1 and len(session) > 1:
                dialog_id += 1
                dialogs.append({
                    'sid': dialog_id,
                    'session': session,
                })
                session = []
            line = line[2:] if turn_num < 10 else line[3:]
            pair = line.rstrip('\n').split('\t')
            if '__SILENCE__' not in pair:
                session.append(pair[0])
            session.append(pair[1])
    dialog_id += 1
    dialogs.append({
        'sid': dialog_id,
        'session': session,
    })
    print('total dialog ', filename, dialog_id) # 17878 

    concept_net = nx.read_gpickle('data/cpnet.graph')
    cpnet_vocab = list(concept_net.nodes)


    ns = multiprocessing.Manager().Namespace()
    ns.all_data = dialogs
    ns.cpnet_vocab = cpnet_vocab
    ns.blacklist = blacklist

    ns.batch_size = int(len(ns.all_data) / 30) + 10
    mylist = [i for i in range(0, len(ns.all_data), ns.batch_size)]  # valid

    p = multiprocessing.Pool()
    func = partial(handleConvAI, ns)
    result = p.map(func, mylist)

    flat_res = [ res for p_result in result for res in p_result ]

    if 'train' in filename:
        with open('data/TGConv/train/raw.json', 'w') as f:
            for line in flat_res[:-2681]:            
                json.dump(line, f)
                f.write('\n')
        with open('data/TGConv/dev/raw.json', 'w') as f:
            for line in flat_res[-2681:]:            
                json.dump(line, f)
                f.write('\n')
    else:
        with open('data/TGConv/test/raw.json', 'w') as f:
            for line in flat_res:            
                json.dump(line, f)
                f.write('\n')


if __name__ == "__main__":
    read_convai('ConvAI2/test')
    read_convai('ConvAI2/train')
    
    read_csv('OTTers/train/') # no_concepts_counter  23
    read_csv('OTTers/dev/') # no_concepts_counter  9
    read_csv('OTTers/test/') # no_concepts_counter  12

# 70 min +
# nohup python -u preprocess/2ground_concepts.py > common.log 2>&1 &