import re
import numpy as np

class Text2Digit():
    def __init__(self, config):

        if config.char_io: scheme = "io"
        elif config.char_bio: scheme = "bio"
        else: scheme = "biotag"

        self.train_vals = load_from_file(config.train, scheme=scheme)
        self.dev_vals = load_from_file(config.dev, scheme=scheme)
        self.test_vals = load_from_file(config.test, scheme=scheme)

        self.token_loc = 0
        self.atag_loc = 1
        self.ptag_loc = 2

        self.tokens = self.train_vals[self.token_loc]+self.dev_vals[self.token_loc]+self.test_vals[self.token_loc]
        self.atags = self.train_vals[self.atag_loc]+self.dev_vals[self.atag_loc]+self.test_vals[self.atag_loc]
        self.atag_dict = prep_tag_dict_skip(self.atags)
        if config.char_io: self.atag_bio_dict = dict([("O",0), ("I", 1)])
        elif config.char_bio: self.atag_bio_dict = dict([("O",0), ("B",1), ("I", 2)])
        else: self.atag_bio_dict = get_bio_tag_dict(self.atag_dict)

        self.ptags = self.train_vals[self.ptag_loc]+self.dev_vals[self.ptag_loc]+self.test_vals[self.ptag_loc]

        if config.use_pretrain:
            pretrain_embedding, pretrain_vocab = loadPretrain(config.pretrain_embed, target_words=None, debug=config.small_debug)
        else:
            pretrain_embedding, pretrain_vocab = generRandom(self.tokens)
        self.unk_id = pretrain_vocab['<unk>']
        self.vocab = pretrain_vocab
        self.pretrain_embedding = pretrain_embedding

        #train_w, train_c, train_triggers = stat_basic_corpus(self.train_vals[self.token_loc], self.train_vals[self.atag_loc])
        #dev_w, dev_c, _ = stat_basic_corpus(self.dev_vals[self.token_loc], self.dev_vals[self.atag_loc])
        #test_w, test_c, test_triggers = stat_basic_corpus(self.test_vals[self.token_loc], self.test_vals[self.atag_loc])
        #stat_unk(train_w, train_c, train_triggers, test_w, test_c, test_triggers, self.vocab, self.char_vocab)
        #sys.exit(0)

        self.get_digit_data()


    def get_digit_data(self):
        '''
            each item in self.train: [sent_token, sent_atag, sent_btag, sent_ptag]
            sent_token: (sent_len), a list of words
            sent_atag: (sent_len), a list of trig tags
            sent_btag: (ent_num), a list of ent tags
            sent_ptag: (sent_len), a list of new word positions.
        '''
        digit_train_tokens = digitalize(self.train_vals[self.token_loc], self.vocab, self.unk_id)
        #digit_train_atags = digitalize(self.train_vals[self.atag_loc], self.atag_dict, None)
        digit_train_atags = digitalize2(self.train_vals[self.atag_loc], self.atag_dict, self.atag_bio_dict)

        digit_dev_tokens = digitalize(self.dev_vals[self.token_loc], self.vocab, self.unk_id)
        #digit_dev_atags = digitalize(self.dev_vals[self.atag_loc], self.atag_dict, None)
        digit_dev_atags = digitalize2(self.dev_vals[self.atag_loc], self.atag_dict, self.atag_bio_dict)

        digit_test_tokens = digitalize(self.test_vals[self.token_loc], self.vocab, self.unk_id)
        #digit_test_atags = digitalize(self.test_vals[self.atag_loc], self.atag_dict, None)
        digit_test_atags = digitalize2(self.test_vals[self.atag_loc], self.atag_dict, self.atag_bio_dict)

        self.train = (digit_train_tokens, digit_train_atags, self.train_vals[self.ptag_loc])
        self.dev = (digit_dev_tokens, digit_dev_atags, self.dev_vals[self.ptag_loc])
        self.test = (digit_test_tokens, digit_test_atags, self.test_vals[self.ptag_loc])

        # check data
        #for i in range(1):
        #    print "------ sent", i
        #    print " ".join(self.test_vals[0][i]).encode("utf8"), digit_test_tokens[i]
        #    print self.test_vals[1][i], digit_test_atags[i]
        #    for wordid, word_tags in enumerate(self.test_vals[3][i]):
        #        print wordid, " ".join(word_tags)
        #        if len([1 for tag in word_tags if tag != "O"]) > 0:
        #            print wordid, word_tags
        #assert False

def prep_tag_dict_skip(tags_in_sents):
    tags_data = sorted(list(set([tag for tags in tags_in_sents for tag in tags if tag != "O" and tag[:2] not in ["B-", "I-"]])))
    tags_data = dict(zip(tags_data, range(1, len(tags_data)+1)))
    tags_data["O"] = 0
    return tags_data

def get_bio_tag_dict(tags_data):
    bio_tags = sorted(["B-"+tag for tag in tags_data.keys() if tag != "O"])
    bio_tags.extend(sorted(["I-"+tag for tag in tags_data.keys() if tag != "O"]))
    bio_tags = dict(zip(bio_tags, range(1, len(bio_tags)+1)))
    bio_tags["O"]=0
    return bio_tags

def prep_tag_dict(tags_in_sents, sepchar=None):
    if sepchar != None:
        tags_data = sorted(list(set([tag.split("#")[0] for tags in tags_in_sents for tag in tags if tag != "O"])))
    else:
        tags_data = sorted(list(set([tag for tags in tags_in_sents for tag in tags if tag != "O"])))
    tags_data = dict(zip(tags_data, range(1, len(tags_data)+1)))
    tags_data["O"] = 0
    return tags_data

# arr: list
def add_st_ed(arr, st_token, ed_token):
    arr.insert(0, st_token)
    arr.append(ed_token)
    return arr

def load_from_file(filename, add_st_ed_token=True, scheme=None):
    '''
        Input:
            one line one sentence format. eg: w1 w2 w3[\t]atag1 atag2 atag3
            atag: event types for words
        Output:
            tokens: [c11, c12,..., w1, c21, ..., w2, c31, ..., w3], mix sequence
            atags:  [O, O, ..., at1,   O, ..., at2,  O, ..., at3], event type tags for mix item
            ptags:  [wp1, wp2, wp3], new position of words
    '''
    content = open(filename, "r").readlines()
    content = [line.rstrip("\n").split("\t") for line in content if len(line.strip())>1]
    data = [[item.strip().split() for item in line] for line in content]

    # trim sents to max_length
    max_len = 70
    tokens = [item[0][:max_len] for item in data]
    atags  = [item[1][:max_len] for item in data]

    tokens = process_tokens(tokens)
    tokens, atags, ptags = append_chars(tokens, atags, scheme=scheme)

    return_values = []
    return_values.append(tokens)
    return_values.append(atags)
    return_values.append(ptags)
    return return_values

def append_chars(tokens, atags, scheme):
    new_tokens = []
    new_atags = []
    ptags = []

    for sentid, (toks, atgs) in enumerate(zip(tokens, atags)):
        new_toks = []
        new_atgs = []
        ptgs = []
        for widx, (tok, atg) in enumerate(zip(toks, atgs)):
            #print tok
            for charid, char in enumerate(tok[:3]):
                new_toks.append(char.encode("utf8"))
                # v1: O for all chars
                #new_atgs.append("O")
                # v2: BIO-trig for chars
                if atg == "O": new_atgs.append("O")
                else:
                    if scheme=="io": new_atgs.append("I")
                    elif scheme=="bio" and charid==0: new_atgs.append("B")
                    elif scheme=="bio" and charid>0: new_atgs.append("I")
                    elif charid == 0: new_atgs.append("B-"+atg)
                    elif charid > 0: new_atgs.append("I-"+atg)

            #    print char,
            #print
            new_toks.append(tok)
            new_atgs.append(atgs[widx])
            ptgs.append(len(new_toks)-1)

        #print " ".join(new_toks)
        #print new_atgs

        #if sentid == 10: break
        new_tokens.append(new_toks)
        new_atags.append(new_atgs)
        ptags.append(ptgs)
    return new_tokens, new_atags, ptags


def loadVocab(filename):
    content = open(filename, "r").readlines()
    words = [w.strip() for w in content]
    vocab = dict(zip(words, range(len(words))))
    vocab["unk"] = len(vocab)   # not sure about 19488, not appeared in training
    vocab["<unk>"] = len(vocab) # 19489 is unk
    return vocab

def generRandom(tokens, word_dim=300):
    unk_word = "<unk>"
    vocab = {}
    words = set([w for item in tokens for w in item])
    vocab = dict(zip(sorted(words), range(len(words))))

    random_embedding = []
    for i in range(len(vocab)):
        random_embedding.append(np.random.uniform(-0.01, 0.01, word_dim).tolist())

     # add unk_word
    if unk_word not in vocab:
        vocab[unk_word] = len(vocab)
        random_embedding.append(np.random.uniform(-0.01, 0.01, word_dim).tolist())

    return np.matrix(random_embedding), vocab

def loadPretrain(filepath, target_words=None, debug=False):
    unk_word = "<unk>"
    st = '<s>'
    ed = '</s>'
    content = open(filepath, "r").readlines()

    if target_words is not None and unk_word not in target_words: target_words.append(unk_word)
    pretrain_embedding = []
    pretrain_vocab = {} # word: word_id
    if debug: content = content[:20000]
    for word_id, line in enumerate(content):
        #word_item = line.decode("utf8").strip().split()
        word_item = line.strip().split()
        if len(word_item) == 2: continue
        word_text = word_item[0]
        if target_words is not None and word_text not in target_words: continue
        embed_word = [float(item) for item in word_item[1:]]
        pretrain_embedding.append(embed_word)
        pretrain_vocab[word_text] = len(pretrain_vocab)

    # add unk_word
    word_dim = len(pretrain_embedding[-1])
    if st not in pretrain_vocab:
        pretrain_vocab[st] = len(pretrain_vocab)
        pretrain_embedding.append(np.random.uniform(-0.01, 0.01, word_dim).tolist())
    if ed not in pretrain_vocab:
        pretrain_vocab[ed] = len(pretrain_vocab)
        pretrain_embedding.append(np.random.uniform(-0.01, 0.01, word_dim).tolist())
    if unk_word not in pretrain_vocab:
        pretrain_vocab[unk_word] = len(pretrain_vocab)
        pretrain_embedding.append(np.random.uniform(-0.01, 0.01, word_dim).tolist())
    return np.matrix(pretrain_embedding), pretrain_vocab

def words2ids(arr, vocab, unk_id, sepchar=None):
    if sepchar is not None:
        new_arr = [str(vocab[witem.split("#")[0]])+"#"+"#".join(witem.split("#")[1:]) if witem.split("#")[0] in vocab else unk_id+"#"+"#".join(witem.split("#")[1:]) for witem in arr]
    else:
        new_arr = [vocab[witem] if witem in vocab else unk_id for witem in arr]
    return new_arr

def digitalize(arr, vocab, unk_id, sepchar=None):
    return [words2ids(item, vocab, unk_id, sepchar=sepchar) for item in arr]

def digitalize2(arr, vocab1, vocab2):
    return [[vocab1[sitem] if sitem in vocab1 else vocab2[sitem] for sitem in item] for item in arr]

def process_word(token):
    word = ""
    if len(token)==4 and token[:2] in ["17", "18", "19", "20"]: # year
        word = token
    else:
        for c in token:
            if c.isdigit(): word += '0'
            else: word += c
        word = re.sub("0+", "0", word)
    return word.lower()

def process_tokens(all_tokens):
    return [[process_word(w) for w in sent] for sent in all_tokens]

def stat_basic_corpus(tokens, evtags):
    print("#sent", len(tokens))
    words = sorted(list(set([w for item in tokens for w in item])))
    print("#word", len(words))
    #for w in words[100:120]:
    #    print(w)
    chars = sorted(list(set([c for item in tokens for w in item for c in w])))
    print("#char", len(chars))
    #for c in chars[100:120]:
    #    print(c)

    evts = [t for item in evtags for t in item if t != 'O']
    print("#triggers", len(evts))

    triggers = [w for toks, tags in zip(tokens, evtags) for w,t in zip(toks, tags) if t!='O']
    print("#triggers", len(triggers))

    return words, chars, triggers

def stat_unk(train_w, train_c, train_triggers, test_w, test_c, test_triggers, vocab, char_vocab):
    test_w_unk = [w for w in test_w if w not in train_w]
    test_c_unk = [c for c in test_c if c not in train_c]
    train_trigger_vocab = set(train_triggers)
    test_trigger_unk_word = [w for w in test_triggers if w not in train_w]
    test_trigger_unk_trig = [w for w in test_triggers if w not in train_trigger_vocab]
    print("#unk w", len(test_w_unk), "%.2f"%(len(test_w_unk)*100.0/len(test_w)))
    print("#unk c", len(test_c_unk), "%.2f"%(len(test_c_unk)*100.0/len(test_c)))
    print("#unk trigger", len(test_trigger_unk_trig), "%.2f"%(len(test_trigger_unk_trig)*100.0/len(test_triggers)))
    print("#unk trigger word", len(test_trigger_unk_word), "%.2f"%(len(test_trigger_unk_word)*100.0/len(test_triggers)))

    #print("----------- unk word")
    #for w in test_w_unk[100:120]: print(w)
    #print("----------- unk char")
    #for w in test_c_unk[:20]: print(w)
    print("----------- unk word (trigger)")
    for w in test_trigger_unk_word[:]: print(w.encode("utf8"))
    #print("----------- unk trigger")
    #for w in test_trigger_unk_trig[:20]: print(w)


    #unk_trig_word_coverbychar = [(w, len(cover_char)) for w, cover_char in unk_trig_word_coverbychar if len(cover_char)>0]
    #print("#unk trigger word cover by char", len(unk_trig_word_coverbychar), "%.2f"%(len(unk_trig_word_coverbychar)*100.0/len(test_trigger_unk_word)))
    #for w, charnum in unk_trig_word_coverbychar[:20]:
    #    print(w, charnum)

    ### unk in pretrain
    #train_w_unk_pretrain = [w for w in train_w if w not in vocab]
    test_w_unk_pretrain = [w for w in test_w if w not in vocab]
    print("#unk w pretrain", len(test_w_unk_pretrain), "%.2f"%(len(test_w_unk_pretrain)*100.0/len(test_w)))
    test_trigger_unk_word_pretrain = [w for w in test_triggers if w not in vocab]
    print("#unk trigger word pretrain", len(test_trigger_unk_word_pretrain), "%.2f"%(len(test_trigger_unk_word_pretrain)*100.0/len(test_triggers)))
    print("----------- unk word (trigger) pretrain")
    for w in test_trigger_unk_word_pretrain[:]: print(w.encode("utf8"))

    #train_c_unk_pretrain = [c for c in train_c if c not in char_vocab]
    #test_c_unk_pretrain = [c for c in test_c if c not in char_vocab]
