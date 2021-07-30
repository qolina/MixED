#coding=utf-8
import torch
import time
import os
import sys
import numpy as np
from util import idsent2words, tensor2var, mask_wpos
from util import cal_prf
from collections import Counter
from mip import *

Tab = "\t"

def eval_model(data_loader, models, loss_function, eval_flag, vocab=None, tags_vocab=None, ana_err=False, use_ilp=False):
    encoder, trig_decoder = models
    encoder.eval()
    trig_decoder.eval()
    loss_all, common_class, common_iden, gold, pred = 0.0, 0, 0, 0, 0
    common_class_ilp, common_iden_ilp, pred_ilp = 0, 0, 0
    if use_ilp:
        ilp_model = Model(sense=MAXIMIZE)

    if ana_err:
        batch_size = -1
        analyze_info = []
        sent_wordids = []
    id2text = dict([(item[1], item[0]) for item in vocab.items()])
    id2tag = dict([(item[1], item[0]) for item in tags_vocab.items()])
    bio_tags = ["O"]
    bio_tags.extend(sorted(["B-"+tag for tag in tags_vocab.keys() if tag != "O"]))
    bio_tags.extend(sorted(["I-"+tag for tag in tags_vocab.keys() if tag != "O"]))

    for batchid, batch in enumerate(data_loader):
        sentence_in, targets, batch_sent_lens, wpos = batch
        bsize, slen = sentence_in.size()
        if batchid == 0: batch_size = bsize
        sentence_in = tensor2var(sentence_in)
        targets = tensor2var(targets)
        encoder_out = encoder(sentence_in, batch_sent_lens, wpos)

        tag_space, bio_tag_space = trig_decoder(encoder_out, batch_sent_lens)
        wpos_mask = tensor2var(mask_wpos(wpos))
        word_targets = targets.view(-1)*wpos_mask.long()
        wpos_space_mask = wpos_mask.unsqueeze(1).expand_as(tag_space)
        char_space_mask = (1-wpos_mask).unsqueeze(1).expand_as(bio_tag_space)
        evt_loss = loss_function(tag_space*wpos_space_mask, word_targets)
        bio_loss = loss_function(bio_tag_space*char_space_mask, targets.view(-1))
        loss = evt_loss + bio_loss
        loss_all += loss.data.item()

        _, tag_outputs = (tag_space*wpos_space_mask).data.max(1)
        gold_targets = word_targets.cpu().view(bsize, slen).numpy()
        pred_outputs = tag_outputs.cpu().view(bsize, slen).numpy()

        # count number of words which are gold, predicted, correctly identified(common_iden), correctly classified(common_class)
        for sentid, (gold_sent, pred_sent) in enumerate(zip(gold_targets, pred_outputs)):
            curr_sentlen = batch_sent_lens[sentid]
            for wordid, (gold_tag, pred_tag) in enumerate(zip(gold_sent[:curr_sentlen], pred_sent[:curr_sentlen])):
                if gold_tag != 0: gold += 1
                if pred_tag != 0: pred += 1
                if gold_tag != 0 and pred_tag != 0: common_iden += 1
                if gold_tag != 0 and gold_tag == pred_tag: common_class += 1

        sentence_in_ids = sentence_in.cpu().data.numpy()
        if ana_err:
            # prepare analyze_info for analyze results
            for sentid, (gold_sent, pred_sent) in enumerate(zip(gold_targets, pred_outputs)):
                curr_sentlen = batch_sent_lens[sentid]
                sent_wordids.append(sentence_in_ids[sentid][:curr_sentlen])
                for wordid, (gold_tag, pred_tag) in enumerate(zip(gold_sent[:curr_sentlen], pred_sent[:curr_sentlen])):
                    analyze_info.append([batchid*batch_size+sentid, wordid, gold_tag, pred_tag])

        if use_ilp:
            _, tag_outputs_char = (bio_tag_space.data*char_space_mask).cpu().max(1)
            pred_outputs_char = tag_outputs_char.view(bsize, slen).numpy()
            tag_prob = tag_space.view(bsize, slen, -1).cpu().data.numpy()
            bio_tag_prob = bio_tag_space.view(bsize, slen, -1).cpu().data.numpy()

            ilp_model.clear()
            ilp_model.verbose=0
            #ilp_model.cut_passes = 1
            #ilp_model.threads = 4
            v = []
            conflict_items = []
            for sentid in range(bsize):
                print_sent_flag=False
                for word_seqid, seqid in enumerate(wpos[sentid]):
                    char_st = wpos[sentid][word_seqid-1]+1 if word_seqid>0 else 0
                    word_len = seqid-char_st
                    if word_len == 1: continue
                    word_ptag = id2tag[pred_outputs[sentid, seqid]]
                    char_ptag_bio = [bio_tags[item] for item in pred_outputs_char[sentid, char_st:seqid]]
                    char_ptag = [item[2:] if item!='O' else item for item in char_ptag_bio]
                    if np.all(np.array(char_ptag) == word_ptag): continue # all char tags consist with word
                    #if np.any(np.array(char_ptag) != char_ptag[0]): continue # all char tags are same
                    if pred_outputs[sentid, seqid]==0 and 0 in pred_outputs_char[sentid, char_st:seqid]: continue
                    conflict_items.append((len(v), char_st, seqid, sentid))
                    if not print_sent_flag:
                        print("-- Sent", sentid, " ".join([id2text[item] for itemid, item in enumerate(sentence_in_ids[sentid]) if itemid in wpos[sentid]]))
                        print_sent_flag = True
                    if 1:
                        word = id2text[sentence_in_ids[sentid, seqid]]
                        word_gtag = id2tag[gold_targets[sentid, seqid]]
                    print("--conflict: bs", batchid, "sent", sentid, "seq", seqid, "word", word, "gold", word_gtag, "pred", word_ptag, "char_pred", char_ptag_bio)

                    # char items' sum_tag_to_1 constraint, add to objective
                    objective = LinExpr()
                    for temp_char_seqid in range(char_st, seqid):
                        item_v = [ilp_model.add_var(var_type=BINARY) for i in range(67)]
                        v.append(item_v)
                        ilp_model += xsum(v[-1]) == 1, "one_tag"
                        objective.add_term(xsum(bio_tag_prob[sentid, temp_char_seqid][j]*item_v[j] for j in range(67)))
                    # word item, sum_tag_to_1 constraint, add to objective
                    item_v = [ilp_model.add_var(var_type=BINARY) for i in range(34)]
                    v.append(item_v)
                    ilp_model += xsum(v[-1]) == 1, "one_tag"
                    objective.add_term(xsum(tag_prob[sentid, seqid][j]*item_v[j] for j in range(34)))
                    ilp_model.objective = objective+objective

                    # non trigger chars -> non trigger word constraints
                    if pred_outputs[sentid, seqid]==1:
                        ilp_model += xsum(xsum(v[i][1:]) for i in range(conflict_items[-1][0], len(v)-1)) >= xsum(v[-1][1:])
                    # trigger chars -> trigger word constraints
                    if 0 not in pred_outputs_char[sentid, char_st:seqid]:
                        ilp_model += xsum(xsum(v[i][1:]) for i in range(conflict_items[-1][0], len(v)-1)) + xsum(v[-1][1:]) >= 1+word_len
            if len(conflict_items)==0:
                print("--Warning, no conflict items")
                continue

            # solve ilp, read results
            if ilp_model.optimize(max_seconds=1, max_solutions=1) in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
                print("------- Feasible solution")
                for id_v, char_st, seqid, sentid in conflict_items:
                    word_len = seqid-char_st
                    for j in range(34):
                        if int(v[id_v+word_len][j].x)==1: ilp_tag = j
                    #print(id_v, sentid, char_st, seqid, ilp_tag, gold_targets[sentid, seqid], pred_outputs[sentid, seqid])
                    if ilp_tag == pred_outputs[sentid, seqid]: #same with before
                        #print(batchid, sentid, seqid, ilp_tag, pred_outputs[sentid, seqid], gold_targets[sentid, seqid], time.asctime())
                        continue
                    else: # changed
                        change_flag = ""
                        if ilp_tag == 0: # changed from trigger to 0
                            pred_ilp += -1
                            change_flag="p-1 "
                            if gold_targets[sentid, seqid] != 0:
                                common_iden_ilp += -1
                                change_flag += "i-1 "
                            if pred_outputs[sentid, seqid] == gold_targets[sentid, seqid]:
                                common_class_ilp += -1
                                change_flag += "c-1"
                        else: # changed from 0/trigger to trigger
                            if pred_outputs[sentid, seqid] == 0:
                                pred_ilp += 1
                                change_flag="p+1 "
                                if gold_targets[sentid, seqid] != 0:
                                    common_iden_ilp += 1
                                    change_flag += "i+1 "
                                if ilp_tag == gold_targets[sentid, seqid]:
                                    common_class_ilp += 1
                                    change_flag += "c+1"
                            else: # pred, iden same
                                change_flag=""
                                if pred_outputs[sentid, seqid] == gold_targets[sentid, seqid]:
                                    change_flag += "c-1"
                                    common_class_ilp += -1
                                elif ilp_tag == gold_targets[sentid, seqid]:
                                    common_class_ilp += 1
                                    change_flag += "c+1"
                        print("--Changed", change_flag, "from", id2tag[pred_outputs[sentid, seqid]], "to", id2tag[ilp_tag], batchid, sentid, seqid, id2tag[gold_targets[sentid, seqid]], time.asctime())
        #if batchid == 0: break

    #if ana_err: output_analyze_trigger(analyze_info, sent_wordids, vocab, tags_vocab)

    if eval_flag=="test": gold += 48
    prf =       cal_prf(common_class, pred, gold)
    prf_iden =  cal_prf(common_iden, pred, gold)
    print("--Eval result", common_class, common_iden, gold, pred, prf, prf_iden)
    prf_ilp =       cal_prf(common_class+common_class_ilp, pred+pred_ilp, gold)
    prf_iden_ilp =  cal_prf(common_iden+common_iden_ilp, pred+pred_ilp, gold)
    print("--Eval result", common_class_ilp, common_iden_ilp, gold, pred_ilp, prf_ilp, prf_iden_ilp)
    return [loss_all, prf, prf_iden]

def output_analyze_trigger(analyze_info, sent_wordids, vocab, tags_vocab):
    content = open("../data/ACE_event_types.txt", "r").readlines()
    content = [line.strip().split("\t") for line in content]
    tagnum2text = dict([(item[0], "_".join(item[1:])) for item in content])
    id2text = dict([(item[1], item[0]) for item in vocab.items()])
    id2tags = dict([(item[1], item[0]) for item in tags_vocab.items()])
    widen_counter = Counter()
    miss_counter = Counter()
    sys_outs = []

    #for sentid, sent_wordid in enumerate(sent_wordids):
    #    sent_text = [id2text[wid] if wid in id2text else "unk" for wid in sent_wordid]
    #    sent_info = [item for item in analyze_info if item[0]==sentid]
    #    #print "------------------------------------ Sentid, len", sentid, len(sent_text)
    #    for word_seqid, word_item in enumerate(sent_info):
    #        sentid_, wordid, gold_tag, pred_tag = word_item
    #        #print word_seqid, sent_text[word_seqid], gold_tag, pred_tag
    #return
    for sentid, sent_wordid in enumerate(sent_wordids):
        sent_text = " ".join([id2text[wid] if wid in id2text else "unk" for wid in sent_wordid])
        sent_info = [item for item in analyze_info if item[0]==sentid and item[-1]+item[-2]>0] # labels
        #sent_info = [item for item in analyze_info if item[0]==sentid and item[-1]+item[-2]>0 and (item[-1]==5 or item[-2]==5)] # conflict-attack
        #sent_info = [item for item in analyze_info if item[0]==sentid and item[-1]+item[-2]>0 and (item[-1]==27 or item[-2]==27)] # movement-transport
        if len(sent_info) == 0: continue
        print("------------------------------------ Sentid, len, text", sentid, len(sent_text.split(" ")))
        print(sent_text)
        for word_seqid, word_item in enumerate(sent_info):
            sentid_, wordid, gold_tag, pred_tag = word_item
            word = sent_text.split(" ")[wordid]
            if gold_tag == 0 and pred_tag == 0: tag_status = "false neg"
            elif gold_tag == 0 and pred_tag != 0: tag_status = "wrong iden"
            elif gold_tag != 0 and pred_tag == 0: tag_status = "missed trig"
            elif gold_tag != 0 and pred_tag != 0 and gold_tag != pred_tag: tag_status = "wrong class"
            elif gold_tag != 0 and gold_tag == pred_tag: tag_status = "true class"

            gold_tag_text = tagnum2text[id2tags[gold_tag]] if gold_tag != 0 else 'O'
            pred_tag_text = tagnum2text[id2tags[pred_tag]] if pred_tag != 0 else 'O'
            if tag_status == "false neg": continue
            concat_tags = gold_tag_text+"#"+pred_tag_text
            #sys_outs.append(concat_tags+"#"+word)
            #if "movement" in concat_tags: sys_outs.append(concat_tags+"#"+word)
            if "conflict" in concat_tags: sys_outs.append(concat_tags+"#"+word)
            if tag_status == "wrong iden": widen_counter[concat_tags] += 1
            if tag_status == "missed trig": miss_counter[concat_tags] += 1
            print("-- wordid, word, gold_tag, pred_tag", wordid, word, tag_status, gold_tag, pred_tag, gold_tag_text, pred_tag_text)
        #break

    widen_num = sum([item[1] for item in widen_counter.items()])
    print("----------------- Statistics of wrong iden triggers", widen_num)
    for item in widen_counter.most_common():
        print(item[0].rjust(40), "\t", item[1], "\t%.2f"%(item[1]*100.0/widen_num))
    miss_num = sum([item[1] for item in miss_counter.items()])
    print("----------------- Statistics of missed triggers", miss_num)
    for item in miss_counter.most_common():
        print(item[0].rjust(40), "\t", item[1], "\t%.2f"%(item[1]*100.0/miss_num))

    print("----------------- Statistics of triggers tags --------------------", len(sys_outs))
    sys_out_dict = dict(Counter(sys_outs).most_common())
    for item in sorted(sys_out_dict.items(), key=lambda a:a[0].split("#")[-1]):
        print("{}\t{}".format(item[0], item[1]).rjust(80))
