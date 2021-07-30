import torch
from util import idsent2words, get_trigger, tensor2var, mask_wpos
from util import cal_prf
from collections import Counter

Tab = "\t"

def eval_model(data_loader, models, loss_function, eval_flag, args):
    encoder, trig_decoder = models
    encoder.eval()
    trig_decoder.eval()

    loss_all, common_class, common_iden, gold, pred = 0.0, 0, 0, 0, 0

    for batchid, batch in enumerate(data_loader):
        sentence_in, targets, batch_sent_lens, wpos = batch
        bsize, slen = sentence_in.size()

        sentence_in = tensor2var(sentence_in)
        targets = tensor2var(targets)
        encoder_out = encoder(sentence_in, batch_sent_lens, wpos)
        wpos_mask = tensor2var(mask_wpos(wpos))
        word_targets = targets.view(-1)*wpos_mask.long()
        if args.single_decoder:
            tag_space = trig_decoder(encoder_out, batch_sent_lens)
            wpos_space_mask = wpos_mask.unsqueeze(1).expand_as(tag_space)
            loss = loss_function(tag_space*wpos_space_mask, word_targets)
        else:
            tag_space, bio_tag_space = trig_decoder(encoder_out, batch_sent_lens)
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

    # auto segmentation losses 48 triggers
    if eval_flag=="test" and args.evt_train.find("autoseg")>0: gold += 48
    prf =       cal_prf(common_class, pred, gold)
    prf_iden =  cal_prf(common_iden, pred, gold)
    print("--Eval result", common_class, common_iden, gold, pred, prf, prf_iden)
    return [loss_all, prf, prf_iden]
