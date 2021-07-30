# This file mainly implements trigger training process
import sys
import time
from data_util import ACEDataset, pad_trig, pad_trig_char
from util import tensor2var, load_models, set_loss_optim, mask_wpos, output_model_result
from eval_util import eval_model
from model import RNNEncoder, TrigDecoder, TrigDecoderw
import torch
import torch.utils.data as torch_data
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

Tab = "\t"

def trainFunc(args, ace_data, debug=False):
    # put ace digit data into pytorch DataLoader
    train_loader = torch_data.DataLoader(ACEDataset(ace_data.train), batch_size=args.batch_size, shuffle=True, collate_fn=pad_trig_char)
    dev_loader  = torch_data.DataLoader(ACEDataset(ace_data.dev), batch_size=args.batch_size, shuffle=False, collate_fn=pad_trig_char)
    test_loader = torch_data.DataLoader(ACEDataset(ace_data.test), batch_size=args.batch_size, shuffle=False, collate_fn=pad_trig_char)

    id2words = dict([(item[1], item[0]) for item in ace_data.vocab.items()])

# init models
    model = RNNEncoder(args)
    if args.use_pretrain:
        model.word_embeddings.weight.data.copy_(torch.from_numpy(ace_data.pretrain_embedding))
    trig_decoder = TrigDecoderw(args) if args.single_decoder else TrigDecoder(args)
    parameters = list(model.parameters()) + list(trig_decoder.parameters())
    loss_function, optimizer = set_loss_optim(parameters, args.loss_flag, args.opti_flag, args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

# training
    best_f1 = -10.0
    best_epoch = -1
    for epoch in range(args.epoch_num):
        loss_all = 0.0
        training_id = 0
        model.train()
        trig_decoder.train()

        for iteration, batch in enumerate(train_loader):
            model.zero_grad()
            trig_decoder.zero_grad()

            sentence_in, targets, batch_sent_lens, wpos = batch
            sentence_in = tensor2var(sentence_in)
            targets = tensor2var(targets)
            bsize, slen = sentence_in.size()

            #if iteration == -1:
            #    print "--------", batch_sent_lens.cpu().tolist()
            #    print wpos
            #print sentence_in[2]
            #for i in sentence_in[2].cpu().data.tolist():
            #    print id2words[i],
            #print
            #print targets[2]
            #print targets.nonzero()
            #assert False

            encoder_out = model(sentence_in, batch_sent_lens, wpos)
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
            loss.backward()
            optimizer.step()

            model_list = [model, trig_decoder]

            training_id += sentence_in.size(0)
            if training_id % args.checkpoint == 0:
                print("## training #inst", training_id, " accumulate loss %.6f"%loss_all, time.asctime())

            if args.small_debug and training_id % args.small_data_size == 0: break
        scheduler.step()

        ## record best result on dev
        eval_results = eval_model(dev_loader, model_list, loss_function, "dev", args)
        if float(eval_results[1][-1]) > best_f1:
            best_f1, best_epoch = float(eval_results[1][-1]), epoch
            print("## New record, best f1", best_f1, "best_epoch", best_epoch)
            torch.save(model_list, args.model_path)
        print('Epoch', epoch, 'result', eval_results, "Best epoch", best_epoch, 'best_f1', best_f1)

        if best_f1 == 100.0 or (epoch-best_epoch > args.early_stop): break

        ## check results on train/test occasionally on some epoch
        if epoch > args.check_test-1 and epoch % args.check_test == 0:
            if epoch % 10 == 0:
                eval_results = eval_model(train_loader, model_list, loss_function, "train", args)
                output_model_result(eval_results, epoch, "train")

            eval_results = eval_model(test_loader, model_list, loss_function, "test", args)
            output_model_result(eval_results, epoch, "test")

# final result on train/test
    best_model_list = load_models(args.model_path)
    eval_results = eval_model(train_loader, best_model_list, loss_function, "train", args)
    output_model_result(eval_results, epoch, "train_final")

    eval_results = eval_model(dev_loader, best_model_list, loss_function, "dev", args)
    output_model_result(eval_results, epoch, "dev_final")

    eval_results = eval_model(test_loader, best_model_list, loss_function, "test", args)
    output_model_result(eval_results, epoch, "test_final")

