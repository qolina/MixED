import sys
import time
from argparse import ArgumentParser
from preprocessing import Text2Digit
from train import trainFunc
from test import testFunc

import torch

Tab = "\t"

def get_args():
    parser = ArgumentParser(description='charater based chinese event detection')
    parser.add_argument('--no_cuda', action='store_false', help='do not use CUDA', dest='gpu')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--loss_flag', type=str, default='cross-entropy')
    parser.add_argument('--opti_flag', type=str, default='sgd') # or use 'adadelta', 'adam', 'sgd'
    parser.add_argument('--analyze', action='store_true', help='test mode, test with pretrained models')
    parser.add_argument('--small_debug', action='store_true', help='whether use small data size for debug')
    parser.add_argument('--no_shuffle_train', action='store_false', help='not shuffle train before train each epoch', dest='shuffle_train')
    parser.add_argument('--test_as_dev', action='store_true', help='use test as dev to get upper bound')
    parser.add_argument('--single_decoder', action='store_true', help='only use decoder_w, without decoder c')

    parser.add_argument('--gru', action='store_true', help='use gru instead of lstm')
    parser.add_argument('--no_use_bilstm', action='store_false', help='not use bilstm', dest='bilstm')
    parser.add_argument('--no_use_pretrain', action='store_false', help='not use pretrain embedding', dest='use_pretrain')

    parser.add_argument('--checkpoint', type=int, default=500, help='used for output training instance number')
    parser.add_argument('--check_test', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=10)

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--lstm_hidden_dim', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)

    # files
    data_dir = '../data/'
    parser.add_argument('-evt_train', type=str, default=data_dir+'train.txt', help="train file path")
    parser.add_argument('-evt_dev', type=str, default=data_dir+'dev.txt', help="dev file path")
    parser.add_argument('-evt_test', type=str, default=data_dir+'test.txt', help="test file path")

    parser.add_argument('-pretrain_embed', type=str, default=data_dir+'sgns.merge.charword.ace', help="pretrain word embed path")
    parser.add_argument('-model_path', type=str, default='', help="store model path")
    parser.add_argument('--trig_model_path', type=str, default='', help="best performance trig model path, used for analyze")

    # set character output tag scheme to be io or bio or bio+tag
    parser.add_argument('--char_bio', action='store_true', help='use bio for each character\'s output')
    parser.add_argument('--char_io', action='store_true', help='use io for each character\'s output')

    args = parser.parse_args()
    return args

def outputParameters(config):
    print("----- python             -----", Tab, Tab, sys.version[0])
    print("----- pytorch            -----", Tab, Tab, torch.__version__)
    print("----- random seed        -----", Tab, Tab, config.random_seed)

    print("----- use gpu            -----", Tab, Tab, config.gpu)
    print("----- shuf train         -----", Tab, Tab, config.shuffle_train)

    print("----- batch size         -----", Tab, Tab, config.batch_size)
    print("----- #iteration         -----", Tab, Tab, config.epoch_num)
    print("----- learn rate         -----", Tab, Tab, config.lr)
    print("----- optim method       -----", Tab, Tab, config.opti_flag)
    print("----- loss calculation   -----", Tab, Tab, config.loss_flag)

    print("----- small_debug        -----", Tab, Tab, config.small_debug)
    print("----- checkpoint         -----", Tab, Tab, config.checkpoint)
    print("----- check test         -----", Tab, Tab, config.check_test)
    print("----- early stop epoch   -----", Tab, Tab, config.early_stop)

    print("----- use gru            -----", Tab, Tab, config.gru)
    print("----- bi-direct          -----", Tab, Tab, config.bilstm)
    print("----- layers num         -----", Tab, Tab, config.num_layers)
    print("----- embeds dim         -----", Tab, Tab, config.embed_dim)
    print("----- hidden dim         -----", Tab, Tab, config.lstm_hidden_dim)
    print("----- dropout            -----", Tab, Tab, config.dropout)

    print("----- train size         -----", Tab, Tab, config.training_size)
    print("----- dev size           -----", Tab, Tab, config.dev_size)
    print("----- test size          -----", Tab, Tab, config.test_size)
    print("----- vocab size         -----", Tab, Tab, config.vocab_size)
    print("----- tags size          -----", Tab, Tab, config.tagset_size)

if __name__ == "__main__":
    print("Program starts", time.asctime())
    param_str = "_" + time.strftime("%Y%m%d%H%M%S", time.gmtime())

    args = get_args()
    if args.random_seed != -1:
        torch.manual_seed(args.random_seed)

    if args.small_debug:
        args.epoch_num = 4
        args.small_data_size = 2000
        args.checkpoint = 500
        args.check_test = 1
        args.use_freq_token = False

    #######################
    ## load datasets
    args.train, args.dev, args.test = args.evt_train, args.evt_dev, args.evt_test
    ace_data = Text2Digit(args)
    if args.test_as_dev:
        ace_data.dev = ace_data.test

    #######################
    ## store and output all parameters
    _, pretrain_embed_dim = ace_data.pretrain_embedding.shape
    if args.use_pretrain: args.embed_dim = pretrain_embed_dim

    args.model_path += param_str

    config = args
    config.vocab_size = len(ace_data.vocab)
    config.tagset_size = len(ace_data.atag_dict)
    config.training_size = len(ace_data.train[0])
    config.dev_size = len(ace_data.dev[0])
    config.test_size = len(ace_data.test[0])
    config.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    outputParameters(config)
    print('--Train/dev/test files', config.train)

    #######################
    # test mode
    if args.analyze:
        testFunc(config, ace_data)
    else:
        # training mode
        trainFunc(config, ace_data)
