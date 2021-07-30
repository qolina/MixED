import time
from data_util import ACEDataset, pad_trig, pad_trig_char
from util import tensor2var, load_models, set_loss_optim, cal_prf, output_model_result
from eval_util_ilp import eval_model

import torch
import torch.utils.data as torch_data

def testFunc(args, ace_data, debug=False):
    test_loader = torch_data.DataLoader(ACEDataset(ace_data.test), batch_size=args.batch_size, shuffle=False, collate_fn=pad_trig_char)

# load models
    if not args.analyze: sys.exit(0)

    if len(args.trig_model_path)>1:
        print('- load pretrained trig model', args.trig_model_path)
        pretrained_model_list = load_models(args.trig_model_path)
        loss_function, optimizer = set_loss_optim(list(pretrained_model_list[0].parameters()), args.loss_flag, args.opti_flag, args.lr)
        eval_results = eval_model(test_loader, pretrained_model_list, loss_function, "human craft", vocab=ace_data.vocab, tags_vocab=ace_data.atag_dict, ana_err=True, use_ilp=True)
        output_model_result(eval_results, -1, "test_final")

