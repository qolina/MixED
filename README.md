# MixED
Author: Yanxia Qin (Donghua University)

Code for Neural Chinese Event Detection (mix sequence + BiLSTM + double decoder + joint inference with ILP)

Softwares: 
    Python 3.X, Pytorch >1.0, python-mip 1.13.0

Data: 
    Prepare ACE_Chinese with three files (train.txt, dev.txt, test.txt). Data format check with sample file in ./data

Run: 
    Pretrained word embedding: update with your own. Put it in ./data
    See ./src/run.sh for training and test commands

