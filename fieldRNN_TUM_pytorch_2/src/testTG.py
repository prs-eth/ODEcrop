import numpy as np
import torch.nn
from utils.dataset_hdf5_TG19 import Dataset
from utils.dataset_eval_TG19 import Dataset_eval
from models.sequenceencoder import LSTMSequentialEncoder
from models.sequenceencoder_star import STARSequentialEncoder
from utils.logger import Logger, Printer, VisdomLogger
import argparse
from utils.snapshot import save, resume
import os
from networks import FCN_CRNN
from eval import evaluate, evaluate_fieldwise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default='./',help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=1 , type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=1, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=100, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default='./trained_models_TG/star_TGyear_t_110_4_64_wd_00005_lrS_2_model.pth', type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default='trained_models', type=str, help="directory to save checkpoints")
    parser.add_argument('-wd', "--weight_decay", default=0.00005, type=float, help="weight_decay")
    parser.add_argument('-hd', "--hidden", default=64, type=int, help="hidden dim")
    parser.add_argument('-nl', "--layer", default=4, type=int, help="num layer")    
    parser.add_argument('-nm', "--name", default='debug', type=str, help="name")
    return parser.parse_args()

def main(
    datadir,
    batchsize = 1,
    workers = 12,
    epochs = 1,
    lr = 1e-3,
    snapshot = None,
    checkpoint_dir = None,
    weight_decay = 0.0000,
    name='debug',
    layer=2,
    hidden=128
    ):

    #traindataset = Dataset_eval("/cluster/scratch/tmehmet/test_set_TG_SG_24x24_debug.hdf5", 0., 'all')
    #testdataset =  Dataset_eval("/cluster/scratch/tmehmet/test_set_TG_SG_24x24_debug.hdf5", 0.0, 'all')    
    traindataset = Dataset_eval("/cluster/scratch/tmehmet/train_set_24X24_debug.hdf5", 0., 'train')
    testdataset =  Dataset_eval("/cluster/scratch/tmehmet/TG_expYear19.hdf5", 0., 'all')   

    
    nclasses = testdataset.n_classes
    #nclasses = 125
    print('Num classes:' , nclasses)
    LOSS_WEIGHT  = torch.ones(nclasses)
    LOSS_WEIGHT[0] = 0


    #Define the model
    #network = LSTMSequentialEncoder(24,24,nclasses=nclasses, input_dim=4, hidden_dim=128)
    network = STARSequentialEncoder(24,24,nclasses=nclasses, input_dim=4, hidden_dim=hidden, n_layers=layer)
    
#    network = FCN_CRNN(fcn_input_size=(traindataset.max_obs,9,24,24), crnn_input_size=(traindataset.max_obs,256,24//4,24//4), crnn_model_name='clstm', 
#                 hidden_dims=256, lstm_kernel_sizes=(3,3), conv_kernel_size=3, lstm_num_layers=1, avg_hidden_states=True, 
#                 num_classes=nclasses, bidirectional=False, pretrained=False, early_feats=True, use_planet=False, resize_planet=True, 
#                 num_bands_dict={'s1': 0, 's2': 9, 'planet': 0, 'all': 9 }, main_crnn=False, main_attn_type='None', attn_dims=32, 
#                 enc_crnn=False, enc_attn=False, enc_attn_type='None')


    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    loss = torch.nn.NLLLoss(weight=LOSS_WEIGHT)

    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()
        loss = loss.cuda()


    if snapshot is not None:        
        resume(snapshot,model=network, optimizer=optimizer)


            
    # evaluate model
    print("\n Eval on test set - 1")
    evaluate_fieldwise(network, traindataset, batchsize) 
    print("\n Eval on test set")
    evaluate_fieldwise(network, testdataset, batchsize) 


if __name__ == "__main__":

    args = parse_args()
    print(args)

    main(
        args.data,
        batchsize=args.batchsize,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.learning_rate,
        snapshot=args.snapshot,
        checkpoint_dir=args.checkpoint_dir,
        weight_decay=args.weight_decay,
        name=args.name,
        layer=args.layer,
        hidden=args.hidden
    )
