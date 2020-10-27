import torch
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn
from utils.dataset_multistage_2 import Dataset
#from utils.dataset_multistage_eval import Dataset_eval
from models.multi_stage_sequenceencoder import multistageSTARSequentialEncoder
from utils.logger import Logger, Printer, VisdomLogger
import argparse
from utils.snapshot import save, resume
import os
#from eval_multistage import evaluate_fieldwise
from eval_multi_stage_RNN import  evaluate_fieldwise_stage1, evaluate_fieldwise2
from models.networkTrans_gt import model_GT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default='./',help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=4 , type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=1, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=50, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default='/home/pf/pfstaff/projects/ozgur_deep_filed/multi_stage/trained_models_multistage_transformer/debug_TRANSFORMER_4_0.001_6_64_2_0.1_0.5_0.0001_1_labelsC.csv_0_model.pth', type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default='trained_models_multistage', type=str, help="directory to save checkpoints")
    parser.add_argument('-wd', "--weight_decay", default=0.0001, type=float, help="weight_decay")
    parser.add_argument('-hd', "--hidden", default=64, type=int, help="hidden dim")
    parser.add_argument('-nl', "--layer", default=6, type=int, help="num layer")    
    parser.add_argument('-lrs', "--lrSC", default=2, type=int, help="lrScheduler")    
    parser.add_argument('-nm', "--name", default='debug_TRANSFORMER', type=str, help="name")
    parser.add_argument('-l1', "--lambda_1", default=0.1, type=float, help="lambda_1")
    parser.add_argument('-l2', "--lambda_2", default=0.5, type=float, help="lambda_2")
    parser.add_argument('-l0', "--lambda_0", default=1, type=float, help="lambda_0")
    parser.add_argument('-stg', "--stage", default=3, type=float, help="num stage")
    parser.add_argument('-sd', "--seed", default=0, type=int, help="random seed")
    parser.add_argument('-fd', "--fold", default=1, type=int, help="5 fold")   
    parser.add_argument('-gt', "--gt_path", default='labelsC.csv', type=str, help="gt file path")
    
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
    layer=6,
    hidden=64,
    lrS=1,
    lambda_1=1,
    lambda_2=1,
    lambda_0=1,
    stage=3,
    fold_num=None,
    gt_path=None
    ):
        
    data_file  = "/scratch/tmehmet/train_set_24X24_debug.hdf5"        
    if not os.path.isfile(data_file):
        data_file  = "/cluster/scratch/tmehmet/train_set_24X24_debug.hdf5"
    
    if not os.path.isfile(data_file):
        data_file  = "/home/pf/pfstaff/projects/ozgur_deep_filed/data_crop_CH/train_set_24x24_debug.hdf5"

    testdataset =  Dataset(data_file , 0., 'test', True, fold_num, gt_path)    
    
    nclasses = testdataset.n_classes
    nclasses_local_1 = testdataset.n_classes_local_1
    nclasses_local_2 = testdataset.n_classes_local_2
    #nclasses = 125
    print('Num classes:' , nclasses)

    #Class stage mappping
    s1_2_s3 = testdataset.l1_2_g
    s2_2_s3 = testdataset.l2_2_g

    #Define the model
    network = multistageSTARSequentialEncoder(24,24, nstage=stage, nclasses=nclasses, nclasses_l1=nclasses_local_1, nclasses_l2=nclasses_local_2, input_dim=4, hidden_dim=hidden, n_layers=layer)
    network_gt = model_GT(nclasses=nclasses, s1_2_s3=s1_2_s3, s2_2_s3=s2_2_s3)
    network.eval()
    network_gt.eval()
    
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Num params: ', params)

    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()
        network_gt = torch.nn.DataParallel(network_gt).cuda()
    
    if snapshot is not None:        
        checkpoint = torch.load(snapshot)
        network.load_state_dict(checkpoint['network_state_dict'])
        network_gt.load_state_dict(checkpoint['network_gt_state_dict'])
 
    print("\n Eval on test set - stage 1 out")
    evaluate_fieldwise2(network, network_gt,testdataset, batchsize=batchsize, viz=False, fold_num=fold_num) 
    
#    print("\n Eval on test set - stage 3 out")
#    evaluate_fieldwise_stage1(network, network_gt,testdataset, batchsize=batchsize, viz=False, fold_num=fold_num) 

if __name__ == "__main__":

    args = parse_args()
    print(args)

    model_name = str(args.name) + '_' + str(args.batchsize) + '_' + str(args.learning_rate) + '_' + str(args.layer) + '_' + str(args.hidden) + '_'  + str(args.lrSC) + '_' + str(args.lambda_1) + '_' + str(args.lambda_2)  + '_' + str(args.weight_decay) + '_' + str(args.fold) + '_' + str(args.gt_path) + '_' + str(args.seed) 
    print(model_name)
      
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(
        args.data,
        batchsize=args.batchsize,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.learning_rate,
        snapshot=args.snapshot,
        checkpoint_dir=args.checkpoint_dir,
        weight_decay=args.weight_decay,
        name=model_name,
        layer=args.layer,
        hidden=args.hidden,
        lrS=args.lrSC,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_0=args.lambda_0,
        stage = args.stage,
        fold_num = args.fold,
        gt_path = args.gt_path
    )
