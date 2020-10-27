import torch
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn
from utils.dataset_multistage_2 import Dataset
#from utils.dataset_eval import Dataset_eval
from models.sequenceencoder_star import STARSequentialEncoder
from utils.logger import Logger, Printer, VisdomLogger
import argparse
from utils.snapshot import save, resume
import os
from eval_multi_stage_baseline_3 import  evaluate_fieldwise
from utils.data_sampler import ImbalancedDatasetSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default='./',help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=4, type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=1, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=30, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default=None, type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default='/home/pf/pfstaff/projects/ozgur_deep_filed/multi_stage/trained_models_baseline_3_4paper', type=str, help="directory to save checkpoints")
    parser.add_argument('-wd', "--weight_decay", default=0.0001, type=float, help="weight_decay")
    parser.add_argument('-hd', "--hidden", default=64, type=int, help="hidden dim")
    parser.add_argument('-nl', "--layer", default=6, type=int, help="num layer")    
    parser.add_argument('-lrs', "--lrSC", default=2, type=int, help="lrScheduler")    
    parser.add_argument('-nm', "--name", default='bs_cb_beta', type=str, help="name")
    parser.add_argument('-da', "--data_aug", default=0, type=int, help="data augment")
    parser.add_argument('-gc', "--clip", default=5, type=float, help="grad clip")
    parser.add_argument('-fd', "--fold", default=4, type=int, help="5 fold")
    parser.add_argument('-sd', "--seed", default=0, type=int, help="random seed")
    parser.add_argument('-gt', "--gt_path", default='labelsC.csv', type=str, help="gt file path")
    parser.add_argument('-bt', "--beta", default=-1, type=float, help="beta for CB loss")
    parser.add_argument('-cb', "--cb_loss", default=-1, type=int, help="CB loss")

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
    data_aug=0,
    grad_clip=-1,
    fold_num=None,
    gt_path=None,
    cb_loss=False,
    beta=-1
    ):

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    data_file  = "/scratch/tmehmet/train_set_24X24_debug.hdf5"        
    if not os.path.isfile(data_file):
        data_file  = "/cluster/scratch/tmehmet/train_set_24X24_debug.hdf5"

    if not os.path.isfile(data_file):
        data_file  = "/home/pf/pfstaff/projects/ozgur_deep_filed/data_crop_CH/train_set_24x24_debug.hdf5"    

    traindataset = Dataset(data_file, 0., 'train', False, fold_num, gt_path)
    testdataset =  Dataset(data_file, 0., 'test', True, fold_num, gt_path)    
    
    nclasses = traindataset.n_classes
    #nclasses = 125
    print('Num classes:' , nclasses)
    LOSS_WEIGHT  = torch.ones(nclasses)
    LOSS_WEIGHT[0] = 0

    if cb_loss in [1,2]:
        #Compute the class frequencies
        class_fq  = torch.zeros(nclasses)
    
        for i in range(len(traindataset)): 
            temp = traindataset[i][1].flatten()
        
            for j in range(nclasses):
               class_fq[j] = class_fq[j] + torch.sum(temp==j) 
          
        for i in range(1,nclasses):
            if class_fq[i] > 0:
                if beta>0:
                    LOSS_WEIGHT[i] = (1.-beta)/(1.-beta**class_fq[i])
                else:
                    LOSS_WEIGHT[i] = 1./class_fq[i]
        
        if cb_loss == 2:    
            print('class balance loss - NORMALIZATION')
            #Normalization
            LOSS_WEIGHT = LOSS_WEIGHT/torch.median(LOSS_WEIGHT)
        
        print('class balance loss, beta: ', beta)
        print(LOSS_WEIGHT)                
        
    
    if data_aug==1:
        traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize, shuffle=False, num_workers=workers, 
                                                      sampler=ImbalancedDatasetSampler(traindataset))
        print('Data augmentation: oversampling for the minority classes')
    else:
        traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize, shuffle=True, num_workers=workers)
        print('No data augmentation')


    testdataloader = torch.utils.data.DataLoader(testdataset,batch_size=batchsize, shuffle=False, num_workers=workers)


    logger = Logger(columns=["loss"], modes=["train", "test"])
    vizlogger = VisdomLogger()

    #Define the model
    network = STARSequentialEncoder(24,24,nclasses=nclasses, input_dim=4, hidden_dim=hidden, n_layers=layer)
    

    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    loss = torch.nn.NLLLoss(weight=LOSS_WEIGHT)

    if lrS == 1:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)
    elif lrS == 2:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)
    elif lrS == 3:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)

    
    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()
        loss = loss.cuda()

    start_epoch = 0
    best_test_acc = 0
    test_acc = -1
    
    if snapshot is not None:        
        state = resume(snapshot,model=network, optimizer=optimizer)

        if "epoch" in state.keys():
            start_epoch = state["epoch"]

        if "data" in state.keys():
            logger.resume(state["data"])

    for epoch in range(start_epoch, epochs):
        logger.update_epoch(epoch)

        print("\nEpoch {}".format(epoch))
        print("train")
        train_epoch(traindataloader, network, optimizer, loss, loggers=(logger,vizlogger), grad_clip=grad_clip)

        print("test")
        test_epoch(testdataloader, network, optimizer, loss, loggers=(logger,vizlogger))

        #call LR scheduler 
        lr_scheduler.step()

        data = logger.get_data()
        vizlogger.update(data)

        # evaluate model
        if epoch>15 and epoch%1 == 0:
            print("\n Eval on test set")
            test_acc = evaluate_fieldwise(network, testdataset, batchsize=batchsize) 
            
            if checkpoint_dir is not None:
                checkpoint_name = os.path.join(checkpoint_dir, name + "_model.pth")
                if test_acc > best_test_acc:
                    print('Model saved! Best val acc:', test_acc)
                    best_test_acc = test_acc
                    save(checkpoint_name, network, optimizer, epoch=epoch, data=data)
                


def train_epoch(dataloader, network, optimizer, loss, loggers, grad_clip):
    logger, vizlogger = loggers

    network.train()

    #printer = Printer(N=len(dataloader))
    logger.set_mode("train")
    mean_loss = 0.
    
    for iteration, data in enumerate(dataloader):
        optimizer.zero_grad()

        input, target, _, _ = data
        

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = network.forward(input)
        l = loss(output, target)
        stats = {"loss":l.data.cpu().numpy()}
        mean_loss += l.data.cpu().numpy()

        l.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
        
        optimizer.step()

        #printer.print(stats, iteration)
        logger.log(stats, iteration)
        #vizlogger.plot_steps(logger.get_data())
    print('Loss: %.4f'%(mean_loss/iteration))
    

def test_epoch(dataloader, network, optimizer, loss, loggers):
    logger, vizlogger = loggers

    #printer = Printer(N=len(dataloader))
    logger.set_mode("train")
    mean_loss = 0.
    
    for iteration, data in enumerate(dataloader):
        optimizer.zero_grad()

        input, target, _, _, _ = data

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = network.forward(input)
        l = loss(output, target)
        stats = {"loss":l.data.cpu().numpy()}
        mean_loss += l.data.cpu().numpy()

        #printer.print(stats, iteration)
        logger.log(stats, iteration)
        #vizlogger.plot_steps(logger.get_data())
    print('Loss: %.4f'%(mean_loss/iteration))    


if __name__ == "__main__":

    args = parse_args()
    print(args)
    
    model_name = str(args.name) + '_' + str(args.batchsize) + '_' + str(args.learning_rate) + '_' + str(args.layer) + '_' + str(args.hidden) + '_'  + str(args.lrSC) + '_' + str(args.weight_decay) + '_' + str(args.cb_loss) + '_' + str(args.beta) + '_' + str(args.fold) + '_' + str(args.gt_path) + '_' + str(args.seed) 
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
        data_aug=args.data_aug,
        grad_clip = args.clip,
        fold_num = args.fold,
        gt_path = args.gt_path,
        beta = args.beta,
        cb_loss = args.cb_loss
    )
