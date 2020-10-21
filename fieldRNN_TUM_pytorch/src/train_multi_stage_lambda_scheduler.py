import numpy as np
import torch.nn
from utils.dataset_multistage_2 import Dataset
#from utils.dataset_multistage_eval import Dataset_eval
from models.multi_stage_sequenceencoder import multistageSTARSequentialEncoder
from utils.logger import Logger, Printer, VisdomLogger
import argparse
from utils.snapshot import save, resume
import os
#from eval_multistage import evaluate_fieldwise
from eval_multi_stage_baseline_3 import  evaluate_fieldwise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default='./',help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=8 , type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=1, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=50, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default=None, type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default='trained_models_multistage', type=str, help="directory to save checkpoints")
    parser.add_argument('-wd', "--weight_decay", default=0.0001, type=float, help="weight_decay")
    parser.add_argument('-hd', "--hidden", default=64, type=int, help="hidden dim")
    parser.add_argument('-nl', "--layer", default=6, type=int, help="num layer")    
    parser.add_argument('-lrs', "--lrSC", default=2, type=int, help="lrScheduler")    
    parser.add_argument('-nm', "--name", default='debug', type=str, help="name")
    parser.add_argument('-l1', "--lambda_1", default=0.8, type=float, help="lambda_1")
    parser.add_argument('-l2', "--lambda_2", default=0.1, type=float, help="lambda_2")
    parser.add_argument('-l0', "--lambda_0", default=0.1, type=float, help="lambda_0")
    parser.add_argument('-stg', "--stage", default=3, type=float, help="num stage")

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
    stage=3
    ):

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    data_file  = "/scratch/tmehmet/train_set_24X24_debug.hdf5"        
    if not os.path.isfile(data_file):
        data_file  = "/cluster/scratch/tmehmet/train_set_24X24_debug.hdf5"
    
    traindataset = Dataset(data_file, 0., 'train', False)
    testdataset =  Dataset(data_file, 0., 'test', True)   
    
    nclasses = traindataset.n_classes
    nclasses_local_1 = traindataset.n_classes_local_1
    nclasses_local_2 = traindataset.n_classes_local_2
    #nclasses = 125
    print('Num classes:' , nclasses)
    LOSS_WEIGHT  = torch.ones(nclasses)
    LOSS_WEIGHT[0] = 0  
    LOSS_WEIGHT_LOCAL_1  = torch.ones(nclasses_local_1)
    LOSS_WEIGHT_LOCAL_1[0] = 0
    LOSS_WEIGHT_LOCAL_2  = torch.ones(nclasses_local_2)
    LOSS_WEIGHT_LOCAL_2[0] = 0

    traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize,shuffle=True,num_workers=workers)
    #testdataloader = torch.utils.data.DataLoader(testdataset,batch_size=batchsize,shuffle=False,num_workers=workers)

    logger = Logger(columns=["loss"], modes=["train", "test"])
    vizlogger = VisdomLogger()

    #Define the model
    network = multistageSTARSequentialEncoder(24,24, nstage=stage, nclasses=nclasses, nclasses_l1=nclasses_local_1, nclasses_l2=nclasses_local_2, input_dim=4, hidden_dim=hidden, n_layers=layer)
    
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Num params: ', params)

    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    loss = torch.nn.NLLLoss(weight=LOSS_WEIGHT)
    loss_local_1 = torch.nn.NLLLoss(weight=LOSS_WEIGHT_LOCAL_1)
    loss_local_2 = torch.nn.NLLLoss(weight=LOSS_WEIGHT_LOCAL_2)

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
        loss_local_1 = loss_local_1.cuda()
        loss_local_2 = loss_local_2.cuda()

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

        for param_group in optimizer.param_groups:
            print('Current learning rate: ' , param_group['lr'])
           
        
        if epoch == 2:
            lambda_0 = 0.1
            lambda_1 =  0.1
            lambda_2 = 0.8
        elif epoch == 4:
            lambda_0 = 0.8
            lambda_1 =  0.1
            lambda_2 = 0.1               
        elif epoch == 15:
            lambda_0 = 1
            lambda_1 =  0.
            lambda_2 = 0.
            
        print("\nEpoch {}".format(epoch))
        print("train")
        train_epoch(traindataloader, network, optimizer, loss, loss_local_1, loss_local_2, 
                    loggers=(logger,vizlogger), lambda_1=lambda_1,lambda_2=lambda_2,lambda_0=lambda_0, stage=stage)
#        print("\ntest")
#        test_epoch(testdataloader, network,loss, loggers=(logger, vizlogger))
    
        #call LR scheduler 
        lr_scheduler.step()

        data = logger.get_data()
        vizlogger.update(data)

        # evaluate model
        if epoch>-15 and epoch%1 == 0:
#            print("\n Eval on train set")
#            evaluate(network, traindataset_2, batchsize=4) 
            print("\n Eval on test set")
            test_acc = evaluate_fieldwise(network, testdataset, batchsize=batchsize) 
            #test_acc = np.sum(np.diag(cm)) / np.sum(cm)
            
            if checkpoint_dir is not None:
                checkpoint_name = os.path.join(checkpoint_dir, name + "_model.pth")
                if test_acc > best_test_acc:
                    print('Model saved! Best val acc:', test_acc)
                    best_test_acc = test_acc
                    save(checkpoint_name, network, optimizer, epoch=epoch, data=data)
                


def train_epoch(dataloader, network, optimizer, loss, loss_local_1, loss_local_2, loggers, lambda_1,lambda_2,lambda_0, stage):
    logger, vizlogger = loggers
    
    #printer = Printer(N=len(dataloader))
    logger.set_mode("train")
    mean_loss_glob = 0.
    mean_loss_local_1 = 0.
    mean_loss_local_2 = 0.
    
    for iteration, data in enumerate(dataloader):
        optimizer.zero_grad()

        input, target_glob, target_local_1, target_local_2 = data
        
        if torch.cuda.is_available():
            input = input.cuda()
            target_glob = target_glob.cuda()
            target_local_1 = target_local_1.cuda()
            target_local_2 = target_local_2.cuda()

        output_glob, output_local_1, output_local_2 = network.forward(input)
        
        l_glob = loss(output_glob, target_glob)
        l_local_1 = loss_local_1(output_local_1, target_local_1) 
        l_local_2 = loss_local_2(output_local_2, target_local_2) 
        
        if stage==3 or stage==1:
            total_loss = lambda_0 * l_glob + lambda_1 * l_local_1 + lambda_2 * l_local_2
        elif stage==2:           
            total_loss = l_glob + lambda_2 * l_local_2 
        else:
            total_loss = l_glob
        
        stats = {"loss":l_glob.data.cpu().numpy()}
        mean_loss_glob += l_glob.data.cpu().numpy()
        mean_loss_local_1 += l_local_1.data.cpu().numpy()
        mean_loss_local_2 += l_local_2.data.cpu().numpy()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()

        #printer.print(stats, iteration)
        logger.log(stats, iteration)
        #vizlogger.plot_steps(logger.get_data())
   
    print('Local Loss 1: %.4f'%(mean_loss_local_1/iteration))
    print('Local Loss 2: %.4f'%(mean_loss_local_2/iteration))
    print('Global Loss: %.4f'%(mean_loss_glob/iteration))


def test_epoch(dataloader, network, loss, loggers):
    logger, vizlogger = loggers

    printer = Printer(N=len(dataloader))
    logger.set_mode("test")
    mean_loss = 0.
    
    with torch.no_grad():
        for iteration, data in enumerate(dataloader):

            input, target = data

            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            output = network.forward(input)
            l = loss(output, target)
            
            stats = {"loss":l.data.cpu().numpy()}
            mean_loss += l.data.cpu().numpy()
            
            printer.print(stats, iteration)
            logger.log(stats, iteration)
            vizlogger.plot_steps(logger.get_data())

        vizlogger.plot_images(target.cpu().detach().numpy(), output.cpu().detach().numpy())
    print('Loss: %.4f'%(mean_loss/iteration)) 

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
        hidden=args.hidden,
        lrS=args.lrSC,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_0=args.lambda_0,
        stage = args.stage
    )
