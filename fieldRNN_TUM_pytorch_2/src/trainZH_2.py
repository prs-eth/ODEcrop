import numpy as np
import torch.nn
from utils.dataset_hdf5_4 import Dataset
from utils.dataset_eval import Dataset_eval
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
    parser.add_argument('-b', "--batchsize", default=16 , type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=1, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=100, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default=None, type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default='trained_models_ZH', type=str, help="directory to save checkpoints")
    parser.add_argument('-wd', "--weight_decay", default=0.0001, type=float, help="weight_decay")
    parser.add_argument('-hd', "--hidden", default=64, type=int, help="hidden dim")
    parser.add_argument('-nl', "--layer", default=4, type=int, help="num layer")    
    parser.add_argument('-lrs', "--lrscheduler", default=1, type=int, help="lrScheduler")    
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
    hidden=128,
    lrS=1
    ):

    traindataset = Dataset("/cluster/scratch/tmehmet/train_set_24X24_debug.hdf5", 0.5, 'train')
    testdataset =  Dataset_eval("/cluster/scratch/tmehmet/train_set_24X24_debug.hdf5", 0., 'test')    
    testdataset2 =  Dataset_eval("/cluster/scratch/tmehmet/train_set_24X24_debug.hdf5", 0.5, 'test')    
    
    nclasses = traindataset.n_classes
    #nclasses = 125
    print('Num classes:' , nclasses)
    LOSS_WEIGHT  = torch.ones(nclasses)
    LOSS_WEIGHT[0] = 0

    traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize,shuffle=True,num_workers=workers)
    #testdataloader = torch.utils.data.DataLoader(testdataset,batch_size=batchsize,shuffle=False,num_workers=workers)

    logger = Logger(columns=["loss"], modes=["train", "test"])
    vizlogger = VisdomLogger()

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
        train_epoch(traindataloader, network, optimizer, loss, loggers=(logger,vizlogger))
#        print("\ntest")
#        test_epoch(testdataloader, network,loss, loggers=(logger, vizlogger))
    
        #call LR scheduler 
        lr_scheduler.step()

        data = logger.get_data()
        vizlogger.update(data)

        # evaluate model
        if epoch>-0 and epoch%1 == 0:
#            print("\n Eval on train set")
#            evaluate(network, traindataset_2) 
            print("\n Eval on test set")
            evaluate_fieldwise(network, testdataset2, batchsize=batchsize) 
            test_acc = evaluate_fieldwise(network, testdataset, batchsize=batchsize) 
            
            if checkpoint_dir is not None:
                checkpoint_name = os.path.join(checkpoint_dir, name + "_model.pth")
                if test_acc > best_test_acc:
                    print('Model saved! Best val acc:', test_acc)
                    best_test_acc = test_acc
                    save(checkpoint_name, network, optimizer, epoch=epoch, data=data)
                


def train_epoch(dataloader, network, optimizer, loss, loggers):
    logger, vizlogger = loggers

    #printer = Printer(N=len(dataloader))
    logger.set_mode("train")
    mean_loss = 0.
    
    for iteration, data in enumerate(dataloader):
        optimizer.zero_grad()

        input, target = data
        

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = network.forward(input)
        l = loss(output, target)
        stats = {"loss":l.data.cpu().numpy()}
        mean_loss += l.data.cpu().numpy()

        l.backward()
        #torch.nn.utils.clip_grad_norm_(network.parameters(), 1)
        optimizer.step()

        #printer.print(stats, iteration)
        logger.log(stats, iteration)
        #vizlogger.plot_steps(logger.get_data())
    print('Loss: %.4f'%(mean_loss/iteration))


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
        lrS=args.lrscheduler
    )
