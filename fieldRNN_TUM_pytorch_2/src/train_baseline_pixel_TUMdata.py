import numpy as np
import torch.nn
from crop_classification_tcn import Crops
#from utils.dataset_eval import Dataset_eval
from utils.logger import Logger, Printer, VisdomLogger
import argparse
from utils.snapshot import save, resume
import os
from eval_baseline_pixel import  evaluate
from tqdm import tqdm
#from utils.data_sampler import ImbalancedDatasetSampler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default='./',help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=1 , type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=1, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=30, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default=None, type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default='/home/pf/pfstaff/projects/ozgur_deep_filed/multi_stage/trained_models_sota_baselines', type=str, help="directory to save checkpoints")
    parser.add_argument('-wd', "--weight_decay", default=0.0001, type=float, help="weight_decay")
    parser.add_argument('-hd', "--hidden", default=64, type=int, help="hidden dim")
    parser.add_argument('-nl', "--layer", default=6, type=int, help="num layer")    
    parser.add_argument('-lrs', "--lrSC", default=2, type=int, help="lrScheduler")    
    parser.add_argument('-nm', "--name", default='baseline_pixel', type=str, help="name")
    parser.add_argument('-l1', "--lambda_1", default=0.1, type=float, help="lambda_1")
    parser.add_argument('-l2', "--lambda_2", default=0.5, type=float, help="lambda_2")
    parser.add_argument('-l0', "--lambda_0", default=1, type=float, help="lambda_0")
    parser.add_argument('-stg', "--stage", default=3, type=float, help="num stage")
    parser.add_argument('-cp', "--clip", default=5, type=float, help="grad clip")
    parser.add_argument('-sd', "--seed", default=0, type=int, help="random seed")
    parser.add_argument('-fd', "--fold", default=1, type=int, help="5 fold")   
    parser.add_argument('-gt', "--gt_path", default='labelsC.csv', type=str, help="gt file path")
    parser.add_argument('-mt', "--model", default='lstm', type=str, help="model_type")
    
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
    clip=1,
    fold_num=None,
    gt_path=None,
    model_type='lstm'
    ):

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    root = r'data/Crops'
    scratch_root1 = r'/scratch/Nando/ODEcrop/Crops'
    scratch_root2 = r'/cluster/scratch/metzgern/ODEcrop/Crops'
    if os.path.exists(scratch_root1):
        root = scratch_root1
    elif os.path.exists(scratch_root2):
        root = scratch_root2
    print("dataroot: " + root)

    traindataset = Crops(root, mode="train", noskip=True)
    testdataset = Crops(root, mode="eval", noskip=True)
    
    """
    train_dataset_obj = SwissCrops(root, mode="train", device=device, noskip=args.noskip,
                                    step=args.step, trunc=args.trunc, nsamples=args.n,
                                    datatype=args.swissdatatype, singlepix=args.singlepix)
    test_dataset_obj = SwissCrops(root, mode="test", device=device, noskip=args.noskip,
                                    step=args.step, trunc=args.trunc, nsamples=args.validn,
                                    datatype=args.swissdatatype, singlepix=args.singlepix) 
    """
    
    nclasses = traindataset.n_classes
    #nclasses = 125
    print('Num classes:' , nclasses)
    LOSS_WEIGHT  = torch.ones(nclasses)
    LOSS_WEIGHT[0] = 1 #changed!!

    traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize, shuffle=True, num_workers=workers)

    logger = Logger(columns=["loss"], modes=["train", "test"])
    vizlogger = VisdomLogger()

    #Define the model
    if model_type == 'lstm':
        from models.LongShortTermMemory import LSTM    

        network = LSTM(input_dim=54, hidden_dims=440, nclasses=nclasses, num_rnn_layers=1, 
                     dropout=0., bidirectional=False,use_batchnorm=False, use_layernorm=False)
    elif model_type == 'tr':
        from models.TransformerModel import TransformerModel    
        
        network = TransformerModel(input_dim=54, sequencelength=20,
                                   d_model=64, d_inner=128,
                                   n_layers=3, n_head=2,
                                   dropout=0., num_classes=nclasses)
    elif model_type == 'tcn':
        from models.tempCNN import TempCNN    


        print("hidden_size: ", hidden)

        network = TempCNN(input_dim=54, num_classes=nclasses, sequencelength=20, kernel_size=3, hidden_dims=hidden, dropout=0.5)

    
    
    
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Num params: ', params)


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
        network.train()
        train_epoch(traindataloader, network, optimizer, loss, loggers=(logger,vizlogger))

        #call LR scheduler 
        lr_scheduler.step()

        data = logger.get_data()
        vizlogger.update(data)

        # evaluate model
        if epoch>=0 and epoch%1 == 0:
            print("\n Eval on test set")
            test_acc = evaluate(network, testdataset, batchsize=batchsize) 
            
            if checkpoint_dir is not None:
                checkpoint_name = os.path.join(checkpoint_dir, name + "_model.pth")
                if test_acc > best_test_acc:
                    print('Model saved! Best val acc:', test_acc)
                    best_test_acc = test_acc
                    #save(checkpoint_name, network, optimizer, epoch=epoch, data=data)
                


def train_epoch(dataloader, network, optimizer, loss, loggers):
    logger, vizlogger = loggers

    printer = Printer(N=len(dataloader))
    logger.set_mode("train")
    mean_loss = 0.
    
    
    for iteration, data in (enumerate(dataloader)):
        optimizer.zero_grad()

        input, a, b, target = data
          
        #Reshape the data
        #input = input.permute(0,3,4,1,2)
        #input = input.contiguous().view(-1, input.shape[3], input.shape[4])
        #target = target.contiguous().view(-1)
        target = torch.argmax(target,1)
        input = input.float()
        
        #lstm_input = nn.utils.rnn.pack_padded_sequence(embeds, [x - context_size + 1 for x in seq_lengths], batch_first=False)
        # pack padded sequece
        
        #lengths = torch.FloatTensor( [torch.max(torch.nonzero(torch.sum(seq,1))) for seq in input] )
        #usedlength = int(torch.min(lengths))

        #input = input[:,:usedlength,:]
        input = input[:,:20,:]
        #x_padded = torch.nn.utils.rnn.pack_padded_sequence(input, lengths=lengths, batch_first=True, enforce_sorted=False)

        if torch.cuda.is_available():
            x_padded = input.cuda()
            target = target.cuda()

        output = network.forward(input)
        #output = network(input)
        l = loss(output, target)
        stats = {"loss":l.data.cpu().numpy()}
        mean_loss += l.data.cpu().numpy()

        if (l != l).item() :
            print('Nan value')
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
            optimizer.step()
        
        printer.print(stats, iteration)
        logger.log(stats, iteration)
        vizlogger.plot_steps(logger.get_data())
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
        clip = args.clip,
        fold_num = args.fold,
        gt_path = args.gt_path,
        model_type = args.model,
    )
