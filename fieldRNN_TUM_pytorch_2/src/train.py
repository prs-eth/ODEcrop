import torch.nn
from utils.dataset import ijgiDataset as Dataset
from models.sequenceencoder import LSTMSequentialEncoder
from utils.logger import Logger, Printer, VisdomLogger
import argparse
from utils.snapshot import save, resume
import os
from networks import FCN_CRNN
from eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default='./',help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=1 , type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=4, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=100, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=1e-3, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default=None, type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default=None, type=str, help="directory to save checkpoints")
    return parser.parse_args()

def main(
    datadir,
    batchsize = 1,
    workers = 12,
    epochs = 100,
    lr = 1e-3,
    snapshot = None,
    checkpoint_dir = None
    ):

    #datadir = "/home/pf/pfstaff/projects/ozgur_deep_filed/data" 
    datadir = "/scratch/tmehmet/data"
    traindataset = Dataset(datadir, tileids="tileids/train_fold0.tileids")
    testdataset = Dataset(datadir, tileids="tileids/test_fold0.tileids")

    nclasses = len(traindataset.classes)
    print('Num classes:' , nclasses)
    LOSS_WEIGHT  = torch.ones(nclasses)
    LOSS_WEIGHT[0] = 0

    traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize,shuffle=True,num_workers=workers)
    testdataloader = torch.utils.data.DataLoader(testdataset,batch_size=batchsize,shuffle=False,num_workers=workers)

    logger = Logger(columns=["loss"], modes=["train", "test"])

    vizlogger = VisdomLogger()

    #Define the model
#    network = LSTMSequentialEncoder(48,48,nclasses=nclasses)
    network = FCN_CRNN(fcn_input_size=(30,13,48,48), crnn_input_size=(30,256,48//4,48//4), crnn_model_name='clstm', 
                 hidden_dims=256, lstm_kernel_sizes=(3,3), conv_kernel_size=3, lstm_num_layers=1, avg_hidden_states=True, 
                 num_classes=18, bidirectional=False, pretrained=False, early_feats=True, use_planet=False, resize_planet=True, 
                 num_bands_dict={'s1': 0, 's2': 13, 'planet': 0, 'all': 13 }, main_crnn=False, main_attn_type='None', attn_dims=32, 
                 enc_crnn=False, enc_attn=False, enc_attn_type='None')



    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss = torch.nn.NLLLoss(weight=LOSS_WEIGHT)
    #loss = torch.nn.NLLLoss()

    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()
        loss = loss.cuda()

    start_epoch = 0

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
        #print("\ntest")
        #test_epoch(testdataloader, network,loss, loggers=(logger, vizlogger))

        # evaluate model
        if epoch>-1 and epoch%1 == 0:
            print("\n Eval on test set")
            cm = evaluate(network, testdataset)

        data = logger.get_data()
        vizlogger.update(data)

        if checkpoint_dir is not None:
            checkpoint_name = os.path.join(checkpoint_dir,"model_{:02d}.pth".format(epoch))
            save(checkpoint_name, network, optimizer, epoch=epoch, data=data)


def train_epoch(dataloader, network, optimizer, loss, loggers):
    logger, vizlogger = loggers

    printer = Printer(N=len(dataloader))
    logger.set_mode("train")

    for iteration, data in enumerate(dataloader):
        optimizer.zero_grad()

        input, target = data
        

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = network.forward(input)

        
        l = loss(output, target)
        stats = {"loss":l.data.cpu().numpy()}

        l.backward()
        optimizer.step()

        printer.print(stats, iteration)
        logger.log(stats, iteration)
        vizlogger.plot_steps(logger.get_data())
        
        #return network

def test_epoch(dataloader, network, loss, loggers):
    logger, vizlogger = loggers

    printer = Printer(N=len(dataloader))
    logger.set_mode("test")

    with torch.no_grad():
        for iteration, data in enumerate(dataloader):

            input, target = data

            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            output = network.forward(input)
            l = loss(output, target)
            
            stats = {"loss":l.data.cpu().numpy()}

            printer.print(stats, iteration)
            logger.log(stats, iteration)
            vizlogger.plot_steps(logger.get_data())

        vizlogger.plot_images(target.cpu().detach().numpy(), output.cpu().detach().numpy())

if __name__ == "__main__":

    args = parse_args()

    main(
        args.data,
        batchsize=args.batchsize,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.learning_rate,
        snapshot=args.snapshot,
        checkpoint_dir=args.checkpoint_dir
    )
