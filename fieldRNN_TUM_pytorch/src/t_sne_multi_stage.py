import numpy as np
import torch.nn
#from utils.dataset_hdf5_4 import Dataset
from utils.dataset_multistage_2 import Dataset
from utils.dataset_eval import Dataset_eval
from models.multi_stage_sequenceencoder import multistageSTARSequentialEncoder
from utils.logger import Logger, Printer, VisdomLogger
import argparse
from utils.snapshot import save, resume
import os
from networks import FCN_CRNN
from eval import evaluate, evaluate_fieldwise
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default='./',help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=16 , type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=1, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=100, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default='./trained_models_multistage/ms_lrs_2_lambda_05_05_s3_model.pth', type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default='trained_models', type=str, help="directory to save checkpoints")
    parser.add_argument('-wd', "--weight_decay", default=0.00005, type=float, help="weight_decay")
    parser.add_argument('-hd', "--hidden", default=64, type=int, help="hidden dim")
    parser.add_argument('-nl', "--layer", default=6, type=int, help="num layer")    
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
    num_samples  = 100

    #testdataset =  Dataset("/scratch/tmehmet/train_set_24X24_debug.hdf5", 0.9, 'test') 
    testdataset =  Dataset('/home/pf/pfstaff/projects/ozgur_data/test_set_TG_SG_24x24_debug.hdf5', 0.9, 'test', True) 
    dataloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=16, num_workers=0)
    
    nclasses = testdataset.n_classes
    nclasses_local_1 = testdataset.n_classes_local_1
    nclasses_local_2 = testdataset.n_classes_local_2
    #nclasses = 125
    print('Num classes:' , nclasses)
    LOSS_WEIGHT  = torch.ones(nclasses)
    LOSS_WEIGHT[0] = 0


    #Define the model
    network = multistageSTARSequentialEncoder(24,24, nstage=3, nclasses=nclasses, nclasses_l1=nclasses_local_1, 
                                              nclasses_l2=nclasses_local_2, input_dim=4, hidden_dim=hidden, n_layers=layer, viz=True)
    


    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    loss = torch.nn.NLLLoss(weight=LOSS_WEIGHT)

    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()
        loss = loss.cuda()


    if snapshot is not None:        
        resume(snapshot,model=network, optimizer=optimizer)


    #Evaluation
    #evaluate(network, testdataset, 16) 


    #Visulization
    X = list()
    y = list()
    for iteration, data in tqdm(enumerate(dataloader)):

        inputs, targets, _, _, _ = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        #_,_,_, _, z, _ = network.forward(inputs)
        z = network.forward(inputs)
        z = z.cpu().detach().numpy()
        gt = targets.cpu().detach().numpy()

        X.append(z)
        y.append(gt)
        
#        if iteration==100:
#            break
        
    X = np.vstack(X)
    y = np.vstack(y)
    
    X = np.transpose(X, (0, 2, 3, 1))
    
#    X = X[:,:,1,1]
#    y = y[:,1,1]
    X = X.reshape((X.shape[0]*X.shape[1]*X.shape[2], X.shape[3]), order='F')
    y = y.reshape((y.shape[0]*y.shape[1]*y.shape[2]), order='F')

    #color = ['w',  'lime' , 'b', 'tab:gray', 'tab:brown','greenyellow', 'c', 'm', 'darkgoldenrod', 'tab:orange', 'r', 'purple', 'y' ,'k', 'tab:green']
    #labels=['Unknown', 'barley', 'maize', 'wheat', 'corn', 'sugar beet', 'potato', 'winter rape', 'sunflower', 'vegi', 'art_me', 'ex_me', 'o_per_pas', 'pas', 'ex_pas']

    color = ['w',  'lime' , 'b', 'tab:gray', 'tab:brown','greenyellow', 'c', 'm', 'darkgoldenrod', 'tab:orange', 
             'r', 'purple', 'y' ,'k', 'tab:green', 'lightgrey', 'gold', 'azure', 'plum', 'teal', 'skyblue', 'orchid', 'crimson',
             'palegreen', 'coral', 'sienna', 'peru', 'forestgreen', 'royalblue', 'blueviolet', 'darkblue', 'lightpink', 'bisque']
    labels = ['Unknown', 'Barley', 'Beets', 'Berries', 'Biodiversity','Chestnut', 'Fallow', 
              'Field bean', 'Forest', 'Gardens', 'Grain', 'Hedge', 'Hemp', 'Hops', 'Linen', 'Maize', 'Meadow', 
              'MixedCrop','Multiple', 'Oat', 'Orchards', 'Pasture', 'Potatoes', 'Rapeseed', 'Rye', 'Sorghum', 
              'Soy', 'Spelt', 'Sugar_beets','Sunflowers', 'Vegetables', 'Vines', 'Wheat']
        
    X_reduced = list()
    y_reduced = list()
    for i in range(1,33):
        print(labels[i])
        val = y == i
        tempx  = X[val]
        tempy  = y[val]
        
        if tempx.shape[0] > num_samples:
            print('More than ' ,  num_samples , ' samples')
            tempx = tempx[:num_samples]
            tempy = tempy[:num_samples]
        
        X_reduced.append(tempx)
        y_reduced.append(tempy)


    X_reduced = np.vstack(X_reduced)
    y_reduced = np.concatenate(y_reduced) 
    y_reduced = y_reduced.flatten()
    


#--------------------PCA--------------------------------
    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(X_reduced)
           
    fig, ax = plt.subplots()
    for i in range(1,33):
        val = y_reduced==i
        ax.scatter(X_embedded[val,0], X_embedded[val,1], c=color[i], s=2, label=labels[i])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=5, fancybox=True, shadow=False)    
    plt.savefig('pca_mul.png')
    plt.close()
    
#--------------------t-SNE--------------------------------
    X_embedded = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(X_reduced)

    fig, ax = plt.subplots()
    for i in range(1,33):
        val = y_reduced==i
        ax.scatter(X_embedded[val,0], X_embedded[val,1], c=color[i], s=2, label=labels[i])
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=5, fancybox=True, shadow=False)    
    plt.savefig('tsne_mul.png')
    plt.close()

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
