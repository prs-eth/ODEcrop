import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pylab import savefig
from PIL import Image
import h5py
import scipy
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sn
import pandas as pd
#plt.rcParams.update({'font.size': 10})



labels = [0, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51]

label_names = ['Unknown', 'Apples', 'Beets', 'Berries', 'Biodiversity area', 'Buckwheat',
 'Chestnut', 'Chicory', 'Einkorn wheat', 'Fallow', 'Field bean', 'Forest',
 'Gardens', 'Grain', 'Hedge', 'Hemp', 'Hops', 'Legumes', 'Linen', 'Lupine',
 'Maize', 'Meadow', 'Mixed crop', 'Multiple', 'Mustard', 'Oat', 'Pasture', 'Pears',
 'Peas', 'Potatoes', 'Pumpkin', 'Rye', 'Sorghum', 'Soy', 'Spelt', 'Stone fruit',
 'Sugar beet', 'Summer barley', 'Summer rapeseed', 'Summer wheat', 'Sunflowers',
 'Tobacco', 'Tree crop', 'Vegetables', 'Vines', 'Wheat', 'Winter barley',
 'Winter rapeseed', 'Winter wheat']

merged = True
output_file = './viz/'    

if merged:
    data1 = np.load('./viz/result_1.npz')
    data2 = np.load('./viz/result_2.npz')
    data3 = np.load('./viz/result_3.npz')
    data4 = np.load('./viz/result_4.npz')
    #data5 = np.load('./viz/result_5.npz')

    target_test1 = data1['targets']
    pred_test1 = data1['predictions2']

    target_test2 = data2['targets']
    pred_test2 = data2['predictions2']
    
    target_test3 = data3['targets']
    pred_test3 = data3['predictions2']
    
    target_test4 = data4['targets']
    pred_test4 = data4['predictions2']

#    target_test5 = data5['targets']
#    pred_test5 = data5['predictions2']    
    
    target_test4 = data4['targets']
    pred_test4 = data4['predictions2']    
    
    target_test = np.concatenate((target_test1, target_test2, target_test4), axis=0)
    pred_test = np.concatenate((pred_test1, pred_test2, pred_test4), axis=0)
    
    
else:
    data = np.load('./viz/result_4.npz')
    target_test = data['targets']
    pred_test = data['predictions2']





valid_idx = target_test != 0  
target_test = target_test[valid_idx]
pred_test = pred_test[valid_idx]

valid_labels = np.unique(target_test)
valid_labels = valid_labels.tolist()
valid_labels_names = []

cm = confusion_matrix(target_test, pred_test, normalize='true', labels=valid_labels)


for l in valid_labels:
    valid_labels_names.append(label_names[labels.index(l)])

print(valid_labels_names)
print('Number of classes: ',len(valid_labels_names))

#fig, ax = plt.subplots()
#ax.tick_params(axis='both', which='major', labelsize=6.8)  # Adjust to fit
#plt.rcParams.update({'font.size': 16})
#cm_display = ConfusionMatrixDisplay(cm,display_labels=valid_labels_names).plot( xticks_rotation='vertical', 
#                                   include_values=False, cmap='Blues', ax=ax)
#savefig('./viz/confusion_matrix.pdf',dpi=600, bbox_inches='tight', pad_inches=-0.)

print(cm.shape)


df_cm = pd.DataFrame(cm, index = [i for i in valid_labels_names],
                  columns = [i for i in valid_labels_names])
plt.figure(figsize = (15,10))
sn.heatmap(df_cm, annot=False,  vmin=0, vmax=1, cmap='Blues')
#plt.xlabel('True label')
#plt.ylabel('Predicted label')
#plt.title('Confusion matrix')
plt.savefig('./viz/cm_merged2.pdf', bbox_inches='tight')






















