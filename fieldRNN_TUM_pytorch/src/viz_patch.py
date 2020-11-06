import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pylab import savefig
from PIL import Image
import h5py
import scipy
#from sklearn.metrics import plot_confusion_matrix

fold = int(2)
patch_num = 5
patch_size = 24
#patch_1 = [2678,1763]
#patch_1 = [262,1924]
#patch_1 = [364,2066]
#patch_1 = [843,2222]
patch_1 = [4091,2115]

model_list = [ 'baseline', 'msConvSTAR', 'cb_beta_9999', 'data_aug',  'gru', 'tr','lstm','tcn', 'fcn']

for model in model_list:
    data = np.load('/home/pf/pfstaff/projects/ozgur_deep_filed/multi_stage/viz/' + model + '_' + str(fold) + '.npz')
    
    
    output_file = './viz/'    
    target_test = data['targets']
    pred_test = data['predictions2']
    
    #gt_list = data['arr_2']
    #gt_list_names = data['arr_3']
    #print(gt_list)
    #print(gt_list_names)
    
    data = h5py.File("/home/pf/pfstaff/projects/ozgur_deep_filed/data_crop_CH/train_set_24x24_debug.hdf5", "r")
    #data = h5py.File('/home/pf/pfstaff/projects/ozgur_data/test_set_TG_SG_24x24_debug.hdf5', "r")
    
    valid_list = data['valid_list'][:]
    n_val = np.sum(valid_list)
    
    pred_test = pred_test * (target_test>0)
    
    test_shape = target_test.shape
    print('data shape: ', test_shape)
         
    target = np.zeros((n_val, test_shape[1], test_shape[2]))
    pred = np.zeros((n_val, test_shape[1], test_shape[2]))
    
    target[test_shape[0]*(fold-1):test_shape[0]*fold,...] = target_test
    pred[test_shape[0]*(fold-1):test_shape[0]*fold,...] = pred_test
    
    
    print(valid_list.shape)
    print(n_val)
    print(target_test.shape)
    print(target.shape)
    
    
    dummy = np.zeros([int(np.sum(valid_list)),test_shape[1],test_shape[2]])*255
    dummy[:pred.shape[0],:,:] = pred
    pred_map = np.zeros([valid_list.shape[0],test_shape[1],test_shape[2]])*255
    pred_map[valid_list.astype(bool)] = dummy
    
    
    dummy = np.zeros([int(np.sum(valid_list)),test_shape[1],test_shape[2]])
    dummy[:pred.shape[0],:,:] = target
    target_map = np.zeros([valid_list.shape[0],test_shape[1],test_shape[2]])
    target_map[valid_list.astype(bool)] = dummy
    
    #Reshape the maps - test ZH
    Mx = 5064//24-1
    My = 4815//24-1
    
    #Reshape the maps - test SG - TG
    #Mx = 7051//24-1
    #My = 4408//24-1
    
    num_patches = Mx*My
    print('Num patches: ', num_patches)
    
    target_map_image = np.zeros([int(Mx*test_shape[1]),int(My*test_shape[2])])
    pred_map_image = np.zeros([int(Mx*test_shape[1]),int(My*test_shape[2])])
    step = test_shape[1]
    patch_size= test_shape[1]
    count=0
    for i_y in range(0,My):
        for i_x in range(0, Mx):
            target_map_image[int(step * i_x):int(step * i_x )+patch_size, int(step * i_y):int(step * i_y )+patch_size] = target_map[count]
            pred_map_image[int(step * i_x):int(step * i_x )+patch_size, int(step * i_y):int(step * i_y )+patch_size] = pred_map[count]
            count+=1
    
    performance_map_p = (target_map_image == pred_map_image) * (target_map_image != 0) * 182
    performance_map_n = (target_map_image != pred_map_image) * (target_map_image != 0) 
    
    performance_map_n = scipy.ndimage.binary_opening(performance_map_n, structure=np.ones((2,2))).astype(performance_map_n.dtype)
    performance_map_n = performance_map_n * 212
    
    if model == 'baseline':
        performance_map_occ = (performance_map_p+performance_map_n) != 0
        #np.save('performance_map_occ.npy',performance_map_occ)
        #performance_map_occ = np.load('./performance_map_occ.npy')
        
        
    target_map_image = target_map_image * performance_map_occ
    performance_map_n = performance_map_n * performance_map_occ
    performance_map_p = performance_map_p * performance_map_occ
    
    performance_map_e = ((performance_map_p+performance_map_n) == 0) * 255
    performance_map = np.zeros([target_map_image.shape[0],target_map_image.shape[1],3])
    
    performance_map[:,:,0] = performance_map_e
    performance_map[:,:,1] = performance_map_e
    performance_map[:,:,2] = performance_map_e
    performance_map[:,:,0] += performance_map_n
    performance_map[:,:,1] += performance_map_p
    
    pred_map_image = pred_map_image.astype(np.int8)
    target_map_image = target_map_image.astype(np.int8)
    performance_map = performance_map.astype(np.int8)
    
    ##Crop images
#    if model == 'msConvSTAR':
#        performance_map_occ = performance_map_occ[patch_1[0]:patch_1[0]+patch_size,patch_1[1]:patch_1[1]+patch_size] 
    performance_map = performance_map[patch_1[0]:patch_1[0]+patch_size,patch_1[1]:patch_1[1]+patch_size,:] 
    target_map_image = target_map_image[patch_1[0]:patch_1[0]+patch_size,patch_1[1]:patch_1[1]+patch_size] 
        
    img = Image.fromarray(np.uint8(performance_map))
    img = img.resize((128,128), Image.NEAREST)
    img.save('./viz/patch/patch_' + str(patch_num) + '_' + model + '_fold_'+str(fold)+'.png')




#-----------------------------------------------------------------------------------------------------------
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

colordict = {'Unknown':[255,255,255],
             'Apples':[128,0,0], 
             'Beets':	[220,20,60], 
             'Berries':[255,107,70], 
             'Biodiversity area':[0,191,255], 
             'Buckwheat':[135,206,235],
             'Chestnut':[0,0,128], 
             'Chicory':[138,43,226], 
             'Einkorn wheat':[255,105,180], 
             'Fallow':[0,255,255], 
             'Field bean':[210,105,30], 
             'Forest':[65,105,225],
             'Gardens': [255,140,0], 
             'Grain':[139,0,139], 
             'Hedge':[95,158,160], 
             'Hemp':[128,128,128], 
             'Hops':[147,112,219], 
             'Legumes':	[85,107,47],
             'Linen':[176,196,222], 
             'Lupine':[127,255,212],
             'Maize':	[100,149,237],
             'Meadow':[240,128,128], 
             'Mixed crop': [255,99,71]	,
             'Multiple':	[220,220,220], 
             'Mustard':[0,128,128], 
             'Oat':[0,206,209], 
             'Pasture':[106,90,205]	, 
             'Pears':[34,139,34],
             'Peas':[186,85,211], 
             'Potatoes':[189,183,107], 
             'Pumpkin':[205,92,92], 
             'Rye':	[184,134,11], 
             'Sorghum':	[0,100,0], 
             'Soy':	[199,21,133], 
             'Spelt':[25,25,112], 
             'Stone fruit':	[0,0,0],
             'Sugar beet':[152,251,152], 
             'Summer barley':	[245,222,179],
             'Summer rapeseed':	[32,178,170],
             'Summer wheat':	[255,69,0],
             'Sunflowers':	[0,0,255],
             'Tobacco':[238,232,170],
             'Tree crop':[255,255,102], 
             'Vegetables':	[255,20,147], 
             'Vines':	[255,0,0], 
             'Wheat':	[255,215,0], 
             'Winter barley':[128,128,0],
             'Winter rapeseed':[154,205,50],
             'Winter wheat':[124,252,0]}

for key in colordict:
    x = colordict[key] 
    x[0]/=255
    x[1]/=255
    x[2]/=255
    colordict[key] = x

from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('xx-small')

valid_labels = np.unique(target_map_image)
valid_labels = valid_labels.tolist()
valid_label_names = []
for i in valid_labels:
    valid_label_names.append(label_names[labels.index(i)])

valid_label_names.remove('Unknown')
colors = [colordict[p] for p in valid_label_names]

legendfig, ax = plt.subplots(1, 1)
legend_elements = [Line2D([0], [0], color=color, lw=8, label=band) for band,color in dict(zip(valid_label_names, colors)).items()]
ax.legend(handles=legend_elements, ncol=3, loc="center", borderpad=1.2, handlelength=0.4, borderaxespad=0.1, prop=fontP, frameon=None)
#ax.legend(handles=legend_elements, ncol=3, loc="center", prop=fontP, frameon=None)

ax.axis("off")
legendfig.savefig("./viz/patch/legend_croped_patch_" +str(patch_num)+ ".png",dpi=600)




target_map_RGB = np.ones([target_map_image.shape[0],target_map_image.shape[1],3])*255

for i_x in range(target_map_RGB.shape[0]):
    for i_y in range(target_map_RGB.shape[1]):
        target_pix_val = target_map_image[i_x,i_y]
        #pred_pix_val = target_map_image[i_x,i_y]
        if target_pix_val==0:
            continue
        
        target_pix_color = colordict[ label_names[labels.index(target_pix_val)] ]
        target_map_RGB[i_x,i_y,:] = np.array(target_pix_color)*255

target_map_image_img = Image.fromarray(np.uint8(target_map_RGB))
target_map_image_img = target_map_image_img.resize((128,128), Image.NEAREST)
target_map_image_img.save('./viz/patch/target_patch_'+str(patch_num)+'.png')


