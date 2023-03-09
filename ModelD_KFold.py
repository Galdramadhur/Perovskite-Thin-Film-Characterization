import pandas as pd
import numpy as np
from numpy.random import seed
import os
import glob
import image_slicer
from PIL import ImageFile
from PIL import ImageDraw, ImageFont,Image
import cv2
import sklearn
from sklearn.utils import shuffle
import tensorflow
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Lambda, Activation
import sys
import random
from scipy.integrate import simps
from sklearn.model_selection import KFold


#Define whether the order of data should be shuffled for the KFold; is passed as an argument alongside the date
YesOrNo = str(sys.argv[1])
if YesOrNo == 'y':
    KFoldShuffle = True
else:
    KFoldShuffle = False
Date = str(sys.argv[2])

#Definition of paths to align with our file structure; if another file structure is used, these have to be altered accordingly!
PL_path=os.getcwd() + "/Restructured_Data/PL"
UVVis_path=os.getcwd() + "/Restructured_Data/UVVis"
Picture_path=os.getcwd() + "/Restructured_Data/Sample_Pictures"
Results_path=os.getcwd() + "/Restructured_Data/Results"


#Create an index and string list for calling the data
speed_step_1 = [5,10,20,30,40,50]
speed_step_2 = [5,10,15,20,25,30,40,50]
air_speed = [0,75,100,150,200,250]
air_temp = [0,130,180,230,330,430,530,630,730]
multis = [1,2]

def stringcaller():
    index_list = []
    string_list = []
    for i in range(len(speed_step_1)):
        for j in range(len(speed_step_2)):
            for k in range(len(air_speed)):
                for l in range(len(air_temp)):
                    for m in range(len(multis)):
                        s1 = str(speed_step_1[i])
                        s2 = str(speed_step_2[j])
                        s3 = str(air_speed[k])
                        s4 = str(air_temp[l])
                        s5 = str(multis[m])
                        index_list.append([i,j,k,l])
                        index_list.append([i,j,k,l,m])
                        string_list.append(s1+"_"+s2+"_"+s3+"_"+s4+"_"+s5)
                    string_list.append(s1+"_"+s2+"_"+s3+"_"+s4)
    return index_list, string_list
                    
index_list, string_list = stringcaller()

#Reduce the index and string list down to the actually existing datapoints
os.chdir(PL_path)
for i in reversed(range(len(string_list))):
    try:
        os.chdir(PL_path+"/"+string_list[i])
    except:
        del index_list[i]
        del string_list[i]


#Datapoint removed due to faulty PL measurement
string_list.remove("10_20_250_180")

#Datapoint removed due to faulty UVvis measurement
string_list.remove("50_5_200_130")

#Datapoints removed due to bad pictures
string_list.remove("10_20_200_430")
string_list.remove("10_20_250_330")


#Loads in all pictures in color. Rows defines the number of pictures it is split into, flip controls the augmentation
os.chdir(Picture_path)

def load_split_flip(rows,flip):
    split_top_image_file_list = []
    split_top_image_list = []
    split_back_image_file_list = []
    split_back_image_list = []
    for i in range(len(string_list)):
        for file in glob.glob(string_list[i]+"**top**"+"**cropped**"):
            split_top_image_file_list.append(file)
            if rows == 1:
                top_image=cv2.imread(file)
                resized = cv2.resize(top_image, (int(1100),int(2000)), interpolation = cv2.INTER_AREA)
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                split_top_image_list.append(resized)
                if flip == 1:
                    flipped = cv2.flip(resized,0)
                    split_top_image_list.append(flipped)
            else:
                tiles=image_slicer.slice(file,row=rows,col=1,save=False)
                for tile in tiles:
                    resized = tile.image.resize((int(1100),int(2000//rows)))
                    split_top_image_list.append(resized)
                    if flip == 1:
                        flipped = resized.transpose(method=Image.FLIP_TOP_BOTTOM)
                        split_top_image_list.append(flipped)
    for i in range(len(string_list)):
        for file in glob.glob(string_list[i]+"**back**"+"**cropped**"):
            split_back_image_file_list.append(file)
            if rows == 1:
                back_image=cv2.imread(file)
                resized = cv2.resize(back_image, (int(1100),int(2000)), interpolation = cv2.INTER_AREA)
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                split_back_image_list.append(resized)
                if flip == 1:
                    flipped = cv2.flip(resized,0)
                    split_back_image_list.append(flipped)
            else:
                tiles=image_slicer.slice(file,row=rows,col=1,save=False)
                for tile in tiles:
                    resized = tile.image.resize((int(1100),int(2000//rows)))
                    split_back_image_list.append(resized)
                    if flip == 1:
                        flipped = resized.transpose(method=Image.FLIP_TOP_BOTTOM)
                        split_back_image_list.append(flipped)
                        
    return     split_top_image_file_list,split_top_image_list,split_back_image_file_list,split_back_image_list,rows,flip

split_top_image_file_list,split_top_image_list,split_back_image_file_list,split_back_image_list,rows,flip = load_split_flip(rows=3,flip=1)

set_size = len(split_top_image_list)


#For converting a list of PIL images to np arrays
for i in range(len(split_top_image_list)):
    split_top_image_list[i]=np.array(split_top_image_list[i])
    split_back_image_list[i]=np.array(split_back_image_list[i])


#Image normalization
split_top_image_list = np.array(split_top_image_list)/255
split_back_image_list = np.array(split_back_image_list)/255


#Reads in all PL data
os.chdir(PL_path)

PL_file_list = []
PL_list = []
for i in range(len(string_list)):
    try:
        os.chdir(PL_path+"/"+string_list[i])
        #print(os.getcwd())
        for file in glob.glob("**TOP**.txt"):
            PL_file_list.append(file)
            pl_df = pd.read_csv(file, skiprows=20, header=None, sep="\t")
            del pl_df[2]
            pl_df.columns = ["Wavelength [nm]", "Dark reference", "Intensity [a.u.]"]
            PL_list.append(pl_df)
        for file in glob.glob("**BOT**.txt"):
            PL_file_list.append(file)
            pl_df = pd.read_csv(file, skiprows=20, header=None, sep="\t")
            del pl_df[2]
            pl_df.columns = ["Wavelength [nm]", "Dark reference", "Intensity [a.u.]"]
            PL_list.append(pl_df)
    except:
        print(string_list[i])


#Find and save all peak positions of the PL measurements
pl_peak_list = []
for i in range(len(PL_list)):
    column = PL_list[i].iloc[:,2]
    peak_id=column[500:].idxmax()
    peak=PL_list[i].iloc[peak_id,:]
    pl_peak_list.append(peak)


#Creation of a non averaged PL-List for rows=2
pl_peak_list_solo = []
for i in range (len(pl_peak_list)):
    peak_slice = pl_peak_list[i][0]
    pl_peak_list_solo.append(peak_slice)


#Average the wavelength of the peaks over 2 measurements from the same sample
w_avg_list =[]
for i in range(int(len(pl_peak_list)/2)):
    wavelength_1 = pl_peak_list[2*i][0]
    wavelength_2 = pl_peak_list[2*i+1][0]
    w_avg = (wavelength_1+wavelength_2)/2
    w_avg_list.append(w_avg)


#Beginning of AUAC pipeline, reading in all UV-Vis  data comes first
os.chdir(UVVis_path)

UVV_file_list = []
UVV_list = []
for i in range(len(string_list)):
    for file in glob.glob(string_list[i]+"**TOP**.txt"):
        UVV_file_list.append(file)
        uvv_df = pd.read_csv(file, skiprows=2, header=None, sep="\s+")
        uvv_df.columns = ["Wavelength [nm]", "Intensity [a.u.]"]
        UVV_list.append(uvv_df)
    for file in glob.glob(string_list[i]+"**MID**.txt"):
        UVV_file_list.append(file)
        uvv_df = pd.read_csv(file, skiprows=2, header=None, sep="\s+")
        uvv_df.columns = ["Wavelength [nm]", "Intensity [a.u.]"]
        UVV_list.append(uvv_df)
    for file in glob.glob(string_list[i]+"**BOT**.txt"):
        UVV_file_list.append(file)
        uvv_df = pd.read_csv(file, skiprows=2, header=None, sep="\s+")
        uvv_df.columns = ["Wavelength [nm]", "Intensity [a.u.]"]
        UVV_list.append(uvv_df)


#read in thicknesses
os.chdir(UVVis_path)
thickness_df_1 = pd.read_excel("Thickness Machine Learning_Perovskite_Blade.xlsx",engine = 'openpyxl', header=None, skiprows=1, nrows=4)
thickness_df_2 = pd.read_excel("thickness machine learning (2).xlsx",engine = 'openpyxl', header=None, skiprows=1, nrows=4)

thickness_df_1=thickness_df_1.transpose()
thickness_df_2=thickness_df_2.transpose()

thickness_df_1.columns = ["Sample_name", "Top d [nm]","Middle d [nm]","Bottom d [nm]"]
thickness_df_2.columns = ["Sample_name", "Top d [nm]","Middle d [nm]","Bottom d [nm]"]

thickness_df_1=thickness_df_1.iloc[1:37]
thickness_df_2=thickness_df_2.iloc[1:35]

thickness_df=pd.concat([thickness_df_1,thickness_df_2],ignore_index=True)

sorterIndex = dict(zip(string_list, range(len(string_list))))
thickness_df['Sample_name_Rank'] = thickness_df['Sample_name'].map(sorterIndex)
thickness_df.sort_values(['Sample_name_Rank','Sample_name', 'Top d [nm]','Middle d [nm]','Bottom d [nm]'],ascending = [True,True,True,True,True], inplace = True)
thickness_df.drop('Sample_name_Rank', 1, inplace = True)
thickness_df.drop(thickness_df.tail(4).index,inplace=True)

#order thicknesses
thickness_df_ordered = []
for i in range(len(np.array(thickness_df))):
    thickness_df_ordered.append(thickness_df.iloc[i]['Top d [nm]'])
    thickness_df_ordered.append(thickness_df.iloc[i]['Middle d [nm]'])
    thickness_df_ordered.append(thickness_df.iloc[i]['Bottom d [nm]'])


#divide the UV-Vis data by the thicknesses; state_thickness_division makes sure this only happens once
state_thickness_division = 0

if state_thickness_division==0:
    for i in range(len(UVV_list)):
        UVV_list[i]["Intensity [a.u.]"]=UVV_list[i]["Intensity [a.u.]"].div(thickness_df_ordered[i])
    state_thickness_division=1
else:
    print("Already divided!")


#Calculate AUAC via simpson integrals
area_list = []   
for i in range(len(UVV_list)):
    y = UVV_list[i]["Intensity [a.u.]"]
    area = simps(y[:int(w_avg_list[int(i//3)])], dx=1)
    area_list.append(area)

#Normalize the values and repeat the list for flip=1
area_max = np.max(area_list,axis=-1)
area_min = np.min(area_list,axis=-1)
area_list = (area_list-area_min)/(area_max-area_min)+1


area_list_aug = np.repeat(area_list,2)


#Convert data to be used to np arrays and reshape them in an appropiate fashion 
split_top_image_list=np.array(split_top_image_list)
split_top_image_list=np.reshape(split_top_image_list,(split_top_image_list.shape[0],split_top_image_list.shape[1],split_top_image_list.shape[2],3))


split_back_image_list=np.array(split_back_image_list)
split_back_image_list=np.reshape(split_back_image_list,(split_back_image_list.shape[0],split_back_image_list.shape[1],split_back_image_list.shape[2],3))


data_list_type = area_list_aug

split_num = len(data_list_type)

data_list_type = np.array(data_list_type)
data_list_type = np.reshape(data_list_type, (len(data_list_type),1))


#Seed values are fixed to improve consistency for training, further shuffling can be achieved by the KFold argument
seed_value = 1
seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
tensorflow.random.set_seed(seed_value)

#Create training and test data by shuffling and splitting
x_top_s,x_back_s,y_s = shuffle(split_top_image_list,split_back_image_list,data_list_type)
del split_top_image_list,split_back_image_list,data_list_type
X_top_train, X_top_test,X_back_train, X_back_test, y_train, y_test =x_top_s[:int(0.95*set_size)],x_top_s[int(0.95*set_size):],x_back_s[:int(0.95*set_size)],x_back_s[int(0.95*set_size):],y_s[:int(0.95*set_size)],y_s[int(0.95*set_size):]
del x_top_s,x_back_s,y_s


# Merge inputs and targets
inputs1 = np.concatenate((X_top_train, X_top_test), axis=0)
inputs2 = np.concatenate((X_back_train, X_back_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=split_num, shuffle=KFoldShuffle)

channels=1
reg_l1=0.0001
reg_l2=0.001


checkpoint_filepath = Results_path + '/tmp/checkpoint'

# K-fold Cross Validation model evaluation
fold_no = 1
test_actual_inputs = []
test_actual_results = []
test_pred_results = []
for train, test in kfold.split(inputs1,inputs2, targets):

    # Define the model architecture
    input_top = keras.Input(shape=(int(2000//rows), int(1100),3)) 
    input_back = keras.Input(shape=(int(2000//rows), int(1100),3))

    top_wing = layers.Conv2D(channels, (5,5), activation = 'relu',kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2),input_shape=(int(2000//rows), int(1100),3))(input_top)
    top_wing = layers.MaxPooling2D((3,3))(top_wing)
    top_wing = layers.Conv2D(channels, (3,3),kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2), padding='valid',activation='tanh')(top_wing)
    top_wing = layers.Conv2D(channels, (3,3),kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2), padding='valid',activation='relu')(top_wing)                   
    top_wing = layers.MaxPooling2D((3,3))(top_wing)
    top_wing = layers.Conv2D(channels, (3,3),kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2), padding='valid',activation='relu')(top_wing)  
    top_wing = layers.Flatten()(top_wing)
    top_wing = layers.Dense(16,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2))(top_wing)
    top_out = layers.Dense(10,activation='relu')(top_wing)

    back_wing = layers.Conv2D(channels, (5,5), activation = 'relu',kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2),input_shape=(int(2000//rows), int(1100),3))(input_back)
    back_wing = layers.MaxPooling2D((3,3))(back_wing)
    back_wing = layers.Conv2D(channels, (3,3),kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2), padding='valid',activation='tanh')(back_wing)
    back_wing = layers.Conv2D(channels, (3,3),kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2), padding='valid',activation='relu')(back_wing)                   
    back_wing = layers.MaxPooling2D((3,3))(back_wing)
    back_wing = layers.Conv2D(channels, (3,3),kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2), padding='valid',activation='relu')(back_wing)
    back_wing = layers.Flatten()(back_wing)
    back_wing = layers.Dense(16,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2))(back_wing)
    back_out = layers.Dense(10,activation='relu')(back_wing)

    combined = layers.Concatenate(axis=1)([top_out, back_out])
    combined = layers.Dense(8,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2))(combined)
    combined = layers.Dense(4,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=reg_l1, l2=reg_l2))(combined)
    combined_out = layers.Dense(1,activation='tanh')(combined)

    combined_out = Lambda(lambda x: x+1 )(combined_out)

    model_kfold = keras.Model(inputs=[input_top,input_back], outputs=combined_out)

    # Compile the model
    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
    model_kfold.compile(optimizer=opt,
              loss="MeanAbsolutePercentageError",
              metrics="MeanAbsolutePercentageError")


    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
    
    early_stopping_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',patience=50)
    
    # Fit data to model
    history = model_kfold.fit([inputs1[train],inputs2[train]], targets[train], epochs=150, validation_data=([inputs1[test],inputs2[test]], targets[test]),batch_size = 8, verbose = 1, callbacks=[model_checkpoint_callback,early_stopping_callback])
    
    model_kfold.load_weights(checkpoint_filepath)
    model_kfold.save(Results_path + '/Models' + "/KFoldD_" + Date + "_" +str(fold_no))
    
    # Generate generalization metrics
    scores = model_kfold.evaluate([inputs1[test],inputs2[test]], targets[test], verbose=1)
    print(f'Score for fold {fold_no}: {model_kfold.metrics_names[0]} of {scores[0]}; {model_kfold.metrics_names[1]} of {scores[1]}')
    
    #test_actual_inputs.append(inputs1[test])
    test_actual_results.append(targets[test])
    test_pred_results.append(model_kfold.predict([inputs1[test],inputs2[test]]))


    # Increase fold number'
    fold_no = fold_no + 1
