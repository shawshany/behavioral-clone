import numpy as np
from keras.optimizers import Adam
from keras.layers.core import Dense,Activation
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,PReLU
from keras.models import Sequential,Model
from keras import backend as K
from keras.regularizers import l2
import os.path
import csv
import cv2
import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from keras import callbacks
seed = 42

data_path = 'data/'
# 打开CSV文件，这里可以用csv库，或者pandas库
with open(data_path+'driving_log.csv','r') as csvfile:
    file_reader = csv.reader(csvfile,delimiter=',')
    log = []
    for row in file_reader:
        log.append(row)
log = np.array(log)
log = log[1:,:] #remove the header
# 打印出log文件中有多少张图片以及转角数据
print('DataSet: \n {} images| Numbers of steering data:{}'.format(len(log)*3,len(log)))
# 验证数量是否匹配
ls_imgs = glob.glob(data_path+'IMG/*.jpg')
assert len(ls_imgs)==len(log)*3 

def horizontal_flip(img,label):
    '''
    随机水平翻转图像，概率为0.5
    img:原始图像 in array type
    label: 原始图像的转角值
    '''
    choice = np.random.choice([0,1])
    if choice==1:
        img,label = cv2.flip(img,1),-label
    return (img,label)

def transf_brightness(img,label):
    '''
    将一张图片的亮度值调暗，比例为0.1-1之间的随机数
    img:原始图像 in array type
    label:原始图像的转角值
    '''
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    alpha = np.random.uniform(low=0.1,high=1.0,size=None)
    v = hsv[:,:,2]
    v = v*alpha
    hsv[:,:,2] = v.astype('uint8')
    rgb = cv2.cvtColor(hsv.astype('uint8'),cv2.COLOR_HSV2RGB)
    return (rgb,label)

def center_RightLeft_swap(img_adress,label,label_corr=0.25):
    '''
    这里我们规定汽车校准行驶距离为4m,中间摄像头距离左右摄像头的距离均为1m
    img_adress:physical location of the original image file
    label:steering angle value of original image
    label_corr:correction of the steering angle to applied.default value=1/4
    '''
    swap = np.random.choice(['L','R','C'])
    
    if swap=='L':
        img_adress = img_adress.replace('center','left')
        corrected_label = label + label_corr
        return (img_adress,corrected_label)
    elif swap=='R':
        img_adress = img_adress.replace('center','right')
        corrected_label = label - label_corr
        return (img_adress,corrected_label)
    else:
        return (img_adress,label)

def filter_zero_steering(label,del_rate):
    '''
    label: list of steering angle value in the original dataset
    del_rate:rate of deletion-del_rate=0.9 means delete 90% of example with steering angle=0
    '''
    steering_zero_idx = np.where(label==0)
    steering_zero_idx = steering_zero_idx[0]
    size_del = int(len(steering_zero_idx)*del_rate)
    
    return np.random.choice(steering_zero_idx,size=size_del,replace=False)

def image_transformation(img_adress,label,data_dir):
    #img swap
    img_adress , label = center_RightLeft_swap(img_adress,label)
    # Read img file
    img = cv2.imread(data_dir+img_adress)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img , label = transf_brightness(img,label)
    #flip image
    img , label = horizontal_flip(img,label)
    return (img,label)
def continuousSteering(img_sz,activation_fn='relu',l2_reg=[10**-3,10**-3]):
    pool_size = (2,2)
    model = Sequential()
    model.add(Conv2D(filters=8,kernel_size=(5,5),strides=(1,1),padding='valid',name='conv1',input_shape=img_sz,kernel_regularizer=l2(l2_reg[1])))
    
    if activation_fn=='elu':
        model.add(Activation('elu'))
    elif activation_fn=='prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size,padding='valid'))
    
    model.add(Conv2D(filters=8,kernel_size=(5,5),strides=(1,1),padding='valid',kernel_regularizer=l2(l2_reg[1])))
    if activation_fn=='elu':
        model.add(Activation('elu'))
    elif activation_fn=='prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size,padding='valid'))

    model.add(Conv2D(filters=8,kernel_size=(5,5),strides=(1,1),padding='valid',kernel_regularizer=l2(l2_reg[1])))
    if activation_fn=='elu':
        model.add(Activation('elu'))
    elif activation_fn=='prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size,padding='valid'))
    
    model.add(Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid',kernel_regularizer=l2(l2_reg[1])))
    if activation_fn=='elu':
        model.add(Activation('elu'))
    elif activation_fn=='prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size,padding='valid'))
    
    model.add(Conv2D(filters=16,kernel_size=(4,4),strides=(1,1),padding='valid',kernel_regularizer=l2(l2_reg[1])))
    if activation_fn=='elu':
        model.add(Activation('elu'))
    elif activation_fn=='prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    
    model.add(Flatten())
    
    model.add(Dense(128,kernel_regularizer=l2(l2_reg[0])))
    if activation_fn=='elu':
        model.add(Activation('elu'))
    elif activation_fn=='prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    
    model.add(Dense(50,kernel_regularizer=l2(l2_reg[0])))
    if activation_fn=='elu':
        model.add(Activation('elu'))
    elif activation_fn=='prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    
    model.add(Dense(10,kernel_regularizer=l2(l2_reg[0])))
    if activation_fn=='elu':
        model.add(Activation('elu'))
    elif activation_fn=='prelu':
        model.add(PReLU())
    else:
        model.add(Activation('relu'))
    model.add(Dense(1,activation='linear',kernel_regularizer=l2(l2_reg[0]),kernel_initializer='he_normal'))
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam,loss='mean_squared_error')
    print('LightWeight Model is created and compiled..-activation:{}'.format(activation_fn))
    return model
def batch_generator(x,y,batch_size,img_sz,training=True,del_rate=0.95,data_dir='data/',monitor=True,yieldXY=True):
    '''
    生成训练batch：利用yield关键词修饰
    数据增强机制：水平翻转、改变亮度、改变视角
    在每一轮训练中，在使用数据增强前，先删除95%的转角为0的数据。
    x:被用作训练的所有图像的地址
    y:转角
    training:如果为True,生成数据增强后的训练集；如果为false，生成验证集
    batch_size:
    img_sz:生成的图像大小（height,width,channel）
    del_rate:需要删除转角为0的样本比例
    data_dir:图片的地址目录
    monitor:是否将特征与标签进行存储
    yieldXY:如果为真，yields (x,y),否则只yields x
    '''
    if training:
        y_bag = []
        x,y = shuffle(x,y)
        rand_zero_idx = filter_zero_steering(y,del_rate)
        new_x = np.delete(x,rand_zero_idx,axis=0)
        new_y = np.delete(y,rand_zero_idx,axis=0)
    else:
        new_x = x
        new_y = y
    offset = 0
    while True:
        X = np.empty((batch_size,*img_sz))
        Y = np.empty((batch_size,1))
        #generate a batch
        for example in range(batch_size):
            img_adress,img_steering = new_x[example+offset],new_y[example+offset]
            assert os.path.exists(data_dir+img_adress),'Image file['+img_adress+'] not found-'
            
            if training:
                img, img_steering = image_transformation(img_adress,img_steering,data_dir)
            else:
                img = cv2.imread(data_dir+img_adress)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #crop、resize、scale
            X[example,:,:,:] = cv2.resize(img[80:140,0:320],(img_sz[0],img_sz[1]))/255.0-0.5
            Y[example] = img_steering
            if training:
                y_bag.append(img_steering)
            '''
            如果到达原始数据集末尾，就从头开始循环
            '''
            if (example+1)+offset > len(new_y)-1:
                x , y = shuffle(x,y)
                rand_zero_idx = filter_zero_steering(y,del_rate=del_rate)
                new_x = x
                new_y = y
                new_x = np.delete(new_x,rand_zero_idx,axis=0)
                new_y = np.delete(new_y,rand_zero_idx,axis=0)
                offset = 0
        if yieldXY:
            yield (X,Y)
        else:
            yield X
        offset = offset+batch_size
        if training:
            np.save('y_bag.npy',np.array(y_bag))
            np.save('Xbatch_sample.npy',X)#这里保存的是最后一个batch的images
#这里设置一些参数
test_size = 0.2
img_sz = (128,128,3)
batch_size = 200
data_agumentation = 200
nb_epoch = 15
del_rate = 0.95
activation_fn = 'relu'
l2_reg = [0.00001,0]
print('Total number of samples per EPOCH:{}'.format(batch_size*data_agumentation))

x_ = log[:,0]
y_ = log[:,3].astype(float)
x_,y_ = shuffle(x_,y_)
X_train,X_val,y_train,y_val = train_test_split(x_,y_,test_size=test_size,random_state=seed)
print('Train set size:{}|Validation set size:{}'.format(len(X_train),len(X_val)))

sample_per_epoch = batch_size*data_agumentation
#这里我们确保验证集大小为batch_size的倍数
nb_val_samples = (len(y_val) - len(y_val)%batch_size)/200
#train model
model = continuousSteering(img_sz,activation_fn = activation_fn,l2_reg=l2_reg)
print(model.summary())
'''
Callbacks:给予验证集损失保存最佳训练轮数
1.如果验证损失降低了，保存模型或者2.当验证损失连续5次epoch不再改善就停止训练并保存模型
'''

model_path = os.path.expanduser('model.h5')
save_best = callbacks.ModelCheckpoint(model_path,monitor='val_loss',verbose=1,save_best_only=True,mode='min')#Save the model after every epoch.
early_stop = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=15,verbose=0,mode='auto')
callbacks_list = [early_stop,save_best]

# batch generator default value:
history = model.fit_generator(batch_generator(X_train,y_train,batch_size,img_sz,training=True,del_rate=del_rate),
                              steps_per_epoch=200,
                             validation_steps= 80,
                             validation_data = batch_generator(X_val,y_val,batch_size,img_sz,training=False,monitor=False),
                             epochs = nb_epoch,verbose=1,callbacks=callbacks_list)
with open('model.json','w') as f:
    f.write(model.to_json())
model.save('model_.h5')
print('Model saved!')
