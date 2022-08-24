import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam

from sklearn.utils import shuffle

from time import time

from dataset_q import *
from model_q import *
from util import *

#def step_decay(epoch):
   #initial_lrate = 0.01
   #drop = 0.5
   #epochs_drop = 10.0
   #lrate = initial_lrate
   #lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   #lrate = np.max((lrate, 0.00051))
   #return lrate

T_max=500 
eta_max=1e-2
eta_min=5e-5
def step_decay(epoch):
    lrate = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
    return lrate
class LearningRate(tf.keras.callbacks.Callback):
    
    def on_train_begin(self,logs={}):
        self.lr_epoch=[]

    def on_epoch_end(self, batch, logs={}):
        self.lr_epoch.append(step_decay(len(self.lr_epoch)+1)) 

def main():
    parser      = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['oxiod', 'euroc', 'broad'], help='Training dataset name (\'oxiod\' or \'euroc\' or \'broad\')')
    parser.add_argument('output', help='Model output name')
    parser.add_argument('ArchNN', choices=['TCN_CuDNNLSTM', 'TCN_LSTM', 'TCN_GRU', 'TCN_CuDNNGRU', 'CNN_GRU', 'CNN_CuDNNGRU', 'CNN_LSTM', 'CNN_CuDNNLSTM', 'CNN_TCN', 'pointnet'], help='Neural Network Architecture (\'TCN_CuDNNLSTM\' or \'TCN_LSTM\' or \'TCN_GRU\' or \'TCN_CuDNNGRU\' or \'CNN_GRU\' or \'CNN_CuDNNGRU\' or \'CNN_LSTM\' or \'CNN_CuDNNLSTM\' or \'CNN_TCN\')')
    args        = parser.parse_args()

    np.random.seed(0)

    window_size = 200
    stride      = 10

    x_gyro      = []
    x_acc       = []

    q   = []
    y_delta_q   = []

    imu_data_filenames  = []
    gt_data_filenames   = []

    if args.dataset == 'oxiod':
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/imu3.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/imu1.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/imu2.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/imu2.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu4.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu4.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu2.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu7.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/imu4.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu5.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu3.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu2.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/imu3.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu1.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu3.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu5.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu4.csv')

        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/vi3.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/vi1.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/vi2.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/vi2.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi4.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi4.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi2.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi7.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/vi4.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi5.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi3.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi2.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/vi3.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi1.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi3.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi5.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi4.csv')
    
    elif args.dataset == 'euroc':
        imu_data_filenames.append('MH_01_easy/mav0/imu0/data.csv')
        imu_data_filenames.append('MH_03_medium/mav0/imu0/data.csv')
        imu_data_filenames.append('MH_05_difficult/mav0/imu0/data.csv')
        imu_data_filenames.append('V1_02_medium/mav0/imu0/data.csv')
        imu_data_filenames.append('V2_01_easy/mav0/imu0/data.csv')
        imu_data_filenames.append('V2_03_difficult/mav0/imu0/data.csv')

        gt_data_filenames.append('MH_01_easy/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('MH_03_medium/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('V1_02_medium/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('V2_01_easy/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('V2_03_difficult/mav0/state_groundtruth_estimate0/data.csv')
        
    elif args.dataset =='broad':
        for i in range(1,25):
            imu_data_filenames.append('BROAD/trial_imu'+str(i)+'.csv')
            
            gt_data_filenames.append('BROAD/trial_gt'+str(i)+'.csv')
            
    

    for i, (cur_imu_data_filename, cur_gt_data_filename) in enumerate(zip(imu_data_filenames, gt_data_filenames)):
        if args.dataset == 'oxiod':
            cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_oxiod_dataset(cur_imu_data_filename, cur_gt_data_filename)
        elif args.dataset == 'euroc':
            cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_euroc_mav_dataset(cur_imu_data_filename, cur_gt_data_filename)
        elif args.dataset =='broad':
            cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_broad_dataset(cur_imu_data_filename, cur_gt_data_filename)

        [cur_x_gyro, cur_x_acc], [q_a] = load_dataset_quat(cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data, window_size, stride)

        x_gyro.append(cur_x_gyro)
        x_acc.append(cur_x_acc)

        q.append(q_a)


    x_gyro      = np.vstack(x_gyro)
    x_acc       = np.vstack(x_acc)

    q   = np.vstack(q)

    print(x_acc.shape)
    print(q.shape)
    x_gyro, x_acc, q = shuffle(x_gyro, x_acc, q)

    if args.ArchNN == 'TCN_LSTM':
        pred_model  = create_pred_model_quat_TCN_LSTM(window_size)
    elif args.ArchNN == 'TCN_CuDNNLSTM':
        pred_model  = create_pred_model_quat_TCN_CuDNNLSTM(window_size)
    elif args.ArchNN == 'TCN_GRU':
        pred_model  = create_pred_model_quat_TCN_GRU(window_size)
    elif args.ArchNN == 'TCN_CuDNNGRU':
        pred_model  = create_pred_model_quat_TCN_CuDNNGRU(window_size)
    elif args.ArchNN == 'CNN_GRU':
        pred_model  = create_pred_model_quat_CNN_GRU(window_size)
    elif args.ArchNN == 'CNN_CuDNNGRU':
        pred_model  = create_pred_model_quat_CNN_CuDNNGRU(window_size)
    elif args.ArchNN == 'CNN_LSTM':
        pred_model  = create_pred_model_quat_CNN_LSTM(window_size)
    elif args.ArchNN == 'CNN_CuDNNLSTM':
        pred_model  = create_pred_model_quat_CNN_CuDNNLSTM(window_size)
    elif args.ArchNN == 'CNN_TCN':
        pred_model  = create_pred_model_quat_CNN_TCN(window_size)
    #elif args.ArchNN == 'pointnet':
        #pred_model = create_pointnet(window_size)


    train_model = pred_model
    #lr_schedule = keras.optimizers.schedules.CosineDecay(0.001, 0.1)
    lr_history=LearningRate()
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)

    #train_model.compile(keras.optimizers.Adam(), loss=None)

    model_checkpoint    = ModelCheckpoint('model_checkpoint.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
    tensorboard         = TensorBoard(log_dir="logs/{}".format(time()))

    history = train_model.fit([x_gyro, x_acc], q , epochs=500, batch_size=1000, verbose=1, callbacks=[lr_history,lrate, model_checkpoint, tensorboard], validation_split=0.1)

    train_model = load_model('model_checkpoint.hdf5', custom_objects={'CustomMultiLossLayer':CustomMultiLossLayer, 'TCN':TCN}, compile=False)

    if args.ArchNN == 'TCN_LSTM':
        pred_model  = create_pred_model_6d_quat_TCN_LSTM(window_size)
    elif args.ArchNN == 'TCN_CuDNNLSTM':
        pred_model  = create_pred_model_6d_quat_TCN_CuDNNLSTM(window_size)
    elif args.ArchNN == 'TCN_GRU':
        pred_model  = create_pred_model_6d_quat_TCN_GRU(window_size)
    elif args.ArchNN == 'TCN_CuDNNGRU':
        pred_model  = create_pred_model_6d_quat_TCN_CuDNNGRU(window_size)
    elif args.ArchNN == 'CNN_GRU':
        pred_model  = create_pred_model_6d_quat_CNN_GRU(window_size)
    elif args.ArchNN == 'CNN_CuDNNGRU':
        pred_model  = create_pred_model_6d_quat_CNN_CuDNNGRU(window_size)
    elif args.ArchNN == 'CNN_LSTM':
        pred_model  = create_pred_model_6d_quat_CNN_LSTM(window_size)
    elif args.ArchNN == 'CNN_CuDNNLSTM':
        pred_model  = create_pred_model_6d_quat_CNN_CuDNNLSTM(window_size)
    elif args.ArchNN == 'CNN_TCN':
        pred_model  = create_pred_model_quat_CNN_TCN(window_size)


    #pred_model = create_pred_model_6d_quat(window_size)
    pred_model.set_weights(train_model.get_weights()[:-2])
    pred_model.save('%s.hdf5' % args.output)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
