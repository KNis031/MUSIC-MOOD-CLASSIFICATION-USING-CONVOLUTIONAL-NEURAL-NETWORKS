import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_avg_loss(file):
    data = pd.read_csv(file)
    train_loss = data[data['val0_tr1'] == 1]
    val_loss = data[data['val0_tr1'] == 0]
    train_loss_avg = [(train_loss[train_loss['Epoch'] == i]).mean()['loss'] for i in range(1, 11)]
    val_loss_avg = [(val_loss[val_loss['Epoch'] == i]).mean()['loss'] for i in range(1, 11)]
    return train_loss['loss'], train_loss_avg

def get_roc(file):
    data = pd.read_csv(file)
    roc_auc = data['roc_auc']
    pr_auc = data['pr_auc']
    return roc_auc, pr_auc

scale = 0.8
#plt.figure(figsize=(16*scale*0.45, 9*scale)) #small
plt.figure(figsize=(16*scale, 9*scale))
epochs_7 = np.arange(4, 11)
epochs_10 = np.arange(0, 10)

### ROC AUC ###
#popular_auc = np.ones(7)*0.5
#plt.title('ROC AUC for different models')
#plt.plot(epochs_7, get_roc('FCN5.csv')[0], label='FCN5')
#plt.plot(epochs_7, get_roc('FCN6.csv')[0], label='FCN6')
#plt.plot(epochs_7, get_roc('FCN7.csv')[0], label='FCN7')
#plt.plot(epochs_7, popular_auc, label='popular', linestyle='dashed')
#plt.xlabel('epoch')
#plt.ylabel('roc auc')

### PR AUC ####
#popular_pr = np.ones(7)*0.0319
#plt.title('PR AUC for different models')
#plt.plot(epochs_7, get_roc('FCN5.csv')[1], label='FCN5')
#plt.plot(epochs_7, get_roc('FCN6.csv')[1], label='FCN6')
#plt.plot(epochs_7, get_roc('FCN7.csv')[1], label='FCN7')
#plt.plot(epochs_7, popular_pr, label='popular', linestyle='dashed')
#plt.xlabel('epoch')
#plt.ylabel('pr auc')

### Training loss ######
plt.title('Training loss for different models')
plt.plot(epochs_10, get_avg_loss('loss_log_FCN5.csv')[1], label='FCN5')
plt.plot(epochs_10, get_avg_loss('loss_log_FCN6.csv')[1], label='FCN6')
plt.plot(epochs_10, get_avg_loss('loss_log_FCN7.csv')[1], label='FCN7')
#plt.plot(0.15, label='FCN5 (TBA)')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend()
plt.show()