import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

class EpochLogger(callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, tasks):
        self.tasks_name = [task['name'] for task in tasks]
        self.tasks_x = [task['x_val'] for task in tasks]
        self.tasks_y = [task['y_val'] if (task['y_val'].ndim == 1 or task['y_val'].shape[1] == 1)
                        else np.argmax(task['y_val'], axis=1) for task in tasks]
        self.reports = {}
        
        for task_name in self.tasks_name :
            self.reports[task_name] = []

    def on_epoch_end(self, epoch, logs={}):
        metrics_log = ''
            
        for i, task_name, task_x, task_y in zip(range(len(self.tasks_y)), self.tasks_name, self.tasks_x, self.tasks_y) :
            y_hat = np.asarray(self.model.predict(task_x))
            y_hat = np.argmax(y_hat, axis=1)
            
            report = classification_report(task_y, y_hat, output_dict=True)
            metrics_log += ' - %s : %.3f'%(task_name, report['accuracy'])
            self.reports[task_name].append(report['accuracy'])
            
        print('step {} ... {}'.format(epoch+1, metrics_log))
    
    def get_result(self):
        return self.reports

def rgb2gray(img):
    return np.expand_dims(np.dot(img, [0.2990, 0.5870, 0.1140]), axis=3)

def visualize_sole_acc(res, maxiter=10, prev_res=None):
    """
    Plot accuracy (one model)
    """
    plt.rc('font',family='DejaVu Sans', size=16)
    fig=plt.figure(figsize=(5, 4.5))
    ax=fig.add_axes([0,0,1,1])
    
    for task, reports in res.items() :
        if prev_res != None and task in prev_res :
            plt.plot(list(range(0, maxiter+1)), [prev_res[task][-1]]+reports[:maxiter], label=task, markersize=8, linewidth=3, clip_on=False)
        else :
            plt.plot(list(range(0, maxiter+1)), [0.0]+reports[:maxiter], label=task, markersize=8, linewidth=3, clip_on=False)

    ax.set_xlim([0, maxiter])
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks(list(range(0, maxiter+1, maxiter//5)))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(range(0, maxiter+1, maxiter//10), minor=True)
    ax.set_yticks([i/10 for i in range(0, 11)], minor=True)
    ax.set_xlabel("Epochs", fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.legend(markerscale=1, fontsize=16, loc='lower right')
    ax.grid(which='both', color='#BDBDBD', linestyle='--', linewidth=1)
    plt.rc('font',family='DejaVu Sans', size=16)

    plt.show()

def visualize_multi_acc(models, maxiter=10, tasks=3):
    """
    Plot accuracy (multiple models)
    """
    fig, axs = plt.subplots(tasks, tasks, figsize=(12, 11.5))
    plt.rc('font',family='DejaVu Sans', size=14)
    
    for i in range(tasks) :
        for j in range(0, i):
            axs[i, j].axis('off')
        for j in range(i, tasks):
            for task_name, reports in models[i].res[j-i].items() :
                if j > i and task_name in models[i].res[j-i-1] :
                    axs[i, j].plot(list(range(0, maxiter+1)),
                                   [models[i].res[j-i-1][task_name][-1]]+reports[:maxiter],
                                   label=task_name, linewidth=3, clip_on=False)
                else :
                    axs[i, j].plot(list(range(0, maxiter+1)), 
                                   [0.0]+reports[:maxiter], 
                                   label=task_name, linewidth=3, clip_on=False)
                axs[i, j].set_xlim([0, maxiter])
                axs[i, j].set_ylim([0.0, 1.0])
                axs[i, j].set_xticklabels(list(range(0, maxiter+1, maxiter//5)), fontsize=14)
                axs[i, j].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
                axs[i, j].set_xticks(range(0, maxiter+1, maxiter//10), minor=True)
                axs[i, j].set_yticks([i/10 for i in range(0, 11)], minor=True)
                if i == j :
                    axs[i, j].set_xlabel("Epochs", fontsize=18)
                    axs[i, j].set_ylabel("Accuracy", fontsize=18)
                axs[i, j].legend(markerscale=1, fontsize=14, loc='lower right')
                axs[i, j].grid(which='both', color='#BDBDBD', linestyle='--', linewidth=1)
    
    fig.tight_layout()
    plt.show()