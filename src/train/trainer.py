import os
import time as timer

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.train.utils.train import TrainUtils

MODEL_DIR = 'outputs/'
LOG_DIR = 'logs/'


def train_wrapper(
        trainer: TrainUtils,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        epochs: int = 3,
        saved_model_filename: str = 'model.pth'):
    model_dir_path = os.path.join(MODEL_DIR)
    log_dir_path = os.path.join(LOG_DIR)

    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    if os.path.splitext(saved_model_filename)[1] is None:
        saved_model_filename = f'{saved_model_filename}.pth'

    if not os.path.splitext(saved_model_filename)[1] == '.pth':
        raise ValueError('Acceptable format for saved model name is .pth only!')

    log_filename = f'log_{saved_model_filename}_{str(int(timer.time()))}.txt'
    fig_filename = f'fig_{saved_model_filename}_{str(int(timer.time()))}.png'
    clf_filename = f'clf_result_{saved_model_filename}_{str(int(timer.time()))}.csv'

    model_filename = os.path.join(model_dir_path, saved_model_filename)
    log_filename = os.path.join(log_dir_path, log_filename)
    fig_filename = os.path.join(log_dir_path, fig_filename)
    clf_filename = os.path.join(log_dir_path, clf_filename)

    train_history = []

    last_train_acc = 0.0
    last_test_acc = 0.0

    with open(log_filename, 'w') as fh:

        print('Begin training!\n')

        # Write log header
        fh.write('epoch\ttrain_acc\ttest_acc\n')

        for epoch in range(epochs):

            # Print epoch status
            print(f'Epoch {epoch + 1} out of {epochs}\n ------------')

            # Train model
            start = timer.time()
            trainer.train(train_dataloader, print_log=False)

            # Get elapsed time
            elapsed_time = timer.time() - start
            print(f'Training time: {elapsed_time:>.2f} seconds')

            # Evaluate model: get training accuracy
            start = timer.time()
            if epoch == (epochs - 1):
                train_accuracy = trainer.test(
                    val_dataloader,
                    print_log=False,
                    print_clf_result=False,
                    print_conf_matrix=True,
                    print_clf_report=True,
                    export_clf_result=False)
            else:
                train_accuracy = trainer.test(
                    val_dataloader,
                    print_log=False,
                    print_clf_result=False,
                    print_conf_matrix=False,
                    print_clf_report=False,
                    export_clf_result=False)
            last_train_acc = train_accuracy

            # Get elapsed time
            elapsed_time = timer.time() - start
            print(f'Validation time: {elapsed_time:>.2f} seconds')

            # Evaluate model: get testing accuracy
            start = timer.time()
            if epoch == (epochs - 1):
                test_accuracy = trainer.test(
                    test_dataloader,
                    print_log=False,
                    print_clf_result=False,
                    print_conf_matrix=True,
                    print_clf_report=True,
                    export_clf_result=True,
                    filename_clf_result=clf_filename)
            else:
                test_accuracy = trainer.test(
                    test_dataloader,
                    print_log=False,
                    print_clf_result=False,
                    print_conf_matrix=False,
                    print_clf_report=False,
                    export_clf_result=False)
            last_test_acc = test_accuracy

            # Get elapsed time
            elapsed_time = timer.time() - start
            print(f'Testing time: {elapsed_time:>.2f} seconds')

            # Append epoch train history
            train_history.append([epoch, train_accuracy, test_accuracy])

            # Write training log
            fh.write(f'{epoch}\t{train_accuracy}\t{test_accuracy}\n')

            # Save model
            torch.save(trainer.get_model(), model_filename)
            print(f'Model {saved_model_filename} stored!\n')

    train_history = pd.DataFrame(train_history, columns=['epoch', 'train_acc', 'test_acc'])
    train_history['epoch'] = train_history['epoch'].apply(lambda x: str(x))

    # Plot accuracy
    plt.figure()
    sns.lineplot(
        data=train_history,
        x='epoch',
        y='train_acc',
        label='Train Accuracy',
        color='#5f0f40')
    sns.lineplot(
        data=train_history,
        x='epoch',
        y='test_acc',
        label='Test Accuracy',
        color='#fb8b24')
    plt.title('Model Accuracy History over Epochs\n', fontdict={
        'fontsize': 15, 'fontweight': 'bold'
    })
    plt.xlabel('Epoch', fontdict={
        'fontsize': 10
    })
    plt.ylabel('Accuracy', fontdict={
        'fontsize': 10
    })
    plt.savefig(fig_filename)
    plt.show()

    print('Train report:')
    print('Last train accuracy:', round(last_train_acc, 3))
    print('Last test accuracy:', round(last_test_acc, 3))

    print("\nDone!")
