import os
import time as timer
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


MODEL_DIR = 'models/'
LOG_DIR = 'logs/'


def train_wrapper(
        trainer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        epochs=3,
        saved_model_name='model.pth',
        log_name='log.txt'):

    model_path = os.path.join(MODEL_DIR)
    log_path = os.path.join(LOG_DIR)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    filename_model = os.path.join(model_path, saved_model_name)
    filename_log = os.path.join(log_path, log_name)
    filename_fig = os.path.join(log_path, f'{log_name.split(".")[0]}.png')
    filename_item_result_val_dataset = os.path.join(log_path, f'item_result_val_dataset_{log_name.split(".")[0]}.csv')
    filename_item_result_test_dataset = os.path.join(log_path, f'item_result_test_dataset_{log_name.split(".")[0]}.csv')

    train_history = []

    last_train_acc = 0.0
    last_test_acc = 0.0

    with open(filename_log, 'w') as fh:

        print('Begin training!\n')

        # Write log header
        fh.write('epoch\ttrain_acc\ttest_acc\n')

        for epoch in range(epochs):

            # Print epoch status
            print(f'Epoch {epoch+1} out of {epochs}\n ------------')

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
                    print_item_result=True,
                    print_conf_matrix=True,
                    print_clf_report=True,
                    export_item_result=True,
                    filename_item_result=filename_item_result_val_dataset)
            else:
                train_accuracy = trainer.test(
                    val_dataloader,
                    print_log=False,
                    print_item_result=False,
                    print_conf_matrix=False,
                    print_clf_report=False,
                    export_item_result=False)
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
                    print_item_result=True,
                    print_conf_matrix=True,
                    print_clf_report=True,
                    export_item_result=True,
                    filename_item_result=filename_item_result_test_dataset)
            else:
                test_accuracy = trainer.test(
                    test_dataloader,
                    print_log=False,
                    print_item_result=False,
                    print_conf_matrix=False,
                    print_clf_report=False,
                    export_item_result=False)
            last_test_acc = test_accuracy

            # Get elapsed time
            elapsed_time = timer.time() - start
            print(f'Testing time: {elapsed_time:>.2f} seconds')

            # Append epoch train history
            train_history.append([epoch, train_accuracy, test_accuracy])

            # Write training log
            fh.write(f'{epoch}\t{train_accuracy}\t{test_accuracy}\n')

            # Save model
            torch.save(trainer.get_model(), filename_model)
            print(f'Model {filename_model} stored!\n')

    train_history = pd.DataFrame(
        train_history, columns=[
            'epoch', 'train_acc', 'test_acc'])
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
    plt.savefig(filename_log)
    plt.show()

    print('Train report:')
    print('Last train accuracy:', round(last_train_acc, 3))
    print('Last test accuracy:', round(last_test_acc, 3))

    print("\nDone!")
