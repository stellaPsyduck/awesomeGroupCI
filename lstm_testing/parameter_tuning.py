import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch.nn.init as init
import numpy as np
from sklearn.metrics import mean_absolute_error

from lstm_dataprep import get_dataloaders
from single_stock_lstm import UnivariantLSTM

epoch_size_2 = 2
epoch_size_3 = 3
epoch_size_5 = 5
epoch_size_30 = 30
epoch_size_50 = 50
epoch_size_75 = 75
epoch_size_100 = 100
epoch_size_200 = 200

hidden_layer_size_25 = 25
hidden_layer_size_50 = 50
baseline_hidden_layer_size = 75
hidden_layer_size_100 = 100
hidden_layer_size_175 = 175
hidden_layer_size_250 = 250
hidden_layer_size_350 = 350

base_line_lr = 0.001
learning_rate_1 = 0.0005
learning_rate_2 = 0.0008
learning_rate_3 = 0.025
learning_rate_4 = 0.5


batch_size = 64
seq_length = 60
features = 1 

# ----------- Tuned Parameters IBM --------------

tuned_epoch = 6
tuned_hidden_layer = 100
tuned_learning_rate = 0.0009
tuned_lstm_layers = 1


# ------------ Tuned Parameters BB.TO ------------

bb_tuned_epoch = 42
bb_tuned_hidden_layer = 25
bb_tuned_learning_rate = 0.0008
bb_lstm_layers = 1


# premo results:

# IBM:
# --- Running LSTM with 1 layer(s) ---
# Results for 1 layers -> Train MAE: 0.2967 | Val MAE: 0.0950 | Test MAE: 0.1918





# BlackBerry:
# --- Running LSTM with 1 layer(s) ---
# Epoch [10/42] complete.
# Epoch [20/42] complete.
# Epoch [30/42] complete.
# Epoch [40/42] complete.
# Results for 1 layers -> Train MAE: 0.0935 | Val MAE: 0.0440 | Test MAE: 0.0371





# Need to import data somehow from lstm_dataprep.py

# Need to import the univariant LSTM model from single_stock_lstm.py

# Need to train based on epochs first, then go further along

# Can change ticker for later testing etc etc
IBM_train, IBM_valid, IBM_test = get_dataloaders(ticker="IBM", seq_length=seq_length, batch_size=batch_size)
# test_dataloader, _, _ = get_dataloader(ticker="AMD", seq_length=seq_length, batch_size=batch_size)
BB_train, BB_valid, BB_test = get_dataloaders(ticker="BB.TO", seq_length=seq_length, batch_size=batch_size)


# Drop the first training data in this

def train_and_evaluate_lstm_for_epoch(model, train_loader, val_loader, test_loader, epochs, device, learning_rate=0.001):
    training_loss_list = []
    validation_loss_list = []
    epoch_list = []

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train() 
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs.view(-1), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        training_loss_list.append(avg_train_loss)
        epoch_list.append(epoch + 1)

        model.eval() 
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        validation_loss_list.append(avg_val_loss)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{epochs}] | Train MAE: {avg_train_loss:.4f} | Val MAE: {avg_val_loss:.4f}')

    plt.figure(figsize=(8, 6))
    plt.plot(epoch_list, training_loss_list, label='Training Loss', color='blue')
    plt.plot(epoch_list, validation_loss_list, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('LSTM Training vs Validation Loss')
    plt.legend()  
    plt.savefig('LSTM_Train_vs_Val_Loss -- IBM.png')
    plt.show()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs.view(-1), batch_y)
            test_loss += loss.item()
            
    avg_test_loss = test_loss / len(test_loader)
    print(f"FINAL TEST MAE: {avg_test_loss:.4f}")
    
    return model





# From this, we can see that the ideal epoch is 50, can say taht in a better way

def evaluate_lstm_hidden_sizes(train_loader, val_loader, test_loader, device, input_size, epochs=tuned_epoch, learning_rate=0.001):
    results = {}
    hidden_size_list = [hidden_layer_size_25, hidden_layer_size_50, baseline_hidden_layer_size, hidden_layer_size_100, hidden_layer_size_175, hidden_layer_size_250, hidden_layer_size_350] 

    for size in hidden_size_list:
        print(f'\n--- Running LSTM with hidden layer size of {size} ---')
        model = UnivariantLSTM(input_size=input_size, hidden_size=size).to(device)

        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train() 
            epoch_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}] complete.')

        final_train_loss = epoch_train_loss / len(train_loader)

        model.eval() 
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                epoch_val_loss += loss.item()

        final_val_loss = epoch_val_loss / len(val_loader)

        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                test_loss += loss.item()
                
        final_test_loss = test_loss / len(test_loader)

        print(f'Results for size {size} -> Train MAE: {final_train_loss:.4f} | Val MAE: {final_val_loss:.4f} | Test MAE: {final_test_loss:.4f}')

        results[size] = {
            'train_mae': final_train_loss,
            'val_mae': final_val_loss,
            'test_mae': final_test_loss
        }

    sizes = list(results.keys())
    train_maes = [results[s]['train_mae'] for s in sizes]
    val_maes = [results[s]['val_mae'] for s in sizes]

    x = np.arange(len(sizes)) 
    width = 0.35 

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, train_maes, width, label='Train MAE')
    rects2 = ax.bar(x + width/2, val_maes, width, label='Validation MAE')

    ax.set_ylabel('Mean Absolute Error')
    ax.set_xlabel('LSTM Hidden Layer Size')
    ax.set_title('MAE by LSTM Hidden Layer Size')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    
    caption_text = "Figure: Comparison of Training and Validation MAE across different LSTM hidden layer sizes."
    plt.figtext(0.5, 0.02, caption_text, ha="center", fontsize=10, wrap=True)


    plt.savefig('LSTM_Hidden_Layers_Comparison -- IBM.png')
    # plt.show()

    return results

# Best so far is Results for size 250 -> Train MAE: 0.0893 | Val MAE: 0.0801 | Test MAE: 0.1313
    # Very slow though, but can work for now

# results for 250 and 350 very similar, for time sake, we stay with 250

def evaluate_lstm_learning_rates(train_loader, val_loader, test_loader, device, input_size, hidden_size=tuned_hidden_layer, epochs=tuned_epoch):
    results = {}

    learning_rate_list = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011]

    for rate in learning_rate_list:
        print(f'\n--- Running LSTM with learning rate of {rate} ---')

        # 1 layer, will have dropout = 0.0 automatically
        model = UnivariantLSTM(input_size=input_size, hidden_size=hidden_size).to(device)

        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=rate)

        for epoch in range(epochs):
            model.train() 
            epoch_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}] complete.')

        final_train_loss = epoch_train_loss / len(train_loader)

        model.eval() 
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                epoch_val_loss += loss.item()

        final_val_loss = epoch_val_loss / len(val_loader)


        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                test_loss += loss.item()
                
        final_test_loss = test_loss / len(test_loader)

        print(f'Results for LR {rate} -> Train MAE: {final_train_loss:.4f} | Val MAE: {final_val_loss:.4f} | Test MAE: {final_test_loss:.4f}')


        results[rate] = {
            'train_mae': final_train_loss,
            'val_mae': final_val_loss,
            'test_mae': final_test_loss
        }

    rates_str = [str(r) for r in results.keys()]
    train_maes = [results[r]['train_mae'] for r in results.keys()]
    val_maes = [results[r]['val_mae'] for r in results.keys()]

    x = np.arange(len(rates_str)) 
    width = 0.35 

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, train_maes, width, label='Train MAE')
    rects2 = ax.bar(x + width/2, val_maes, width, label='Validation MAE')

    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_xlabel('Learning Rate')
    ax.set_title('MAE by Learning Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(rates_str)
    ax.legend()
    
    caption_text = "Figure: A visualization of the training and validation MAE for different learning rates."
    plt.figtext(0.5, 0.02, caption_text, ha="center", fontsize=10, wrap=True)

    plt.savefig('Learning_Rates_Comparison -- IBM.png')
    # plt.show()

    return results

# 0.001 is consistently the best over many iterations


# Don't take in model, need to change layers per model
def evaluate_lstm_num_layers(train_loader, val_loader, test_loader, device, input_size, hidden_size=64, epochs=tuned_epoch, learning_rate=tuned_learning_rate):
    results = {}
    
    layer_list = [1, 2, 3, 4, 5, 6]

    for layers in layer_list:
        print(f'\n--- Running LSTM with {layers} layer(s) ---')

        model = UnivariantLSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=layers,
            dropout=0.0
        ).to(device)

        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train() 
            epoch_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}] complete.')

        final_train_loss = epoch_train_loss / len(train_loader)

        model.eval() 
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                epoch_val_loss += loss.item()

        final_val_loss = epoch_val_loss / len(val_loader)

        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                test_loss += loss.item()
                
        final_test_loss = test_loss / len(test_loader)

        print(f'Results for {layers} layers -> Train MAE: {final_train_loss:.4f} | Val MAE: {final_val_loss:.4f} | Test MAE: {final_test_loss:.4f}')


        results[layers] = {
            'train_mae': final_train_loss,
            'val_mae': final_val_loss,
            'test_mae': final_test_loss
        }

    layers = list(results.keys())
    train_maes = [results[l]['train_mae'] for l in layers]
    val_maes = [results[l]['val_mae'] for l in layers]

    x = np.arange(len(layers)) 
    width = 0.35 

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, train_maes, width, label='Training MAE')
    rects2 = ax.bar(x + width/2, val_maes, width, label='Validation MAE')

    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_xlabel('Number of LSTM Layers')
    ax.set_title('MAE by Number of LSTM Layers')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    
    
    caption_text = "Figure : A visualization of the training and validation MAE across different amounts of stacked LSTM layers."
    plt.figtext(0.5, 0.02, caption_text, ha="center", fontsize=10, wrap=True)

    plt.savefig('LSTM_Layers_Comparison -- IBM.png')
    # plt.show()

    return results

# Best results seem to come from 3 layers

# This comes from looking at the validation



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU is available")
    else:
        device = torch.device('cpu')
        print(f"GPU is not available")

    model = UnivariantLSTM(input_size=features, hidden_size=tuned_hidden_layer, output_size=1)
    model.to(device)

    # train_and_evaluate_lstm_for_epoch(model, train_loader=IBM_train, val_loader=IBM_valid, 
    #                     test_loader=IBM_test, epochs=200, device=device, learning_rate=0.001)
    
    # evaluate_lstm_hidden_sizes(train_loader=IBM_train, val_loader=IBM_valid, test_loader=IBM_test, input_size=features, device=device, learning_rate=0.001)

    # evaluate_lstm_learning_rates(train_loader=IBM_train, val_loader=IBM_valid, test_loader=IBM_test, input_size=features, device=device)

    evaluate_lstm_num_layers(train_loader=IBM_train, val_loader=IBM_valid, test_loader=IBM_test, input_size=features, hidden_size=tuned_hidden_layer, epochs=tuned_epoch, learning_rate=tuned_learning_rate, device=device)

    # train_and_evaluate_lstm_for_epoch(model, train_loader=BB_train, val_loader=BB_valid, 
    #                     test_loader=BB_test, epochs=200, device=device, learning_rate=0.001)

    # evaluate_lstm_hidden_sizes(train_loader=BB_train, val_loader=BB_valid, test_loader=BB_test, input_size=features, device=device, epochs=bb_tuned_epoch, learning_rate=0.001)

    # evaluate_lstm_learning_rates(train_loader=BB_train, val_loader=BB_valid, test_loader=BB_test, input_size=features,  epochs=bb_tuned_epoch, hidden_size=bb_tuned_hidden_layer, device=device)

    # evaluate_lstm_num_layers(train_loader=BB_train, val_loader=BB_valid, test_loader=BB_test, input_size=features, hidden_size=bb_tuned_hidden_layer, epochs=bb_tuned_epoch, learning_rate=bb_tuned_learning_rate, device=device)

# DO BB.TO for next tuning