import argparse
import json
import logging
import os
from time import time
import numpy as np
import dgl
import numpy as np
import random 
import torch
import torch.nn
import torch.nn.functional as F
from dgl.data import LegacyTUDataset
from dgl.dataloading import GraphDataLoader
from network import get_network
from torch.utils.data import random_split
from utils import get_stats, boxplot, acc_loss_plot, set_random_seed
from data import GraphDataset
from test_stanford_networks import test_networks
import h5py

def parse_args():
    parser = argparse.ArgumentParser(description="GNN for network classification")
    parser.add_argument("--dataset", type=str, default="dataset", help="just naming of the data added to the info after training the model")
    parser.add_argument("--plot_statistics", type=bool, default=False, help="do plots about acc/loss/boxplot")
    parser.add_argument("--feat_type", type=str, default="ones_feat", choices=["ones_feat", "noise_feat", "degree_feat", "identity_feat"], help="ones_feat/noies_feat/degree_feat/identity_feat")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay of the learning rate over epochs for the optimizer")
    parser.add_argument("--pool_ratio", type=float, default=0.25, help="pooling ratio")
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden size, number of neuron in every hidden layer but could change for currten type of networks")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout ratio")
    parser.add_argument("--epochs", type=int, default=100, help="max number of training epochs")
    parser.add_argument("--patience", type=int, default=-1, help="patience for early stopping, -1 for no stop")
    parser.add_argument("--device", type=str, default="cuda", help="device cuda or cpu")
    parser.add_argument("--architecture",type=str,default="hierarchical",choices=["hierarchical", "global", "gnn", "gin"],help="model architecture",)
    parser.add_argument("--dataset_path", type=str, default="./data", help="path to dataset")
    parser.add_argument("--num_layers", type=int, default=3, help="number of conv layers")
    parser.add_argument("--print_every",type=int,default=10,help="print train log every k epochs, -1 for silent training",)
    parser.add_argument("--num_trials", type=int, default=1, help="number of trials")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--k", type=int, default="4", help="for ID-GNN where control the depth of the generated ID features for helping detecting cycles of length k-1 or less")
    parser.add_argument("--output_activation", type=str, default="log_softmax", help="output_activation function")
    parser.add_argument("--optimizer_name", type=str, default="Adam", help="optimizer type default adam")
    parser.add_argument("--save_hidden_output_train", type=bool, default=False, help="saving the output before output_activation applied for the model in training")
    parser.add_argument("--save_hidden_output_test", type=bool, default=False, help="saving the output before output_activation applied for the model testing/validation")
    parser.add_argument("--save_last_epoch_hidden_output", type=bool, default=False, help="saving the last epoch hidden output only if it is false that means save for all epochs this applied to train and test if they are True")
    parser.add_argument("--loss_name", type=str, default='nll_loss', help='choose loss function corrlated to the optimization function')
    args = parser.parse_args()

    # device
    args.device = "cpu" if args.device == -1 else "cuda"
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, use CPU for training.")
        args.device = "cpu"

    # print every
    if args.print_every == -1:
        args.print_every = args.epochs + 1

    # paths
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.patience == -1:
        args.patience = args.epochs+1
    name = "Data_{}_Hidden_{}_Arch_{}_Pool_{}_WeightDecay_{}_Lr_{}.log".format(
        args.dataset,
        args.hidden_dim,
        args.architecture,
        args.pool_ratio,
        args.weight_decay,
        args.lr,
    ) 
    args.output = os.path.join(args.output_path, name)
    
    return args


def train(model: torch.nn.Module, optimizer, trainloader, device, args, trial, e):
    model.train()
    total_loss = 0.0
    num_batches = len(trainloader)
    list_hidden_output = []
    for batch in trainloader:
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        if args.save_hidden_output_train == True and (args.save_last_epoch_hidden_output == False or e == args.epochs-1):
            out, hidden_feat = model(batch_graphs)
            hidden_feat = hidden_feat.cpu().detach().numpy()
            list_hidden_output.append(hidden_feat)
            del hidden_feat
        else:
            out, _ = model(batch_graphs)
        loss = getattr(F, args.loss_name)(out, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if args.save_hidden_output_train == True and (args.save_last_epoch_hidden_output == False or e == args.epochs-1):
        with h5py.File("{}/save_hidden_output_train_trial{}.h5".format(args.output_path, trial), 'a') as hf:
            hf.create_dataset('epoch_{}'.format(e), data=np.concatenate(list_hidden_output))

    return total_loss / num_batches


@torch.no_grad()
def test(model: torch.nn.Module, loader, device, args, trial, e, if_test):
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    list_hidden_output = []

    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        if args.save_hidden_output_test == True and if_test and (args.save_last_epoch_hidden_output == False or e == args.epochs-1):
            out, hidden_feat = model(batch_graphs)
            hidden_feat = hidden_feat.cpu().detach().numpy()
            list_hidden_output.append(hidden_feat)
            del hidden_feat
        else:
            out, _ = model(batch_graphs)
        pred = out.argmax(dim=1)
        loss += F.nll_loss(out, batch_labels, reduction="sum").item()
        correct += pred.eq(batch_labels).sum().item()

    if args.save_hidden_output_test == True and if_test and (args.save_last_epoch_hidden_output == False or e == args.epochs-1):
        with h5py.File("{}/save_hidden_output_test_trial{}.h5".format(args.output_path, trial), 'a') as hf:
            hf.create_dataset('epoch_{}'.format(e), data=np.concatenate(list_hidden_output))
    
    return correct / num_graphs, loss / num_graphs


def main(args, seed, save=True):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #

    dataset = GraphDataset(device=args.device)
    dataset.load(args.dataset_path)
    if args.feat_type == 'ones_feat':
        dataset.add_ones_feat()
    elif args.feat_type == 'noise_feat':
        dataset.add_noise_feat()
    elif args.feat_type == "identity_feat":
        dataset.add_identity_feat(args.k)
    else:
        dataset.add_degree_feat()

    # add self loop. We add self loop for each graph here since the function "add_self_loop" does not
    # support batch graph.
    for i in range(len(dataset)):
        dataset.graphs[i] = dgl.add_self_loop(dataset.graphs[i])

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - num_val - num_training
    
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [num_training, num_val, num_test], generator=generator)

    #train_set, val_set, test_set = dgl.data.utils.split_dataset(dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=seed)
    
    train_loader = GraphDataLoader(
        train_set, batch_size=args.batch_size, 
    )
    val_loader = GraphDataLoader(
        val_set, batch_size=args.batch_size, 
    )
    test_loader = GraphDataLoader(
        test_set, batch_size=args.batch_size, 
    )

    device = args.device
    
    set_random_seed(seed)

    # Step 2: Create model =================================================================== #
    num_feature, num_classes, _ = dataset.statistics()
    model_op = get_network(args.architecture)
    model = model_op(
        in_dim=num_feature,
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        num_layers=args.num_layers,
        pool_ratio=args.pool_ratio,
        dropout=args.dropout,
        output_activation = args.output_activation
    ).to(device)
    args.num_feature = int(num_feature)
    args.num_classes = int(num_classes)

    # Step 3: Create training components ===================================================== #
    if hasattr(torch.optim, args.optimizer_name):
        optimizer = getattr(torch.optim, args.optimizer_name)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # Replace `parameters` with your specific parameters
    else:
        print(f"Optimizer '{args.optimizer_name}' not found in torch.optim.")
    
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Step 4: training epoches =============================================================== #
    bad_cound = 0
    best_val_loss = float("inf")
    final_test_acc = 0.0
    best_epoch = 0
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    train_times = []
    
    for e in range(args.epochs):
        s_time = time()
        train_loss = train(model, optimizer, train_loader, device, args, seed, e)
        scheduler.step()
        train_times.append(time() - s_time)
        train_acc, _ = test(model, train_loader, device, args, seed, e, 0)
        val_acc, val_loss = test(model, val_loader, device, args, seed, e, 0)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc, _ = test(model, test_loader, device, args, seed, e, 1)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            final_test_acc = test_acc
            if save == True:
                torch.save(model.state_dict(), '{}/best_model_weights_trail{}_{}_{}.pth'.format(args.output_path, seed, args.dataset, args.feat_type))
            bad_cound = 0
            best_epoch = e + 1
        else:
            bad_cound += 1
        if bad_cound >= args.patience and (e+1)%5 == 0:
            args.epochs = e+1
            break

        if (e + 1) % args.print_every == 0:
            log_format = (
                "Epoch {}: loss={:.4f}, train_acc={:.4f}, val_acc={:.4f}, final_test_acc={:.4f}"
            )
            print(log_format.format(e + 1, train_loss, train_acc, val_acc, final_test_acc))
    print(
        "Best Epoch {}, final test acc {:.4f}".format(
            best_epoch, final_test_acc
        )
    )
    if save == True:
        torch.save(model.state_dict(), '{}/last_model_weights_trail{}_{}_{}.pth'.format(args.output_path, seed, args.dataset, args.feat_type))
    
    return final_test_acc, sum(train_times) / len(train_times), [train_loss_list, train_acc_list, val_acc_list] 


def stats():
    boxplot(accs, args.output_path, "Test_Accurcy", args.feat_type)
    acc_loss_plot(stat_list, args.epochs, 5, args.num_trials, args.output_path, args.feat_type, args.dataset)

if __name__ == "__main__":
    global args 
    args = parse_args()
    accs = []
    train_times = []
    idx = None
    best_acc = -1
    stat_list = []
    for i in range(args.num_trials):
        print("Trial {}/{}".format(i + 1, args.num_trials))
        acc, train_time, stat = main(args, i)
        accs.append(acc)
        if best_acc < acc:
            idx = i
            best_acc = acc
            
        train_times.append(train_time)
        stat_list.append(stat)
    

    if args.plot_statistics == True:
        stats()

    print("best trail model is : model_weights_trail{}_{}_{}.pth".format(idx, args.dataset, args.feat_type))

    mean, err_bd = get_stats(accs)
    print("mean acc: {:.4f}, error bound: {:.4f}".format(mean, err_bd))

    out_dict = {
        "hyper-parameters": vars(args),
        "result": "{:.4f}(+-{:.4f})".format(mean, err_bd),
        "train_time": "{:.4f}".format(sum(train_times) / len(train_times)),
    }

    with open(args.output, "w") as f:
        json.dump(out_dict, f, sort_keys=True, indent=4)














 