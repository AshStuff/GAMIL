import argparse
from dataset import GraphDataset
from model import GAMIL
import os, glob
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train(model, optim, criterion, train_loader, tfboard, epoch, lr_step):
    model.train()
    train_loss = []
    length = len(train_loader)
    for i, batch_dict in tqdm.tqdm(enumerate(train_loader), total=length):
        inputs = batch_dict["data"]
        targets = batch_dict["label"]
        classifier_out = model(inputs)
        optim.zero_grad()
        loss = criterion(classifier_out, targets)
        loss.backward()
        print(f"Iter: {i}/{length}, train loss: {loss.item()}")
        optim.step()
        train_loss.append(loss.item())
    mean_loss = np.mean(train_loss)
    lr_step.step(mean_loss)
    print(f"loss at epoch: {mean_loss} is {epoch}")
    tfboard.add_scalar('train/gamil_loss', mean_loss, epoch)


def validate(model, criterion, val_loader, tfboard, epoch):
    model.eval()
    val_loss = []
    length = len(val_loader)
    for i, batch_dict in tqdm.tqdm(enumerate(val_loader), total=length):
        inputs = batch_dict["data"]
        targets = batch_dict["label"]
        classifier_out = model(inputs)
        loss = criterion(classifier_out, targets)
        val_loss.append(loss.item())
    mean_loss = np.mean(val_loss)
    print(f"===> val loss at epoch: {mean_loss} is {epoch}")
    tfboard.add_scalar('val/gamil_loss', mean_loss, epoch)
    return mean_loss


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    train_data = glob.glob(os.path.join(args.data_path, "train", "*", "*"))
    val_data = glob.glob(os.path.join(args.data_path, "val", "*", "*"))
    train_dataset = GraphDataset(train_data, args.num_graphs, args.max_num_nodes, args.distance)
    val_dataset = GraphDataset(val_data, args.num_graphs, args.max_num_nodes, args.distance)

    print(f"Total number of train data: {len(train_dataset)}")
    print(f"Total number of validation data: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               num_workers=args.num_workers,
                                               batch_size=args.batch_size,
                                               pin_memory=False,
                                               drop_last=True,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             num_workers=args.num_workers,
                                             batch_size=args.batch_size,
                                             pin_memory=False,
                                             shuffle=False)
    model = GAMIL(args.num_graphs, args.max_num_nodes, args.input_dim,
                  args.hidden_dim, args.label_dim, args.output_dim, True, True, args.hidden_dim,
                  assign_ratio=args.assign_ratio, pred_hidden_dims=[50], concat=True,
                  gcn_name=args.gcn_name, collect_assign=args.visualization, load_data_sparse=False,
                  norm_adj=args.norm_adj, activation=args.activation,
                  drop_out=args.drop_out, jk=args.jump_knowledge)
    parallel_model = torch.nn.DataParallel(model)
    cudnn.benchmark = False

    os.makedirs(os.path.join(args.logs_path, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    optim = torch.optim.Adam(parallel_model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    train_tfboard = SummaryWriter(os.path.join(args.logs_path, 'logs', 'train'))
    lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min')
    current_epoch = 1
    best_loss = np.Inf
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        parallel_model.load_state_dict(checkpoint["state_dict"])
        current_epoch = checkpoint['current_epoch']
        best_loss = checkpoint["loss"]
        optim = optim.load_state_dict(checkpoint["optim"])
    criterion = nn.CrossEntropyLoss()
    for epoch in range(current_epoch, args.num_epochs):
        train(parallel_model, optim, criterion, train_loader, train_tfboard, epoch, lr_step)
        if epoch % 1 == 0:
            loss = validate(parallel_model, criterion, val_loader, train_tfboard, epoch)
            if loss < best_loss:
                best_loss = loss
                torch.save({"state_dict": parallel_model.state_dict(),
                            "optim": optim.state_dict(),
                            "loss": best_loss,
                            "current_epoch": current_epoch}, os.path.join(args.save_path, f"best_loss.pth"))

        if epoch % 1 == 0:
            path = os.path.join(args.save_path, f"epoch_{current_epoch}.pth")
            torch.save({"state_dict": parallel_model.state_dict(),
                        "optim": optim.state_dict(),
                        "loss": best_loss,
                        "current_epoch": current_epoch}, os.path.join(args.save_path, f"epoch_{current_epoch}.pth"))
            print(f"saved model at {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', dest='save_path', default='results',
                        help='path where models are saved', type=str)
    parser.add_argument('--data_path', dest='data_path',
                        help='path where data is stored', type=str)
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--batch-size', dest='batch_size', default=1, type=int,
                        help='Batch size.')
    parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
                        help='ratio of number of nodes in consecutive layers')
    parser.add_argument('--num_graphs', dest='num_graphs', type=int,
                        help='num_graphs')
    parser.add_argument('--distance', dest='distance', default=0.5, type=int,
                        help='distance')
    parser.add_argument('--epochs', dest='num_epochs', default=100, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type', default='cl',
                        help='[c, ca, cal, cl] c: coor, a:appearance, l:soft-label')
    parser.add_argument('--input-dim', dest='input_dim', default=128, type=int,
                        help='Input feature dimension')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--label_dim', type=int, default=2, help='number of classes')

    parser.add_argument('--max_num_nodes', dest='max_num_nodes', default=500, type=int,
                        help='max_num_nodes')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=50, type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', default=50, type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int, help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', default=3, type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float, help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--sample-ratio', dest='sample_ratio', default=1, )
    parser.add_argument('--sample-time', dest='sample_time', default=1)
    parser.add_argument('--visualization', action='store_const', const=True, default=False,
                        help='use assignment matrix for visualization')
    parser.add_argument('--method', dest='method', help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix', help='suffix added to the output filename')
    parser.add_argument('--input_feature_dim', dest='input_feature_dim', type=int,
                        help='the feature number for each node', default=8)
    parser.add_argument('--resume', default=False, )
    parser.add_argument('--optim', dest='optimizer', help='name for the optimizer, [adam, sgd, rmsprop] ')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--step_size', default=10, type=int, metavar='N',
                        help='stepsize to decay learning rate (>0 means this is enabled)')
    parser.add_argument('--skip_train', action='store_const', const=True, default=False, help='only do evaluation')
    parser.add_argument('--normalize', default=False, help='normalize the adj matrix or not')
    parser.add_argument('--load_data_list', action='store_true', default=False)
    parser.add_argument('--load_data_sparse', action='store_true', default=False)
    parser.add_argument('--name', default='')
    parser.add_argument('--gcn_name', default='SAGE')
    parser.add_argument('--active', dest='activation', default='relu')
    parser.add_argument('--dynamic_graph', dest='dynamic_graph', action='store_const', const=True, default=False, )
    parser.add_argument('--sampling_method', default='random', )
    parser.add_argument('--test_epoch', default=5, type=int)
    parser.add_argument('--logs_path', dest='logs_path', default=".",
                        help='path where models are saved', type=str)
    parser.add_argument('--norm_adj', action='store_const', const=True, default=False, )
    parser.add_argument('--readout', default='max', type=str)
    parser.add_argument('--task', default='colon', type=str)
    parser.add_argument('--mask', default='cia', type=str)
    parser.add_argument('--n', dest='neighbour', default=8, type=int)
    parser.add_argument('--sample_ratio', default=0.5, type=float)
    parser.add_argument('--drop', dest='drop_out', default=0., type=float)
    parser.add_argument('--noise', dest='add_noise', action='store_const', const=True, default=False, )
    parser.add_argument('--valid_full', action='store_const', const=True, default=False, )
    parser.add_argument('--dist_g', dest='distance_prob_graph', action='store_const', const=True, default=False, )
    parser.add_argument('--jk', dest='jump_knowledge', action='store_const', const=True, default=False)
    parser.add_argument('--g', dest='graph_sampler', default='knn', type=str)
    parser.add_argument('--cv', dest='cross_val', default=1, type=int)

    args = parser.parse_args()
    main(args)
