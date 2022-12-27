import  os
import glob
import itertools
import argparse
import tqdm
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from dataset import AEDataset
from model import AutoEncoderVGG
from augmentation import Transforms
from loss import CombinedLoss

def train(model, optim, criterion, train_loader, tfboard, epoch, lr_step):
    model.train()
    train_loss = []
    length = len(train_loader)
    for i, batch in tqdm.tqdm(enumerate(train_loader), total=length):
        inputs1, inputs2 = batch
        inputs = torch.cat((inputs1, inputs2), 0)
        decoder_out, features = model(inputs)
        optim.zero_grad()
        loss = criterion(decoder_out, features, inputs)
        loss.backward()
        print(f"Iter: {i}/{length}, train loss: {loss.item()}")
        optim.step()
        train_loss.append(loss.item())
    mean_loss = np.mean(train_loss)
    lr_step.step(mean_loss)
    print(f"loss at epoch: {mean_loss} is {epoch}")
    tfboard.add_scalar('train/clu_loss', mean_loss, epoch)

def validate(model, criterion, val_loader, tfboard, epoch):
    model.eval()
    val_loss = []
    length = len(val_loader)
    for i, batch in tqdm.tqdm(enumerate(val_loader), total=length):
        inputs1, inputs2 = batch
        inputs = torch.cat((inputs1, inputs2), 0)
        decoder_out, features = model(inputs)
        loss = criterion(decoder_out, features, inputs)
        val_loss.append(loss.item())
    mean_loss = np.mean(val_loss)
    print(f"===> val loss at epoch: {mean_loss} is {epoch}")
    tfboard.add_scalar('val/clu_loss', mean_loss, epoch)
    return mean_loss



def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    train_data = glob.glob(os.path.join(args.data_path, "train", "*"))
    val_data = glob.glob(os.path.join(args.data_path, "val", "*"))
    transforms = Transforms()
    train_dataset = AEDataset(train_data,
                              transforms.train_transform,
                              is_train=True)
    val_dataset = AEDataset(val_data,
                            transforms.val_transform,
                            is_train=False)

    print(f"Total number of train data: {len(train_dataset)}")
    print(f"Total number of validation data: {len(val_dataset)}")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               num_workers=args.num_workers,
                                               batch_size=args.batch_size,
                                               pin_memory=True,
                                               drop_last=True,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               num_workers=args.num_workers,
                                               batch_size=args.batch_size,
                                               pin_memory=True,
                                               shuffle=False)

    model = AutoEncoderVGG()
    parallel_model = torch.nn.DataParallel(model)
    cudnn.benchmark = False

    os.makedirs(os.path.join(args.logs_path, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    optim = torch.optim.Adam(parallel_model.parameters(), lr=args.lr,
                             weight_decay=args.wd)
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

    criterion = CombinedLoss(1, 0.1, args.batch_size)
    for epoch in range(current_epoch, args.max_epochs):
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train deep cluster')

    parser.add_argument('--save_path', dest='save_path', default='results',
                        help='path where models are saved', type=str)
    parser.add_argument('--data_path', dest='data_path', required=True,
                        help='path where models are saved', type=str)
    parser.add_argument('--logs_path', dest='logs_path', required=True,
                        help='path where models are saved', type=str)
    parser.add_argument('--resume', dest='resume', default=None,
                        help='load from path', type=str)
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rates')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')

    parser.add_argument('--num_workers', type=int, default=0, help='num workers')

    parser.add_argument('--max_epochs', type=int, default=100, help='max epochs')
    args = parser.parse_args()
    main(args)
