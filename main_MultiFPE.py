import argparse
import os
import pdb
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.model import FacialPalsyClassificationModel
import matplotlib
from sklearn.metrics import confusion_matrix
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader.dataset_SCB_palsy_allframes import train_data_loader, test_data_loader
import random


def set_random_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


set_random_seed()

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=3, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=4, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--data_set', type=str, default='MEEI_1')

args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")

dataset = args.data_set.split('_')[0]
data_set_number = args.data_set.split('_')[-1]
data_set_number = (f"{data_set_number}")

log_txt_path = f'log/model1/{dataset}/{data_set_number}.txt'
log_curve_path = f'log/model1/{dataset}/{data_set_number}.png'
checkpoint_path = f'checkpoint/model1/{dataset}/{data_set_number}.pth'
best_checkpoint_path = f'best_checkpoint/model1/{dataset}/{data_set_number}.pth'
best_confusion_matrix_path = f'log/model1/{dataset}/{data_set_number}_best_confusion_matrix.png'


def main():
    print('=' * 40)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 40)

    with open(log_txt_path, 'a', encoding='utf-8') as f:
        f.write('=' * 40 + '\n')
        f.write('seed=no' + '\n')
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')
        f.write('=' * 40 + '\n')

    best_acc = 0
    best_conf_matrix = None
    recorder = RecorderMeter(args.epochs)
    print('The training time: ' + now.strftime("%m-%d %H:%M"))
    print('The training set: set ' + str(args.data_set))
    with open(log_txt_path, 'a', encoding='utf-8') as f:
        f.write('The training time: ' + now.strftime("%m-%d %H:%M") + '\n')
        f.write('The training set: set ' + str(args.data_set) + '\n')

    model = FacialPalsyClassificationModel(output_dim=3).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.7)

    with open(log_txt_path, 'a', encoding='utf-8') as f:
        f.write('lr_scheduler:' + '\n')
        f.write(f'Step size:' + str(scheduler.milestones) + '\n')
        f.write(f'Gamma size:' + str(scheduler.gamma) + '\n')

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']

            if recorder.epoch_losses.shape[0] < args.epochs:
                new_size = args.epochs
                recorder.epoch_losses = np.resize(recorder.epoch_losses, (new_size, 2))
                recorder.epoch_accuracy = np.resize(recorder.epoch_accuracy, (new_size, 2))

            best_acc = best_acc.cuda()
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            current_learning_rate = 0.01
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_learning_rate
            print(f"Manually set learning rate to: {current_learning_rate}")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0], gamma=0.1)

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    train_data = train_data_loader(data_set=args.data_set)
    test_data = test_data_loader(data_set=args.data_set)



    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers,
                                             pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a', encoding='utf-8') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args)
        val_acc, val_los, conf_matrix = validate(val_loader, model, criterion, args)

        scheduler.step()

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_conf_matrix = conf_matrix

        if best_conf_matrix is not None:
            plot_confusion_matrix(best_conf_matrix, best_confusion_matrix_path)

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        save_checkpoint({'epoch': epoch + 1,
                         'model': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best)

        epoch_time = time.time() - start_time

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.1f}s'.format(epoch_time))
        with open(log_txt_path, 'a', encoding='utf-8') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write(f'An epoch time: {epoch_time:.1f}s\n')


def train(train_loader, model, criterion, optimizer, epoch, args):

    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    all_preds = []
    all_targets = []

    for i, (images, static_image, landmarks_tensor, left_iris_tensor, right_iris_tensor, static_landmarks_tensor, static_left_iris_tensor, static_right_iris_tensor, target) in enumerate(train_loader):

        images = images.cuda()
        static_image = static_image.cuda()
        landmarks_tensor = landmarks_tensor.cuda()
        left_iris_tensor = left_iris_tensor.cuda()
        right_iris_tensor = right_iris_tensor.cuda()
        static_landmarks_tensor = static_landmarks_tensor.cuda()
        static_left_iris_tensor = static_left_iris_tensor.cuda()
        static_right_iris_tensor = static_right_iris_tensor.cuda()
        target = target.cuda()

        output = model(images, static_image, landmarks_tensor, left_iris_tensor, right_iris_tensor, static_landmarks_tensor, static_left_iris_tensor, static_right_iris_tensor)

        loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        _, preds = torch.max(output, 1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(target.detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            progress.display(i)

    class_correct = np.zeros(3)
    class_total = np.zeros(3)
    for t, p in zip(all_targets, all_preds):
        if t == p:
            class_correct[t] += 1
        class_total[t] += 1

    with open(log_txt_path, 'a', encoding='utf-8') as f:
        conf_matrix = confusion_matrix(all_targets, all_preds)
        conf_matrix_normalized = confusion_matrix(all_targets, all_preds, normalize='true')

        precisions, recalls, f1_scores, class_accs = [], [], [], []

        for cls in range(3):
            TP = conf_matrix[cls, cls]
            FP = conf_matrix[:, cls].sum() - TP
            FN = conf_matrix[cls, :].sum() - TP

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            class_acc = 100 * class_correct[cls] / class_total[cls]

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            class_accs.append(class_acc)

            line = f'Class {cls} Accuracy: {class_acc:.2f}%;F1score: {f1 * 100:.2f}%;Precision: {precision * 100:.2f}%;Recall: {recall * 100:.2f}%'
            print(line)
            f.write(line + '\n')

        print('Train Confusion Matrix:')
        print(conf_matrix_normalized)
        f.write('Train Confusion Matrix:\n')
        f.write(np.array2string(conf_matrix_normalized) + '\n')

        total_accuracy = 100 * np.diag(conf_matrix).sum() / conf_matrix.sum()
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1_scores)

        total_line = f'Train Accuracy: {total_accuracy:.2f}%;F1score: {macro_f1 * 100:.2f}%;Precision: {macro_precision * 100:.2f}%;Recall: {macro_recall * 100:.2f}%'
        print(total_line)
        f.write(total_line + '\n')

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (images, static_image, landmarks_tensor, left_iris_tensor, right_iris_tensor, static_landmarks_tensor, static_left_iris_tensor, static_right_iris_tensor, target) in enumerate(val_loader):
            images = images.cuda()
            static_image = static_image.cuda()
            landmarks_tensor = landmarks_tensor.cuda()
            left_iris_tensor = left_iris_tensor.cuda()
            right_iris_tensor = right_iris_tensor.cuda()
            static_landmarks_tensor = static_landmarks_tensor.cuda()
            static_left_iris_tensor = static_left_iris_tensor.cuda()
            static_right_iris_tensor = static_right_iris_tensor.cuda()
            target = target.cuda()

            output = model(images, static_image, landmarks_tensor, left_iris_tensor, right_iris_tensor, static_landmarks_tensor, static_left_iris_tensor, static_right_iris_tensor)

            loss = criterion(output, target)

            acc1, _ = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0 or i == len(val_loader) - 1:
                progress.display(i)

        class_correct = np.zeros(3)
        class_total = np.zeros(3)

        for t, p in zip(all_targets, all_preds):
            if t == p:
                class_correct[t] += 1
            class_total[t] += 1

        with open(log_txt_path, 'a', encoding='utf-8') as f:
            conf_matrix = confusion_matrix(all_targets, all_preds)
            conf_matrix_normalized = confusion_matrix(all_targets, all_preds, normalize='true')

            precisions, recalls, f1_scores, class_accs = [], [], [], []

            for cls in range(3):
                TP = conf_matrix[cls, cls]
                FP = conf_matrix[:, cls].sum() - TP
                FN = conf_matrix[cls, :].sum() - TP

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                class_acc = 100 * class_correct[cls] / class_total[cls]

                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                class_accs.append(class_acc)

                line = f'Class {cls} Accuracy: {class_acc:.2f}%;F1score: {f1 * 100:.2f}%;Precision: {precision * 100:.2f}%;Recall: {recall * 100:.2f}%'
                print(line)
                f.write(line + '\n')

            print('Validation Confusion Matrix:')
            print(conf_matrix_normalized)
            f.write('Validation Confusion Matrix:\n')
            f.write(np.array2string(conf_matrix_normalized) + '\n')

            total_accuracy = 100 * np.diag(conf_matrix).sum() / conf_matrix.sum()
            macro_precision = np.mean(precisions)
            macro_recall = np.mean(recalls)
            macro_f1 = np.mean(f1_scores)

            total_line = f'Validation Accuracy: {total_accuracy:.2f}%;F1score: {macro_f1 * 100:.2f}%;Precision: {macro_precision * 100:.2f}%;Recall: {macro_recall * 100:.2f}%'
            print(total_line)
            f.write(total_line + '\n')

    return top1.avg, losses.avg, conf_matrix_normalized


def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(log_txt_path, 'a', encoding='utf-8') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 4000, 2000
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.current_epoch)])  # epochs
        y_axis = np.zeros(self.current_epoch)

        plt.xlim(0, self.current_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.current_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:self.current_epoch, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:self.current_epoch, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:self.current_epoch, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:self.current_epoch, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def plot_confusion_matrix(conf_matrix, save_path):
    plt.figure(figsize=(10, 7))
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(conf_matrix.shape[0])
    plt.xticks(tick_marks, range(conf_matrix.shape[0]))
    plt.yticks(tick_marks, range(conf_matrix.shape[0]))

    # Loop over data dimensions and create text annotations.
    threshold = conf_matrix.max() / 2.
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, f"{conf_matrix[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > threshold else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    main()
