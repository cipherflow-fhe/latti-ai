import os
import sys
TEST_FACE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(TEST_FACE_DIR, '..', '..'))
sys.path.insert(0, TEST_FACE_DIR)
sys.path.insert(0, REPO_ROOT)
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import numpy as np


import argparse
from model.facenet_mobilenetv2 import FaceEmbeddingExportWrapper, FaceNetMobileNetV2
from training.nn_tools import replace_activation_with_poly, replace_maxpool_with_avgpool
from training.nn_tools.activations import PolyAct, RangeNormPoly2d
from training.nn_tools.replace import count_activations



def resize_image(image, size, letterbox_image=True):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image
    return image.resize((w, h), Image.BICUBIC)


class LFWDataset(Dataset):
    def __init__(self, lfw_dir, pairs_path, image_size):
        self.lfw_dir = lfw_dir
        self.pairs_path = pairs_path
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.validation_images = self.get_lfw_paths()

    def read_lfw_pairs(self):
        pairs = []
        with open(self.pairs_path, 'r') as f:
            for line in f.readlines()[1:]:
                pairs.append(line.strip().split())
        return pairs

    def get_lfw_paths(self):
        pairs = self.read_lfw_pairs()
        path_list = []
        skipped = 0
        for pair in pairs:
            if len(pair) == 3:
                path0 = os.path.join(
                    self.lfw_dir, pair[0], f'{pair[0]}_{int(pair[1]):04d}.jpg')
                path1 = os.path.join(
                    self.lfw_dir, pair[0], f'{pair[0]}_{int(pair[2]):04d}.jpg')
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(
                    self.lfw_dir, pair[0], f'{pair[0]}_{int(pair[1]):04d}.jpg')
                path1 = os.path.join(
                    self.lfw_dir, pair[2], f'{pair[2]}_{int(pair[3]):04d}.jpg')
                issame = False
            else:
                skipped += 1
                continue
            if os.path.exists(path0) and os.path.exists(path1):
                path_list.append((path0, path1, issame))
            else:
                skipped += 1
        if skipped > 0:
            print('Skipped %d image pairs' % skipped)
        return path_list

    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        image = resize_image(
            image, [self.image_size[1], self.image_size[0]], letterbox_image=True)
        return self.transform(image)

    def __getitem__(self, index):
        path_1, path_2, issame = self.validation_images[index]
        return self.load_image(path_1), self.load_image(path_2), issame

    def __len__(self):
        return len(self.validation_images)


def k_fold_split(n_items, n_splits):
    indices = np.arange(n_items)
    folds = np.array_split(indices, n_splits)
    for fold in folds:
        train_set = np.setdiff1d(indices, fold, assume_unique=True)
        yield train_set, fold


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(
        predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, distances, labels, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros(nrof_folds)
    best_thresholds = np.zeros(nrof_folds)

    for fold_idx, (train_set, test_set) in enumerate(k_fold_split(nrof_pairs, nrof_folds)):
        acc_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, distances[train_set], labels[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold,
                distances[test_set],
                labels[test_set],
            )
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index],
            distances[test_set],
            labels[test_set],
        )
    return np.mean(tprs, 0), np.mean(fprs, 0), accuracy, best_thresholds


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(
        predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0, 0
    return float(true_accept) / float(n_same), float(false_accept) / float(n_diff)


def calculate_val(thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    for fold_idx, (train_set, test_set) in enumerate(k_fold_split(nrof_pairs, nrof_folds)):
        far_train = np.zeros(len(thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, distances[train_set], labels[train_set])
        if np.max(far_train) >= far_target:
            unique_far, unique_indices = np.unique(
                far_train, return_index=True)
            unique_thresholds = thresholds[unique_indices]
            if len(unique_far) > 1:
                threshold = np.interp(
                    far_target, unique_far, unique_thresholds)
            else:
                threshold = unique_thresholds[0]
        else:
            threshold = 0.0
        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, distances[test_set], labels[test_set])
    return np.mean(val), np.std(val), np.mean(far)


def evaluate(distances, labels, threshold_max=2.0, nrof_folds=10):
    thresholds = np.arange(0, threshold_max, 0.001)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(
        thresholds, distances, labels, nrof_folds=nrof_folds)
    val, val_std, far = calculate_val(
        thresholds, distances, labels, 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far, best_thresholds


def load_checkpoint_state_dict(path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get(
        'state_dict', checkpoint.get('net', checkpoint))
    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def infer_num_classes(state_dict):
    for key, value in state_dict.items():
        if key.endswith('classifier.weight'):
            return value.shape[0]
    raise ValueError(
        'Cannot infer num_classes from checkpoint classifier.weight')


def initialize_poly_buffers(model, input_shape):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
        model(dummy, mode='predict')
    model.train(was_training)


def build_poly_model(args, num_classes):
    model = FaceNetMobileNetV2(
        num_classes=num_classes,
        embedding_size=args.embedding_size,
        width_mult=args.width_mult,
        dropout_keep_prob=args.dropout_keep_prob,
        use_embedding_bn=not args.no_embedding_bn,
        normalize_embedding=False,
    )
    poly_cls = RangeNormPoly2d if args.poly_module == 'RangeNormPoly2d' else PolyAct
    n_maxpool = count_activations(model, nn.MaxPool2d)
    replace_maxpool_with_avgpool(model)
    n_relu = count_activations(model, nn.ReLU)
    replace_activation_with_poly(
        model,
        old_cls=nn.ReLU,
        new_module_factory=poly_cls,
        upper_bound=args.upper_bound,
        degree=args.degree,
    )
    print('Poly convert: MaxPool2d=%d, ReLU=%d, poly=%s, upper_bound=%s, degree=%s' % (
        n_maxpool,
        n_relu,
        poly_cls.__name__,
        args.upper_bound,
        args.degree,
    ))
    return model


@torch.no_grad()
def run_lfw(model, loader, device):
    labels, distances = [], []
    for data_a, data_p, label in tqdm(loader, desc='LFW'):
        data_a = data_a.to(device)
        data_p = data_p.to(device)
        out_a = F.normalize(model(data_a), p=2, dim=1)
        out_p = F.normalize(model(data_p), p=2, dim=1)
        # out_a = data_a
        # out_p = data_p
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
        distances.append(dists.cpu().numpy())
        labels.append(label.cpu().numpy())
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    return evaluate(distances, labels)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate output_poly MobileNetV2 FaceNet on LFW')
    parser.add_argument(
        '--checkpoint', default='examples/test_face/output_poly/best.pth')
    parser.add_argument('--lfw-dir', default='/home/zhongy/lfw-aligned/')
    parser.add_argument(
        '--lfw-pairs-path', default='/home/zhongy/facenet-pytorch-change/model_data/lfw_pair.txt')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0, help='-1 for CPU')
    parser.add_argument('--input-shape', type=int, nargs=3,
                        default=[256, 256, 3], metavar=('H', 'W', 'C'))
    parser.add_argument('--width-mult', type=float, default=1.0)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--dropout-keep-prob', type=float, default=0.5)
    parser.add_argument('--no-embedding-bn', action='store_true')
    parser.add_argument('--poly-module', default='RangeNormPoly2d',
                        choices=['RangeNormPoly2d', 'PolyAct'])
    parser.add_argument('--upper-bound', type=float, default=3.0)
    parser.add_argument('--degree', type=int, default=4, choices=[2, 4, 8])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 and torch.cuda.is_available(
    ) else torch.device('cpu')
    state_dict = load_checkpoint_state_dict(args.checkpoint)
    model = build_poly_model(args, infer_num_classes(state_dict))
    initialize_poly_buffers(model, args.input_shape)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print('Missing keys: %d' % len(missing))
    if unexpected:
        print('Unexpected keys: %d' % len(unexpected))
    model = FaceEmbeddingExportWrapper(model).eval().to(device)

    dataset = LFWDataset(args.lfw_dir, args.lfw_pairs_path, args.input_shape)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    _, _, accuracy, val, val_std, far, best_thresholds = run_lfw(
        model, loader, device)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % np.mean(best_thresholds))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))


if __name__ == '__main__':
    main()
