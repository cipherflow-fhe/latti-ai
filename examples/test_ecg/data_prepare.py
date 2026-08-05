
import json
from pathlib import Path

import numpy as np
import wfdb
from tqdm import tqdm
from sklearn.model_selection import train_test_split

"""
ECG Data Preparation and Preprocessing Module.

This module processes MIT-BIH Arrhythmia Database records to extract heartbeats,
convert them to 2D representations, and split the dataset into training, validation,
and test sets. It supports three variants: original data, undersampled (1:1 balanced),
and oversampled (1:1 balanced) training sets for handling class imbalance.
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
from tqdm import tqdm


EXAMPLE_DIR = Path(__file__).resolve().parent
BEAT_LENGTH = 256
IMAGE_SIZE = 16
LEAD_INDEX = 0
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
SEED = 42

ALL_RECORDS = [
    '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
    '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
    '213', '214', '215', '219', '220', '221', '222', '223', '228', '230',
    '231', '232', '233', '234',
]

NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}
ABNORMAL_SYMBOLS = {'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q'}

CLASS_TO_ID = {'normal': 0, 'abnormal': 1}
ID_TO_CLASS = {0: 'normal', 1: 'abnormal'}


def symbol_to_binary_label(sym: str):
    if sym in NORMAL_SYMBOLS:
        return CLASS_TO_ID['normal']
    if sym in ABNORMAL_SYMBOLS:
        return CLASS_TO_ID['abnormal']
    return None


def zscore_normalize(x: np.ndarray):
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        std = 1.0
    return (x - mean) / std


def beat_to_2d(beat_1d: np.ndarray, image_size: int = 16):
    beat_2d = beat_1d.reshape(image_size, image_size)
    return beat_2d[np.newaxis, :, :]


def extract_beats_from_record(record_path: Path, beat_length: int, image_size: int, lead_index: int):
    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), 'atr')

    signal = record.p_signal[:, lead_index].astype(np.float32)
    samples = annotation.sample
    symbols = annotation.symbol

    half_left = 99
    half_right = beat_length - half_left - 1

    x_items = []
    y_items = []

    for r_peak, sym in zip(samples, symbols):
        label = symbol_to_binary_label(sym)
        if label is None:
            continue

        start = r_peak - half_left
        end = r_peak + half_right + 1

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]
        if len(beat) != beat_length:
            continue

        beat = zscore_normalize(beat)
        beat = beat_to_2d(beat, image_size).astype(np.float32)

        x_items.append(beat)
        y_items.append(label)

    if len(x_items) == 0:
        return (
            np.empty((0, 1, image_size, image_size), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    x_all = np.stack(x_items, axis=0).astype(np.float32)
    y_all = np.array(y_items, dtype=np.int64)
    return x_all, y_all


def process_all_records(data_dir: Path, records: List[str], beat_length: int, image_size: int, lead_index: int):
    x_all = []
    y_all = []

    for rec in tqdm(records, desc='Processing records'):
        rec_path = data_dir / rec
        x_rec, y_rec = extract_beats_from_record(rec_path, beat_length, image_size, lead_index)
        if len(x_rec) > 0:
            x_all.append(x_rec)
            y_all.append(y_rec)

    if len(x_all) == 0:
        raise RuntimeError(f'No samples were extracted from data_dir={data_dir}')

    x_all = np.concatenate(x_all, axis=0).astype(np.float32)
    y_all = np.concatenate(y_all, axis=0).astype(np.int64)
    return x_all, y_all


def print_distribution(name: str, y: np.ndarray):
    total = len(y)
    print(f'\n{name}')
    for cls_id in sorted(np.unique(y)):
        count = int(np.sum(y == cls_id))
        ratio = count / total if total > 0 else 0.0
        print(f'class={cls_id} ({ID_TO_CLASS[int(cls_id)]}) count={count} ratio={ratio:.6f}')
    print(f'total={total}')


def undersample_train(x_train: np.ndarray, y_train: np.ndarray, seed: int, target_per_class: Optional[int] = None):
    rng = np.random.default_rng(seed)

    idx_normal = np.where(y_train == 0)[0]
    idx_abnormal = np.where(y_train == 1)[0]
    n_normal = len(idx_normal)
    n_abnormal = len(idx_abnormal)

    target = min(n_normal, n_abnormal) if target_per_class is None else int(target_per_class)
    if target > n_normal or target > n_abnormal:
        raise ValueError(
            f'Undersample target exceeds class counts: target={target}, '
            f'normal={n_normal}, abnormal={n_abnormal}'
        )

    sel_normal = rng.choice(idx_normal, size=target, replace=False)
    sel_abnormal = rng.choice(idx_abnormal, size=target, replace=False)
    sel = np.concatenate([sel_normal, sel_abnormal])
    rng.shuffle(sel)
    return x_train[sel], y_train[sel]


def oversample_train(x_train: np.ndarray, y_train: np.ndarray, seed: int, target_per_class: Optional[int] = None):
    rng = np.random.default_rng(seed)

    idx_normal = np.where(y_train == 0)[0]
    idx_abnormal = np.where(y_train == 1)[0]
    target = max(len(idx_normal), len(idx_abnormal)) if target_per_class is None else int(target_per_class)

    def sample_to_target(indices, target_num):
        n = len(indices)
        if n == target_num:
            return indices
        if n > target_num:
            return rng.choice(indices, size=target_num, replace=False)
        extra = rng.choice(indices, size=target_num - n, replace=True)
        return np.concatenate([indices, extra])

    sel_normal = sample_to_target(idx_normal, target)
    sel_abnormal = sample_to_target(idx_abnormal, target)
    sel = np.concatenate([sel_normal, sel_abnormal])
    rng.shuffle(sel)
    return x_train[sel], y_train[sel]


def save_processed(save_dir: Path, x_train, y_train, x_val, y_val, x_test, y_test, info: dict):
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / 'X_train.npy', x_train)
    np.save(save_dir / 'y_train.npy', y_train)
    np.save(save_dir / 'X_val.npy', x_val)
    np.save(save_dir / 'y_val.npy', y_val)
    np.save(save_dir / 'X_test.npy', x_test)
    np.save(save_dir / 'y_test.npy', y_test)

    (save_dir / 'split_info.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(f'\nSaved: {save_dir}')
    print(f'X_train={x_train.shape}, y_train={y_train.shape}')
    print(f'X_val={x_val.shape}, y_val={y_val.shape}')
    print(f'X_test={x_test.shape}, y_test={y_test.shape}')

    print_distribution(f'{save_dir.name} train', y_train)
    print_distribution(f'{save_dir.name} val', y_val)
    print_distribution(f'{save_dir.name} test', y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='MIT-BIH Arrhythmia Database directory')
    parser.add_argument('--output-root', type=str, default=str(EXAMPLE_DIR / 'data'))
    parser.add_argument('--beat-length', type=int, default=BEAT_LENGTH)
    parser.add_argument('--image-size', type=int, default=IMAGE_SIZE)
    parser.add_argument('--lead-index', type=int, default=LEAD_INDEX)
    parser.add_argument('--train-size', type=float, default=TRAIN_SIZE)
    parser.add_argument('--val-size', type=float, default=VAL_SIZE)
    parser.add_argument('--test-size', type=float, default=TEST_SIZE)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--under-target-per-class', type=int, default=None)
    parser.add_argument('--over-target-per-class', type=int, default=None)
    parser.add_argument('--records', nargs='*', default=None)
    args = parser.parse_args()

    if abs(args.train_size + args.val_size + args.test_size - 1.0) > 1e-8:
        raise ValueError('--train-size + --val-size + --test-size must equal 1.0')

    if args.beat_length != args.image_size * args.image_size:
        raise ValueError('--beat-length must equal --image-size * --image-size')

    data_dir = Path(args.data_dir).expanduser()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    processed_original_dir = output_root / 'processed_original'
    processed_under_dir = output_root / 'processed_under_1to1'
    processed_over_dir = output_root / 'processed_over_1to1'

    records = args.records or ALL_RECORDS
    x_all, y_all = process_all_records(
        data_dir=data_dir,
        records=records,
        beat_length=args.beat_length,
        image_size=args.image_size,
        lead_index=args.lead_index,
    )

    print_distribution('all samples', y_all)

    val_relative = args.val_size / (args.val_size + args.test_size)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x_all,
        y_all,
        test_size=(1.0 - args.train_size),
        random_state=args.seed,
        stratify=y_all,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=(1.0 - val_relative),
        random_state=args.seed,
        stratify=y_temp,
    )

    base_info = {
        'data_dir': str(data_dir),
        'train_size': args.train_size,
        'val_size': args.val_size,
        'test_size': args.test_size,
        'seed': args.seed,
        'beat_length': args.beat_length,
        'image_size': args.image_size,
        'lead_index': args.lead_index,
        'records': records,
    }

    save_processed(
        processed_original_dir,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        {**base_info, 'variant': 'original', 'output_dir': str(processed_original_dir), 'train_balance': 'none'},
    )

    x_train_under, y_train_under = undersample_train(
        x_train,
        y_train,
        seed=args.seed,
        target_per_class=args.under_target_per_class,
    )

    save_processed(
        processed_under_dir,
        x_train_under,
        y_train_under,
        x_val,
        y_val,
        x_test,
        y_test,
        {
            **base_info,
            'variant': 'under_1to1',
            'output_dir': str(processed_under_dir),
            'train_balance': 'undersample_train_only',
            'under_target_per_class': args.under_target_per_class,
        },
    )

    x_train_over, y_train_over = oversample_train(
        x_train,
        y_train,
        seed=args.seed,
        target_per_class=args.over_target_per_class,
    )

    save_processed(
        processed_over_dir,
        x_train_over,
        y_train_over,
        x_val,
        y_val,
        x_test,
        y_test,
        {
            **base_info,
            'variant': 'over_1to1',
            'output_dir': str(processed_over_dir),
            'train_balance': 'oversample_train_only',
            'over_target_per_class': args.over_target_per_class,
        },
    )

    print('\nAll done')
    print(processed_original_dir)
    print(processed_under_dir)
    print(processed_over_dir)


if __name__ == '__main__':
    main()

