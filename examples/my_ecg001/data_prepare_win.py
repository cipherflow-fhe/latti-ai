import os
import json
from pathlib import Path

import numpy as np
import wfdb
from tqdm import tqdm
from sklearn.model_selection import train_test_split


DATA_DIR = r'D:\Learn\python\PythonProject\MachineLearn\Latti-Ai\text2\mit-bih-arrhythmia-database-1.0.0'
OUTPUT_ROOT = r'D:\Learn\python\PythonProject\MachineLearn\Latti-Ai\text2\my_ecg0'

BEAT_LENGTH = 256
IMAGE_SIZE = 16
LEAD_INDEX = 0
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
SEED = 42

UNDER_TARGET_PER_CLASS = None
OVER_TARGET_PER_CLASS = None


ALL_RECORDS = [
    '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
    '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
    '213', '214', '215', '219', '220', '221', '222', '223', '228', '230',
    '231', '232', '233', '234'
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


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        std = 1.0
    return (x - mean) / std


def beat_to_2d(beat_1d: np.ndarray, image_size: int = 16) -> np.ndarray:
    beat_2d = beat_1d.reshape(image_size, image_size)
    return beat_2d[np.newaxis, :, :]


def extract_beats_from_record(record_path: str, beat_length: int, image_size: int, lead_index: int):
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, lead_index].astype(np.float32)
    samples = annotation.sample
    symbols = annotation.symbol

    half_left = 99
    half_right = beat_length - half_left - 1

    X = []
    y = []

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

        X.append(beat)
        y.append(label)

    if len(X) == 0:
        return np.empty((0, 1, image_size, image_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


def process_all_records(data_dir: str, records: list[str], beat_length: int, image_size: int, lead_index: int):
    X_all = []
    y_all = []

    for rec in tqdm(records, desc='Processing records'):
        rec_path = os.path.join(data_dir, rec)
        x_rec, y_rec = extract_beats_from_record(rec_path, beat_length, image_size, lead_index)
        if len(x_rec) > 0:
            X_all.append(x_rec)
            y_all.append(y_rec)

    if len(X_all) == 0:
        raise RuntimeError('没有提取到任何样本，请检查数据路径')

    X_all = np.concatenate(X_all, axis=0).astype(np.float32)
    y_all = np.concatenate(y_all, axis=0).astype(np.int64)
    return X_all, y_all


def print_distribution(name: str, y: np.ndarray):
    total = len(y)
    print(f'\n{name}')
    for cls_id in sorted(np.unique(y)):
        count = int(np.sum(y == cls_id))
        ratio = count / total if total > 0 else 0.0
        print(f'class={cls_id} ({ID_TO_CLASS[int(cls_id)]}) count={count} ratio={ratio:.6f}')
    print(f'total={total}')


def undersample_train(X_train: np.ndarray, y_train: np.ndarray, seed: int, target_per_class=None):
    rng = np.random.default_rng(seed)

    idx_normal = np.where(y_train == 0)[0]
    idx_abnormal = np.where(y_train == 1)[0]

    n_normal = len(idx_normal)
    n_abnormal = len(idx_abnormal)

    if target_per_class is None:
        target = min(n_normal, n_abnormal)
    else:
        target = int(target_per_class)

    if target > n_normal or target > n_abnormal:
        raise ValueError(f'欠采样目标超过当前训练集某类数量: target={target}, normal={n_normal}, abnormal={n_abnormal}')

    sel_normal = rng.choice(idx_normal, size=target, replace=False)
    sel_abnormal = rng.choice(idx_abnormal, size=target, replace=False)

    sel = np.concatenate([sel_normal, sel_abnormal])
    rng.shuffle(sel)

    return X_train[sel], y_train[sel]


def oversample_train(X_train: np.ndarray, y_train: np.ndarray, seed: int, target_per_class=None):
    rng = np.random.default_rng(seed)

    idx_normal = np.where(y_train == 0)[0]
    idx_abnormal = np.where(y_train == 1)[0]

    n_normal = len(idx_normal)
    n_abnormal = len(idx_abnormal)

    if target_per_class is None:
        target = max(n_normal, n_abnormal)
    else:
        target = int(target_per_class)

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

    return X_train[sel], y_train[sel]


def save_processed(save_dir: Path, X_train, y_train, X_val, y_val, X_test, y_test, info: dict):
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / 'X_train.npy', X_train)
    np.save(save_dir / 'y_train.npy', y_train)
    np.save(save_dir / 'X_val.npy', X_val)
    np.save(save_dir / 'y_val.npy', y_val)
    np.save(save_dir / 'X_test.npy', X_test)
    np.save(save_dir / 'y_test.npy', y_test)

    with open(save_dir / 'split_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f'\n已保存: {save_dir}')
    print(f'X_train={X_train.shape}, y_train={y_train.shape}')
    print(f'X_val={X_val.shape}, y_val={y_val.shape}')
    print(f'X_test={X_test.shape}, y_test={y_test.shape}')

    print_distribution(f'{save_dir.name} train', y_train)
    print_distribution(f'{save_dir.name} val', y_val)
    print_distribution(f'{save_dir.name} test', y_test)


def main():
    if abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) > 1e-8:
        raise ValueError('TRAIN_SIZE + VAL_SIZE + TEST_SIZE 必须等于 1.0')

    if BEAT_LENGTH != IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError('BEAT_LENGTH 必须等于 IMAGE_SIZE * IMAGE_SIZE')

    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    processed_original_dir = output_root / 'processed_original'
    processed_under_dir = output_root / 'processed_under_1to1'
    processed_over_dir = output_root / 'processed_over_1to1'

    X_all, y_all = process_all_records(
        data_dir=DATA_DIR,
        records=ALL_RECORDS,
        beat_length=BEAT_LENGTH,
        image_size=IMAGE_SIZE,
        lead_index=LEAD_INDEX
    )

    print_distribution('all samples', y_all)

    val_relative = VAL_SIZE / (VAL_SIZE + TEST_SIZE)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all,
        y_all,
        test_size=(1.0 - TRAIN_SIZE),
        random_state=SEED,
        stratify=y_all
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_relative),
        random_state=SEED,
        stratify=y_temp
    )

    info_original = {
        'variant': 'original',
        'data_dir': DATA_DIR,
        'output_dir': str(processed_original_dir),
        'train_size': TRAIN_SIZE,
        'val_size': VAL_SIZE,
        'test_size': TEST_SIZE,
        'seed': SEED,
        'beat_length': BEAT_LENGTH,
        'image_size': IMAGE_SIZE,
        'lead_index': LEAD_INDEX,
        'train_balance': 'none'
    }

    save_processed(
        processed_original_dir,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        info_original
    )

    X_train_under, y_train_under = undersample_train(
        X_train, y_train,
        seed=SEED,
        target_per_class=UNDER_TARGET_PER_CLASS
    )

    info_under = {
        'variant': 'under_1to1',
        'data_dir': DATA_DIR,
        'output_dir': str(processed_under_dir),
        'train_size': TRAIN_SIZE,
        'val_size': VAL_SIZE,
        'test_size': TEST_SIZE,
        'seed': SEED,
        'beat_length': BEAT_LENGTH,
        'image_size': IMAGE_SIZE,
        'lead_index': LEAD_INDEX,
        'train_balance': 'undersample_train_only',
        'under_target_per_class': UNDER_TARGET_PER_CLASS
    }

    save_processed(
        processed_under_dir,
        X_train_under, y_train_under,
        X_val, y_val,
        X_test, y_test,
        info_under
    )

    X_train_over, y_train_over = oversample_train(
        X_train, y_train,
        seed=SEED,
        target_per_class=OVER_TARGET_PER_CLASS
    )

    info_over = {
        'variant': 'over_1to1',
        'data_dir': DATA_DIR,
        'output_dir': str(processed_over_dir),
        'train_size': TRAIN_SIZE,
        'val_size': VAL_SIZE,
        'test_size': TEST_SIZE,
        'seed': SEED,
        'beat_length': BEAT_LENGTH,
        'image_size': IMAGE_SIZE,
        'lead_index': LEAD_INDEX,
        'train_balance': 'oversample_train_only',
        'over_target_per_class': OVER_TARGET_PER_CLASS
    }

    save_processed(
        processed_over_dir,
        X_train_over, y_train_over,
        X_val, y_val,
        X_test, y_test,
        info_over
    )

    print('\n全部完成')
    print(processed_original_dir)
    print(processed_under_dir)
    print(processed_over_dir)


if __name__ == '__main__':
    main()