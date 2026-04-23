import sys
import time
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ========== Linux 模块导入适配（核心！解决找不到模块报错）==========
FILE_ROOT = Path(__file__).resolve().parent
if str(FILE_ROOT) not in sys.path:
    sys.path.insert(0, str(FILE_ROOT))

from dataset import ECGNpyDataset
from model import build_model
from losses import build_loss
from augment import ECGAugment
from utils_exp import (
    set_seed, ensure_dir, save_json, save_checkpoint,
    compute_class_weights, EarlyStopping, load_checkpoint
)

# =========================================================
# 适配你的Linux路径，无需改盘符，上传后直接可用
# =========================================================
# 数据路径：my_ecg001目录下的processed_over_1to1
PROCESSED_DIR = os.getenv('ECG_PROCESSED_DIR', str(FILE_ROOT / 'processed_over_1to1'))
# 输出路径：统一存在my_ecg001/runs下，和旧项目隔离
RUN_ROOT = os.getenv('ECG_RUN_ROOT', str(FILE_ROOT / 'runs/exp_over009'))

MODEL_NAME = 'two_conv'  # 你的模型：tiny_cnn / tiny_cnn8 / two_conv / mlp_head
NUM_CLASSES = 2
INPUT_SHAPE = [1, 16, 16]  # 固定和你的数据匹配

EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001
NUM_WORKERS = 4  # Linux多核加速，无GPU可改成2
TORCH_NUM_THREADS = 4
SEED = 42
LOSS_NAME = 'weighted_ce'  # ce / weighted_ce / focal
FOCAL_GAMMA = 2.0
USE_AUGMENT = True
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    pbar = tqdm(loader, desc='Train', leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_num += x.size(0)

        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / total_num, total_correct / total_num


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    pbar = tqdm(loader, desc='Eval ', leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        # ========== 修复致命bug：之前漏了这行，会报preds未定义 ==========
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_num += x.size(0)

        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / total_num, total_correct / total_num


def main():
    set_seed(SEED)
    torch.set_num_threads(TORCH_NUM_THREADS)

    # Linux自动检测GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] Using {device} (GPU count: {torch.cuda.device_count()})')

    # 路径标准化
    processed_dir = Path(PROCESSED_DIR).resolve()
    run_root = Path(RUN_ROOT).resolve()
    model_dir = run_root / 'model'
    log_dir = run_root / 'logs'
    export_dir = run_root / 'task/server'  # 提前建好poly导出目录
    ensure_dir(model_dir)
    ensure_dir(log_dir)
    ensure_dir(export_dir)

    # 校验数据路径
    if not (processed_dir / 'X_train.npy').exists():
        raise FileNotFoundError(f'数据文件不存在，请检查路径：{processed_dir}')
    print(f'[Data] 加载数据路径：{processed_dir}')

    # 数据加载
    train_transform = ECGAugment(enable=USE_AUGMENT, seed=SEED) if USE_AUGMENT else None
    train_set = ECGNpyDataset(
        x_path=str(processed_dir / 'X_train.npy'),
        y_path=str(processed_dir / 'y_train.npy'),
        transform=train_transform
    )
    val_set = ECGNpyDataset(
        x_path=str(processed_dir / 'X_val.npy'),
        y_path=str(processed_dir / 'y_val.npy'),
        transform=None
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )

    # 模型初始化
    model = build_model(num_classes=NUM_CLASSES, model_name=MODEL_NAME).to(device)
    # 多GPU适配
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'[Model] 启用多GPU并行：{torch.cuda.device_count()} 张GPU')

    # 损失函数&优化器
    class_weights = compute_class_weights(
        y_path=str(processed_dir / 'y_train.npy'),
        num_classes=NUM_CLASSES
    ).to(device)
    criterion = build_loss(
        loss_name=LOSS_NAME,
        class_weights=class_weights if LOSS_NAME in ['weighted_ce', 'focal'] else None,
        focal_gamma=FOCAL_GAMMA
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early_stopper = EarlyStopping(mode='max', patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA)

    # 训练初始化
    best_val_acc = -1.0
    best_state = None
    history = []
    config = {
        'processed_dir': str(processed_dir),
        'run_root': str(run_root),
        'model_name': MODEL_NAME,
        'num_classes': NUM_CLASSES,
        'input_shape': INPUT_SHAPE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'num_workers': NUM_WORKERS,
        'torch_num_threads': TORCH_NUM_THREADS,
        'seed': SEED,
        'loss_name': LOSS_NAME,
        'focal_gamma': FOCAL_GAMMA,
        'use_augment': USE_AUGMENT,
        'device': str(device)
    }
    save_json(config, log_dir / 'train_config.json')

    # 训练循环
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== Epoch {epoch}/{EPOCHS} =====')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # 记录日志
        row = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'lr': float(optimizer.param_groups[0]['lr']),
        }
        history.append(row)
        print(f'Epoch {epoch}:')
        print(f'  train_loss={train_loss:.6f}, train_acc={train_acc:.6f}')
        print(f'  val_loss  ={val_loss:.6f}, val_acc  ={val_acc:.6f}')

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                'epoch': epoch,
                'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }
            save_checkpoint(best_state, model_dir / 'train_baseline.pth')
            print(f'[Saved] 最优模型已更新，val_acc={best_val_acc:.6f}')

        # 早停
        if USE_EARLY_STOPPING:
            stop = early_stopper.step(val_acc)
            if stop:
                print(f'[EarlyStopping] 早停触发，epoch={epoch}，最优val_acc={early_stopper.best:.6f}')
                break

    # 训练结束
    elapsed = time.time() - start_time
    print(f'\n训练完成！耗时：{elapsed / 60:.2f} min，最优验证准确率：{best_val_acc:.6f}')
    save_json(history, log_dir / 'train_history.json')
    save_json({
        'best_val_acc': float(best_val_acc),
        'elapsed_minutes': float(elapsed / 60.0),
    }, log_dir / 'best_metrics.json')


if __name__ == '__main__':
    main()