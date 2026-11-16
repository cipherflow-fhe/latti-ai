"""Verify final decrypted weights against an equally unrolled plaintext run."""
import argparse
import json
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=Path)
    parser.add_argument('--report', type=Path, default=None)
    parser.add_argument('--allow-failure', action='store_true', help='write diagnostics even if numeric threshold fails')
    args = parser.parse_args()
    data = json.loads((args.task / 'data.json').read_text())
    result = json.loads((args.task / 'result.json').read_text())
    manifest = json.loads((args.task / 'experiment.json').read_text())
    dag = json.loads((args.task / 'training_dag.json').read_text())
    resources_path = args.task / 'resource_metrics.json'
    resources = json.loads(resources_path.read_text()) if resources_path.exists() else {}
    w = np.array(result['weights']); ref = np.array(data['expected_weights'])
    assert w.shape == ref.shape, 'Unexpected output weight shape'
    assert np.all(np.isfinite(w)), 'Nonfinite CKKS output'
    error = float(np.max(np.abs(w-ref)))
    if error >= 1e-5 and not args.allow_failure:
        raise AssertionError(f'Weight error {error} exceeds 1e-5')
    xt = np.c_[np.ones(len(data['x_test'])), data['x_test']]
    yt = np.array(data['y_test'])
    def confusion(pred):
        return [[int(np.sum((yt == i) & (pred == j))) for j in (0, 1)] for i in (0, 1)]
    plain_pred = (xt @ ref >= 0)
    ckks_pred = (xt @ w >= 0)
    result.update(weight_max_absolute_error=error,
                  weight_rmse=float(np.sqrt(np.mean((w-ref)**2))),
                  plaintext_accuracy=float(np.mean((xt @ ref >= 0) == yt)),
                  ckks_accuracy=float(np.mean((xt @ w >= 0) == yt)),
                  steps=manifest['steps'], logit_ranges=data['logit_ranges'],
                  ct_multiplication_depth=dag['ct_multiplication_depth'],
                  bootstrapping_count=manifest['bootstrapping_count'])
    result['prediction_disagreement_count'] = int(np.sum(plain_pred != ckks_pred))
    result['prediction_agreement'] = float(np.mean(plain_pred == ckks_pred))
    result['plaintext_confusion_matrix'] = confusion(plain_pred)
    result['ckks_confusion_matrix'] = confusion(ckks_pred)
    result.update(resources)
    history = []
    for step in range(1, manifest['steps'] + 1):
        if not manifest.get('checkpoints') and step != manifest['steps']:
            continue
        if step == manifest['steps']:
            actual = w
        elif manifest.get('checkpoints'):
            actual = np.array([result['checkpoint_weights'][f'step{step}_w{j}'] for j in range(len(w))])
        else:
            continue
        expected = np.array(data['expected_weight_history'][step - 1]) if 'expected_weight_history' in data else ref
        step_error = float(np.max(np.abs(actual - expected)))
        agreement = float(np.mean((xt @ actual >= 0) == (xt @ expected >= 0)))
        if (not np.all(np.isfinite(actual)) or step_error >= 1e-5 or agreement != 1.0) and not args.allow_failure:
            raise AssertionError(f'Step {step}: error {step_error}, agreement {agreement}')
        history.append(dict(step=step, weight_max_absolute_error=step_error,
                            prediction_agreement=agreement,
                            plaintext_accuracy=float(np.mean((xt @ expected >= 0) == yt)),
                            ckks_accuracy=float(np.mean((xt @ actual >= 0) == yt))))
    result['per_step_verification'] = history
    result['plaintext_training_seconds'] = data.get('plaintext_training_seconds')
    result['plaintext_traced_peak_bytes'] = data.get('plaintext_traced_peak_bytes')
    result['sklearn_baseline'] = data.get('sklearn_baseline')
    result['exact_sigmoid_reference'] = data.get('exact_sigmoid_reference')
    result['experiment'] = manifest
    result['fhe_parameter'] = json.loads((args.task / 'fhe_parameter.json').read_text())
    result['graph_operation_counts'] = dag['operation_counts']
    result['verification'] = dict(weight_error_threshold=1e-5,
        all_checkpoint_weights_passed=all(x['weight_max_absolute_error'] < 1e-5 for x in history),
        all_checkpoint_predictions_identical=all(x['prediction_agreement'] == 1.0 for x in history),
        checked_steps=[r['step'] for r in history])
    result['all_steps_weight_max_absolute_error'] = max(r['weight_max_absolute_error'] for r in history)
    result['task_storage_bytes'] = sum(p.stat().st_size for p in args.task.rglob('*') if p.is_file())
    (args.task / 'verified_result.json').write_text(json.dumps(result, indent=2))
    text = f'''# 逻辑回归密文训练 GPU 固定基准

## 测试目的与实现

将 {manifest['steps']} 次全批量梯度更新静态展开为一个 DAG。权重在更新之间保持 CKKS 密文，服务端没有解密、重加密或 Python 训练循环。一次 GPU task 执行整个图，最后由客户端解密权重；测试集准确率在客户端明文计算。

每轮显式包含：特征与权重密文乘法、求和得到 logits、现有 PolyRelu0D 的 PS/BSGS 多项式、减去加密标签、梯度密文乘法、旋转归约、学习率缩放和权重更新。第 t+1 轮依赖第 t 轮的权重节点。

## 参数

| 参数 | 值 |
|---|---|
| 数据集 | sklearn 内置 Breast Cancer Wisconsin Diagnostic |
| 训练 / 测试样本 | 426 / 143；分层划分，seed=42 |
| 特征 | 30 个；StandardScaler 仅在训练集拟合；另加截距 |
| 明文 / 密文更新次数 | {manifest['steps']} / {manifest['steps']} |
| 学习率 / 初始化 | {manifest['learning_rate']} / 全零 |
| 优化器 | 全批量梯度下降，无正则化 |
| sigmoid | {manifest['degree']} 阶；{manifest.get('approximation', 'taylor')}；明密文均不裁剪 |
| 加密数据 | X、y、权重全部加密，多项式系数公开 |
| SIMD | 每特征一密文，数据零填充；权重复制到所有槽 |
| CKKS | 标准环，logN=15，16384 槽，默认 scale=2^40 |
| GPU / UUID / 驱动 / 显存容量 MiB | {resources.get('gpu_environment', '本次未采集').replace(chr(10), '<br>')} |
| 软件 | CUDA 12.8，sm_89，Release 构建 |
| 运行时间 UTC | {resources.get('started_utc', '本次未采集')} |

完整模数和参数由任务目录的 fhe_parameter.json 保存。每轮系数按对应 level 生成编码 scale。

固定多项式系数（幂次 → 系数）：`{manifest.get('polynomial_coefficients', {})}`。
区间拟合使用 [-{manifest.get('approximation_radius', 10)}, {manifest.get('approximation_radius', 10)}] 上的 4097 个等距点，不使用测试标签选择系数。原三阶 Taylor 在学习率 0.05 下的 20 轮明文训练会发散，故本基准改用固定区间最小二乘近似，明文与密文同时更换。

## 实测结果

| 指标 | 值 |
|---|---:|
| 明文测试准确率 | {result['plaintext_accuracy']:.10f} |
| CKKS 测试准确率 | {result['ckks_accuracy']:.10f} |
| 权重最大绝对误差 | {error:.10g} |
| 所有轮次中的最大权重误差 | {result['all_steps_weight_max_absolute_error']:.10g} |
| 权重 RMSE | {result['weight_rmse']:.10g} |
| 明文训练时间（不含加载和评估） | {data.get('plaintext_training_seconds', 0):.6f} s |
| 明文训练 tracemalloc 峰值（非进程 RSS） | {data.get('plaintext_traced_peak_bytes', '未采样')} bytes |
| 密钥生成 | {result['keygen_seconds']:.6f} s |
| 编码与加密 | {result['encode_encrypt_seconds']:.6f} s |
| GPU task 墙钟时间（含初始化） | {result['gpu_task_wall_seconds']:.6f} s |
| 解密 | {result['decrypt_seconds']:.6f} s |
| 总时间（不含编译） | {result['total_seconds']:.6f} s |
| 主机峰值 RSS | {result['host_peak_rss_bytes']} bytes |
| GPU 进程采样峰值显存 | {resources.get('gpu_process_sampled_peak_bytes', '尚未采样')} bytes |
| 输入 / 最终输出 level | {manifest['input_level']} / {manifest['outputs'][-1]['level']} |
| 输入与最终输出 level 差（不含自举恢复） | {manifest['level_consumption']} |
| ct×ct 乘法深度（依赖路径分析） | {dag['ct_multiplication_depth']} |
| 自举次数 | {manifest['bootstrapping_count']} |
| 批量自举节点数 | {manifest.get('bootstrap_batch_count', 0)} |

权重误差通过阈值为 1e-5 的断言。level 消耗包含 ct×pt 缩放，不应称作纯 ct×ct 乘法深度。此为单次功能测试，不用于比较运行性能。原始结果见 verified_result.json。

自举在第 {[r['before_step'] for r in manifest.get('bootstrap_placements', [])]} 轮之前执行，每个节点刷新全部权重密文。自举次数按被刷新的密文数量计数；批量节点数单独列出。自举前加倍权重，自举后乘 1/2 并 rescale，降低权重尺度上的绝对刷新误差，每次额外消耗一个 level。ct×ct 深度为训练算术的累计依赖深度，不含自举内部电路。显存通过 nvidia-smi 每约 0.2 秒加查询耗时采样，共 {resources.get('gpu_memory_sample_count', 0)} 个有效样本，采样峰值可能低于瞬时峰值。主机 RSS 包含密钥、编码、解密和逐轮诊断输出；明文时间与完整密文流水线不构成公平加速比。

逐轮诊断权重在完整 GPU task 结束后统一解密，不反馈给训练图。

常用 sklearn 参考：`{data.get('sklearn_baseline', '未采集')}`。它采用 L2 正则化与 LBFGS 收敛求解，不作为明密文逐轮一致性对照。同轮次、同学习率、无正则化的精确 sigmoid 明文参考：`{data.get('exact_sigmoid_reference', '未采集')}`。精确 sigmoid 与多项式之间的差异属于近似方案差异，不能计为 CKKS 数值误差。

软件版本：NumPy {manifest.get('numpy_version', '未采集')}，sklearn {manifest.get('sklearn_version', '未采集')}。tracemalloc 从训练循环前开始，未覆盖预先加载的数据和 Python/NumPy 运行时，不能与 CKKS 主机进程 RSS 直接比较。

| 轮次 | 明文准确率 | CKKS 准确率 | 权重最大绝对误差 | 预测一致率 |
|---|---:|---:|---:|---:|
{chr(10).join(f"| {r['step']} | {r['plaintext_accuracy']:.10f} | {r['ckks_accuracy']:.10f} | {r['weight_max_absolute_error']:.6g} | {r['prediction_agreement']:.1%} |" for r in history)}

所有轮次训练 logits 的范围为 [{min(r[0] for r in data['logit_ranges']):.10f}, {max(r[1] for r in data['logit_ranges']):.10f}]；是否位于拟合区间需根据上述范围核对。逐轮范围保存在 GPU_RESULT.json。首轮 logits 为零，后续轮次验证非零输入下的多项式及误差传播。

## 复现

```bash
python examples/logistic_regression/prepare_ckks.py --steps {manifest['steps']} --approximation {manifest.get('approximation', 'taylor')} --degree {manifest['degree']} --learning-rate {manifest['learning_rate']} --radius {manifest.get('approximation_radius', 10)} --checkpoints --output /tmp/logistic-ckks-benchmark
cmake -S . -B /tmp/logistic-gpu-build -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/logistic-gpu-build --target ckks_step_gpu -j8
python examples/logistic_regression/run_gpu_benchmark.py /tmp/logistic-gpu-build/examples/ckks_step_gpu /tmp/logistic-ckks-benchmark
python examples/logistic_regression/verify_ckks.py /tmp/logistic-ckks-benchmark
```

GPU 构建需开启 INFERENCE_SDK_ENABLE_GPU=ON、LATTISENSE_CUDA_ARCH=89，运行环境必须可访问 NVIDIA 设备。执行器仅链接 lattisense 聚合库，避免重复加载 GPU backend。

## 验证边界

本基准验证固定数据、参数、轮次下连续密文更新与明文的一致性，不代表模型已收敛，也不代表任意数据集和轮数均受支持。较长训练需要继续检查 logits 区间、多项式质量及自举误差。本次最终配置只有一次完整实测，未验证多组随机密钥下误差阈值的稳定通过率，计时也不提供均值或方差。

开发过程中的自举精度诊断记录在 BOOTSTRAP_PRECISION.json：不放大权重时最终误差约 1.40e-5，32 倍放大时约 6.43e-4，均未通过原定 1e-5 阈值；最终使用 2 倍放大配置。这些是不同配置的诊断，不能当作相同配置的重复性能试验。

## 编译器接入状态

图构建入口位于 `training/model_compiler/logistic_training.py`。生成任务时同步导出 `training_dag.json`，包含基础节点、依赖、level/scale、轮次边界、操作计数及 ct×ct 路径深度。结构测试覆盖两轮与二十轮，验证每轮 logits 依赖上一轮全部权重，并检查自举位置、计数与最终 level。

当前接入的是 LattiSense primitive DAG，自举位置手动安排并作为显式节点展开；尚未适配 Latti-ai 的 LayerAbstractGraph 自动调度入口。下一步需要统一两层图的算子、布局及深度契约。
'''
    report = args.report or args.task / 'RESULTS.md'
    report.write_text(text)
    print(json.dumps({k:v for k,v in result.items() if k!='weights'}, indent=2))
