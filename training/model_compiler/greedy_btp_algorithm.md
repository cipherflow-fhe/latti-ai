# Greedy BTP Insertion Algorithm

本文记录一个先不考虑 `drop_level` 和多输入 level 对齐的 greedy BTP 插入方案。

## 目标

当前先不考虑算子时间代价，只希望用一个快速、简单的策略插入 bootstrapping：

- 初始输入 feature level 设为最大 level。
- 按拓扑顺序遍历 feature node。
- 对每个 feature node，只看它后继 compute node 的 `level_cost`。
- 如果当前 feature 的 level 不足以支撑任意一个后继 compute，就在该 feature 后插入 BTP。
- compute node 被访问时，更新其输出 feature 的 level。

该版本暂不处理：

- `drop_level` 插入；
- 多输入 compute 的输入 level 对齐；
- fan-out 中只给部分边 refresh；
- BTP scale gamma 对可见 max level 的影响；
- BTP 全局数量最优性证明。

## 基本定义

设：

```python
min_level = 1 if config.mpc_refresh or config.graph_type == "mpc" or config.set_btp_scale is not None else 0
max_level = config.fhe_param.max_level
btp_out_level = max_level
```

这里要和 DP 模式保持一致：

- `set_btp_scale is None` 时，普通 BTP 图允许 feature level 用到 `0`。
- `set_btp_scale is not None` 时，BTP 输入最小 level 是 `1`；同时 `set_fhe_param()` 已经把 `config.fhe_param.max_level` 减 1，所以 BTP 后可见 `btp_out_level` 使用当前 `config.fhe_param.max_level` 即可。

对于一个 compute node `c_node`：

```python
level_cost = dag.nodes[c_node]["level_cost"]
```

一个输入 feature 能够执行该 compute 的条件是：

```text
current_level(feature) - level_cost >= min_level
```

当 `min_level = 1` 时，等价于：

```text
current_level(feature) - level_cost > 0
```

因此，如果：

```text
current_level = 1
level_cost = 2
```

则：

```text
1 - 2 = -1 <= 0
```

必须先插入 BTP。

## Greedy 策略

对每个 `FeatureNode f_node`：

1. 获取 `f_node` 的所有后继 compute node。
2. 取这些后继 compute node 中最大的 `level_cost`。
3. 如果 `current_level(f_node) - max_level_cost >= min_level`，说明该 feature 当前 level 足够喂给所有后继 compute，继续。
4. 否则，在 `f_node` 后插入：

```text
f_node -> bootstrapping -> refreshed_f_node
```

5. 插入后，`refreshed_f_node` 的 level 设为：

```python
btp_out_level
```

6. 后续 compute 使用 refreshed feature。

## Compute 输出 level 更新

当一个 compute node `c_node` 的所有输入 feature 都已经有 level 后，可以更新它的输出 feature level。

暂不考虑多输入对齐时，输出 level 按输入 feature 的最小 level 计算：

```python
input_level = min(feature_level[p] for p in preds)
out_level = input_level - level_cost
feature_level[out_f_node] = out_level
```

其中：

```python
preds = list(dag.predecessors(c_node))
out_f_node = next(dag.successors(c_node))
level_cost = dag.nodes[c_node]["level_cost"]
```

## 伪代码

```python
def greedy_insert_btp(graph: LayerAbstractGraph):
    dag = graph.dag

    min_level = 1 if config.mpc_refresh or config.graph_type == "mpc" or config.set_btp_scale is not None else 0
    max_level = config.fhe_param.max_level
    btp_out_level = max_level

    feature_level: dict[FeatureNode, int] = {}
    processed_compute: set[ComputeNode] = set()

    topo_nodes = list(nx.topological_sort(dag))

    # 1. 初始化源 feature level
    for node in topo_nodes:
        if isinstance(node, FeatureNode) and dag.in_degree(node) == 0:
            feature_level[node] = max_level
            dag.nodes[node]["level"] = max_level

    # 2. 按拓扑顺序遍历 feature node
    for node in topo_nodes:
        if not isinstance(node, FeatureNode):
            continue

        if node not in feature_level:
            continue

        current_f = node
        current_level = feature_level[current_f]

        # 3. 查看后继 compute node 的最大 level_cost
        succ_computes = [
            succ
            for succ in dag.successors(current_f)
            if isinstance(succ, ComputeNode)
        ]

        if succ_computes:
            max_level_cost = max(
                dag.nodes[c].get("level_cost", 0)
                for c in succ_computes
            )

            # 4. 如果当前 level 不足，插入 BTP
            if current_level - max_level_cost < min_level:
                btp_node, refreshed_f = insert_btp_after_feature(
                    dag,
                    current_f,
                    btp_out_level=btp_out_level,
                )

                dag.nodes[btp_node]["level_cost"] = current_level - btp_out_level
                dag.nodes[refreshed_f]["level"] = btp_out_level
                feature_level[refreshed_f] = btp_out_level

                current_f = refreshed_f
                current_level = btp_out_level

        # 5. 尝试处理 current_f 的后继 compute
        for c_node in list(dag.successors(current_f)):
            if not isinstance(c_node, ComputeNode):
                continue

            if c_node in processed_compute:
                continue

            preds = list(dag.predecessors(c_node))

            # 多输入情况下，只有所有输入 feature level 已知时才处理
            if not all(p in feature_level for p in preds):
                continue

            level_cost = dag.nodes[c_node].get("level_cost", 0)
            input_level = min(feature_level[p] for p in preds)

            # 理论上前面的 feature-level 检查应保证这里足够
            if input_level - level_cost < min_level:
                raise RuntimeError(
                    f"insufficient level before {c_node.layer_id}: "
                    f"input_level={input_level}, level_cost={level_cost}"
                )

            succs = list(dag.successors(c_node))
            if len(succs) != 1:
                raise RuntimeError(f"compute node {c_node.layer_id} should have one output feature")

            out_f = succs[0]
            out_level = input_level - level_cost

            dag.nodes[out_f]["level"] = out_level
            feature_level[out_f] = out_level
            processed_compute.add(c_node)

    return graph
```

## `insert_btp_after_feature` 伪代码

当前先使用 feature-level refresh，即一个 feature 后面只要需要 BTP，就让所有后继 compute 共享这个 refreshed feature。

```python
def insert_btp_after_feature(dag, f_node, btp_out_level):
    refreshed_f = copy.deepcopy(f_node)
    refreshed_f.node_id = make_unique_id(f"{f_node.node_id}_refreshed")

    btp_node = ComputeNode(
        layer_id=f"{f_node.node_id}_bootstrap",
        layer_type="bootstrapping",
        channel_input=f_node.channel,
        channel_output=f_node.channel,
    )

    old_successors = list(dag.successors(f_node))

    dag.add_node(btp_node, name=btp_node.layer_id)
    dag.add_node(refreshed_f, level=btp_out_level)

    for c_node in old_successors:
        edge_attrs = dict(dag.edges[f_node, c_node])
        dag.remove_edge(f_node, c_node)
        dag.add_edge(refreshed_f, c_node, **edge_attrs)

    dag.add_edge(f_node, btp_node)
    dag.add_edge(btp_node, refreshed_f)

    return btp_node, refreshed_f
```

## 当前版本的行为特点

### 优点

- 复杂度接近线性，速度远快于 DP。
- 规则简单，容易 debug。
- 对同一个 feature 的多个后继 compute，天然共享一次 BTP。

### 限制

- 不保证全局 BTP 数量最少。
- 暂不处理多输入 compute 的输入 level 对齐。
- 暂不处理 `drop_level`。
- 如果 BTP 要求输入必须是 level 1，该版本还需要在插 BTP 前补充 drop-to-1 逻辑。
- 如果某些后继 compute 已经处理过，再对同一个 feature 做整体 refresh，可能改变已处理边；实际实现时需要保证拓扑遍历和插入时机不会破坏已经处理过的 compute。

## 后续增强方向

1. 加入 `drop_level`：
   - BTP 前如果输入 level 大于 1，先 drop 到 1。
   - 多输入 compute 前，将较高 level 的输入 drop 到最低输入 level。

2. 加入多输入 level 对齐：
   - 对每个 multi-input compute，要求所有输入 feature level 一致。
   - 输出 level 使用对齐后的 input level 减去 `level_cost`。

3. 加入 edge-specific BTP：
   - 当不希望 refresh 某个 feature 的全部后继时，只替换指定边：

```text
f_node -> c_node
```

   变为：

```text
f_node -> bootstrapping -> refreshed_f_node -> c_node
```

4. 加入 BTP 共享优化：
   - 对同一个 feature，如果多个后继都需要 refresh，可以共享一个 refreshed feature。
   - 如果只有部分后继需要 refresh，可以只重连这些后继。
