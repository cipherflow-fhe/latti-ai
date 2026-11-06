# TAA WASM 0.2.0 OPFS 流式 ZIP32/ZIP64 设计

日期：2026-11-06。状态：已实现，ZIP64 专项与两任务真实服务互操作验收通过，证据见[验收记录第 3、4 节](../research/taa_v6_acceptance_report.md)。上位契约见[浏览器协议规范](taa_v6_browser_client_design.md)。格式依据 [PKWARE APPNOTE](https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT) 4.3.9、4.3.14–4.3.16、4.4.1、4.5.3；来源 `src/zip.js` 为仓库自写代码，本次升级该模块，不假定使用同名第三方库即可完成。

## 1. 必做范围与内存边界

所有 ZIP 创建、读取、解压、导入和导出经 OPFS 分块完成，支持 ZIP32 与 ZIP64：64 位原始/压缩大小、中央目录中的 64 位 relative local-header offset、ZIP64 中央目录扩展、ZIP64 EOCD record/locator、超过 65,535 条目。归档/单 entry 可超过 4 GiB，不能以 WASM 内存上限或批次限制回避。

输出使用 STORED；输入支持 STORED 与 raw DEFLATE，支持外部 data descriptor。只支持单卷、未加密、server 约定的扁平文件名；不新增压缩算法、分卷、自解压或 ZIP 层加密。未支持的结构给出明确错误。

内存模型为：最多 4 MiB I/O 块 + 当前 CRC/解压状态 + 当前密码学工作区 + 有预算的元数据索引。禁止全归档/全大 entry/全中央目录 `arrayBuffer()`、`new Blob(allChunks)`、`concat(entries)`、`Response(stream).arrayBuffer()` 或返回 payload 数组。`getFile()` 文件快照和有界 `slice().arrayBuffer()` 可以使用。

## 2. 最小模块接口

`src/zip.js` 依赖小型 seekable OPFS source/sink，拟定以下职责，不引入通用文件系统：

| 操作 | 语义 |
| --- | --- |
| source.readAt(offset, maxBytes) | BigInt 位置，块长 ≤4 MiB；返回实际字节，允许 EOF，固定结构短读报错 |
| sink.append(chunk) | 顺序写并等待完成，更新 BigInt 逻辑位置 |
| sink.patch(offset, chunk) | 覆盖预留范围，不改变逻辑追加位置 |
| writer.addEntry(name, source, options) | 一次一个 entry；source 是已完成的 OPFS 文件或有界流，边读边 CRC/计数/写盘 |
| writer.finish() | 关闭中央目录 spool、复制、写终结记录、回填与关闭；返回待发布的归档元数据 |
| reader.entries() / readEntry | 有界迭代 central record；定位 local header，校验/解压到当前 OPFS entry 文件 |

OPFS 基线用异步 `createWritable()`，顺序 write/定位 write 均 await。位置写不能破坏追加 cursor；按记录的逻辑位置显式恢复或指定后续位置。只有性能证据需要时才用 Worker sync access；必须循环处理短写/短读，flush/close 并遵守独占锁。

总大小/位置/count 始终 BigInt；JSON/Worker 消息为十进制字符串。CRC32、传统 u16/u32 字段才用 JS 位运算。读写 uint64 用 LE getBigUint64/setBigUint64；禁止对总数使用 `>>>0` 或 `Number` 后再转回 BigInt。

浏览器定位 API 接受 number 时，仅在 `0 <= offset <= Number.MAX_SAFE_INTEGER` 且区间终点同样安全后转换；否则报 `IO_RANGE_UNSUPPORTED`。uint64 元数据仍须精确解析，不能截断。此边界远高于 4 GiB，不能作为不支持真实 ZIP64 的理由。

## 3. OPFS 工作区与写归档流程

```text
job-<unique>/
  incoming.zip                 # 外部输入流式导入
  current-feature.part         # 当前 Feature，允许 core write-at
  current-entry.part           # 当前解压结果
  archive.part                 # 输出 ZIP
  central.spool                # 逐条中央目录记录
  entries.jsonl                # 业务条目元数据
  manifest.json                # 最后关闭的 committed manifest
```

文件名仅在本 job 目录内固定。一个 job 仅一个归档写者，公开/私有角色与发布规则沿用浏览器规范。

写入顺序：

1. 当前 case 的 Feature 由 core 流式写到 OPFS 临时文件，完成长度回填、关闭和校验。
2. ZIP writer 记 local offset，写本条 local header/extra/name。已知小条目可用 ZIP32；未知长度在开始时就预留 ZIP64 header，不能跨 4 GiB 后再扩展已写头。
3. 从完成的当前 Feature 流式读入归档，增量计算 CRC32、实际原始/压缩大小；STORED 两大小相等。复制完 patch local header 的最终 CRC/size，生成一个 central record 写入 `central.spool`。
4. 当前 Feature 临时文件可释放，继续下一条。只保留当前记录，不保留中央目录数组。
5. 全部条目结束后，关闭 spool writer，记 central offset，顺序复制 spool 文件流入 archive，得到 central size；写必要的 EOCD64/locator 和传统 EOCD。
6. 完成 patch、关闭 archive，检查长度和结构；关闭条目清单和 committed manifest 后发布 handle。spool/当前临时文件随后删除。

采用当前 Feature 暂存是为了让 ZIP CRC 基于**最终回填后的字节**。不能对回填前的流计算 CRC 后直接使用；本次不实现 CRC 补丁数学或第二套 direct-write 优化。内存保持有界，额外的当前 entry 磁盘占用与复制耗时须如实计入。

## 4. ZIP 字段与 ZIP64 扩展

所有格式数值均 LittleEndian；传统字段的宽度不变，不能把传统 header 中 u32 直接改为 u64。

| 结构 | writer 与 reader 的共同约束 |
| --- | --- |
| Local header `0x04034b50` | ZIP64 version-needed ≥45；传统压缩/原始大小写 `0xffffffff`，extra `0x0001` 包含原始 u64、压缩 u64，local 大小扩展必须成对；CRC 为 u32 |
| Central record `0x02014b50` | 传统 size/relative local-header offset 超界或保留 sentinel 时写 `0xffffffff`；真实值放本记录 ZIP64 extra，不给 local header 新增 offset 字段 |
| Central ZIP64 extra `0x0001` | 按对应传统字段 sentinel 决定存在项；固定顺序：原始大小 u64、压缩大小 u64、relative local-header offset u64、disk start u32；不把 local 的双大小必选规则套到 central |
| ZIP64 EOCD `0x06064b50` | remaining-record-size u64 不含前导 12 字节，最少 44；版本字段、disk=0、central-start-disk=0、两份 u64 count、u64 central size/offset；reader 接受有界跳过可变扩展，不硬编码整个 record=56 字节 |
| ZIP64 locator `0x07064b50` | 位于传统 EOCD 之前，disk=0、EOCD64 offset u64、total disks=1；校验其指向范围和 record 签名/长度 |
| 传统 EOCD `0x06054b50` | 始终保留；ZIP64 的 count/size/offset 对应字段使用 sentinel，从 EOCD64 读取；正常 ZIP32 直接读取传统值 |
| Data descriptor / bit 3 | reader 接受有/无 `0x08074b50` 签名；CRC u32，ZIP32 大小 u32×2，ZIP64 大小 u64×2；通过 central 大小定位，不在 payload 搜索签名 |
| 名称与 flags | 输出 UTF-8 字节名，bit 11=1；name length 按 UTF-8 bytes 检查 ≤65535；基线输出 bit 3=0，使用 seek 回填；输入 bit 3 可为 1 |

当 size/offset ≥`0xffffffff` 或 count ≥`0xffff` 时进入 ZIP64（sentinel 本身不表示普通数值）。未知长度/强制 ZIP64 测试也可提前使用扩展。中央目录可混合 ZIP32/ZIP64 entry；若任一条目用了 ZIP64，writer 同时输出 EOCD64/locator，传统 EOCD 字段按上述规则交给 EOCD64。reader 还须接受 entry 为 ZIP64 而传统 EOCD 中可表示总量仍为实际值的外部文件，不能仅靠 EOCD sentinel 判定所有 entry。

校验 ZIP64 extra 长度与所需字段一致，不把 uint64 当成 signed32。发现缺失必要扩展、重复歧义扩展或冲突值报格式错误；允许跳过完整且不影响本范围的未知 extra。disk start 与所有 disk 字段必须符合单卷约束。

## 5. 读取、解压与边界检查

1. 外部 File/handle 的 `stream()` 分块复制到 OPFS incoming，关闭后取得快照；不在 UI 线程读整包。
2. 从尾部最多 `22+65535` 字节搜索传统 EOCD，验证 comment length 对应真实文件终点，避免 payload 中伪签名。ZIP64 locator 可另作有界范围读取，不能假定 central/EOCD64 全在尾窗。
3. 读取 locator/EOCD64，验证 uint64 加法、记录终点、central 范围/count、单卷与各记录相对位置。large extension 逐块跳过，不按声明分配总 buffer。
4. 从 central offset 逐条读固定 header，再读有界 name/extra/comment；消费字节总数与 central size、实际条目数与 count 一致。支持小 ZIP32、强制小 ZIP64、混合记录、变量 EOCD64 扩展。
5. 按各 entry local offset 定位 local header，核对名称、方法、影响数据解释的 flags、ZIP64 大小和数据起点；bit 3 时允许 local CRC/size 未定，但 central、descriptor、实际值最终必须一致。校验数据不越过文件/central、不重叠其他 header/data，不允许指向 central 内的伪 payload。
6. STORED 按大小分块复制；DEFLATE 用原生 `DecompressionStream('deflate-raw')` 或已声明能力的增量实现，将真实输出写 OPFS entry，同时累计 CRC/原始大小。能力不可用则明确拒绝 DEFLATE，不收集全 entry 后解压。
7. 解压输出完成、CRC/压缩消耗长度/真实大小通过后关闭当前文件，再提供给 core 分块解密；解密 CSV 分块落盘。损坏数据不产生 committed artifact。

拒绝绝对路径、`/`/`\\` 分隔符、`.`/`..`、NUL、非法 UTF-8 和不满足 server 扁平命名的条目；按 NFC + 不依赖 locale 的小写键检查重复/大小写冲突，保留原名用于 case 映射。不能借解压做路径创建。规范化冲突、central/local 不一致、加密、多卷、未知压缩方法均给出带 entry 的错误。

## 6. 受控元数据和解压预算

本次选择**有明确预算的精简索引**，不建设磁盘 B-tree。writer 的 central records 一律 spool；reader/case 匹配只保留 name、offset/size、CRC、必要 flags 和区间。默认 metadata 预算 64 MiB，按字符串实际 UTF-16 存储估计加保守对象开销记账；实现时通过 heap 测量校准，不能只按文件名字节数宣称恒定内存。超过预算报 `METADATA_BUDGET_EXCEEDED`，可由调用方增大预算。

支持 >65,535 entries 的真实验收必须在默认或公开声明的配置下通过；预算不能固定成 ZIP32 entry 上限。若今后真实任务超预算，再基于实际需要增加磁盘索引，本次不预建复杂分页基础设施。

调用方可配置 `maxEntries`、`maxEntryOutputBytes`、`maxTotalOutputBytes`、`maxMetadataBytes`。初始化时明确验证/展示实际值；默认大小预算不得低于 8 GiB 单条、32 GiB 总输出，条目预算不得低于 100,000，设备配额不足仍以实际 I/O 错误处理。测试可显式调整预算覆盖更大输入；格式 uint64 上限与运行预算分开。

解压逐块按**真实输出**执行预算，不能只信 central 声明。磁盘预估至少包含 incoming + 最终 archive + central spool + 当前最大 Feature/解压文件 + CSV/导出副本，storage estimate 只是提示。配额耗尽或取消停止生产并在 finally 关句柄，清理仅限本 job 未提交产物。

## 7. 验收矩阵

| 层级 | 必须验证 |
| --- | --- |
| 快速格式 | `0xffffffff`/`0x100000000` 前后 size/offset，`0xffff` 前后 count；local/central extra 组合、ZIP32/64 混合、强制小 ZIP64、EOCD comment/可变扩展、locator、descriptor 有/无签名 |
| 精度 | full uint64 精确解析；超过 2^53−1 的 IO 明确拒绝；禁止 wrap/近似位置；UTF-8 byte length |
| 真实 Chromium/OPFS | 分块生成并读回 >4 GiB 单 entry；第二用例让后续 local header 实际 offset >4 GiB；生成/读回 >65,535 entries；测试文件不整体驻内存 |
| 独立互操作 | 浏览器 ZIP64 经 Python zipfile/7-Zip 流式验证 size/count/CRC/offset；独立实现生成的 STORED/DEFLATE/descriptor/ZIP64 由浏览器读取，不能只自读自写 |
| 真实 server | ResNet 单输入、Retina 三输入由本 writer 生成 ZIP64，现有服务接受，返回 ZIP 可直接解密；与任务规范共同验收 |
| 失败 | CRC/真实大小不符、截断 header/extra/central/EOCD64、重叠区间、重复名、配额、短写/close、解压超额、取消/Worker 崩溃；无成功半成品 |
| 内存/磁盘 | 相同 chunk/密码学参数下增加总 payload，JS/WASM/RSS payload 内存不线性增长；索引单独记账；加密 ZIP 与读/解压/解密分别记录峰值与磁盘占用 |

边界 mock 和小文件强制 ZIP64 用于快速反馈，不能代替真实 >4 GiB、64 位 offset 和大条目数证据。外部 server 如仅支持 ZIP32，列为未完成依赖并安排其必要修改；不能拆包/截断或改成内存路径后宣称本需求完成。
