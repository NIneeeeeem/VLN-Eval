# Linux 部署与推理

简体中文 | [English](deployment.md)

实验与机器资源分别配置。当前可直接使用已有 Python/Conda 环境运行；Docker launcher
需要用户对 daemon 的访问权以及事先构建并验证的镜像 digest。本仓库不提供已发布的
真实模型/仿真镜像。

## 控制面验证

```bash
python -m nav_eval plugins list
python -m nav_eval doctor
python -m nav_eval plan \
  --config configs/experiments/navida-r2r.json \
  --resources configs/resources/my-host.json
```

这些命令只验证插件发现、入口和资源规划，不启动模型或仿真器。真实方法各自保留
上游依赖环境，并在 `run` 的 preflight/prepare 阶段检查。

## 真实运行

```bash
python -m nav_eval plan --config configs/experiments/navida-r2r.json --resources configs/resources/my-host.json
python -m nav_eval run --config configs/experiments/navida-r2r.json --resources configs/resources/my-host.json
python -m nav_eval run --config configs/experiments/navida-vlnverse.json --resources configs/resources/my-host.json
python -m nav_eval resume --run runs/<run-id>
python -m nav_eval evaluate --run runs/<run-id>
```

机器专属资源映射（`*.this-host.json`）不入库。任何机器都从
`configs/resources/example.json` 建立自己的资源文件。运行前检查其中物理 GPU 的
空闲显存；GPU 分配只影响本次 worker，不停止其他进程。

资源可按 plugin ID 放到 `runtimes`，或用 `method/environment` 覆盖。
每个角色可指定 python、gpu、pythonpath、library_paths、cwd、env、timeout_s、
min_free_memory_mib，以及 checkpoint/repo_path/data_root/challenge_repo 等 settings。
资源 settings 只接受 manifest 的 requires.paths/resource_settings；相机、动作和成功
阈值等实验参数必须写入实验配置，不能经本机映射绕过比较键与兼容检查。
模型和渲染进程可以使用不同 GPU；物理卡通过 CUDA_VISIBLE_DEVICES 映射到进程内 0。

Preflight 检查路径、解释器、GPU 和磁盘。隔离进程的模型加载与环境初始化并发进行；
全部副本 prepare 和身份校验通过后才开始 episode。单副本日志写入 `method.log/environment.log`，失败摘要
写入 `failure.json`。Python launcher 清理独立进程组，包含解释器脚本启动的子进程。
模型跨 episode 常驻，方法历史和缓存按上游 reset 规则清理。

## StreamVLN R2R

资源映射绑定 StreamVLN v1-3 权重与 Habitat 0.2.4 独立环境：

```bash
python -B -m nav_eval run \
  --config configs/experiments/streamvln-r2r.json \
  --resources configs/resources/my-host.json
```

该配置选择 val_unseen 全部 11 个场景中各 3 条不同路线，共 33 个诊断 episode；
不是 1839 条全集评测。删除 `episodes` 字段可运行完整 split。资源文件中的 GPU 4
是本机测试时的显式选择，启动前会检查模型与渲染进程合计 32000 MiB 的显存预算。
长轨迹曾观测到模型进程占用 27258 MiB（26.6 GiB），短烟测不足以估算并行容量；
预算不足时应等待资源或更换显卡。该预算已按长轨迹观测从初始试验配置上调。
其他机器须修改解释器、上游源码、权重、数据及 GPU 路径/分配。

默认 `method_settings` 为 `preprocess_mode: lazy`、`tokenizer_mode: reuse`：
保存每步原始 RGB，按上游相同规则读取当前/历史帧，并复用专用提示 tokenizer 副本。
不改变模型精度、生成参数或随机连接词。`eager`、`upstream` 是可复现的性能对照开关。
两条烟测配置分别为 `streamvln-r2r-smoke.json` 和 `streamvln-r2r-lazy-validation.json`。
两次运行后可检查完整动作、评测证据、指标和身份锁：

```bash
python -B scripts/compare_inference.py BASE_RUN OPTIMIZED_RUN \
  --streamvln-preprocessing --output comparison.json
```

该开关仅允许上述两项 StreamVLN 预处理设置不同，其他身份/语义仍须匹配。
StreamVLN 当前使用单 session 的流式 KV 缓存；多副本可走下面的独立进程模式，
尚不支持 `inference.mode: shared`。

## 并行推理

实验配置增加 `"parallelism": 2`，资源文件增加两个显式分配：

```json
{
  "replicas": [
    {"method": {"gpu": 0}, "environment": {"gpu": 0}},
    {"method": {"gpu": 1}, "environment": {"gpu": 1}}
  ]
}
```

这些条目与原有 `runtimes`、`method/environment` 资源配置合并，不能单独替代权重和
解释器配置。可直接修改 [parallel.example.json](../configs/resources/parallel.example.json)。
`replicas` 数量必须等于 `parallelism`，需要 GPU 的每个角色必须明确写出设备。
同一副本的方法和环境可以放在不同卡上；也可以显式将多个副本放到同一张卡。
副本只能覆盖部署字段，不能改 checkpoint、方法设置或 benchmark 设置。

默认 `inference.mode: replicas`，每个副本由独立的 method/environment 进程组成，常驻模型只加载一次。所有副本共享
一个 episode 队列，完成当前任务就领取下一个任务；长轨迹不会让其他空闲副本等待。
每个 worker 仍只持有一个 session，episode 内严格按观察、推理、动作、反馈的顺序执行。
因此 NaVid 的上游全局历史、StreamVLN 的缓存等不需要改成线程安全对象。
此模式的每份权重占用独立显存；共享权重模式需要下面的显式配置。

GPU 角色必须配置正数 `min_free_memory_mib`，代表该 worker 的保守显存预算。
Preflight 按物理 GPU UUID 汇总所有模型和渲染进程的预算，检查总和是否小于可用显存。
数字索引和 UUID 指向同一卡时合并计算。它是启动前的容量检查，不是系统级显存预留，
仍需给最长历史和其他作业留余量。多副本共用一张卡的收益取决于模型大小和 GPU 负载，
不能预设并发数翻倍就能获得两倍速度。

示例（副本 GPU 分配来自你的资源映射，可参照
[parallel.example.json](../configs/resources/parallel.example.json) 构建）：

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r-parallel.json \
  --resources configs/resources/my-host.json
```

该示例固定 GPU 7、两个副本与 4 个诊断 episode。扩展到完整 split 前，需要确认全部
场景存在并修改 episode 选择。其他已接入方法使用相同 `parallelism/replicas` 配置，
替换基础资源映射即可；模型各自的真实并行数值一致性仍须验证。

并行日志在 `workers/000/`、`workers/001/` 等目录；`replicas.lock.json` 锁定每个副本
的运行时与资产。执行前要求所有副本身份及 episode 清单一致；重启只影响发生基础设施
错误或清理失败的 worker 对，并再次检查身份。调度器统一提交 attempt，结果、评分和
resume 仍在一个 run 目录中。Ctrl-C 会回收本次启动的进程组；已提交 attempt 保留，
未提交任务从 episode 边界恢复。`local` launcher 保持单副本，避免 RNG 和仿真线程混用。

## 共享模型与合批

已适配的方法可使用一份模型服务多个独立环境，当前生产方法中只有 NaVIDA 声明该能力。
实验增加以下配置；资源仍复用 `replicas` 以明确指定每个环境和共享模型的设备：

```json
{
  "parallelism": 2,
  "inference": {"mode": "shared", "max_batch_size": 1, "max_wait_ms": 0}
}
```

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r-shared.json \
  --resources configs/resources/my-host.json
```

所有 `replicas[].method` 合并后的部署资源必须相同；模型只加载一份，显存预算也只计
一次，环境预算逐个累计。`parallelism` 在此模式中表示环境/session 上限，
`max_batch_size` 是一次合批的请求数上限，默认 1，必须在 1 到 parallelism 之间。
默认共享模式允许环境与模型重叠，保留单样本生成形状；设为 2 或更大才启用真实张量合批。
`max_wait_ms` 是从首个就绪请求开始等待更多请求的时间上限，默认 5 ms；最后一个
episode 不必等待其他环境。排队和 GPU 执行时间仍可能更长，这不是 RPC 总延迟上限。

只合并不同 episode 的当前请求，每个 episode 内仍等待执行和新观测。历史、动作队列、
generation 和 observation sequence 按 session 隔离。NaVIDA 多输入采用左 padding，
沿用原来的 JPEG、帧采样及动作解析；不跨 episode 复用 KV 或响应。
队列合并的请求可能包含 pending action，因此实际模型 batch 数要看 generation 计数。
模型日志写入 `workers/shared-method/method.log`，环境日志仍在 `workers/000/` 等目录。

基础设施或清理故障会暂停领取新 episode，等待所有在途 attempt 完成并提交，然后
统一重启共享模型及环境并再次核对身份。策略错误只终结对应 episode，不触发模型重启。
共享模型进程崩溃会影响使用它的在途 episode；需要更强故障隔离时使用默认副本模式。
量化、张量并行、vLLM/SGLang 替换和随机采样合批本轮均未启用。

NaVIDA 的 `decoding: upstream` 是原有实验的默认值，保留上游的
`use_model_defaults=True`。本机检查点会把 do_sample=False 覆盖成采样模式。
默认共享模式保留 upstream，且逐 session 保存/恢复 CPU/CUDA RNG，防止其他 episode
的 reset 或推理改变采样序列。大于 1 的 batch 要求显式 `decoding: greedy`，使用
`use_model_defaults=False` 阻止采样覆盖，同时继承正确的 BOS/EOS/pad token。
greedy 是被冻结的方法设置，实测与 upstream 轨迹不同；不能混用基线。即使明确
greedy，浮点批量计算也不保证逐 token 等价。较大 batch 是诊断选项，必须与相同
decoding 的单样本轨迹对照后再决定是否用于实验。

## 仿真与资产

Isaac/InternUtopia 使用单个 Kit App，通过 reset 更换选中场景。绑定默认允许
`max_control_steps=500` 次控制决策、`max_task_steps=25000` 个上游任务步；后者不是
模型决策次数。两项均属于 benchmark_settings，进入比较键。上游终止原因及任务
计数独立记录。`navida-vlnverse-two-scenes.json` 是每场景 20 次决策的部署诊断配置，
不能用于报告完整 benchmark 分数。本机 5.0 安装包的确切 rc/build 身份保存在锁文件；
模型初始化、Kit 启动、场景重置与实际推理耗时分开统计。

首次运行会计算权重和选中资产内容摘要，其时间与模型初始化计时分开；文件存在
不等于身份已验证。运行目录包含 resolved 配置、各角色最小配置、环境/模型/评分锁、
episode attempts、轨迹、私有 evidence、timing 和独立 evaluations。

## Docker

实验选择 `launcher: docker`，各角色资源设置：

```json
{
  "image": "registry.example/navida@sha256:<64位真实摘要>",
  "container_python": "python",
  "gpu": 0,
  "mounts": [
    {"source": "<宿主机权重目录>", "target": "/weights", "read_only": true}
  ],
  "settings": {"checkpoint": "/weights/navida"}
}
```

宿主机侧挂载源是宿主机路径唯一出现的位置，且只存在于你未入库的资源映射中。

路径检查发生在宿主，容器内对应资源使用相同绝对路径挂载，避免隐式路径转换。
选中的外部插件会随代码快照装入容器，无需额外挂载。数据/权重用只读挂载，模型和
shader 缓存显式设置可写 mount。源码与每个角色的配置只读挂载；不会挂载整个项目，
method 不接收 environment 的 role config。
镜像需要预装相应 SDK 和原模型依赖；镜像不存在时 Docker 根据固定 digest 拉取。

两个容器通过宿主 loopback 端口供本机控制进程连接，不开放公共网络 API。
Launcher 仅停止本 run 生成的随机名称容器。Docker 的命令构造可测试，但 daemon
权限不足时必须把容器实跑标记为未验证。

## 恢复与吞吐

恢复单位为 episode，默认基础设施重试额度为 0；需要重试时在原始实验里设置
`max_infrastructure_retries`。策略错误不重试；未提交的 episode 可从头重跑。
已完成 attempt 不重复推理，源代码、插件、模型或资产变化时拒绝混入旧 run。
基础设施失败后，会在下次允许的 attempt 前重建 worker，并再次核对运行时、资产和
episode 清单。中断后的离线评分直接读取冻结指标配置，不依赖之前已生成评分报告。

`transport: binary` 采用无损 buffer framing；`json` 为参考协议。不启用有损压缩、
量化或替换推理引擎。`shards/shard_index` 可将同一冻结集合分给独立运行；每个 shard
内部还可设置 `parallelism`。先筛选 episode/limit，再分 shard，再由副本队列动态分配，
不会让不同副本重复采集同一 episode。不同 run 的设备容量仍由调用者协调。

`timing.json` 保存冷启动、reset、act、step 时间；episode 记录包括 RPC 总耗时。
NaVIDA 还报告 preprocess/generate/decode 时间、实际 generation 次数、generation batch
大小直方图及输入/输出 token 数；pending action
执行不冒充一次模型生成。每 episode 只提交新增 attempt，退出或恢复时重建 JSONL。

`timing.json.sessions` 记录每次 run/resume 的启动时间、rollout 墙钟时间、已提交的
决策数/完成数以及每个副本的 `attempts/episode_wall_time_s/busy_fraction` 和阶段计时。
`decisions_per_second` 按实际 rollout 墙钟计算，pending action 仍是决策而非 generation。
顶层 `method/environment` 计时仅保留副本 0 的兼容视图。各 session 的 `method_workers`
提供按实际模型 worker 去重的 `current/retired` 计数；`replicas` 的 `method_worker_index`
标明归属，共享模式中不能把每个环境看到的同一模型计数重复相加。
`batcher.batch_sizes/requests/queue_wait_s` 描述队列批次和请求等待时间总和，
`phases.generation_batch_sizes` 才是实际模型生成批次。

`collection.json.episodes_per_hour` 是完成数除以累计 rollout 墙钟时间，包含并发、
reset、通信、提交和期间的重启；`collection_episodes_per_hour` 另外包含资源哈希、
加载与身份验证。后者不含结束时的进程回收和离线评分。`episode_wall_time_sum_s`
单独保留逐 episode 时间之和，不能作为并发吞吐的分母。恢复时累计各次墙钟时间，
已完成 episode 不重复计数；若硬中断导致计时缺失、计数无法对齐，则吞吐返回 null。
模型数值等价、跨 GPU 一致性与吞吐收益需要各自测量。
