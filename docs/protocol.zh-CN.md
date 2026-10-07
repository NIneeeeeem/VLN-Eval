# 协议与生命周期

简体中文 | [English](protocol.md)

当前 wire envelope 为 `nav-eval/0.2`，新增实验配置为 `nav-eval-experiment/1`，
插件声明为 `nav-eval-plugin/1`，原子提交产物为 `nav-eval-rollout/0.4`。范围是可信
单机评测——TLS、公共网络鉴权、多租户调度与 step 级 exactly-once 投递不在范围内。

## Worker 与会话

worker 启动首先输出本机 endpoint；进程存在不等于就绪。`prepare` 加载真实模型或
初始化真实场景，成功后 `describe.ready=true`；`/health` 未就绪返回 503。
`describe` 包含角色、插件身份、传感器/动作、capture schema、clock、session capacity
与 codec；控制端验证它们与 resolved plan 一致。

```text
prepare → describe / attest → episodes
  → environment.reset(episode_id, seed, session_id)
  → method.reset(public_context, session_id)
  → [method.act → environment.step → public transition]*
  → environment.finish → close_episode → atomic attempt commit
  → offline evaluate
```

默认一个 worker 同时只服务一个活动 session。每次 attempt 使用不同 session_id；方法
内部 generation 也重新生成。MethodBoundaryAdapter 检查 episode、generation 和
观测序号，拒绝迟到结果；支持 transition 的方法收到每个已执行子动作的公开反馈。
方法保持权重常驻，reset/close 清理历史、地图、缓存、pending actions 等算法状态。

并行运行创建多个独立 worker 对，通过动态队列分配完整 episode。单 worker 的独占
会话约束不变；控制面只在前一次 attempt 提交后才给该副本分配下一条任务。所有副本
执行前核对相同资产、运行时和 episode 清单；每条记录保留 `replica_index`。

显式 `inference.mode: shared` 可让多个环境连接同一方法 worker。方法必须声明
batching 能力并实现 `act_batch`；SDK 为每个 session 创建独立的
原有方法边界。对外 RPC 格式不变，`describe.session_capacity` 改为配置上限；对内
仅合并不同 session 的就绪 act，请求数与等待时间有界，同一 session 不能同时有两个
未完成调用。`act_batch` 是多 session 推理批次，与一个决策返回的动作 chunk 不同。
共享推理的 `method_worker_index` 用于去重计时，故障时先排空在途任务再协调重启。
默认 max_batch_size=1，方法可声明 independent_sessions 并隔离随机状态。更大批次
要求 independent_greedy；同一方法不同 decoding 设置的轨迹不能混为同一基线。

## 公开观测与时间

观测含 `episode_id / sequence / sim_time_s / sensors / sensor_specs`，可含
`control_tick`。每个 sensor 声明 modality、dtype、shape、unit、frame、source、calibration。
验证包含 shape/dtype、RGB 范围、相机尺寸与 hfov；depth 单位必须是 m。
`rendered / measured / estimated` 是允许的来源；privileged ground truth 不可作为输入。

有模拟时钟的任务使用 `sim_time_s` 秒。Habitat/VLNVerse 的离散任务目前设置
`sim_time_s=null`，用 `control_tick` 记录控制步，避免把动作次数冒充物理秒。
目标坐标、reference path、测地距离及评分信息仅进入 environment 的私有 evidence。

Python 进程隔离用于分离角色，不是针对不可信代码的安全沙箱。Docker 默认只挂载代码
快照、当前角色配置及显式配置的资源——私有 benchmark 数据不要挂进 method 容器。

## 动作与失败

动作 batch 含 episode、observation_sequence 与 1–16 个动作。primitive.forward
以 m 计，left/right 以 rad 计；twist 使用 m/s、rad/s 和秒。STOP 为独立动作。
camera_tilt/wait 虽有类型，也必须有实际 binding 才能执行。控制器转换必须显式选择。
NaN、Inf、非法幅度、未知动作和 frame 不会被悄悄裁剪成合法动作。

Habitat 的 R2R/RxR binding 在 `supported_action_semantics` 中明确声明可支持的
15°/30° 转向。`track: native` 按方法声明选择幅度，并冻结到 binding、环境配置与
`compare_key`；NaVid / Uni-NaVid 使用 30°，其他现有方法使用 15°。一次 30° 转向
仍是一个控制步。`standardized` 保留 benchmark 默认幅度，冲突在计划阶段拒绝；
不同幅度的 native 运行不能当作完全相同的评测协议。

terminated/truncated 都停止剩余 chunk。`PolicyViolation` 保留为 policy_error；
协议错误、超时、worker 故障为 infrastructure_error。策略失败不重试；基础设施
重试预算随配置冻结。同一 episode 首个 completed/policy_error 是 canonical attempt，
否则展示最后一个基础设施失败；所有已提交 attempt 保留。

HTTP 400 是协议或策略错误，500 是 worker 内部错误；完整异常保留在角色日志。
不重发无法确认执行状态的 step。恢复从 episode 边界开始；它不是中间仿真状态恢复，
也没有承诺恰好执行一次。监控采集失败不会覆盖已提交的 episode 结果。

## 无损张量传输

参考 codec 为 JSON/base64：显式 `tensor_b64 / dtype / shape / byte_order`，支持
uint8、float32、int32，wire 固定为 little-endian。ndarray 的非本机字节序会显式转换。

`transport: binary` 先通过 describe 协商，然后发送 4 字节 header 长度、JSON header
和 tensor 原始 buffers。解码验证长度、索引与边界，恢复相同张量。JSON 限制 8 MiB，
binary 限制 64 MiB；不改变图像内容、尺寸、压缩率或历史帧选择。
NaVIDA 自身的 JPEG prompt 图像属于算法预处理，仍按上游行为执行。

HTTP 默认绑定 loopback；Docker 发布到宿主 loopback 的随机端口。该实现适合可信
单机评测；TLS、认证、多租户资源调度和跨机集群不在本次实现范围内。
