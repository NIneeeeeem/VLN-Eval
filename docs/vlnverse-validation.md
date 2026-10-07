# VLNVerse、InternVLA-N1 与插件修复验证（2026-10-07）

在四步 smoke 之上，使用真实模型、真实图像与完整预算完成闭环导航验证，并修复外部
插件注册/资源解析边界。下表均为小样本诊断，不代表全 split 或论文性能复现。
GA-VLN 不支持当前 Isaac/VLNVerse 策略接口，因为该接口不提供位姿。

## 实测结果

| 方法与环境 | 完成 / 选定 | SR | SPL | NE (m) | OSR | 结束原因 |
|---|---:|---:|---:|---:|---:|---|
| StreamVLN，VLNVerse，物理 H1 | 12 / 12 | 33.33% | 28.65% | 4.255 | 33.33% | 8 STOP、4 跌倒 |
| StreamVLN，VLNVerse，flash + 碰撞检查 | 12 / 12 | 58.33% | 47.79% | 2.863 | 66.67% | 8 STOP、4 导航预算耗尽 |
| InternVLA-N1，R2R-CE，Habitat 0.3.0 | 9 / 9 | 55.56% | 50.49% | 4.396 | 77.78% | 9 STOP |
| InternVLA-N1，VLNVerse，物理 H1 | 12 / 12 | 41.67% | 34.87% | 3.721 | 58.33% | 10 STOP、2 跌倒 |
| InternVLA-N1，VLNVerse，flash + 碰撞检查 | 12 / 12 | 58.33% | 51.04% | 2.995 | 75.00% | 9 STOP、3 导航预算耗尽 |

上述 5 组共 57 次 episode 的 policy/infrastructure error 均为零。R2R 的两条路线执行了 527、593
次公共控制，但分别只有 91、101 次导航动作，证明相机探测不会再提前耗尽 500 次
导航预算。已核对最终产物 SHA256、逐条事件计数以及离线指标：VLNVerse 与捕获的
上游 SR/SPL/NE/OSR/nDTW 最大误差 `8.89e-16`；R2R SR/SPL/NE 最大误差
`1.30e-8`。R2R 上游未捕获 oracle_success，故 OSR 没有直接上游对照。
VLNVerse 上游 success 只检查终点距离，不要求 STOP；预算耗尽或跌倒也可能记为
SR=1。因此保留上游指标定义，并单列结束原因，不将 SR 等同于正常停止率。
InternVLA flash 的 7 条成功中，6 条成功 STOP，另 1 条 `kujiale_0020_0_0`
在终点附近耗尽导航预算；若额外要求成功 STOP，则是 6/12（50.00%）。该组没有
跌倒，全部路线均进入最终评分，未剔除 3 条预算耗尽的路线。

运行产物位于仓库本机目录（`runs/` 不纳入源码发布）：

- StreamVLN 物理 H1：`runs/vlnverse-repair-20261007/streamvln-12/20261007T044714Z-3f2656b9/`。
- StreamVLN 碰撞 flash：`runs/vlnverse-repair-20261007/streamvln-flash-12/20261007T055256Z-b24d120e/`。
- InternVLA R2R：`runs/vlnverse-repair-20261007/internvla-r2r-9/20261007T053417Z-ffe55a75/`。
- InternVLA 物理 H1：`runs/vlnverse-repair-20261007/internvla-12-fixed/20261007T060856Z-8c36c0ef/`。
- InternVLA 碰撞 flash：`runs/vlnverse-repair-20261007/internvla-flash-fixed-12/20261007T062830Z-7c9da98e/`（中断后经 `resume` 续跑完成 12/12）。
- 指标/哈希/动作计数核验：`runs/vlnverse-repair-20261007/metric-parity.json`，
  同目录 `verify_results.py` 可重新核验完成的运行。

## 固定样本与复现

VLNVerse fine-unseen 原 split 有 825 条、33 个场景。先固定前 3 个字典序场景，
每场景取前 4 个数值路线编号，再取该路线最小指令编号；选择不依赖测试成绩：

```text
kujiale_0020_0_0  kujiale_0020_1_3  kujiale_0020_2_3  kujiale_0020_3_2
kujiale_0030_0_0  kujiale_0030_1_3  kujiale_0030_2_3  kujiale_0030_3_1
kujiale_0031_0_0  kujiale_0031_3_2  kujiale_0031_4_4  kujiale_0031_6_3
```

选择、split SHA256、1341 个场景资产文件的 SHA256 在评分前冻结于
`runs/vlnverse-repair-20261007/cohort.json`。split SHA256 为
`5de7d1fedcdc09630ba0dee713e2f09d161907a8a14663e4253a4ba58cb02212`。
0020 使用已有本地资产；0030、0031 通过 `data/scene_data/vlnverse/` 下的链接
只读接入外部共享场景库。H1 使用已有 `data/Embodiments/vln-pe/h1/h1_internvla.usd`；
未新申请账号授权，资产来源记录与在线下载收据分开保存。

已完成环境和资产登记后，在仓库根目录运行（GPU 编号按本机空闲卡选择）：

```bash
python -B -m nav_eval run \
  --config configs/experiments/streamvln-vlnverse-diagnostic.json \
  --gpu 1 --output runs/streamvln-vlnverse-diagnostic
python -B -m nav_eval run \
  --config configs/experiments/internvla-n1-vlnverse-diagnostic.json \
  --gpu 2 --output runs/internvla-vlnverse-diagnostic
```

两份配置都是 `track: native`、seed 0、640×480、90° HFOV、初始俯仰 0°、
500 导航动作；InternVLA 开启 depth/tilt，控制上限 5000、物理步上限 50000；
StreamVLN 控制上限 500、物理步上限 25000。当前绑定预热为 30 步。
原 smoke 配置仍然是短程连通性检查。
R2R 诊断取前三个字典序场景的前三条不同路线，最低指令编号：
`10,16,43,220,289,349,127,196,202`；运行配置保存在同一证据目录的
`internvla-r2r.json`。

复现离散 flash 对照时，在相同诊断配置的 `benchmark_settings` 中增加
`"robot_flash": true`，其余保持不变；已运行的完整配置为证据目录内
`streamvln-flash.json` 和 `internvla-flash.json`。碰撞检查始终开启，默认配置仍是物理 H1。

## 修复内容与必要边界

InternVLA-N1 的模型布局保持已训练 checkpoint 的 `visual.*`、
`model.embed_tokens/layers/norm.*`、`model.latent_queries`、`model.navdp.*`
与 `lm_head.*`，避免新版 Transformers 的视觉/文本嵌套布局改写权重键。
加载时检查这些权重的 missing/unexpected/mismatched 诊断，兼容字符串和 shape
元组两类诊断格式；不允许静默丢弃核心权重继续评测。实际 checkpoint 验证中各类
加载诊断均为空，latent 为 `[1,16,3584]`，NavDP 轨迹为 `[32,32,3]`。
`scripts/verify_internvla_n1.py` 可检查真实 RGB 文本生成和 NavDP 张量计算；其中
NavDP 独立检查使用合成零 RGB/depth，仅验证张量计算，导航质量由闭环运行评估。
增强验证覆盖两枚新 token（覆盖生成缓存），确认最终模型无 meta 参数、
latent 与 NavDP 输出均有限；报告为证据目录内
`internvla/verification-cached.json`。

两个 VLN 绑定分别记 `navigation_steps` 与 `control_steps`：primitive 和 STOP
计入两者，camera_tilt 只计控制；保留旧 `steps`、`control_tick` 的全部控制计数。
返回的 `decision_limit` 和 Habitat 内部步数上限使用控制上限，绑定自行限制导航
动作。tilt 无限循环仍由有限控制上限终止。30° 逻辑转向的两次 15° 后端调用只算
一次导航决策。记录和 evidence 同时保存预算与结束原因。

完整 RGB-D 测试另发现上游 STOP 分支只返回 pose/metrics，不渲染相机；原绑定
读取 `depth` 会抛 KeyError，将正常终止误记为基础设施错误。现终止时若缺传感器，
通过上游零时间增量渲染补采新帧，不新增物理动作或 metric update，也不复用旧图像。
非终止观测缺少所需传感器仍明确报错。原失败产物保留在
`runs/vlnverse-repair-20261007/internvla-12/20261007T055819Z-aa9df18b/`；
修复后的运行使用新目录，不覆盖失败证据。

Isaac 相机校准使用真实光轴方向计算初始俯仰，`camera_pitch_deg: 0` 在站立时
接近水平；实际世界俯仰会随机器人步态变化。默认不指定时保留 USD 的 −30° 安装。
真实 RGB/depth、上下俯仰恢复、前进和转向已通过 renderer/pose 探针验证，图像位于
证据目录的 `sensor-probe*`。物理控制名义前进 0.25 m，但启动阶段实测位移较短，
转向存在残余平移；该偏差影响导航，行为以闭环运行验证为准。

flash 模式下，相机俯仰原先借用 `stand_still` 刷新画面，会在 flash 刚重置关节后
额外调用物理步态控制器。原 flash InternVLA 前三条均在 tilt 期间跌倒；现 flash
探测只旋转相机并零时间渲染，不切入物理站立控制，仍计入有限公共控制预算。
describe/record 的 `camera_tilt_advances_physics` 明确标识两种模式差异。
物理模式保留原行为；flash primitive 的碰撞检查不变。
真实探针 `sensor-probe-flash-fixed/probe.json` 已验证：起始、向下 15°、恢复三个
观测的底座位置/姿态及上游任务步计数相同，而向下后的 RGB SHA256 改变；随后
STOP 成功返回 RGB-D。旧 flash 诊断保留在 `internvla-flash-12/`，修复后单独重跑。
修复后的固定 12 条完整预算评测已全部完成：无跌倒、无策略/基础设施错误，
SR/SPL 为 58.33%/51.04%；预算耗尽与成功 STOP 的区别见上表及其说明。

外部插件继续使用现有 Registry 和 manifest：

- `configure --plugin-dir PATH` 与实验 `plugin_dirs` 发现同一个外部 bundle，
  精确保留插件 ID，仅保留明确的 `uni-navid` 历史别名。
- 本机资源目录允许保存未加载的外部插件；解析/冻结实验时只带入选定 method 和
  simulator。无关插件的资产设置不再改变当前计划或导致未知插件错误。
- `requires.paths` 与 `resource_paths` 统一决定路径注入及规范化，去掉方法 manifest
  的重复 `resource_settings` 路径列表；旧版非路径 `resource_settings` 仍兼容。
- 外部 bundle 已通过注册、解析、独立 Python worker 与完成 episode 的测试；
  宿主安装配置在运行后保持不变。

安装资源与实验语义分离；模型与仿真 worker 保持独立解释器，避免 PyTorch、
Transformers、Habitat、Isaac 的依赖冲突；策略可见观测与评测真值保持分离。
沿用既有工具与边界，未引入新的插件管理器或依赖。

## 回归验证

`python -B -m unittest discover -s tests -q`：103 项中 94 项通过、9 项按环境
条件跳过。Habitat 0.2.4 与 0.3.0 各自解释器的俯仰配置测试各 4 项通过；
11 种方法均通过 `tests/check_method_runtime.py` 的隔离导入检查，禁止借用外部
方法 checkout 或导入仿真器。外部插件测试实际启动独立 Python worker。
全部 5 组运行的产物哈希、逐条动作计数、固定 cohort 与离线指标均已重新核验，
结果全部通过。

## 性能结论的限制与后续优化

当前结果证明闭环与评分正常工作，尚不能证明论文级“正常性能”。固定 12 条只占
fine-unseen 的小部分，且物理 H1 会跌倒。安装的官方参考来自
`sihaoevery/vlnverse_emr` 的 `vlnverse` revision（源码归档 SHA256
`fe11dbf202f190c5ad5ef9eaa258a9f5d14f5f9a2bafb4b2f4a33081cafe6e24`）。
其 InternVLA fine-unseen 配置使用 flash + continuous trajectory，本仓库当前方法
适配器输出离散动作，且上游配置预热为 500 步、当前绑定为 30 步，不能将两者
原始分数直接视为性能对等。预热对物理 H1 稳定性的影响尚未单独量化。

公开数字也需区分数据集和版本：[VLNVerse 论文表 4](https://arxiv.org/html/2512.19021v1#S5.T4)
中的 InternNav-N1 是 fine **test** SR 28.95%、SPL 25.00%，不是当前 12 条
fine-val-unseen 的相同协议基线；该表没有 StreamVLN。
[当前 InternNav 官方 README](https://github.com/InternRobotics/InternNav#-benchmark-results)
列出的 NavDP* 双系统 R2R-CE 为 SR 64.1%、SPL 58.1%，而本机 9 条 R2R 只是
功能与性能诊断，checkpoint 版本及全 split 尚未对齐。不能据这些不同设置的数字
宣称已经复现官方成绩。

实验选项 `robot_flash: true` 复用上游离散 flash 控制器，并强制保留碰撞检查。
它与物理步态、以及上游连续轨迹协议分别报告。此次同路线 StreamVLN 对照中，
SR 从 33.33% 变为 58.33%，但仍有 4 条碰到导航预算上限。初始前进探针曾被
碰撞检查拒绝；碰撞检查保持开启，全部选定路线计入评分。
相机零时间渲染不移动机器人底座或俯视碰撞相机；下一次 primitive
仍使用对应当前位姿的上游碰撞图。StreamVLN flash 日志已有 2221 次 collision
abort，`kujiale_0030_3_1` 的 358 次前进中 357 次没有位移，说明重复请求被阻挡的
前进也是预算耗尽的重要原因。上游像素区域碰撞判定可能偏保守；本仓库保留其
协议，未据低分修改碰撞阈值或添加策略逃逸动作。
上游名为 nDTW 的此实现实际是参考路径邻近度平均，短轨迹也可能很高；不将它
单独作为导航达标标准。NaVIDA 等方法更深层的算法忠实度对齐暂未处理。

后续优化按优先级排列：

1. 先匹配官方方法的相机、本体、控制模式、checkpoint 与预算，建立同路线基线，
   再扩大场景/全 split；当前分数不替代这一步。
2. 将源码身份从全仓散列收敛到选定插件显式声明的传递依赖，减少无关源码变化
   对 `resume` 的影响，同时保留共享依赖变更的失效检查。
3. 从原子 attempt 文件增量刷新汇总；当前 `collection.json` 在运行中不是实时
   成绩。沿用现有记录源即可，不需要增加第二套状态管理。
4. 主进程意外退出后及时回收其 worker，防止空闲模型和仿真器继续占用 GPU；
   中断后可用原生 `resume` 续跑，已完成 attempt 保留源码、资源与运行时身份检查。
5. 仅对确认逐字节相同的运行时代码做共享提取，并将回归测试纳入版本管理；
   避免为减少文件数量而合并依赖冲突的模型/仿真环境。
