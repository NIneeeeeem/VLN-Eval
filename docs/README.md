# Nav-Eval Documentation / 项目文档

[English](#english) | [简体中文](#简体中文)

## English

Find docs by question:

**I want to understand how the project is organized**
→ [Architecture](architecture.md): the single execution pipeline, plugin boundaries,
protocol and authenticity, deployment and storage model.

**I want to run an evaluation**
→ [Deployment](deployment.md): resource maps, real runs, parallel inference, shared
models, Docker, resume and throughput conventions.
For a first run see the [Quickstart](../README.md#quickstart).

**I want to plug in my own component**
→ [Plugin integration guide](integration.md):

| I want to add a… | Guide |
|---|---|
| Method (model) | [Add a method](integration.md#add-a-method) |
| Simulator | [Add a simulator](integration.md#add-a-simulator) |
| Benchmark | [Add a benchmark](integration.md#add-a-benchmark) |
| Metric | [Add a metric](integration.md#add-a-metric) |
| Controller | [Add a controller](integration.md#add-a-controller) |

**I want the details of worker interactions**
→ [Protocol](protocol.md): wire envelope, session lifecycle, public observations and
time, action and failure semantics, lossless tensor transport.

## 简体中文

按问题查找文档：

**我想了解这个项目是怎么组织的**
→ [架构](architecture.zh-CN.md)：单一执行链路、插件边界、协议与真实性、部署与存储模型。

**我想跑一次评测**
→ [部署](deployment.zh-CN.md)：资源映射、真实运行、并行推理、共享模型、Docker、恢复与吞吐口径。
快速上手见仓库根目录 [README](../README.zh-CN.md#快速开始)。

**我想接入自己的东西**
→ [插件接入指南](integration.zh-CN.md)：

| 我想添加… | 指南 |
|---|---|
| 方法（model） | [添加方法](integration.zh-CN.md#添加方法) |
| 仿真器 | [添加仿真器](integration.zh-CN.md#添加仿真器) |
| Benchmark | [添加 Benchmark](integration.zh-CN.md#添加-benchmark) |
| 指标 | [添加指标](integration.zh-CN.md#添加指标) |
| 控制器 | [添加控制器](integration.zh-CN.md#添加控制器) |

**我想了解 worker 之间的交互细节**
→ [协议](protocol.zh-CN.md)：wire envelope、会话生命周期、公开观测与时间、动作与失败语义、无损张量传输。

---

## All documents / 全部文档

| Doc / 文档 | English | 简体中文 |
|---|---|---|
| Architecture / 架构 | [architecture.md](architecture.md) | [architecture.zh-CN.md](architecture.zh-CN.md) |
| Plugin integration / 插件接入 | [integration.md](integration.md) | [integration.zh-CN.md](integration.zh-CN.md) |
| Deployment / 部署 | [deployment.md](deployment.md) | [deployment.zh-CN.md](deployment.zh-CN.md) |
| Protocol / 协议 | [protocol.md](protocol.md) | [protocol.zh-CN.md](protocol.zh-CN.md) |
