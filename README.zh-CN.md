<h1 align="center">agent-guard</h1>

<p align="center"><b>别让你的 agent 一夜烧掉 400 美元</b><br>
给 AI agent 的预算上限、死循环检测与熔断<br>
零依赖 · 不做代理 · 数据不出你的进程</p>

<p align="center"><img src="https://raw.githubusercontent.com/yaoyuxiang-gnn/agent-guard/main/docs/demo.svg" width="600" alt="终端里运行 examples/basic.py：预算 0.05 美元的 agent 在第三次调用时被拦下，报告显示预算条已达 114%"></p>

一个不再有进展的 agent 并不会停下来。它用同样的参数反复调用同一个工具，或者在两个工具之间来回弹跳，每一轮都重新买一遍完整的上下文。`agent-guard` 就是你的循环里负责**发现这件事**的那部分——外加一张告诉你花了多少钱的账单。

```bash
pip install agent-budget-guard-py
```

```python
from agentguard import Guard

guard = Guard(max_usd=1.00, max_steps=25)

with guard:
    while True:
        with guard.step() as step:
            response = client.chat.completions.create(...)
            step.record(response)                     # 自动提取 token 与花费
            with step.tool("search", {"q": query}):   # 生成指纹，用于死循环检测
                results = search(query)

print(guard.report())
```

三件事会自动发生：`step.record(response)` 从**任何** SDK 响应里取出 token 数和模型名；`step.tool(...)` 给这次调用生成指纹，于是重复调用会在**第 4 次真正执行之前**被拦下；`guard.report()` 打印账单。Python 3.10+，无运行时依赖——连 provider SDK 都不需要。

> **快速跳转：** [看它跑起来](#看它跑起来) · [它能拦下什么](#它能拦下什么) ·
> [接进你的技术栈](#接进你的技术栈) · [它不认识的模型](#它不认识的模型) ·
> [放进-ci](#放进-ci) · [诚实的回答](#诚实的回答) ·
> [全部细节](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/docs/DETAILS.md)

---

## 问题在哪

一次**调用之后**才做的预算检查，只能告诉你已经花了多少。等上限发现 agent 卡住了，钱已经花掉了——而真正的故障通常不是崩溃，是一个**看起来很忙**的循环。

有两样东西能拦住它，而它们拦的不是同一件事：

| | 拦下什么 | 什么时候触发 |
|---|---|---|
| **天花板** — `max_usd`、`max_tokens`、`max_steps`、`max_seconds` | 一直往上涨的花费 | 越过它的那次调用**之后** |
| **死循环检测器** — 四个，默认开启 | 在重复自己的 agent | 重复**真正执行之前** |

`agent-guard` 两样都做，跑在你的进程里，只用标准库。没有代理、没有服务端、没有账号，什么都不往外发。

## 看它跑起来

一个研究型 agent 把同一个问题问了三遍。它在第三次一模一样的 `search_web` 调用处停下了——不是因为预算耗尽，而是因为这次调用和前面两次**完全相同**。下面这一次运行就是整个库：

```python
guard = Guard(max_usd=5.00, name="research-agent")
with guard.step(tag="search") as step:
    step.record("gpt-4o", input_tokens=8_000, output_tokens=400)
    with step.tool("search_web", {"query": "weather in oslo"}):
        results = search("weather in oslo")        # 第 3 次：在这里被拦下
```

```text
turn 1: searched, spent $0.0240
turn 2: searched, spent $0.0480

LoopDetected: Loop detected [repeat]: the same call appeared 3 times in the
last 3 steps: search_web({"query":"weather in oslo"}).
```

```text
agentguard  research-agent
================================================================
  wall time   0ms             steps     3
  llm calls   3               tokens    25,200  (in 24,000 / out 1,200)

  limits
    budget   $0.072 / $5                   1.4%  [................]

  by model
    gpt-4o   3 calls      $0.072      24,000 in / 1,200 out

  tripped: loop [repeat] the same call appeared 3 times in the last 3 steps: ...
```

整个思路就是这么简单，而它决定了这是「0.07 美元的失误」还是「一整夜的失误」。上面两段输出都是真实的——异常那一行是为了排版折了行，报告同样做了截断。有两个示例可以离线跑出同一套机制：

```bash
python examples/loop_detection.py   # 四个检测器，外加一次必须不被拦下的健康运行
python examples/basic.py            # 换成预算上限：在 $0.057 / $0.05 处被拦下
```

## 它能拦下什么

每个上限都是可选的、互相独立的。一个不设任何上限的 guard 依然会检测死循环，也依然会生成报告。

| 上限 | 何时触发 |
|---|---|
| `max_usd=1.00` | 已知花费超过 1 美元 |
| `max_tokens=500_000` | 输入 + 输出 token 超过额度 |
| `max_steps=25` | 第 26 次 `guard.step()` 被打开 |
| `max_seconds=300` | 自 guard 创建以来的墙上时钟时间 |

四个检测器，以及每一个到底是干什么用的：

| 检测器 | 抓什么 | 第几次触发 |
|---|---|---|
| `RepeatDetector` | 同一个调用、参数完全相同 | 第 3 次完全相同的调用 |
| `CycleDetector` | `A, B, A, B`——两个工具来回弹跳 | 第 2 个完整来回 |
| `SimilarityDetector` | 换了个说法的重复：`search("python asyncio")` → `search("python asyncio ")` | 第 4 次近似相同的调用 |
| `NoProgressDetector` | 一个永远不动的进度标记 | 第 6 次未变化的标记 |

每一条判定都是一句人话，因为「你为什么杀了我的 agent」是所有人问的第一个问题。下面是 `examples/loop_detection.py` 的真实输出：

```text
exact repeat                     -> repeat
                                    the same call appeared 3 times in the last 3 steps: search_web({"query":"weather in oslo"})
two-step ping-pong               -> cycle
                                    a 2-step pattern repeated 2 times: read_file({"path":"app.py"}) -> write_file(...)
paraphrased calls                -> similarity
                                    4 near-identical calls (>= 95% similar) in the last 4 steps: search_web(...)
no progress                      -> no-progress
                                    the progress marker did not change for 6 consecutive observations: {"rows_written":0}

healthy varied work              -> clean, as it should be
```

最后一行和前面几行同样重要：**一个会在健康工作上误报的检测器，就是一个会被你关掉的检测器。** 每个阈值都被调到「自己的场景够得着、别的场景碰不到」，而且每一个都可以按 guard 单独调整。

## 接进你的技术栈

三种深度，只有第一种是必须的。

**1. 手动** — 什么框架都能用，包括你手写的 `while` 循环：

```python
guard = Guard(max_usd=1.0)
guard.record("gpt-4o", input_tokens=1200, output_tokens=300)
guard.check()
```

**2. 结构化** — 每次迭代一个 step，工具指纹自动生成。就是上面那段快速开始。

**3. 包裹客户端** — 一行调用点都不用改，每次调用都被记账：

```python
from openai import OpenAI
from agentguard.adapters.openai import guard_openai

client = guard_openai(OpenAI(), max_usd=1.0, max_steps=25)
response = client.chat.completions.create(...)   # 自动记账
```

适配器是纯鸭子类型——它们从不 import provider SDK——所以同一个包裹器覆盖 **Anthropic、LiteLLM、OpenRouter、vLLM、Together、Groq 和 Azure OpenAI**：

```python
from agentguard.adapters.anthropic import guard_anthropic

client = guard_anthropic(Anthropic(), max_usd=2.0)
```

**流式没问题。** 传 `stream=True` 时响应会被包起来，chunk 原样透传，用量在流被消费完时**恰好记录一次**。Anthropic 的 `messages.stream()` 和异步客户端（`async for`）用同样的方式覆盖。

**LangGraph** 只需要一个回调处理器：

```python
from agentguard.integrations.langgraph import guard_langgraph

handler = guard_langgraph(Guard(max_usd=1.0, max_steps=25))
graph.invoke(inputs, config={"callbacks": [handler]})
```

**或者用装饰器**，适合请求处理器而不是循环：

```python
from agentguard import current_guard, guarded

@guarded(max_usd=0.50, max_steps=20)
def summarise(url: str) -> str:
    guard = current_guard()
    ...
```

`@guarded(max_usd=...)` 每次调用创建一个**全新的 guard**——这是请求处理器的正确默认值：一个调用方把预算用光，不该让下一个调用方也停下来。当花费需要跨调用累积时，传 `guard=` 给一个共享的 guard。

**在付钱之前拒绝一次调用。** 事后检查只能报告超支；`preflight()` 会拒绝一次「最坏情况塞不进剩余额度」的调用：

```python
client = guard_openai(OpenAI(), max_usd=0.05, preflight=True)
# BudgetExceeded: refused before spending: a gpt-4o call could reach $0.6,
# over the $0.05 limit (already spent $0)
```

**重启时不重置预算。** 一个会给自己打检查点的 agent，也可以给花费打检查点，这样重启不会白送它一份新预算：

```python
write_checkpoint({"cursor": 41, "guard": guard.snapshot()})

# 之后，在新进程里
guard = Guard.from_snapshot(read_checkpoint()["guard"], max_usd=5.0)
guard.remaining_usd      # 真正还剩多少，而不是整份额度
```

有三个决定值得知道，细节都在[详细文档](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/docs/DETAILS.md#checkpointing)里：检测器的滑动窗口**会**跨检查点存活（跨检查点的循环依然是循环）；墙上时钟时间**不会**（`max_seconds` 限制的是**当前进程**，恢复已耗时会让你为没花过的时间被拦下）；读不精确的检查点会被**拒绝**，而不是应用一半。

## 它不认识的模型

内置表里有 **119 个模型**——OpenAI、Anthropic、Google、xAI、DeepSeek、Mistral 和 Qwen 的当代系列，`PRICING_AS_OF` 标注为 `2026-09`。但它不可能知道你的微调模型、你网关的别名、或你谈下来的价格：

```bash
agentguard config set my-finetune-v3 3 12 --cached 0.3
agentguard config alias acme/fast claude-3-5-haiku
agentguard config disable gpt-4          # 不信任这个内置价格
```

**它从没听说过的模型，绝不会按猜出来的价格计费。** 这种调用会被记为 *unpriced*，排除在预算计算之外，并大声报出来——因为一个默默假设 `$0.00` 的安全工具，比没有安全工具更糟：

```text
  by model
    gpt-4o            1 call       $2.52   1,000,000 in / 2,000 out
    acme-rerank-v3    1 call    unpriced      40,000 in / 0 out

  ! 1 call(s) had no known price and are excluded from the budget:
      acme-rerank-v3
    Price them with `agentguard config set <model> <input> <output>`,
    or pass Guard(pricing={...}) in code.
```

被 `disable` 掉的模型同样变成 unpriced，于是被排除的花费是**看得见的**，而不是悄悄按一个你拒绝过的数字计费。所有东西都能在代码里覆盖，而代码永远优先于文件。

## 放进 CI

worker 里 `guard.save()`，CI 里 `agentguard report`——读取端什么都不用装：

```bash
agentguard report run.json           # 渲染 guard.save(...) 存下的报告
agentguard report run.json --json
agentguard pricing                   # 生效中的价格表，每个模型带来源
agentguard pricing gpt-4o
agentguard config path               # 配置从哪读、忽略了什么
```

## 诚实的回答

**零依赖，而且会一直保持。** 没有 `pydantic`、没有 `httpx`、没有 provider SDK——只有标准库。一旦 wheel 里出现运行时依赖，CI 会直接让构建失败。所以它可以被塞进一个 vendor 了自己依赖的技术栈、钉在旧版 Python 上、或者丢进 Lambda，都不需要动 lockfile。

**价格表会过期。** 这一份是有日期的，而 provider 可能在你更新完的第二天就改价或下线某个模型。已退役的模型保留最后一次公布的价格，而不是被删掉——因为删掉一个名字会悄悄让它所有调用变成 unpriced。任何你要拿去出账的数字都该自己核实，任何你依赖的模型都该自己配置。那个诚实的解法——可选、带校验和的 `pricing --update`——**还没做**，它是 [roadmap](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/ROADMAP.md) 上的第一项。

**它不是什么。** 不是可观测性平台（什么都不往外发，没有服务端、没有后台线程）；不是代理（它看不到没人告诉它的流量）；不是分词器（预检的输入估算只是启发式）；也不能替代 provider 侧的消费限额——两个都要用。agent-guard 拦住**你的**循环；provider 的限额才是在你进程死掉、请求还在飞的时候救你的那一道。

**局限，直说。** 不报告用量的响应无法计价：它会警告一次，记为 unpriced，而不是编一个数字出来。流是在被消费完时计价的，所以 OpenAI 兼容的流请传 `stream_options={"include_usage": True}`。预检的输入 token 数是从序列化后的 prompt 估算的。

## 文档

| | |
|---|---|
| [docs/DETAILS.md](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/docs/DETAILS.md) | 全部内容：每个检测器及其调参、完整的价格配置与信任模型、检查点格式、完整 CLI 参考、设计原则 |
| [examples/](https://github.com/yaoyuxiang-gnn/agent-guard/tree/main/examples) | 八个可直接运行的程序——预算上限、四个检测器、包裹客户端、流式、LangGraph、自定义模型、检查点，以及一份真实报告 |
| [CHANGELOG.md](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/CHANGELOG.md) | 版本历史 |
| [ROADMAP.md](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/ROADMAP.md) | 后续计划 |
| [CONTRIBUTING.md](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/CONTRIBUTING.md) | 这个库遵守的四条约束 |

包里每一个 docstring 示例都会作为测试运行，所以文档不可能和行为脱节。`python -m unittest discover -s tests -t .` 跑完整套件——563 个测试，不联网，不需要下载任何 fixture。

## 许可证

MIT — 见 [LICENSE](https://github.com/yaoyuxiang-gnn/agent-guard/blob/main/LICENSE)。
