# 音乐对话式推荐挑战赛 — 基线系统

**RecSys Challenge 2026 对话式音乐推荐系统挑战赛**官方评估框架。Music-CRS 聚焦于音乐发现领域的新趋势：静态推荐列表正在被动态的、对话式的交互所取代。随着用户越来越多地通过自然语言与 AI 互动，迫切需要能够将自然语言理解（NLU）与高精度推荐系统（RecSys）无缝集成的系统。本挑战赛旨在推动 AI 理解用户细腻偏好、通过对话探索音乐品味并提供上下文相关曲目推荐的能力边界。

本仓库提供标准化工具，用于在 **TalkPlay Data Challenge** 数据集上评估音乐推荐系统。参赛者必须遵循下面指定的严格推理 JSON 格式，以确保提交结果能够被正确评估。

- **ACM RecSys 官网**: [https://www.recsyschallenge.com/](https://www.recsyschallenge.com/)
- **挑战赛官网**: [https://nlp4musa.github.io/music-crs-challenge/](https://nlp4musa.github.io/music-crs-challenge/)
- **挑战赛数据集**: [talkpl-ai/talkplay-data-challenge](https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge)

---

## 时间线

| 日期 | 里程碑 |
|------|---------|
| 2026年3月31日 | 网站上线 |
| 2026年4月10日 | 挑战赛启动 — 发布数据集（训练集、开发集、盲测集A） |
| 2026年4月15日 | 提交系统开放 — 排行榜上线（使用盲测集A） |
| 2026年6月15日 | 盲测集B发布，激活盲测集B的提交系统 |
| 2026年6月30日 | 挑战赛结束 |
| 2026年7月6日 | 最终排行榜与获奖者 — EasyChair 开放论文提交 |
| 2026年7月9日 | 上传最终预测代码 |
| 2026年7月20日 | 论文投稿截止 |
| 2026年8月3日 | 论文录用通知 |
| 2026年8月10日 | 终稿提交 |
| 2026年9月 | RecSys Challenge Workshop 在 ACM RecSys 2026 举办 |

---

## 整体架构

### 两阶段流水线

系统采用**两阶段流水线**处理每一轮对话：

```
用户查询 + 对话历史
        │
        ▼
┌─────────────────────────────────┐
│  阶段1：检索（推荐系统）          │
│  BM25 或 BERT 检索器            │
│  搜索全部曲库                    │
└──────────────┬──────────────────┘
               │ top-20 曲目ID
               ▼
┌─────────────────────────────────┐
│  元数据查找                      │
│  MusicCatalogDB                 │
│  ID → 人类可读的字符串           │
└──────────────┬──────────────────┘
               │ top-1 曲目元数据字符串
               ▼
┌─────────────────────────────────┐
│  阶段2：LLM 回复生成             │
│  Llama-3.2-1B-Instruct          │
│  生成自然语言回复                │
└──────────────┬──────────────────┘
               │
               ▼
  {
    "retrieval_items": [top-20 曲目ID],
    "recommend_item": "track_id: ..., track_name: ...",
    "response": "根据你的心情，我推荐..."
  }
```

### 核心组件

| 组件 | 文件 | 描述 |
|---|---|---|
| **流水线编排器** | `mcrs/crs_baseline.py` | 将检索与LLM串联，管理会话记忆 |
| **LLM** | `mcrs/lm_modules/llama.py` | Llama-3.2-1B-Instruct 用于自然语言回复生成 |
| **稀疏检索** | `mcrs/retrieval_modules/bm25.py` | BM25 词汇检索（关键词匹配） |
| **稠密检索** | `mcrs/retrieval_modules/bert.py` | BERT 嵌入检索（语义匹配） |
| **曲目数据库** | `mcrs/db_item/music_catalog.py` | 曲目元数据：ID → 格式化字符串 |
| **用户数据库** | `mcrs/db_user/user_profile.py` | 用户画像：ID → 人口统计字符串 |
| **系统提示词** | `mcrs/system_prompts/` | 3个控制LLM行为的提示词模板 |
| **配置文件** | `config/` | 不同实验设置的YAML文件 |
| **推理脚本** | `run_inference_*.py` | 批量推理入口 |
| **下界基线** | `lowerbound/` | 流行度和随机基线 |
| **改进建议** | `tips/` | 改进基线系统的扩展指南 |

### 数据来源（Hugging Face 数据集）

| 数据集 | 用途 | URL |
|---|---|---|
| TalkPlayData-Challenge-Dataset | 对话数据（训练/测试划分） | `talkpl-ai/TalkPlayData-Challenge-Dataset` |
| TalkPlayData-Challenge-Track-Metadata | 曲目信息（名称、艺人、专辑、标签、发行日期） | `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` |
| TalkPlayData-Challenge-User-Metadata | 用户画像（年龄段、性别、国家） | `talkpl-ai/TalkPlayData-Challenge-User-Metadata` |
| TalkPlayData-Challenge-Blind-A | 盲测集A（无标签） | `talkpl-ai/TalkPlayData-Challenge-Blind-A` |

---

## 逐文件详细解释

---

### 项目根目录文件

#### `pyproject.toml` — 项目配置与依赖

这是 Python 项目描述文件，使用 setuptools 作为构建后端。

- **`[build-system]`**: 使用 `setuptools>=61.0`，构建后端为 `setuptools.build_meta`。
- **`[project]`**:
  - `name = "mcrs"` — 可安装的包名
  - `requires-python = ">=3.8"` — Python 版本要求
  - **依赖包**:
    | 包名 | 版本 | 用途 |
    |---|---|---|
    | `bm25s` | `>=0.2.13` | 带磁盘缓存的快速BM25稀疏检索 |
    | `datasets` | `>=3.1.0` | Hugging Face datasets 库，用于加载对话/元数据 |
    | `omegaconf` | `>=2.3.0` | 支持变量插值的YAML配置加载器 |
    | `psutil` | `>=7.2.2` | 系统资源监控工具 |
    | `torch` | `>=2.5.1` | PyTorch 深度学习框架 |
    | `torchaudio` | `>=2.5.1` | PyTorch 音频 I/O 与变换 |
    | `torchvision` | `>=0.20.1` | PyTorch 视觉工具 |
    | `transformers` | `>=4.46.3` | Hugging Face Transformers（Llama、BERT、分词器） |
- **`[tool.setuptools]`**: 声明 `mcrs` 为唯一要安装的包。运行 `pip install -e .` 后，`mcrs` 即可作为 Python 包导入。

#### `.gitignore`

从版本控制中排除：
- `.venv` — 虚拟环境目录
- `cache/` — 本地模型和检索索引缓存
- `__pycache__/` — 编译后的 Python 字节码
- `*.egg-info/` — setuptools 构建元数据
- `uv.lock` — `uv` 包管理器的锁文件
- `inference/` — 推理输出目录

---

### 核心包: `mcrs/`

#### `mcrs/__init__.py` — 包入口与工厂函数

导出唯一的函数 `load_crs_baseline()`，作为创建完整配置的CRS流水线的**工厂函数**。它接受11个参数，全部具有合理的默认值：

```python
def load_crs_baseline(
    lm_type="meta-llama/Llama-3.2-1B-Instruct",
    retrieval_type="bm25",
    item_db_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    user_db_name="talkpl-ai/TalkPlayData-Challenge-User-Metadata",
    track_split_types=["all_tracks"],
    user_split_types=["all_users"],
    corpus_types=["track_name", "artist_name", "album_name"],
    cache_dir="./cache",
    device="cuda",
    attn_implementation="eager",
    dtype=torch.bfloat16
)
```

它简单地实例化并返回一个带有所有给定参数的 `CRS_BASELINE` 对象。推理脚本调用的就是这个函数。

---

#### `mcrs/crs_baseline.py` — 核心流水线编排器

这是**整个仓库最重要的文件**。`CRS_BASELINE` 类是中央编排器，将所有组件（检索、LLM、数据库、提示词）组合成一个可工作的对话式推荐系统。

**`__init__()` — 初始化**（第30-77行）:

1. 将所有配置参数存储为实例属性（`self.lm_type`、`self.retrieval_type` 等）
2. **加载LLM模块** 通过 `load_lm_module()` — 下载并初始化 Llama-3.2-1B-Instruct
3. **加载检索模块** 通过 `load_retrieval_module()` — 下载曲目元数据并构建/加载搜索索引
4. **创建 `MusicCatalogDB`** — 用于将曲目ID转换为人类可读的元数据字符串
5. **创建 `UserProfileDB`** — 用于查询用户人口统计信息
6. **加载3个系统提示词模板** 从 `mcrs/system_prompts/` 到 `self.role_prompt` 字典中：
   - `"role_play"` → `roleplay.txt`: 设定AI角色
   - `"personalization"` → `personalization.txt`: 使用用户画像的指令
   - `"response_generation"` → `response_generation.txt`: 格式化回复的规则
7. 初始化空的 `self.session_memory = []` 用于跟踪对话状态

**`_get_system_prompt(user_id)` — 系统提示词组装**（第89-100行）:

构建完整的系统提示词字符串：
1. 始终以 `role_play` + `response_generation` 提示词开头
2. 如果提供了 `user_id`，则追加 `personalization` 提示词，后跟用户的格式化画像字符串（例如 `user_id: USR001\nage_group: 25-34\ngender: female\ncountry_name: united states`）
3. 返回完整的系统提示词

这意味着个性化是**可选的** — 如果没有给出 user_id，系统将在没有用户特定上下文的情况下运行。

**`chat(user_query, user_id)` — 单轮对话**（第102-130行）:

执行CRS流水线一个完整轮次的核心方法：

```
步骤0: 将用户消息追加到 session_memory
步骤1: 构建系统提示词（可选个性化）
步骤2: 将所有 session_memory 拼接为 "role: content" 格式 → 检索查询
步骤3: 调用 self.retrieval.text_to_item_retrieval() → top-20 曲目ID
步骤4: 对 top-1 曲目调用 self.item_db.id_to_metadata() → 格式化的元数据字符串
步骤5: 调用 self.lm.response_generation() → 自然语言回复
步骤6: 返回包含 user_id、user_query、retrieval_items、recommend_item、response 的字典
```

**`batch_chat(batch_data)` — 批量推理**（第132-191行）:

为提高效率同时处理多个对话轮次：

1. **准备阶段**: 对于批次中的每个项目，复制会话记忆，追加用户查询，构建系统提示词，并格式化检索输入字符串。返回3个并行列表：`sys_prompts`、`retrieval_inputs`、`session_memories`。
2. **阶段1 — 批量检索**: 检查检索模块是否有 `batch_text_to_item_retrieval()` 方法；如果有则使用，否则退化为顺序调用。
3. **元数据查找**: 对每个结果的 top-1 曲目，调用 `self.item_db.id_to_metadata()`。
4. **阶段2 — 批量生成**: 检查LLM是否有 `batch_response_generation()` 方法；如果有则使用，否则退化为顺序调用。
5. **结果组装**: 构建与单轮格式匹配的结果字典列表。

这种使用 `hasattr` 检查的设计意味着自定义检索/LLM模块只需要实现批量方法即可自动受益于更快的批量推理。

**`_reset_session_memory()`**（第79-82行）: 清除内存中的对话历史。在切换独立会话时有用。

**`_upload_session_memory(chat_history)`**（第84-87行）: 用提供的聊天历史替换当前会话记忆。推理脚本使用此方法从数据集设置对话上下文。

---

### 检索模块: `mcrs/retrieval_modules/`

#### `mcrs/retrieval_modules/__init__.py` — 检索工厂

`load_retrieval_module()` 函数根据 `retrieval_type` 字符串分派到相应的检索器类：

| `retrieval_type` | 类 | 描述 |
|---|---|---|
| `"bm25"` | `BM25_MODEL` | 稀疏词汇检索（关键词匹配） |
| `"bert"` | `BERT_MODEL` | 稠密语义检索（含义匹配） |
| 其他任何值 | `ValueError` | 不支持的类型错误 |

要添加新的检索器（例如 ColBERT），需要在此处添加另一个 `elif` 分支并创建相应的模块文件。

---

#### `mcrs/retrieval_modules/bm25.py` — BM25 稀疏词汇检索器

使用 `bm25s` 库实现 **BM25**（Best Match 25）检索。BM25 是一种基于词频（TF）和逆文档频率（IDF）的"词袋"排序函数。它匹配查询与文档之间的精确词重叠。

**`BM25_MODEL` 类:**

**`__init__(dataset_name, split_types, corpus_types, cache_dir)`**（第17-40行）:
1. 通过用下划线连接语料库类型来构建 `corpus_name`（例如 `"track_name_artist_name_album_name_release_date"`）
2. 通过 `_load_corpus()` 从 Hugging Face 加载曲目元数据语料库
3. 检查 `cache/bm25/{corpus_name}/` 是否存在缓存的 BM25 索引
4. 如果已缓存：通过 `_load_bm25()` 加载
5. 如果未缓存：先调用 `build_index()` 创建索引，然后加载

**`_load_corpus()`**（第53-61行）:
- 从 Hugging Face 加载数据集
- 拼接指定的划分（例如 `["all_tracks"]`）
- 返回 `dict[track_id → metadata_dict]` 映射

**`_stringify_metadata(metadata)`**（第63-76行）:
- 将曲目的元数据字典转换为用于索引的多行字符串
- 对于 `corpus_types` 中的每个字段，格式化 `field_name: value\n`
- 如果值是列表（例如标签），用 `", "` 连接
- 示例输出：
  ```
  track_name: yesterday
  artist_name: the beatles
  album_name: help!
  release_date: 1965
  ```

**`build_index()`**（第78-93行）:
1. 遍历所有曲目，将每个曲目的元数据字符串化
2. 使用 `bm25s.tokenize()` 对语料库进行分词（小写化+单词分割）
3. 创建 `bm25s.BM25` 对象并对分词结果调用 `.index()`
4. 创建目录 `cache/bm25/{corpus_name}/`
5. 通过 `.save()` 将 BM25 索引和语料库保存到磁盘
6. 保存 `track_ids.json`（与索引顺序匹配的有序曲目ID列表）

**`text_to_item_retrieval(query, topk)`**（第95-106行）:
1. 对查询进行分词（小写化）
2. 调用 `bm25_model.retrieve()` 获取 top-k 匹配文档
3. 使用 `self.track_ids` 将返回的文档索引映射回曲目ID
4. 返回曲目ID字符串列表

**`batch_text_to_item_retrieval(queries, topk)`**（第108-122行）:
- 批量版本：一次性对所有查询进行分词，对每个查询进行检索，返回 `List[List[str]]`

**BM25 的关键特点：**
- 构建和查询速度快（仅CPU，不需要GPU）
- 适合精确匹配："jazz piano" 会找到包含这些确切词汇的曲目
- 无法捕捉语义相似性："sad music" 无法匹配 "melancholy songs"
- 索引存储在磁盘上并加载到内存中

---

#### `mcrs/retrieval_modules/bert.py` — BERT 稠密语义检索器

使用 BERT（`bert-base-uncased`）实现**稠密检索**。它不是匹配关键词，而是将查询和曲目元数据都编码为稠密向量嵌入，并计算余弦相似度。

**`BERT_MODEL` 类:**

**`__init__(dataset_name, split_types, corpus_types, cache_dir, model_name, device, batch_size, max_length)`**（第23-68行）:
1. 将索引目录设置为 `cache/bert/{corpus_name}/`
2. 通过 `_load_corpus()` 加载曲目元数据语料库
3. 初始化 BERT 分词器（`AutoTokenizer`）和模型（`AutoModel`）
4. 将模型移至 GPU 并设置为评估模式
5. 检查缓存的嵌入（`embeddings.pt` + `track_ids.json`）
6. 如果已缓存：加载它们；否则：先调用 `build_index()` 然后加载

**`_load_corpus()`**（第79-87行）: 与 BM25 相同 — 从 Hugging Face 加载曲目元数据并创建 `track_id → metadata` 字典。

**`_stringify_metadata(metadata)`**（第89-102行）: 与 BM25 相同 — 将元数据字段转换为用于编码的字符串。

**`_mean_pool(last_hidden_states, attention_mask)`**（第104-115行）:
- 获取 BERT 的最后一层隐藏状态 `[batch, seq_len, hidden]`
- 扩展注意力掩码以匹配隐藏维度
- 计算加权求和的 token 嵌入，除以 token 数量
- 这为每个输入序列生成一个固定大小的向量 `[batch, hidden]`
- `clamp(min=1e-9)` 防止除以零

**`build_index()`**（第117-146行）:
1. 将所有曲目元数据字符串化
2. 按 `batch_size`（默认32）批次迭代
3. 对于每个批次：分词，通过 BERT，对输出进行 mean-pool
4. **对每个嵌入进行 L2 归一化**（`F.normalize(pooled, p=2, dim=1)`）— 这使得余弦相似度等价于点积
5. 将所有批次嵌入拼接为 `[N, hidden]` 矩阵
6. 保存 `embeddings.pt`（嵌入矩阵）和 `track_ids.json`

**`text_to_item_retrieval(query, topk)`**（第148-167行）:
1. 将查询通过 BERT → mean-pool → L2归一化 → `[hidden]` 向量进行编码
2. 计算 `scores = embeddings @ query_emb` — 由于两者都经过L2归一化，这给出了 `[-1, 1]` 范围内的余弦相似度
3. 按分数返回 top-k 曲目ID

**`batch_text_to_item_retrieval(queries, topk)`**（第169-191行）:
- 在一个批次中编码所有查询 → `[batch, hidden]`
- 计算 `scores = embeddings @ query_embs.T` → `[N, batch]`
- 对于每个查询，从对应的列中取 top-k
- 返回 `List[List[str]]`

**BERT 的关键特点：**
- 捕捉语义含义："sad music" ≈ "melancholy songs"
- 构建索引（编码所有曲目）和推理需要GPU
- 索引构建速度慢（每个曲目都要通过BERT编码），但会缓存到磁盘
- 所有曲目的嵌入矩阵必须放入RAM（~10万曲目 × 768维 × 4字节 ≈ 300MB，对于 bert-base）

---

### 语言模型模块: `mcrs/lm_modules/`

#### `mcrs/lm_modules/__init__.py` — LLM 工厂

`load_lm_module()` 函数根据 `lm_type` 字符串进行分派。目前只支持一个模型：

| `lm_type` | 类 | 描述 |
|---|---|---|
| `"meta-llama/Llama-3.2-1B-Instruct"` | `LLAMA_MODEL` | 1B 参数的 Llama 3.2 指令微调模型 |
| 其他任何值 | `ValueError` | 不支持的类型 |

要添加新的LLM（例如 Qwen），在此处添加 `elif` 分支并创建相应的模块。

---

#### `mcrs/lm_modules/llama.py` — Llama-3.2-1B-Instruct 封装器

封装 Hugging Face Transformers，用于加载、格式化和生成 Llama-3.2-1B-Instruct 的输出。

**`LLAMA_MODEL` 类:**

**`__init__(model_name, device, attn_implementation, dtype)`**（第6-13行）:
1. 调用 `_load_model()` 下载/加载分词器和模型
2. 将模型设置为评估模式
3. 将模型移至目标设备并转换为指定的 dtype（默认 `bfloat16`）

**`_load_model()`**（第15-18行）:
- 使用 `padding_side="left"` 加载分词器（对于自回归模型的批量生成至关重要 — 模型从右向左生成，所以填充必须在左侧）
- 使用指定的 `attn_implementation`（`"flash_attention_2"` 用于加速，`"eager"` 用于兼容性）加载模型

**`_format_chat_history(sys_prompt, chat_history, recommend_item)`**（第20-25行）:

这是关键的提示词格式化方法。它按以下结构构建LLM的输入：

```
[system]  ← roleplay + response_generation (+ personalization + 用户画像)
[user]    ← 用户消息1
[assistant] ← 助手回复1（或前一轮的曲目元数据）
[user]    ← 用户消息2
...
[assistant] ← 推荐的曲目元数据（top-1 检索结果）
             ← 生成提示词 token（模型从此处开始生成）
```

关键洞察：推荐的曲目元数据字符串作为 **assistant 消息** 追加在生成提示词之前。这意味着 LLM "看到" 曲目推荐就好像是自己产生的输出一样，然后生成自然语言回复来**解释和语境化**该推荐。这是一种巧妙的方法，使检索结果看起来像是 LLM 自己产生的。

**`response_generation(sys_prompt, chat_history, recommend_item, max_new_tokens=512)`**（第27-35行）:
1. 将聊天历史格式化为模板字符串
2. 对格式化后的字符串进行分词
3. 运行 `lm.generate()` 生成最多 512 个新 token
4. 仅解码新生成的部分（跳过输入token）
5. 返回生成的文本字符串

**`batch_response_generation(sys_prompts, chat_histories, recommend_items, max_new_tokens=64)`**（第37-73行）:
1. 并行格式化所有聊天历史
2. 确保设置了 pad_token（如果未配置则使用 eos_token）
3. 使用 `padding=True` 和 `truncation=True` 进行分批次分词
4. 设置 `pad_token_id` 运行 `lm.generate()`
5. 仅解码每个项目的新生成token
6. 返回 `List[str]`

注意 `max_new_tokens` 的差异：单条生成为 **512**，批量生成为 **64**。这是性能权衡 — 批量回复为了速度保持更短。

**关于左侧填充的重要细节**: 分词器使用 `padding_side="left"` 加载。对于自回归（因果）语言模型，填充token必须在左侧，因为模型从左向右生成。如果填充在右侧，模型需要先生成填充token，会产生无效输出。

---

### 数据库模块

#### `mcrs/db_item/__init__.py`

从 `music_catalog.py` 重新导出 `MusicCatalogDB`。`__all__` 列表控制使用 `from mcrs.db_item import *` 时导出什么。

#### `mcrs/db_item/music_catalog.py` — 曲目元数据库

**`MusicCatalogDB` 类:**

**`__init__(dataset_name, split_types, corpus_types)`**（第7-15行）:
1. 从 Hugging Face 加载曲目元数据集（例如 `talkpl-ai/TalkPlayData-Challenge-Track-Metadata`）
2. 拼接所有指定的划分（例如 `["all_tracks"]`）
3. 构建内存中的字典：`{track_id: metadata_dict}`
4. 存储 `corpus_types` 供后续格式化使用

**`id_to_metadata(track_id, use_semantic_id=False)`**（第17-24行）:

将曲目ID转换为人类可读的元数据字符串。这是**检索系统不透明的曲目ID与LLM自然语言理解之间的桥梁**。

转换示例：
- 输入: `"TRK12345"`
- 输出: `"track_id: TRK12345, track_name: yesterday, artist_name: the beatles, album_name: help!, release_date: 1965"`

该方法：
1. 在 `self.metadata_dict` 中查找 track_id
2. 以 `"track_id: {id}"` 开头
3. 对于 `corpus_types` 中的每个字段，追加 `, {field_name}: {value}`
4. 如果字段值是列表（如标签），用 `", "` 连接
5. 将所有值小写

`use_semantic_id` 参数是为未来生成式检索功能预留的（参见 `tips/use_genrec_semantic_ids.md`）。

此方法在两个地方被调用：
1. **在聊天流水线中**（`crs_baseline.py:121`）: 获取 top-1 推荐曲目的元数据字符串以提供给LLM
2. **在聊天历史解析器中**（`run_inference_*.py:42`）: 将历史中的 `music` 角色消息（仅包含曲目ID）转换为完整元数据字符串，使LLM能够看到自己之前的"推荐"的完整细节

---

#### `mcrs/db_user/__init__.py`

从 `user_profile.py` 重新导出 `UserProfileDB`。

#### `mcrs/db_user/user_profile.py` — 用户画像数据库

**`UserProfileDB` 类:**

**`__init__(dataset_name, split_types)`**（第7-14行）:
1. 从 Hugging Face 加载用户元数据集
2. 拼接指定的划分
3. 定义 `default_columns = ['user_id', 'age_group', 'gender', 'country_name']`
4. 构建内存中的字典：`{user_id: profile_dict}`

**`id_to_profile(user_id)`**（第16-18行）: 返回用户的原始画像字典。

**`id_to_profile_str(user_id)`**（第20-23行）:
将用户的画像转换为系统提示词的格式化字符串：

```
user_id: USR001
age_group: 25-34
gender: female
country_name: united states
```

当启用个性化时，此字符串会被追加到系统提示词中，为LLM提供关于用户的人口统计上下文。

---

### 系统提示词: `mcrs/system_prompts/`

三个纯文本文件，通过提示词工程控制LLM的行为。它们在初始化时由 `CRS_BASELINE` 加载，并拼接形成系统提示词。

#### `roleplay.txt`

```
You are an expert music recommendation assistant. Your task is to understand user preferences and provide personalized music recommendations.
```

这是**角色设定提示词** — 它定义了LLM的身份和核心使命。始终作为系统提示词的第一部分包含在内。

#### `response_generation.txt`

关于LLM如何格式化回复的六条详细指令：

1. **必须基于之前推荐的曲目来回复** — LLM被指示讨论由推荐系统检索到的曲目，而不是自行发明。
2. **如果不匹配，要道歉** — 如果检索到的曲目与用户查询不匹配，LLM应承认问题并将其归因于"推荐系统"（与检索组件分离）。
3. **如果是好的匹配，要热情** — 自信、积极的语气。
4. **分享关键细节** — 标题、艺人、流派、情绪、风格、"显著特点"。
5. **解释匹配原因** — 为什么这首曲目符合用户的请求。
6. **邀请进一步互动** — 保持对话继续。

该提示词本质上是要求LLM充当**推荐解释器** — 它不自己做推荐，而是解释和语境化检索系统找到的内容。

#### `personalization.txt`

```
Consider the following user profile when generating your response. Take into account the user's age group, country, and gender to personalize your recommendations and communication style appropriately.
```

此提示词仅在提供 `user_id` 时包含，实际的用户画像字符串紧随其后追加。它指示LLM根据人口统计信息来调整其沟通风格和推荐。

---

### 配置文件: `config/`

所有四个YAML配置文件共享相同的结构。它们仅在两个维度上有所不同：**检索类型**（bm25 vs bert）和**评估数据集**（devset vs blindset_A）。

#### 配置结构说明

| 字段 | 类型 | 描述 |
|---|---|---|
| `lm_type` | `str` | LLM 的 Hugging Face 模型ID |
| `retrieval_type` | `str` | `"bm25"` 或 `"bert"` |
| `test_dataset_name` | `str` | 用于评估的 Hugging Face 数据集ID |
| `item_db_name` | `str` | 曲目元数据的 Hugging Face 数据集ID |
| `user_db_name` | `str` | 用户画像的 Hugging Face 数据集ID |
| `track_split_types` | `list[str]` | 曲目数据集划分（**必须是 `["all_tracks"]`**） |
| `user_split_types` | `list[str]` | 用户数据集划分 |
| `corpus_types` | `list[str]` | 用于检索索引的元数据字段 |
| `cache_dir` | `str` | 缓存索引的本地目录 |
| `device` | `str` | `"cuda"` 或 `"cpu"` |
| `attn_implementation` | `str` | `"flash_attention_2"` 或 `"eager"` |

#### 四个配置文件

| 文件 | 检索方式 | 测试数据集 |
|---|---|---|
| `llama1b_bm25_devset.yaml` | `bm25` | `talkpl-ai/TalkPlayData-Challenge-Dataset` |
| `llama1b_bert_devset.yaml` | `bert` | `talkpl-ai/TalkPlayData-Challenge-Dataset` |
| `llama1b_bm25_blindset_A.yaml` | `bm25` | `talkpl-ai/TalkPlayData-Challenge-Blind-A` |
| `llama1b_bert_blindset_A.yaml` | `bert` | `talkpl-ai/TalkPlayData-Challenge-Blind-A` |

所有四个都使用：
- `lm_type: "meta-llama/Llama-3.2-1B-Instruct"`
- `track_split_types: ["all_tracks"]` — 这是有效评估的强制要求
- `corpus_types: ["track_name", "artist_name", "album_name", "release_date"]`
- `attn_implementation: "flash_attention_2"` — 需要 `flash-attn` 包
- `device: "cuda"`, `cache_dir: "./cache"`

---

### 推理脚本

#### `run_inference_devset.py` — 开发集推理

在**开发集**（带有真实标签用于评估）上运行推理的主入口。

**`chat_history_parser(conversations, music_crs, target_turn_number)`**（第16-49行）:

将数据集中的原始对话数据解析为CRS流水线期望的格式。

输入：对话轮次列表，如：
```json
[
  {"turn_number": 1, "role": "user", "content": "我想听一些放松的音乐"},
  {"turn_number": 1, "role": "music", "content": "TRK12345"},
  {"turn_number": 2, "role": "user", "content": "来点更欢快的"},
  ...
]
```

对于给定的 `target_turn_number`（例如第3轮）的处理：
1. 筛选 `turn_number < target_turn_number` 的轮次作为历史
2. 对于每个历史轮次：
   - 如果 `role == "music"`: 将角色改为 `"assistant"`，并通过 `music_crs.item_db.id_to_metadata()` 将裸曲目ID转换为完整元数据 — 这使LLM将之前的推荐视为自己的详细输出
   - 否则：保持原始角色和内容
3. 用户查询是 `turn_number == target_turn_number` 时的内容
4. 返回 `(chat_history_list, user_query_string)`

**`main(args)`**（第51-119行）:

完整执行流程：
1. **清除缓存**（`rm -rf cache`）— 确保每次运行构建新索引
2. **加载配置** 从 `config/{args.tid}.yaml` 通过 OmegaConf
3. **初始化 `music_crs`** — 这会触发模型下载、索引构建等
4. **加载测试数据集** 从 Hugging Face
5. **准备批量数据**: 对数据集中的每个会话，创建8个推理项（每个轮次一个，第1-8轮）。对于每个轮次，解析到该轮为止的聊天历史。这会产生 `会话数 × 8` 个推理项。
6. **批量推理循环**: 按 `batch_size` 分块处理 `batch_data`，对每个分块调用 `music_crs.batch_chat()`
7. **收集结果**: 将每个结果映射回其 `session_id`、`user_id`、`turn_number`
8. **保存输出** 到 `exp/inference/devset/{tid}.json`

**输出格式:**
```json
[
  {
    "session_id": "SESS001",
    "user_id": "USR001",
    "turn_number": 1,
    "predicted_track_ids": ["TRK001", "TRK002", ..., "TRK020"],
    "predicted_response": "根据你对放松音乐的偏好，我推荐..."
  },
  ...
]
```

**命令行参数:**
- `--tid`: 匹配配置文件名的工作标识符（默认: `"llama1b_bm25_testset"`）
- `--batch_size`: 推理批大小（默认: 16）
- `--save_path`: 结果保存的基础目录（暂未使用）

---

#### `run_inference_blindset.py` — 盲测集推理（用于提交）

与开发集脚本非常相似，但设计用于**盲测集**，其中真实标签对参赛者隐藏。

**与开发集脚本的关键区别:**

1. **额外参数** `--eval_dataset`（默认: `"blindset_A"`）: 控制输出子目录名称
2. **不同的历史解析**（第92-94行）: 不是预测所有8轮，盲测集每个会话只提供一个需要预测的轮次。脚本取 `conversations[:-1]` 作为历史，`conversations[-1]` 作为查询。
3. **输出路径**: `exp/inference/{eval_dataset}/{tid}.json`（例如 `exp/inference/blindset_A/llama1b_bm25_blindset_A.json`）

**`chat_history_parser()` 函数**: 与开发集版本相同 — 它在两个文件中都有定义，但只在开发集中实际使用（盲测集脚本定义了它，但在 `main()` 中以不同的方式解析历史）。

**命令行参数:**
- `--tid`（默认: `"llama1b_bm25_blindset_A"`）
- `--eval_dataset`（默认: `"blindset_A"`）
- `--batch_size`（默认: 16）
- `--save_path`（暂未使用）

两个推理脚本共享挑战赛提交所需的相同**输出JSON格式**:
```json
[
  {
    "session_id": "...",
    "user_id": "...",
    "turn_number": 1,
    "predicted_track_ids": ["id1", "id2", ..., "id20"],
    "predicted_response": "我推荐..."
  }
]
```

---

### 下界基线: `lowerbound/`

不使用任何机器学习的简单基线 — 它们确立了挑战赛的**最低预期性能**。

#### `lowerbound/popularity.py` — 流行度基线

无论用户查询或上下文如何，始终推荐训练集中**出现频率最高的前20首曲目**。

**`load_popularity_track()`**（第8-18行）:
1. 加载对话数据集的**训练划分**
2. 遍历所有对话，从 `music` 角色轮次中提取曲目ID（训练中实际推荐的曲目）
3. 使用 `collections.Counter` 统计频率
4. 返回前20个最常见的曲目ID

**`main()`**（第20-37行）:
1. 获取前20首流行曲目
2. 加载**测试划分**
3. 对每个会话 × 8轮，输出相同的20首曲目
4. `predicted_response` 始终为空字符串（无LLM）
5. 保存到 `exp/inference/popularity.json`

该基线回答的问题是："如果我们总是推荐最流行的歌曲，效果会有多好？"

---

#### `lowerbound/random_sample.py` — 随机基线

对每轮从整个曲库中**随机采样20首曲目**。

**`load_track_pools()`**（第7-15行）:
- 从曲目元数据数据集的 `all_tracks` 划分中加载**所有曲目**
- 返回所有曲目ID的列表

**`main()`**（第17-52行）:
1. 获取完整的曲目ID池
2. 加载测试划分
3. 对每个会话 × 轮次，调用 `random.sample(track_pools, 20)` 随机选择20首曲目
4. `predicted_response` 始终为空
5. 保存到 `exp/inference/random.json`

该基线回答："最差的合理表现是什么？"（随机应该比流行度更差；任何机器学习系统都应该超过两者。）

---

### 改进建议与扩展指南: `tips/`

三个 markdown 文件提供了如何改进基线系统的指导。

#### `tips/add_reranker.md` — 添加第二阶段重排序器

建议在初始检索之后添加一个**重排序模块**来精炼候选项：

- **方案A — 基于嵌入的重排序**:
  - 使用来自用户收听历史的嵌入来计算用户-物品相似度
  - 跨模态重排序：组合文本相关性 + 音频相似度 + 用户偏好信号
- **方案B — 基于LLM的重排序**:
  - 使用更大的LLM来判断top-k候选项的相关性
  - 提示词："根据相关性对以下曲目进行排序：{user_query}"
  - 提到 Llama-3-8B、Qwen-7B 作为候选重排序模型

提供了代码草图，展示了 `crs_baseline.py` 中的集成点：检索 top-100，重排序到 top-20。

引用了带有用户/曲目嵌入的额外 Hugging Face 数据集。

---

#### `tips/improve_item_representation.md` — 更好的物品表示

两种改进曲目表示方式的策略：

**1. 添加更多曲目信息:**
- 当前使用：track_name、artist_name、album_name、release_date
- 建议添加：流派标签（`tag_list`）、情绪标签、流行度分数
- 实现方式：编辑配置中的 `corpus_types` + 更新 `_stringify_metadata()`
- 进阶：使用 CLAP（对比语言-音频预训练）来结合实际音频特征

**2. 使用更强的嵌入模型:**
- 替换 `bert-base-uncased` 为：
  - `Qwen2.5-Embedding` — 多语言支持
  - `Contriever` — 强大的零样本检索能力
  - `E5` 或 `BGE` — 当前最先进的文本嵌入模型
  - `ColBERT` — token级别的后期交互，实现精确匹配

---

#### `tips/use_genrec_semantic_ids.md` — 生成式检索

最雄心勃勃的方向：**用单个端到端生成模型替换两阶段流水线**。

概念：不是先检索后解释，而是训练LLM**直接生成曲目标识符**来响应用户查询。

**语义ID方法:**
- 为曲目分配层级化的ID（例如 `jazz/smooth/piano/0042`）
- ID结构编码了音乐相似性（相似曲目有相似的ID）
- 训练LLM生成相关的曲目ID

**优势:**
- 统一架构（一个模型取代检索+生成）
- 利用LLM推理能力处理复杂用户意图
- 可能更好地处理细腻的查询

---

## 如何运行

### 前置条件

- **Python 3.10+**（推荐；技术上支持 3.8+）
- **支持 CUDA 的 GPU**，至少 8GB 显存（用于 bf16 的 Llama-3.2-1B）
- **Hugging Face 账号**，具有：
  - [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 的访问权限（需要接受 Meta 的许可协议）
  - TalkPlayData-Challenge 数据集集合的访问权限
- **uv** 包管理器（或 pip）

### 步骤1：安装

```bash
cd music-crs-baselines

# 创建并激活虚拟环境
uv venv .venv --python=3.10
source .venv/bin/activate

# 安装 mcrs 包及所有依赖
uv pip install -e .

# （可选但推荐）安装 Flash Attention 2 以加速 LLM 推理
uv pip install flash-attn --no-build-isolation
```

如果没有 `uv`，使用 pip：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install flash-attn --no-build-isolation
```

### 步骤2：Hugging Face 认证

```bash
huggingface-cli login
# 粘贴你的 HF token（从 https://huggingface.co/settings/tokens 获取）
```

### 步骤3：运行推理

#### 开发集

```bash
# BM25 基线（稀疏检索）
python run_inference_devset.py --tid llama1b_bm25_devset --batch_size 16

# BERT 基线（稠密检索）
python run_inference_devset.py --tid llama1b_bert_devset --batch_size 16
```

结果保存到：
- `exp/inference/devset/llama1b_bm25_devset.json`
- `exp/inference/devset/llama1b_bert_devset.json`

#### 盲测集A（用于排行榜提交）

```bash
# BM25 基线
python run_inference_blindset.py --tid llama1b_bm25_blindset_A --batch_size 16

# BERT 基线
python run_inference_blindset.py --tid llama1b_bert_blindset_A --batch_size 16
```

结果保存到：
- `exp/inference/blindset_A/llama1b_bm25_blindset_A.json`
- `exp/inference/blindset_A/llama1b_bert_blindset_A.json`

#### 下界基线

```bash
# 流行度基线
python lowerbound/popularity.py
# 输出: exp/inference/popularity.json

# 随机基线
python lowerbound/random_sample.py
# 输出: exp/inference/random.json
```

### 步骤4：运行时发生了什么

当你运行推理脚本时，会发生以下顺序的操作：

**阶段1 — 初始化（首次运行需要数分钟）:**
1. **加载配置** 从 `config/{tid}.yaml`
2. **清除缓存** — 脚本运行 `rm -rf cache` 以确保干净的状态
3. **下载曲目元数据** — `MusicCatalogDB` 从 Hugging Face 获取
4. **下载用户画像** — `UserProfileDB` 从 Hugging Face 获取
5. **构建检索索引**（仅首次运行）:
   - BM25: 对所有曲目元数据字符串进行分词并构建 BM25 索引 → 保存到 `cache/bm25/`
   - BERT: 通过 BERT 编码所有曲目元数据，mean-pool，归一化 → 保存到 `cache/bert/`
6. **下载并加载 LLM** — Llama-3.2-1B-Instruct（~2.5GB）被下载并移至 GPU
7. 在后续运行中（如果删除 `rm -rf cache` 行），缓存的索引被复用，启动速度大大加快

**阶段2 — 数据准备:**
1. 从 Hugging Face 加载测试数据集
2. 对于每个对话会话，解析聊天历史：
   - **开发集**: 独立预测所有8轮，每轮使用真实的前几轮作为历史（"神谕历史"评估）
   - **盲测集**: 只预测最后一轮，使用所有前几轮作为历史

**阶段3 — 批量推理:**
对于每个批次：
1. 所有查询通过检索模块 → 每个查询的 top-20 曲目ID
2. 通过 `MusicCatalogDB` 查找 top-1 曲目的元数据
3. 所有格式化的提示词通过 LLM → 自然语言回复

**阶段4 — 输出:**
- 结果以挑战赛提交格式保存为 JSON

### 步骤5：创建自定义配置

在 `config/` 中创建一个新的YAML文件：

```yaml
# config/my_model.yaml
lm_type: "meta-llama/Llama-3.2-1B-Instruct"
retrieval_type: "bm25"                      # 或 "bert"
test_dataset_name: "talkpl-ai/TalkPlayData-Challenge-Dataset"
item_db_name: "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
user_db_name: "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
track_split_types:
  - "all_tracks"                            # 强制要求：始终使用 all_tracks
user_split_types:
  - "all_users"
corpus_types:
  - "track_name"
  - "artist_name"
  - "album_name"
  - "release_date"
  - "tag_list"                              # 添加更多字段以实现更丰富的检索
cache_dir: "./cache"
device: "cuda"
attn_implementation: "flash_attention_2"
```

然后运行：
```bash
python run_inference_devset.py --tid my_model --batch_size 16
```

### 重要注意事项

- **始终在 `track_split_types` 中使用 `all_tracks`**。在推理过程中过滤曲目会使评估无效。
- **缓存每次运行都会被清除**，因为推理脚本中有 `os.system("rm -rf cache")` 这一行。要复用缓存的索引，请注释掉这一行。这一点很重要，因为根据曲库大小，构建 BERT 索引可能需要 10-30 分钟。
- **首次运行较慢**: 构建检索索引（尤其是BERT）和下载LLM需要大量时间。后续使用缓存索引的运行会快得多。
- **GPU显存**: Llama-3.2-1B 在 bf16 下需要约 2.5GB 显存。BERT 编码器也使用 GPU。如果遇到 OOM，请减小 `--batch_size`。
- **磁盘空间**: Llama 模型约 2.5GB，BERT-base 约 440MB。嵌入缓存的大小与曲目数量成正比。
- **Python版本**: 项目指定 `>=3.8`，但推荐 3.10+ 以获得完整的依赖兼容性。

---

## 评估

要评估系统的输出，使用官方评估器仓库：
- **评估器**: [https://github.com/nlp4musa/music-crs-evaluator](https://github.com/nlp4musa/music-crs-evaluator)

评估器会衡量：
1. **推荐质量** — 对预测曲目ID的 Recall@K、NDCG@K
2. **回复质量** — 对预测回复的自然语言生成指标

---

## 完整文件索引

| 文件 | 行数 | 用途 |
|---|---|---|
| `pyproject.toml` | 31 | 项目配置、包名、所有依赖 |
| `.gitignore` | 7 | 排除 venv、cache、bytecode、egg-info |
| `readme.md` | — | 本文档 |
| `mcrs/__init__.py` | 17 | `load_crs_baseline()` 工厂函数，11个默认参数 |
| `mcrs/crs_baseline.py` | 192 | 核心 `CRS_BASELINE` 编排器：初始化、系统提示词组装、单条/批量聊天 |
| `mcrs/lm_modules/__init__.py` | 7 | LLM 工厂，将 `"meta-llama/Llama-3.2-1B-Instruct"` 分派到 `LLAMA_MODEL` |
| `mcrs/lm_modules/llama.py` | 74 | `LLAMA_MODEL`: 加载 Llama，格式化聊天模板，单条/批量回复生成 |
| `mcrs/retrieval_modules/__init__.py` | 17 | 检索工厂，将 `"bm25"`/`"bert"` 分派到 `BM25_MODEL`/`BERT_MODEL` |
| `mcrs/retrieval_modules/bm25.py` | 123 | `BM25_MODEL`: 加载语料库，构建/查询缓存的 BM25 索引，单条/批量检索 |
| `mcrs/retrieval_modules/bert.py` | 192 | `BERT_MODEL`: 加载语料库，通过 BERT 编码曲目，mean-pool，余弦相似度检索 |
| `mcrs/db_item/__init__.py` | 2 | 重新导出 `MusicCatalogDB` |
| `mcrs/db_item/music_catalog.py` | 24 | `MusicCatalogDB`: 加载曲目元数据，`id_to_metadata()` 将 ID 转换为格式化字符串 |
| `mcrs/db_user/__init__.py` | 2 | 重新导出 `UserProfileDB` |
| `mcrs/db_user/user_profile.py` | 23 | `UserProfileDB`: 加载用户画像，`id_to_profile_str()` 格式化人口统计信息 |
| `mcrs/system_prompts/roleplay.txt` | 1 | 系统提示词：将 AI 角色设定为专家音乐推荐助手 |
| `mcrs/system_prompts/response_generation.txt` | 7 | 系统提示词：6条回复格式化规则 |
| `mcrs/system_prompts/personalization.txt` | 1 | 系统提示词：利用用户画像进行个性化 |
| `config/llama1b_bm25_devset.yaml` | 17 | BM25 + 开发集 配置 |
| `config/llama1b_bert_devset.yaml` | 17 | BERT + 开发集 配置 |
| `config/llama1b_bm25_blindset_A.yaml` | 17 | BM25 + 盲测集A 配置 |
| `config/llama1b_bert_blindset_A.yaml` | 17 | BERT + 盲测集A 配置 |
| `run_inference_devset.py` | 145 | 开发集推理：神谕历史，全部8轮，保存到 `exp/inference/devset/` |
| `run_inference_blindset.py` | 152 | 盲测集推理：最后一轮预测，保存到 `exp/inference/{eval_dataset}/` |
| `lowerbound/popularity.py` | 40 | 流行度基线：始终推荐训练集中出现频率最高的前20首曲目 |
| `lowerbound/random_sample.py` | 52 | 随机基线：每轮随机采样20首曲目 |
| `tips/add_reranker.md` | 39 | 指南：添加基于嵌入或基于LLM的重排序器 |
| `tips/improve_item_representation.md` | 44 | 指南：更丰富的元数据字段、音频特征、更强的嵌入模型 |
| `tips/use_genrec_semantic_ids.md` | 26 | 指南：使用语义ID的端到端生成式检索 |

---

祝你在挑战赛中取得好成绩！
