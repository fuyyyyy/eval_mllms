# eval_mllms

一个通用的多模态大模型评测脚手架，支持：

- 从 Hugging Face 加载图像或图文数据集
- 统一评测闭源 API 模型
- 统一评测本地 Hugging Face 开源多模态模型
- 统一评测通过 `vllm serve` 暴露出的开源多模态模型
- 输出逐样本预测结果和整体准确率

当前默认适合类似 `fuyyy74/EmoSet2k` 这类图像情感分类数据集，也支持带 `question/prompt` 字段的 VQA/指令式数据集。

## 当前支持的模型接入方式

- `openai`：OpenAI 官方多模态 API
- `anthropic`：Anthropic 官方多模态 API
- `gemini`：Google Gemini 官方多模态 API
- `huggingface_local`：本地 `transformers` 多模态模型
- `vllm_server`：通过 `vllm serve` 暴露的 OpenAI-compatible 服务

## 推荐实验入口

如果你的研究目标是比较 `thinking vs no thinking`，建议优先使用：

- [configs/models.thinking.yaml](/Users/fangyiyang/Desktop/eval_mllms/configs/models.thinking.yaml)

它只保留适合做这类对照实验的模型，并且已经按“同一模型家族下的 thinking / no-thinking”方式配好了。

直接运行：

```bash
bash scripts/run_thinking_models.sh
```

## 模型清单

### 闭源模型

见 [configs/models.closedsource.yaml](/Users/fangyiyang/Documents/New%20project/eval_mllms/configs/models.closedsource.yaml)。

- `gpt-4.1-mini`
- `claude-sonnet-4-0`
- `gemini-2.5-flash`

### 开源模型

见 [configs/models.opensource.yaml](/Users/fangyiyang/Documents/New%20project/eval_mllms/configs/models.opensource.yaml)。

- `Qwen2.5-VL`: `7B` 和 `72B`
- `Qwen3-VL`: `8B` 和 `30B-A3B`
- `InternVL3`: `2B` 和 `38B`
- `LLaVA-OneVision`: `0.5B` 和 `72B`
- `DeepSeek-VL2`: `Tiny` 和 `Full`

说明：

- 我把开源配置重写成“每个家族两档尺寸”，方便你做小模型/大模型对照。
- 这些开源模型默认都按 `vllm_server` 来组织，更适合你后面迁到另一套环境跑。
- `Qwen3-VL` 这里选的是官方 `8B` 和 `30B-A3B` 两档。
- `DeepSeek` 这里接的是开源视觉模型 `DeepSeek-VL2` 系列。
- 我没有把 DeepSeek 闭源 API 写进“闭源 MLLM”列表，因为我查到的官方 API 文档目前没有图像输入的正式文档；现阶段更稳妥的做法是把 DeepSeek 作为开源 VLM 来测。

## Thinking 对照模型

见 [configs/models.thinking.yaml](/Users/fangyiyang/Desktop/eval_mllms/configs/models.thinking.yaml)。

- `Claude Sonnet 4`: `standard` vs `thinking`
- `Gemini 2.5 Flash`: `thinking_budget=0` vs `thinking_budget=1024`
- `Qwen3-VL-8B`: `Instruct` vs `Thinking`
- `Qwen3-VL-30B-A3B`: `Instruct` vs `Thinking`

说明：

- 这套配置更适合回答“thinking 到底带来了什么变化”这个问题。
- 其中 API 模型通过参数切换 thinking 模式，开源 Qwen3-VL 通过 `Instruct` 和 `Thinking` 两个模型版本形成对照。
- 我暂时没有把 DeepSeek 和 LLaVA 放进这套 thinking 对照配置，因为我这里没有查到同样清晰、稳定的 no-thinking 对照入口。

你只需要改配置，不需要改脚本。

## 安装依赖

```bash
cd /Users/fangyiyang/Documents/New\ project/eval_mllms
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你要评测闭源模型，请先设置环境变量：

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

## 运行示例

### 1. 跑闭源模型

```bash
python3 evaluate.py \
  --config configs/dataset_emoset2k.yaml \
  --models configs/models.closedsource.yaml
```

### 2. 跑开源模型

```bash
python3 evaluate.py \
  --config configs/dataset_emoset2k.yaml \
  --models configs/models.opensource.yaml
```

### 3. 只跑所有开源模型

```bash
python3 evaluate.py \
  --config configs/dataset_emoset2k.yaml \
  --models configs/models.all.yaml \
  --only-source open_source
```

### 4. 只跑一个模型

```bash
python3 evaluate.py \
  --config configs/dataset_emoset2k.yaml \
  --models configs/models.opensource.yaml \
  --only qwen3_vl_8b_vllm
```

### 5. 限制样本数做快速 smoke test

```bash
python3 evaluate.py \
  --config configs/dataset_emoset2k.yaml \
  --models configs/models.opensource.yaml \
  --max-samples 20
```

### 6. 一条命令测试所有开源模型

如果你已经把开源模型相关的 `vllm serve` 和本地模型环境准备好了，可以直接运行：

```bash
bash scripts/run_all_open_models.sh
```

这个脚本默认会评测 [configs/models.opensource.yaml](/Users/fangyiyang/Documents/New%20project/eval_mllms/configs/models.opensource.yaml) 里的全部开源模型。

## 当前开源模型端口规划

为了让每个模型都能单独起一个 `vllm serve`，我在配置里默认分配了这些端口：

- `8000`: `Qwen2.5-VL-7B`
- `8001`: `Qwen2.5-VL-72B`
- `8002`: `Qwen3-VL-8B`
- `8003`: `Qwen3-VL-30B-A3B`
- `8004`: `InternVL3-2B`
- `8005`: `InternVL3-38B`
- `8006`: `LLaVA-OneVision-0.5B`
- `8007`: `LLaVA-OneVision-72B`
- `8008`: `DeepSeek-VL2-Tiny`
- `8009`: `DeepSeek-VL2`

## 使用 vLLM

先启动一个模型服务，例如：

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --port 8000 \
  --limit-mm-per-prompt '{"image":1}'
```

然后在模型配置里使用：

```yaml
- name: qwen3_vl_8b_vllm
  provider: vllm_server
  source_type: open_source
  model: Qwen/Qwen3-VL-8B-Instruct
  base_url: http://127.0.0.1:8000/v1
  api_key: EMPTY
```

如果你要同时评多个 `vllm` 服务，最简单的方法是给每个服务一个不同端口，然后在 YAML 里分别填不同的 `base_url`。

## 输出结果

每个模型会在 `outputs/<dataset_name>/<model_name>/` 下生成：

- `predictions.jsonl`：逐条样本预测
- `metrics.json`：聚合指标

## 配置说明

### 数据集配置

见 [configs/dataset_emoset2k.yaml](/Users/fangyiyang/Documents/New%20project/eval_mllms/configs/dataset_emoset2k.yaml)。

关键字段：

- `dataset.path`：Hugging Face 数据集名
- `dataset.name`：可选子集名
- `dataset.split`：例如 `train`、`validation`、`test`
- `dataset.image_column`：图像列名
- `dataset.question_column`：可选，如果为空则走纯图像分类提示词
- `dataset.label_column`：标签列名
- `dataset.label_names`：可选，若数据集元信息里没有类别名就手动填写
- `dataset.prompt_template`：提示词模板，支持 `{label_space}` 和 `{question}`

### 模型配置

见 [configs/models.all.yaml](/Users/fangyiyang/Documents/New%20project/eval_mllms/configs/models.all.yaml)。

支持的 `provider`：

- `openai`
- `openai_compatible`
- `vllm_server`
- `anthropic`
- `gemini`
- `huggingface_local`

额外字段：

- `source_type`：`open_source` 或 `closed_source`
- `model_family`：模型家族名，方便你人工分组
- `model_size`：模型尺寸标记，方便你人工识别
- `base_url`：OpenAI-compatible 服务地址，常用于 `vllm_server`

## 适配不同任务

### 图像分类数据集

给出：

- `image_column`
- `label_column`
- `label_names`

脚本会自动构造“从候选标签中选一个”的 prompt。

### 图文问答数据集

如果数据里已经有 `question` 或 `prompt` 列，只要把 `question_column` 指过去即可。

## 注意事项

- 本地开源模型通常需要 GPU，尤其是 7B 以上模型。
- `huggingface_local` 适配器默认使用 `transformers` 的 `AutoProcessor + AutoModelForImageTextToText`，适合大多数新式 VLM。
- `vllm_server` 适合你在另一台环境里统一部署多个开源模型，然后用同一套评测脚本远程打分。
- 如果某个模型必须开启 `trust_remote_code`，请在模型配置里设为 `true`。
- 不同数据集字段名不一致时，优先改 YAML 配置，不建议改 Python 脚本。
