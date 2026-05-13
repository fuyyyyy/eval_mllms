# eval_mllms

一个面向视觉语言模型的统一评测脚手架，重点支持你现在要做的三类推理对比实验：

- `no_cot`：不显示思维链，直接回答
- `cot`：显式输出思维链，再给最终答案
- `latent_cot`：要求模型先内部思考，但最终只输出答案

它可以统一评测：

- 闭源 API 模型：OpenAI / Anthropic / Gemini
- 开源本地模型：`transformers` 直连
- 开源服务化模型：通过 `vllm serve` 暴露的 OpenAI-compatible 接口

当前默认数据集是 `fuyyy74/EmoSet2k`，但结构上也支持一般图像分类和图文问答。

## 你现在最适合直接用的配置

- 推荐模型清单：[configs/models.recommended.yaml](/Users/fangyiyang/Documents/New%20project/eval_mllms/configs/models.recommended.yaml)
- 推理模式配置：[configs/reasoning.default.yaml](/Users/fangyiyang/Documents/New%20project/eval_mllms/configs/reasoning.default.yaml)
- 数据集配置：[configs/dataset_emoset2k.yaml](/Users/fangyiyang/Documents/New%20project/eval_mllms/configs/dataset_emoset2k.yaml)

推荐模型里已经包含：

- 你指定的主测模型：`Qwen2.5-VL`、`Qwen3-VL`、`LLaVA-OneVision`
- 额外开源视觉语言模型：`InternVL3`、`DeepSeek-VL2`
- 闭源模型：`gpt-4.1-mini`、`gpt-5-mini`、`claude-sonnet-4-0`、`gemini-2.5-flash`

## 安装

```bash
cd /Users/fangyiyang/Documents/New\ project/eval_mllms
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

闭源模型需要 API Key：

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

## 一键跑推荐实验

```bash
bash scripts/run_recommended_benchmark.sh
```

等价命令：

```bash
python3 evaluate.py \
  --config configs/dataset_emoset2k.yaml \
  --models configs/models.recommended.yaml \
  --reasoning-config configs/reasoning.default.yaml
```

## 常用命令

只跑开源模型：

```bash
python3 evaluate.py \
  --config configs/dataset_emoset2k.yaml \
  --models configs/models.all.yaml \
  --reasoning-config configs/reasoning.default.yaml \
  --only-source open_source
```

只跑一个模型：

```bash
python3 evaluate.py \
  --config configs/dataset_emoset2k.yaml \
  --models configs/models.recommended.yaml \
  --reasoning-config configs/reasoning.default.yaml \
  --only qwen3_vl_8b_vllm
```

只跑一种推理模式：

```bash
python3 evaluate.py \
  --config configs/dataset_emoset2k.yaml \
  --models configs/models.recommended.yaml \
  --reasoning-config configs/reasoning.default.yaml \
  --only-mode latent_cot
```

快速 smoke test：

```bash
python3 evaluate.py \
  --config configs/dataset_emoset2k.yaml \
  --models configs/models.recommended.yaml \
  --reasoning-config configs/reasoning.default.yaml \
  --max-samples 20
```

## 输出结构

每个模型、每种模式都会单独落盘：

```text
outputs/<dataset_tag>/<model_name>/<reasoning_mode>/
```

其中包含：

- `predictions.jsonl`：逐样本结果
- `metrics.json`：该模型该模式的指标

总表会写到：

- `outputs/<dataset_tag>/summary.json`
- `outputs/<dataset_tag>/summary.csv`

`summary.csv` 很适合后续直接拿去画表格或做统计检验。

## 推理模式设计

默认的三种模式在 [configs/reasoning.default.yaml](/Users/fangyiyang/Documents/New%20project/eval_mllms/configs/reasoning.default.yaml) 里：

- `no_cot`：要求模型只返回 `<answer>...</answer>`
- `cot`：要求模型返回 `<reasoning>...</reasoning>` 和 `<answer>...</answer>`
- `latent_cot`：提示模型内部思考，但只返回 `<answer>...</answer>`

对支持隐藏思考参数的接口，还会自动附加 provider 级请求参数：

- OpenAI：`reasoning.effort`
- Anthropic：`thinking`
- Gemini：`thinking_budget`
- `vllm_server` / `openai_compatible`：`extra_body.chat_template_kwargs.enable_thinking`

这意味着：

- 统一 prompt 层实验可以直接做
- 支持隐藏思考开关的模型还能进一步做更接近“latent CoT”的实验
- 对不支持隐藏思考 API 的模型，也能退化成“内部思考但不展示”的提示式实验

## vLLM 运行示例

例如启动 `Qwen3-VL-8B-Instruct`：

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --port 8001 \
  --limit-mm-per-prompt '{"image":1}'
```

如果你要比较 `Qwen3-VL` 的 `no_cot` 和 `latent_cot`，默认配置会分别给 OpenAI-compatible 请求附加：

- `enable_thinking: false`
- `enable_thinking: true`

如果某个后端不支持该字段，直接删掉 [configs/reasoning.default.yaml](/Users/fangyiyang/Documents/New%20project/eval_mllms/configs/reasoning.default.yaml) 里对应 provider 的 `request_overrides` 即可。

## 配置说明

数据集配置关键字段：

- `dataset.path`：Hugging Face 数据集名
- `dataset.split`：如 `test`
- `dataset.image_column`：图像列
- `dataset.question_column`：可选，VQA 类任务使用
- `dataset.label_column`：标签列
- `dataset.label_names`：可选，手工指定标签空间
- `dataset.prompt_template`：基础任务 prompt

模型配置关键字段：

- `provider`：`openai` / `anthropic` / `gemini` / `huggingface_local` / `vllm_server`
- `source_type`：`open_source` 或 `closed_source`
- `model_family`：便于分组统计
- `model_size`：便于对照记录
- `base_url`：OpenAI-compatible 服务地址
- `disabled_modes`：可选，跳过不想跑的推理模式
- `request_defaults`：可选，模型级额外请求参数

## 适合你后续继续扩展的方向

- 增加更多数据集，比如 DocVQA、MMMU、ScienceQA、ChartQA
- 在 `summary.csv` 基础上再加 bootstrap 或 McNemar 检验
- 按模型家族生成横向对比图
- 记录 token 用量、响应时延和失败率

## 注意事项

- `vllm_server` 适合批量挂开源模型做统一评测。
- 大模型是否支持“隐藏思考”取决于具体模型和后端实现，不是所有模型都能严格等价实现 `latent_cot`。
- 如果你想把“latent 思维链”定义得更严格，建议后续把支持原生 hidden reasoning 的模型单独列成一组分析。
