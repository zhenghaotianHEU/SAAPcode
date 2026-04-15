# SAAPcode

## 简单说明

这是一个基于 **SAAP** 的模型剪枝代码仓库。

## 基本环境

建议在 Linux 环境下运行，并使用支持 CUDA 的 GPU 环境。

当前使用和推荐的环境版本如下：

- 操作系统：Linux
- Python：3.9.18
- CUDA：11.8
- PyTorch：2.0.1
- transformers：4.28.1
- datasets：2.16.1
- cryptography：41.0.7
- numpy：1.26.2
- scipy：1.13.1
- accelerate：1.10.1
- sentencepiece：0.2.1
- tokenizers：0.13.3
- safetensors：0.7.0

如果使用 Conda，建议准备一个单独环境，例如：

```bash
conda create -n torch201-py39-cuda118 python=3.9.18 -y
conda activate torch201-py39-cuda118
```

当前脚本默认使用的 Python 环境是：

- `/opt/anaconda3/envs/torch201-py39-cuda118/bin/python`

模型默认路径是：

- `/home/easyai/llm_weights/decapoda-llama-7b-hf`

## 运行方式

进入项目目录：

```bash
cd /home/easyai/下载/SAAP_code/SAAP/SAAPcode
```

直接运行：

```bash
./start_saap.sh
```

## 日志目录

运行后的日志和结果默认保存在：

```bash
core/prune_log/prunellm/
```

每次运行会在这个目录下生成一个新的时间子目录。

## 可选参数

如果要指定模型路径或日志名称，可以这样运行：

```bash
/opt/anaconda3/envs/torch201-py39-cuda118/bin/python saap.py \
  --base_model /home/easyai/llm_weights/decapoda-llama-7b-hf \
  --save_ckpt_log_name prunellm
```

## 目录说明

- `core/`：核心代码目录
- `core/saap_core/`：SAAP 主流程、配置和运行逻辑
- `core/saap_core/pruneflow/`：剪枝主流程代码
- `core/pruner/`：具体剪枝方法实现
- `core/datasets/`：数据加载与样本构造
- `core/models/`：模型相关代码
- `core/utils/`：日志和通用工具
- `core/evaluator/`：评测相关代码
- `core/templates/`：模板和提示相关内容
- `core/torch_pruning/`：剪枝依赖代码
- `datasets/`：运行时数据缓存目录
- `saap.py`：主入口脚本
- `start_saap.sh`：启动脚本

## 备注

推荐直接使用：

```bash
./start_saap.sh
```

这样最简单。
