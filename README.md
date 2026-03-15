# d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation 🚀

[![Paper](https://img.shields.io/badge/Paper-arXiv:2601.07568-orange)](https://arxiv.org/abs/2601.07568)
[![Blog](https://img.shields.io/badge/Blog-text--diffusion-blue)](https://hao-ai-lab.github.io/blogs/text-diffusion/)
[![Demo](https://img.shields.io/badge/Demo-d3LLM--Demo-purple)](https://d3llm-team.github.io/)
[![d3LLM-Dream](https://img.shields.io/badge/🤗-d3LLM--Dream-yellow)](https://huggingface.co/d3LLM/d3LLM_Dream)
[![d3LLM-LLaDA](https://img.shields.io/badge/🤗-d3LLM--LLaDA-yellow)](https://huggingface.co/d3LLM/d3LLM_LLaDA)
[![d3LLM-Coder](https://img.shields.io/badge/🤗-d3LLM--Coder-yellow)](https://huggingface.co/d3LLM/d3LLM_Dream_Coder)
[![dLLM-Leaderboard](https://img.shields.io/badge/📊-dLLM--Leaderboard-blue)](https://huggingface.co/spaces/d3LLM/dLLM_Leaderboard)

This is the official implementation of the paper [d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation](https://arxiv.org/abs/2601.07568), where we introduce a novel recipe for building an ultra-fast diffusion language model named ***d3LLM*** (_pseuDo-Distilled Diffusion LLM_) 🚀.

## 📣 News
- `[2026/03/15]`: 🏎️ SGLang support is here! d3LLM models are now supported in the SGLang engine (PR [#20615](https://github.com/sgl-project/sglang/pull/20615)) — try it out [here](#efficient-serving-with-sglang-engine)! (Thanks to [Hao-Cong Wu](https://github.com/flowermouse) for the contribution!)
- `[2026/02/01]`: We have updated our d3LLM paper on 📄ArXiv. See our [updated paper](https://arxiv.org/abs/2601.07568).
- `[2026/01/12]`: We release the paper on 📄ArXiv! See our [paper](https://arxiv.org/abs/2601.07568).
- `[2026/12/29]`: We release a Leaderboard📊 of diffusion LLMs, see our [dLLM Leaderboard](https://huggingface.co/spaces/d3LLM/dLLM_Leaderboard).
- `[2025/12/11]`: We release the models on Huggingface 🤗, see our [d3LLM-LLaDA](https://huggingface.co/d3LLM/d3LLM_LLaDA), [d3LLM-Dream](https://huggingface.co/d3LLM/d3LLM_Dream), and [d3LLM-Dream-Coder](https://huggingface.co/d3LLM/d3LLM_Dream_Coder).
- `[2025/12/11]`: We release the training scripts, training datasets, and evaluation code for d3LLM, see [our GitHub repo](https://github.com/hao-ai-lab/d3LLM).
- `[2025/12/10]`: We release the 🌐 [blog](https://hao-ai-lab.github.io/blogs/text-diffusion/).

## ✨ Demo

Demo of d3LLM: Achieve up to 5× speedup over autoregressive models (Qwen-2.5-7B-it) on H100 GPU and 3.6× speedup on A100 GPU, and 10× speedup over the vanilla Dream/LLaDA. **You can try 🕹️ [our demo](https://d3llm-team.github.io/).**

<div align="center">

![d3LLM Demo](asset/imgs/demo.gif)

</div>


## 📖 What is d3LLM?

**d3LLM** (_pseuDo-Distilled Diffusion LLM_) is a novel framework for building ultra-fast diffusion language models with negligible accuracy degradation. d3LLM achieves **5× speedup** over autoregressive models on H100 GPUs while maintaining competitive performance.


## 🎯 Getting Started

### Installation

We recommend creating a dedicated `~/Codes` directory to maintain consistent paths during evaluation:

```bash
# Create workspace directory
mkdir -p ~/Codes
cd ~/Codes

# Clone the repository
git clone https://github.com/hao-ai-lab/d3LLM.git
cd d3LLM

# Install dependencies
# It is important to check the version of transformers==4.49.0, lm_eval==0.4.9, datasets==3.2.0, and flash_attn==2.7.4.post1
pip install -r requirements.txt
```

> **Note:** We recommend cloning in `~/Codes/d3LLM`, which ensures `eval_scripts` work out-of-the-box with consistent paths.

### Try d3LLM Instantly

Chat with d3LLM models using our simple chat scripts:

```bash
# Chat with d3LLM-Dream
python chat/chat_d3llm_dream.py

# Or chat with d3LLM-LLaDA
python chat/chat_d3llm_llada.py
```

> Note that because our distillation data primarily consists of **coding** and **math reasoning** tasks, acceleration may only appear on prompts of these tasks.


## 🔬 How d3LLM Works

The d3LLM framework combines two key innovations:


### (i) Pseudo-Trajectory Distillation 📚

Instead of random masking, we extract the teacher model's decoding order—the sequence in which it unmasks tokens. This pseudo-trajectory guides the student model to learn efficient generation patterns.

- **Pseudo-Trajectory Extraction** → 18% TPF improvement
- **Progressive Noise Schedule** → Additional 12% TPF boost
- **Progressive Window Sizing** → Another 8% TPF gain

<div align="center">

![Distillation Process](asset/imgs/fig_distillation.png)
*Our pseudo-trajectory-based distillation*

</div>


### (ii) Multi-Block Decoding Strategy ⚡

We enable parallel decoding across multiple blocks simultaneously using entropy-based token selection.

- **Entropy-Based Multi-Block Decoding** → 30% TPF improvement
- **KV-Cache with Periodic Refresh** → 35% TPS boost in long contexts
- **Early Stopping on EOS** → 5% TPF gain

<div align="center">

![Multi-Block Decoding](asset/imgs/fig_decoding.png)
*Entropy-based multi-block decoding with KV-cache and refresh.*

</div>

Together, these innovations achieve **5-10× speedup** on TPF (tokens per forward) over vanilla diffusion models while maintaining accuracy.
Based on the d3LLM framework, we have released three models on 🤗 HuggingFace: [d3LLM-LLaDA](https://huggingface.co/d3LLM/d3LLM_LLaDA), [d3LLM-Dream](https://huggingface.co/d3LLM/d3LLM_Dream), and [d3LLM-Coder](https://huggingface.co/d3LLM/d3LLM_Dream_Coder).

## 🏋️‍♀️ Training d3LLM Models

We provide the training scripts for d3LLM-Dream and d3LLM-LLaDA. You can use the following commands to train the models.

```bash
# Training d3LLM-Dream
deepspeed --num_gpus=4 d3llm/d3llm_DREAM/distill_2_training/d3llm_dream_train.py

# Training d3LLM-LLaDA
deepspeed --num_gpus=4 d3llm/d3llm_LLaDA/distill_2_training/d3llm_llada_train.py
```

The trajectory dataset is already extracted and uploaded to HuggingFace (see [Dream Trajectory](https://huggingface.co/datasets/d3LLM/trajectory_data_dream_32) and [LLaDA Trajectory](https://huggingface.co/datasets/d3LLM/trajectory_data_llada_32)). You can also generate the pseudo-trajectory dataset
using the script in `distill_1_data_prepare/` folder.

## 🧪 Evaluation on Standard Benchmarks

All evaluation scripts are in the `eval_scripts/` folder—just install the environment and run! We include comprehensive evaluation codes for:

- ✅ **d3LLM** (our method)
- ✅ [**AR Model (e.g., Qwen-2.5-7B-it)**](https://arxiv.org/abs/2412.15115) - Autoregressive baselines
- ✅ [**Vanilla LLaDA**](https://arxiv.org/abs/2502.09992) - Original LLaDA model
- ✅ [**Vanilla Dream**](https://arxiv.org/abs/2508.15487) - Original Dream model
- ✅ [**Fast-dLLM**](https://arxiv.org/abs/2505.22618) - Training-free acceleration with KV cache
- ✅ [**D2F**](https://arxiv.org/abs/2508.09192) - Discrete diffusion forcing
- ✅ [**dParallel**](https://arxiv.org/abs/2509.26488) - Distilled dLLMs
- ✅ [**Fast-dLLM v2**](https://arxiv.org/abs/2509.26328) - Block-wise diffusion

See [eval_scripts](eval_scripts/) for more details.

## 📊 Benchmark Results

Our d3LLM achieves the highest AUP ([_Accuracy Under Parallelism_](https://hao-ai-lab.github.io/blogs/text-diffusion/)) scores across multiple dLLMs and tasks:

<div align="center">

<table>
<tr>
<td align="center"><img src="asset/imgs/data_llada_aup_radar.png" width="100%"/><br/><b>LLaDA-based Models</b></td>
<td align="center"><img src="asset/imgs/data_dream_aup_radar.png" width="100%"/><br/><b>Dream-based Models</b></td>
<td align="center"><img src="asset/imgs/data_dream_coder_aup_radar.png" width="100%"/><br/><b>Coder Models</b></td>
</tr>
</table>

*Radar plots comparing AUP scores across different methods and benchmarks*

</div>

### Acceleration Highlights (on GSM8K-CoT Dataset, with Huggingface Backend)

<div align="center">

| Model | H100's TPS | A100's TPS | Speedup vs. AR |
|-------|:----------:|:----------:|:---------------:|
| Qwen-2.5-7B (AR) | 57.32 | 50.36 | 1.00× |
| d3LLM-LLaDA | **288.89** | **183.33** | **3.47×~5.04×** |
| d3LLM-Dream | **235.34** | **128.19** | **2.55×~4.67×** |

</div>

> **Want more details?** Check out our dLLM leaderboard and comprehensive results at 🌐 **[this blog](https://hao-ai-lab.github.io/blogs/text-diffusion/)**.



## Efficient Serving with SGLang Engine

We provide SGLang support for d3LLM inference (see PR [#20615](https://github.com/sgl-project/sglang/pull/20615)). Install the patched version with:

```bash
pip install uv
uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git@refs/pull/20615/head#subdirectory=python"
```

Then launch the server:

```bash
# d3LLM-LLaDA
python -m sglang.launch_server \
    --model d3LLM/d3LLM_LLaDA \
    --trust-remote-code \
    --attention-backend flashinfer \
    --dllm-algorithm FullAttnMultiBlock \
    --mem-fraction-static 0.8 \
    --cuda-graph-max-bs 32

# d3LLM-Dream
python -m sglang.launch_server \
    --model d3LLM/d3LLM_Dream \
    --trust-remote-code \
    --attention-backend flashinfer \
    --dllm-algorithm FullAttnMultiBlock \
    --mem-fraction-static 0.8 \
    --cuda-graph-max-bs 32
```

### Acceleration Highlights using [SGLang Engine](https://github.com/sgl-project/sglang/pull/20615)

**Dataset:** GSM8K-CoT (zero-shot)  
**Decoding Method:** FullAttnMultiBlock  
**TP Size:** 1

| Model | Threshold | Batch Size | B200 TPS | H800 TPS | A800 TPS | TPF | Accuracy |
|-------|-----------|------------|----------|----------|----------|-----|----------|
| Qwen2.5-7B-Instruct | / | 1 | 274.7 | 108.6 | 96.8 | 1 | 74.1% |
| Qwen3-8B | / | 1 | 234.2 | 98.3 | 90.0 | 1 | 93.63% |
| d3LLM-LLaDA (8B dense) | 0.5 | 1 | 1240.99 | 545.31 | 251.61 | 9.91 | 75.36% |
| d3LLM-LLaDA (8B dense) | 0.5 | 4 | 1310.18 | 551.87 | 249.98 | 8.56 | 75.12% |
| d3LLM-Dream (7B dense) | 0.4 | 1 | 586.77 | 280.48 | 125.57 | 4.89 | 80.89% |
| d3LLM-Dream (7B dense) | 0.4 | 4 | 676.81 | 281.82 | 127.85 | 4.22 | 80.76% |

> **TPS** = Tokens Per Second, **TPF** = Tokens Per Forward (average forward passes per token)



## 🏆 Diffusion LLM Leaderboard

We further present a leaderboard that compares different diffusion LLMs across five representative benchmark tasks, using the AUP score (Accuracy Under Parallelism) as the primary evaluation metric, which is a hardware-independent metric that measures both the efficiency and the performance of a
dLLM. **More details can be found in [AUP_leaderboard](AUP_leaderboard/) and 🌐 [this blog](https://hao-ai-lab.github.io/blogs/text-diffusion/).**




## 🙏 Acknowledgments

This project builds upon excellent open-source work:
- [LLaDA](https://arxiv.org/abs/2502.09992) - Large Language Diffusion Models
- [Dream](https://arxiv.org/abs/2508.15487) - Diffusion Large Language Models
- [Fast-dLLM](https://arxiv.org/abs/2505.22618) - Training-free acceleration
- [D2F](https://arxiv.org/abs/2508.09192) - Discrete diffusion forcing
- [dParallel](https://arxiv.org/abs/2509.26488) - Distilled dLLMs
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Evaluation framework

## 📝 Citation

If you find our d3LLM or the AUP metric useful for your research, please star our project and cite our work.

```bibtex
@article{arxiv'26:d3llm,
  title   = {d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation},
  author  = {Yu-Yang Qian and Junda Su and Lanxiang Hu and Peiyuan Zhang and Zhijie Deng and Peng Zhao and Hao Zhang},
  journal = {ArXiv preprint},
  volume  = {arXiv:2601.07568},
  year    = {2026}
}
```



<div align="center">

⭐ Star us on GitHub and cite our paper if you find this project helpful!

</div>
