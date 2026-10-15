# OSLR
Rethinking Sampling for SAR-to-Optical Translation: When One-Step Regression Suffices

OSLR 用**一次网络前向**把 SAR 图像翻译成光学图像。它在冻结的 [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) 自编码器潜空间里，学习从 SAR 潜变量到光学潜变量的确定性映射。推理时没有噪声输入、没有时间步、也没有迭代采样。

配对的 SAR–光学评测用一张参考图给每张预测打分，采样多样性没有收益。OSLR 因此去掉条件分布采样，直接回归这张参考。在 SAR 与光学信息充足、空间也对齐的数据上，它在 SSIM、PSNR、LPIPS 上比多步扩散更贴近参考，推理代价也低得多。FID 上的小幅差距是感知–失真权衡，不是配对指标失效。

> 论文：*Rethinking Sampling for SAR-to-Optical Translation: When One-Step Regression Suffices*  
> 代码：<https://github.com/nastyjam02/OSLR>  
> 作者：陈瑞、刘鹏（复旦大学）

定性对比（SAR、8 步 DPM-Solver++、OSLR、光学真值）见论文图 3。

<!-- ![SEN12 定性对比](assets/figures/teaser.png) -->

## SEN12 上的结果

下表是 SEN12 测试集（3,000 对）。推理时间是单张 NVIDIA A800-SXM4-80GB、batch size 32、预热 20 次、CUDA 同步后的摊销毫秒/张。计时包含 SAR 编码、UNet 前向和 VAE 解码，不含磁盘 I/O。其他 GPU 或 batch size 测得的 `ms/img` 不要和本表比较。

| 方法 | 类型 | NFE | SSIM ↑ | PSNR ↑ | LPIPS ↓ | FID ↓ | ms/img |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LDM (DPM-Solver++) | 扩散 | 8 | 0.3599 | 18.43 | 0.3226 | **35.47** | 60.7 |
| **OSLR (full)** | 回归 | **1** | **0.4724** | **20.31** | **0.2612** | 37.69 | **21.4** |

相对 8 步扩散参考，SSIM 提高 0.113，PSNR 提高 1.88 dB，LPIPS 下降 0.061。吞吐约 2.8 倍，FID 升高 2.2。

## 什么时候值得试一步回归

训练之前，用配对图像算出两个统计量，用来判断确定性回归是否值得作为第一次尝试：

| 数据集 | 像素互信息 ↑（nat） | Chamfer ↓（px） | 结果 |
| --- | ---: | ---: | --- |
| SEN12 | 0.133 | 1.88 | OSLR 的 SSIM、PSNR、LPIPS 更好，FID 只略升。 |
| QXS-SAROPT | 0.085 | 3.67 | SSIM、PSNR 仍上升，但结果发糊，LPIPS 和 FID 一起变差。 |

互信息高、Chamfer 距离小，说明光学目标能从 SAR 预测出来，配对也空间一致，值得训一次前向。两个量都不利时，保留生成式方法，或先改善配准。

## 环境

Python 3.10。记录的环境名是 `nasty-home`。

```bash
conda env create -f environment/environment.nasty-home.yml
conda activate nasty-home
pip install -r environment/requirements-nasty-home.txt
```
凡是要报告 FID，都必须设置 `FID_WEIGHTS`。文件是 torch-fidelity 发布的 Inception v3 权重，SHA256 为 `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`。

## 数据

本仓库不分发 SAR / 光学原图。

| 数据集 | 论文中的用法 | 获取途径 |
| --- | --- | --- |
| SEN1-2（论文中称 SEN12） | 13,000 对训练、3,000 对测试，四类地物，resize 到 256×256 
| QXS-SAROPT | 16,000 对训练，以及 seed 42 固定的 4,000 对测试子集 

目录约定：

```text
SEN12_ROOT/{train,test}/{agri,barrenland,grassland,urban}/{s1,s2}/
QXS_ROOT/splits/{train,valtest}/qxs/{s1,s2}/
```


## 快速开始

冒烟测试用 16 张类别均衡的 SEN12 图像（`seed=42`）检查环境、OSLR (full) 检查点和指标管线是否一致。

```bash
bash scripts/run_oslr_smoke.sh
```
三个层级都固定 `seed=42`：


| 项目 | 取值 |
| --- | --- |
| 优化器 | AdamW，\(\beta_1=0.9\)，\(\beta_2=0.999\)，权重衰减 0.01 |
| 学习率 | \(1\times 10^{-4}\)，100 步线性 warmup，余弦衰减 |
| 硬件 | 2× A800-SXM4-80GB，每卡 batch size 16（全局 32） |
| SEN12 上的 OSLR | 先 30,000 步潜空间 \(\ell_1\)，再 20,000 步第二阶段目标。OSLR (full) 是 L1+FFL+LPIPS 这一行。 |
| SEN12 扩散基线 | 单独训练 50,000 步 |
| 跨数据集表中的 QXS 模型 | 50,000 步 |

训练入口是 `src/sar2opt_onestep/train.py`（OSLR）和 `src/sar2opt_latent/train.py`（潜空间扩散）。复现表中指标不需要重训，用上面的检查点即可。

## 目录

```text
src/sar2opt_onestep/     OSLR 的训练、推理和评测
src/sar2opt_latent/      潜空间扩散、求解器和共用指标
src/archive/             像素互信息和 Chamfer
src/baselines/           对已有预测目录计算指标
assets/manifests/        固定的 SEN12 与 QXS 样本清单
assets/results/          论文表格的审计 JSON
assets/statistics/       MI / Chamfer 汇总和逐张 CSV
environment/             nasty-home 的 conda 环境与锁定依赖
scripts/                 清单生成，以及 smoke / fixed / full 评测
```

## 引用

```bibtex
@article{chen2026oslr,
  author  = {Chen, Rui and Liu, Peng},
  title   = {Rethinking Sampling for {SAR}-to-Optical Translation: When One-Step Regression Suffices},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026},
  note    = {Manuscript received April 9, 2026}
}
```


## 许可

| 材料 | 许可 |
| --- | --- |
| 本仓库代码 | 待定。公开前补上 `LICENSE`。 |
| SEN1-2 | CC-BY 4.0 |
| QXS-SAROPT | CC-BY，需通过作者问卷获取 |
| Stable Diffusion 2.1 | CreativeML OpenRAIL-M |
| FID Inception 权重 | 遵循 torch-fidelity 的发布许可 |

