# Few-Shot Disease Query Adaptation for Frozen Chest X-ray Vision–Language Models

本目录整理了论文 `latex_ours/main_with_supplement.tex` 对应的实验代码、结果汇总和表格核验工具，覆盖 **10 个模型、11 个数据集、51 张表（正文 4 张、附录 47 张）**。方法保持骨干权重冻结，使用有标签支持集适配疾病查询（Q），并可结合双视图融合（V）；完整方法为 Q+V。

**复现状态：** 使用随包汇总数据重建表格已经验证；完整图像推理仍需要补齐原始数据、权重、冻结划分、模型资源和历史缓存。下面给出完整实验链路及各阶段命令，但当前包不是下载后即可一键完成全部 GPU 实验的独立发行版。尤其是，日志不能替代逐图预测、特征缓存或支持／测试清单。

## 目录与两种复现方式

```text
README.md                     本文档
source/                       原工作区相对布局的实验和表格代码
source/latex_ours/supplement_data/  已保存的汇总数值
source/latex_ours/sec/         当前论文使用的表格及附录源码
resources/MODEL_ASSETS.json    模型源码版本、权重文件名及已有哈希
ENVIRONMENTS.json             十个骨干环境的历史实测核心版本
environments/                各环境依赖锁文件和安装配置
tools/                       环境准备、依赖检查、表格重建和核验工具
tables/                      51 张表的独立 LaTeX 环境
TABLE_INDEX.csv              表号 → 实验入口 → 生成方式
TABLE_LOG_INDEX.csv           表号 → 原始运行日志目录
LOG_FILES.csv                原工作区日志／结果文件位置与大小
RESULT_PROVENANCE.json        固定支持结果来源和原文件哈希
SOURCE_MANIFEST.json          整理后的 source 文件哈希
PORTABILITY_AUDIT.json        本机路径／本地运行限制候选位置
VALIDATION.json               已完成验证的范围
```

- **方式 A：表格数值复现。** 只需 Python 3.10+ 标准库，读取已保存的汇总结果，不运行模型。
- **方式 B：实验复现。** 下载模型和数据，恢复冻结协议及缓存，运行模型／比较方法，保存预测，再重新计算指标、区间和表格。需要 GPU 和多套环境。

`LOG_FILES.csv` 是原工作区的完整索引。随仓库发布的 **241 份文本日志** 位于 `logs/<原相对路径>`，逐文件路径、大小与 SHA-256 见 [PUBLISHED_LOGS.json](PUBLISHED_LOGS.json)。主控制器只归档本论文固定支持实验对应的 `phase6_*.log`；索引中的其他历史日志和结果文件不代表均已上传。`source/` 中包含原适配器及辅助程序，以 [TABLE_INDEX.csv](TABLE_INDEX.csv) 指定入口为准，不能将所有历史脚本视作当前论文实验。

## 1. 先验证随包表格

在本 README 所在目录运行；输出目录必须尚不存在：

```bash
python tools/rebuild_tables.py --output ./build
python tools/verify_manual_tables.py
```

第一条在临时工作区运行 `build_supplement.py` 和 `build_classification_table.py`。9 个当前使用的自动生成文件与论文快照逐字节一致；另外生成的中文表和已隐藏表没有随包基准快照，单独记录。

第二条核验正文表 2、4 及附录 C.9、D.5、D.7、D.8 的 **227 个显示数值**，包括均值和区间端点。敏感性结果使用修正后的 `sensitivity_grid_summary_corrected.csv`，不使用历史旧版 `sensitivity_parameter_summary.csv`。

这些检查不重新计算图像级指标，也不证明新机器上的模型推理数值一致。

## 2. 隔离环境与包版本

### 2.1 历史骨干环境

版本来自本地实测记录和保存的依赖锁文件。CUDA 列是 **PyTorch wheel 的 CUDA 构建版本**，不是要求安装同版本系统 CUDA Toolkit。需要兼容这些 wheel 的 NVIDIA 驱动；安装命令依据 [PyTorch 历史版本说明](https://pytorch.org/get-started/previous-versions/)。

| 环境 | Python | PyTorch / CUDA | torchvision | NumPy | transformers | 完整包版本 |
|---|---|---|---|---|---|---|
| AlphaRAD | 3.10.22 | 2.9.0 / 12.8 | 0.24.0 | 1.24.3 | 4.57.1 | [AlphaRAD.txt](environments/AlphaRAD.txt) |
| RadZero | 3.12.7 | 2.2.1 / 12.1 | 0.17.1 | 1.26.4 | 4.39.3 | [RadZero_i2cl.txt](environments/RadZero_i2cl.txt) |
| BiomedCLIP | 3.10.22 | 2.2.2 / 12.1 | 0.17.2 | 1.26.4 | 4.35.2 | [BiomedCLIP.txt](environments/BiomedCLIP.txt) |
| BioViL-T | 3.10.22 | 2.2.2 / 12.1 | 0.17.2 | 1.26.4 | 4.35.2 | [BioViL-T.txt](environments/BioViL-T.txt) |
| CheXzero | 3.10.22 | 2.2.2 / 12.1 | 0.17.2 | 1.26.4 | 4.35.2 | [CheXzero.txt](environments/CheXzero.txt) |
| MedKLIP | 3.10.22 | 2.2.2 / 12.1 | 0.17.2 | 1.26.4 | 4.35.2 | [MedKLIP.txt](environments/MedKLIP.txt) |
| MAVL | 3.10.22 | 2.2.2 / 12.1 | 0.17.2 | 1.26.4 | 4.35.2 | [MAVL.txt](environments/MAVL.txt) |
| KAD | 3.10.22 | 2.2.2 / 12.1 | 0.17.2 | 1.26.4 | 4.35.2 | [KAD.txt](environments/KAD.txt) |
| MaCo | 3.10.12 | 2.2.2 / 12.1 | 0.17.2 | 1.23.5 | 4.26.1 | [MaCo.txt](environments/MaCo.txt) |
| CARZero | 3.10.12 | 2.2.2 / 12.1 | 0.17.2 | 1.24.4 | 4.35.2 | [CARZero.txt](environments/CARZero.txt) |

OpenCV-headless：AlphaRAD 4.8.0.76；MaCo 4.10.0.84；CARZero 4.8.1.78；其余上述骨干为 4.11.0.86。MaCo 的原 freeze 早于空间融合依赖安装，提供的安装文件补入了实测记录中的 4.10.0.84，并保留注释。

### 2.2 TaskRes／SS-Text+ 与统计环境

两种比较方法共享单独环境，不需要再下载一种胸片骨干权重：它们消费 BiomedCLIP／AlphaRAD 的冻结特征。

| Python | PyTorch / CUDA | NumPy | SciPy | scikit-learn | OmegaConf | 版本文件 |
|---|---|---|---|---|---|---|
| 3.12.7 | 2.5.1 / 12.4 | 1.26.4 | 1.13.1 | 1.5.1 | 2.3.0 | [TaskRes_SS_Text.txt](environments/TaskRes_SS_Text.txt) |

历史环境用 `venv --system-site-packages` 继承已安装 PyTorch。这里的安装工具为新机器创建**不继承系统包**的环境，并显式安装同版本 PyTorch；这是安装方式调整，尚未完成新环境端到端实验回归。其余依赖包括 ftfy 6.3.1、yacs 0.1.8、pytest 7.4.4。

不要直接安装历史 `requirements-resolved.txt` 的完整系统导出：其中包含无关 Anaconda 包和本机构建路径。CPU 调度／统计也要固定环境，记录实际 Python、NumPy、SciPy、scikit-learn 版本；本文提供的 solver 环境可作为新运行的统计环境起点，但不宣称它替代了每一历史批次的环境取证。

### 2.3 新机器创建工作区

先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 `uv`，准备 `git`。环境创建使用 [uv 的独立 venv](https://docs.astral.sh/uv/pip/environments/)，不修改现有环境。

从发布目录设置路径：

```bash
export RELEASE_ROOT="$(pwd)"
# 修改为一个尚不存在的新目录；不要指向已有 medical 实验目录。
export PAPER_ROOT="$RELEASE_ROOT/../paper_workspace"

# 先查看将执行的源码下载命令。
python tools/prepare_workspace.py --workspace "$PAPER_ROOT"
# 正式下载固定版本源码，再覆盖放入本包适配器和实验代码。
python tools/prepare_workspace.py --workspace "$PAPER_ROOT" --execute
```

源码恢复使用 [MODEL_ASSETS.json](resources/MODEL_ASSETS.json) 中的提交版本，以及固定版本 TaskRes／SS-Text 仓库。AlphaRAD 的提交来自本地 Git；RadZero 的提交来自本地上游 `RadZero` 仓库，`RadZero_i2cl` 工作副本本身没有 Git 元数据。该步骤不能自动恢复所有历史本地兼容性修改，模型接口需要用后续 parity／smoke 检查验证。遇到上游 API 差异，应恢复对应原始补丁，而不是放宽数值核验阈值。

创建单个环境：

```bash
# 默认只打印安装命令。
python "$RELEASE_ROOT/tools/setup_environment.py" AlphaRAD --workspace "$PAPER_ROOT"
# --execute 才实际安装；已有 .venv 时拒绝覆盖。
python "$RELEASE_ROOT/tools/setup_environment.py" AlphaRAD --workspace "$PAPER_ROOT" --execute
```

创建全部环境：

```bash
for model in AlphaRAD RadZero_i2cl BiomedCLIP BioViL-T CheXzero MedKLIP MAVL KAD MaCo CARZero TaskRes_SS_Text; do
  python "$RELEASE_ROOT/tools/setup_environment.py" "$model" --workspace "$PAPER_ROOT" --execute || break
done
```

安装流程为：下载指定 Python → 创建 `.venv` → 从 `cu121`／`cu128`／`cu124` wheel 索引安装指定 PyTorch → 安装对应版本锁 → `pip check`。旧版科学计算包和模型代码未验证 Windows／macOS；使用 Linux x86-64。锁文件是保存的版本快照，不包含所有 wheel 哈希；本次文档整理没有在空环境中实际安装这 11 套环境。

模型入口使用各自 `local_setup/python_local.sh`。这些入口启用了离线模式，因此**权重、tokenizer、配置和远程模型代码必须在运行前下载完整**。避免只下载一个 `.pt` 文件。

## 3. 模型源码与预训练权重

以下为本实验采用的发布来源；文件名及可用 SHA-256 见 [MODEL_ASSETS.json](resources/MODEL_ASSETS.json)，八个 baseline 的详细说明另见 `source/baselines/<模型>/local_setup/manifest.json`。权重文件不随 Git 仓库分发。

| 模型 | 官方代码 | 权重入口 | 工作区中的主要文件／目录 |
|---|---|---|---|
| AlphaRAD | [GitHub](https://github.com/jz5426/ECCV-2026-AlphaRAD) | [Hugging Face](https://huggingface.co/maxxyouu/AlphaRAD) | `AlphaRAD/local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar` |
| RadZero | [GitHub](https://github.com/deepnoid-ai/RadZero) | [Hugging Face](https://huggingface.co/Deepnoid/RadZero) | `RadZero_i2cl/local_assets/model/`，含 `model.safetensors`、config 和实现文件 |
| BiomedCLIP | [OpenCLIP](https://github.com/mlfoundations/open_clip) | [模型卡与文件](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | `baselines/BiomedCLIP/checkpoints/`，含 `open_clip_pytorch_model.bin`、tokenizer、`open_clip_config.json`、本地 `text_config/` |
| BioViL-T | [HI-ML](https://github.com/microsoft/hi-ml) | [模型卡与文件](https://huggingface.co/microsoft/BiomedVLP-BioViL-T) | `baselines/BioViL-T/checkpoints/`，含图像权重 `biovil_t_image_model_proj_size_128.pt` 和文本权重 `pytorch_model.bin` |
| CheXzero | [GitHub](https://github.com/rajpurkarlab/CheXzero) | [作者权重目录](https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno) | `baselines/CheXzero/checkpoints/`；使用 `checkpoint_listing.json` 指定的 **10 个成员集成** |
| MedKLIP | [GitHub](https://github.com/MediaBrain-SJTU/MedKLIP) | [checkpoint_final.pth](https://drive.google.com/file/d/1tPpWoxpmBZadCsbqnrmyxIgacRm7S1Xf/view) | `baselines/MedKLIP/local_assets/checkpoint_final.pth` |
| MAVL | [GitHub](https://github.com/hieuvmphan/CVPR2024_MAVL) | [epoch 46](https://drive.google.com/file/d/1kV3kEujvNArFzpdwylFnjfvWXSuMFjs4/view)、[epoch 40](https://drive.google.com/file/d/1kjhLqon_GVNwC6mvJOrWVPcCE9uHxH5N/view) | `baselines/MAVL/local_assets/checkpoint_full_46.pth`、`checkpoint_full_40.pth`；按原 adapter 的任务分支加载 |
| KAD | [GitHub](https://github.com/xiaoman-zhang/KAD) | [KAD-224](https://drive.google.com/file/d/1EOuWd6v4Us3lAYEwGcSPtH29616pELED/view) | `baselines/KAD/local_assets/KAD_224_best_valid.pt`，不要替换成 512／1024 版本 |
| MaCo | [GitHub](https://github.com/SZUHvern/MaCo) | [作者权重](https://drive.google.com/file/d/1B87JuLJnjh_CoQpjT7RWGTp6alfXV8_0/view) | `baselines/MaCo/checkpoints/maco.pth` |
| CARZero | [GitHub](https://github.com/laihaoran/CARZero) | [作者权重](https://drive.google.com/file/d/1kYF-k5otW5DHwz1En5d_ScV3zu2E27Ch/view) | `baselines/CARZero/pretrain_model/CARZero_best_model.ckpt` |

额外资源：

- AlphaRAD／RadZero 的视觉编码器：[StanfordAIMI/dinov2-base-xray-224](https://huggingface.co/StanfordAIMI/dinov2-base-xray-224)；文本编码器：[all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)。RadZero 使用 `local_assets/pretrained/StanfordAIMI/dinov2-base-xray-224` 和 `local_assets/pretrained/sentence-transformers/all-mpnet-base-v2`。
- MedKLIP／MAVL 需要 [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) 及原疾病描述、概念资源；扩展查询使用本实验冻结的描述，不能从当前网页重新生成。
- KAD 需要原 Med-KEBERT tokenizer/config 资源，按其 manifest 和 adapter 的 `local_assets/Med-KEBERT` 布局恢复；完整训练后文本权重已在 KAD checkpoint 中。
- CARZero 使用 [Laihaoran/BioClinicalMPBERT](https://huggingface.co/Laihaoran/BioClinicalMPBERT) 的配套资源，路径为 `baselines/CARZero/local_assets/BioClinicalMPBERT`。
- 比较方法源码：[TaskRes 固定提交](https://github.com/geekyutao/TaskRes/tree/5195359661d6bef661090c33b81aacfc7d7cc181)、[SS-Text 固定提交](https://github.com/jusiro/SS-Text/tree/4bc0c9e3b7b9e0f090079bcc51d7aa852806e91f)。本实验为 feature-level binary OvR 移植，不使用额外 CONCH 编码器。

链接来自本地下载记录和官方仓库；页面访问不代表已验证大文件可下载。AlphaRAD 的 Hugging Face 页面本次未能通过网页工具访问，保留官方 GitHub 指向的原链接。Google Drive／Kaggle／PhysioNet 可能需要账号和访问授权。

## 4. 数据集、固定划分与预处理

下表中的测试数量是**本论文固定支持协议**，不是数据集总大小。逐图身份、类别顺序、支持样本、患者／重复图像分组以冻结清单为准。

| 数据集 | 获取入口 | 论文固定测试范围 | 原工作区图像目录 |
|---|---|---|---|
| ChestXray14 | [NIH 数据发布](https://www.kaggle.com/datasets/nih-chest-xrays/data) | 21,101 图，14 类 | `data/alpharad/ChestXray14/` |
| ChestXDet10 | [作者数据仓库](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset) | 分类 492 图；PG 409 图 | `data/alpharad/ChestXDet10/` |
| RSNA Pneumonia | [RSNA 官方入口](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) | 固定支持修正后 5,297 图 | `data/alpharad/RSNA/`，及原 DICOM／NIH 患者映射 |
| SIIM-ACR Pneumothorax | [竞赛数据](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation) | 分别报告 1,704 和 1,694 图协议 | `data/alpharad/SIIM/`，以及 RadZero 原 DICOM |
| CheXpert | [Stanford 官方入口](https://stanfordmlgroup.github.io/competitions/chexpert/) | 596 图，12 类 | `data/alpharad/CheXpert/` |
| Shenzhen | [NLM 数据入口](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/) | 652 图 | `data/alpharad/ShenzhenXray/` |
| Montgomery | [NLM 数据入口](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html) | 128 图 | `data/alpharad/Montgomery/` |
| CovidKaggle | [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | 1,381 图；本地下载记录为 Kaggle v5 | `data/alpharad/CovidKaggle/` |
| Node21 | [官方挑战](https://node21.grand-challenge.org/) | PG 1,129 图 | `data/alpharad/Node21/`，保留作者 MHA 数据格式 |
| QaTaCOV19 | [作者数据发布](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset) | 测试 2,113 图 | `data/alpharad/QaTaCOV19/`，含 v2 与 Control Group |
| CheXchoNet | [PhysioNet v1.0.0](https://physionet.org/content/chexchonet/1.0.0/) | 7,407 图／2,470 患者，三个二元标签 | `data/alpharad/chexchonet-custom-test-7494/` |

CheXchoNet 需按 PhysioNet 的受限访问流程获取；论文的三个标签共享一个测试队列。OpenI 只保留在历史开发／补评核验依赖中，**不计入当前 11 个报告数据集**；若运行仍认证整个历史批次的脚本，还需恢复其原资产，入口为 [OpenI](https://openi.nlm.nih.gov/)。

### 4.1 不可替换的协议细节

- 固定支持 RSNA：保留原十张支持图，排除支持患者另外 30 张胸片，5,327 → **5,297**。已经隔离患者的五支持抽样／路径实验仍使用 **5,327 图／3,720 患者**。
- 表 1 同时保留两种 SIIM 协议；表 3 每个模型只使用 1,704 图协议，避免重复计权。
- AlphaRAD CheXpert 的完整原始参考为 668 图，论文配对比较为 596 图；两者不能直接相减。
- 五支持种子固定为 42–46，不重新采样或挑选最优种子。SIIM 路径额外留出集为 1,000 图，独立于两个固定支持协议。
- QaTa 验证选阈值使用 473 图／200 患者的验证掩码；测试集仍为 2,113 图。附录 A 的历史测试集选阈值结果与 D.5 验证集选阈值结果分开。
- 支持图只用原视图。测试中心裁剪在模型原生预处理后的张量上执行，边长比例 0.8；不得擅自改成文件级裁剪、不同 DICOM 转换或新版预处理。
- CovidKaggle 使用本地记录中的公开图像和映射；原上游未发布的 `covid_kaggle_256` 副本不能被当作已恢复。复现对象是本论文保存的实际预处理和文件哈希。

### 4.2 必须恢复的实验资产

主要路径如下；本发布草稿未包含这些完整清单／缓存：

| 资产 | 用途 |
|---|---|
| `baselines/evaluation/matched_phase6_20261022/` | 固定测试身份、类别和历史评价范围 |
| `baselines/evaluation/query_transfer_20261022_v1/` | 八模型固定支持与测试协议 |
| `AlphaRAD/local_setup/chexchonet_20261023/`、`baselines/evaluation/chexchonet_20261023/` | 心脏任务支持／测试与查询资源 |
| `AlphaRAD/local_setup/chexpert_query_data_20261022_shot5_seed42/` | AlphaRAD CheXpert 配对协议 |
| `experiments/query_search_20261021_152010/assets/` | 原生查询、参考预测与 `vision_tokens_fp32/` |
| `experiments/fewshot_multiseed_20261024/protocol_v1/`、`cache_reuse_sources_v1.json` | 五支持抽样及缓存认证 |
| `experiments/rsna_chest_stability_20261024/runs/full_20261024_v1/` | 五支持比较复用的原生预测／特征来源 |
| `experiments/fewshot_multiseed_20261024/runs/full_20261024_v2/` | TaskRes／SS-Text+ 消费的冻结特征 |
| 各路径／分割实验的 `protocol_v1/`、SIIM 的 `protocol_siim_v1/` | 固定支持、测试和分组 |
| `experiments/query_resource_completion_20261103/query_descriptions.json` | 肺结核／历史 OpenI 补充描述 |
| `baselines/paper_draft_zh/CHEXCHONET_VERIFIED.json` 和完整 `verified_results.json` | 原始结果认证及汇总 |

还需要各模型配置、tokenizer、资源 manifest 和 protocol 目录的资源链接。**新机器修改路径后，原 JSON 内容哈希可能改变。** 必须在单独的新协议／运行清单中重新记录来源与哈希，保留原审计文件；不能修改旧哈希使错误资产通过检查。

`prepare_protocol.py` 是协议构建实现，不是“重新随机划分也算复现”的入口；它们还依赖上游支持池、身份映射和缓存。当前没有发布脱离这些资产的完整原图冷启动流水线。

运行主要依赖存在性检查：

```bash
python "$RELEASE_ROOT/tools/preflight_reproduction.py" --workspace "$PAPER_ROOT"
```

非零退出表示列出的主要资产有缺失；全部存在也不等于完整依赖已闭合，具体 worker 还会检查文件哈希、类别顺序和原生数值一致性。

## 5. 所有实验命令的公共约定

以下命令在**恢复完资产的工作区**运行，不在发布包根目录运行：

```bash
cd "$PAPER_ROOT"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# 给 CPU 调度／统计提供明确环境，模型 worker 仍调用各自环境。
export PATH="$PAPER_ROOT/experiments/fewshot_taskres_sstext_20261109/.venv/bin:$PATH"
export REPRO_RUN="$PAPER_ROOT/experiments/reproduction_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPRO_RUN"
```

运行顺序为：固定支持评价及补评 → 原生五支持稳定性缓存 → 五支持方法比较 → TaskRes／SS-Text+ → 查询路径干预 → 分割验证阈值 → 参数敏感性 → 统计认证 → 表格。

历史主调度器限制物理 GPU 4–7，心脏路径限制 6、7；敏感性和 TaskRes 接受其他编号。下面保留原入口允许的编号。只有一张卡或设备编号不同的机器，需要先把调度器及 `reserve_gpu` 的本地限制参数化并记录变更；仅设置 `CUDA_VISIBLE_DEVICES=0` 不会自动解除物理编号校验。直接 worker 命令不负责 GPU 锁，只在确认对应 GPU 可用时运行。

所有新结果使用新目录；完整与 smoke 输出分开。不使用历史 `full_*` 目录作为新运行输出，也不要并行启动争用同一 GPU 的实验。

## 6. 固定支持主结果与组件消融

对应正文表 1、3；附录 A.1–A.11、D.1。四分支为 Original、Q、V、Q+V。

### 6.1 八个 baseline

```bash
bash baselines/scripts/eval_query_refinement.sh --gpus 4 5 6 7
```

该入口自动创建新运行目录。BiomedCLIP／BioViL-T 使用 `baselines/protocols/alpharad_binary/` 的二元提示计分版本。只需审查一个模型时可追加 `--models BiomedCLIP`；完整论文应运行全部八个模型。这里四分支已包含 Original，不需要再额外运行一次原始模型取得配对基线。

### 6.2 AlphaRAD／RadZero 原生四分支

先检查 AlphaRAD 的调用计划，不加载模型：

```bash
bash AlphaRAD/local_setup/python_local.sh baselines/paper_draft_zh/scripts/replay_phase6.py \
  --model alpharad --dataset chest14 --gpu 6 --mode four-arm \
  --output "$REPRO_RUN/alpha_dry_run" --dry-run
```

依次执行两个循环：

```bash
for dataset in chest14 chestxdet10 chestxdet10_gt rsna_pneumonia siim_pneumothorax node21 qatacovid_seg shenzhenxray montgomery covidkaggle; do
  bash AlphaRAD/local_setup/python_local.sh baselines/paper_draft_zh/scripts/replay_phase6.py \
    --model alpharad --dataset "$dataset" --gpu 6 --mode four-arm \
    --output "$REPRO_RUN/alpharad/$dataset" || break
done

for dataset in ChestXDet10 ChestXray14 SIIM RSNA; do
  bash RadZero_i2cl/local_setup/python_local.sh baselines/paper_draft_zh/scripts/replay_phase6.py \
    --model radzero --dataset "$dataset" --gpu 7 --mode four-arm \
    --output "$REPRO_RUN/radzero/$dataset" || break
done
```

包装器解除原调度截止时间，但原生特征、查询、视图和计分不变。它明确以 `--cache-mode read` 使用已认证 token 缓存，**不是仅下载图像后即可执行的冷启动入口**。AlphaRAD batch size 8，RadZero 16。OpenI 不在当前报告范围；旧聚合器若认证整个历史目录，则需要保留其原结果。

### 6.3 CheXpert 与六项报告范围内补评

```bash
bash AlphaRAD/scripts/eval_chexpert_query_refinement.sh --gpus 6 7

CUDA_VISIBLE_DEVICES=6 bash AlphaRAD/local_setup/python_local.sh \
  experiments/table2_completion_20261103/alpharad/evaluate_missing.py \
  --dataset siim_radzero_holdout --output "$REPRO_RUN/completion/alpharad_siim1694"

CUDA_VISIBLE_DEVICES=6 bash RadZero_i2cl/local_setup/python_local.sh \
  experiments/table2_completion_20261103/radzero/evaluate.py \
  --datasets chexpert shenzhenxray montgomery \
  --output "$REPRO_RUN/completion/radzero_part1"

CUDA_VISIBLE_DEVICES=7 bash RadZero_i2cl/local_setup/python_local.sh \
  experiments/table2_completion_20261103/radzero/evaluate.py \
  --datasets siim_pneumothorax covidkaggle \
  --output "$REPRO_RUN/completion/radzero_part2"
```

补评脚本会回归历史分支预测，因此还需要原批次参考文件。历史 `verify_completion.py` 固定认证七项（含 OpenI），而当前报告只纳入其中六项；审计旧批次需保留历史 OpenI 记录，不能把七项硬校验改成六项而不记录协议变更。

### 6.4 MedKLIP／MAVL 肺结核查询补全

```bash
for dataset in shenzhenxray montgomery; do
  CUDA_VISIBLE_DEVICES=5 bash baselines/MedKLIP/local_setup/python_local.sh \
    experiments/query_resource_completion_20261103/medklip/evaluate_missing_cached.py \
    --dataset "$dataset" --output "$REPRO_RUN/query_resources/medklip/$dataset" || break
done

CUDA_VISIBLE_DEVICES=7 bash baselines/MAVL/local_setup/python_local.sh \
  experiments/query_resource_completion_20261103/mavl/evaluate.py \
  --datasets shenzhenxray montgomery --output "$REPRO_RUN/query_resources/mavl"
```

Original／Q／V／Q+V 必须共享扩展后的查询上下文。历史核验器还检查 OpenI 补全记录，处理方式与上一节一致。

### 6.5 十模型 CheXchoNet

依次运行，不同时争用 GPU：

```bash
bash experiments/chexchonet_tools/run.sh "$REPRO_RUN/chexchonet_native"
bash baselines/scripts/eval_chexchonet.sh --gpus 4 5 6 7
```

第一条运行 AlphaRAD／RadZero，第二条运行八个 baseline。论文只读取 7,407 图配对范围，不能用 7,494 图原始参考代替。三个任务 CSLV／CDLV／CCOM 保留各自预测，但表 3 将其先平均为一个数据集。

### 6.6 固定 RSNA 患者排除后重新计分

```bash
python experiments/rsna_fixed_patient_cleanup_20261110/recompute.py \
  --root "$PAPER_ROOT" --output "$REPRO_RUN/rsna_patient_cleanup"
```

该程序读取 `baselines/paper_draft_zh/verified_results.json` 指向的原始固定支持预测及患者映射，不重新推理。默认读取的是**历史注册来源**，不会自动选择第 6.1–6.5 节产生的新目录。重计新运行前，必须建立指向新预测的完整来源注册表并保留新哈希。随包的精简 `verified_results.json` 只支持表格渲染，不能用于这一步。

## 7. 五支持种子主比较与 TaskRes／SS-Text+

对应正文表 2，附录 C.1–C.9；BiomedCLIP、AlphaRAD × RSNA、ChestXDet10 × seeds 42–46。

先生成／恢复被主比较认证的原生稳定性缓存：

```bash
bash experiments/rsna_chest_stability_20261024/run.sh "$REPRO_RUN/native_stability"
```

然后运行主比较：

```bash
bash experiments/fewshot_multiseed_20261024/run.sh "$REPRO_RUN/fewshot_multiseed"
```

主比较包括 Original、Prototype、Prototype score mixing、Tip-Adapter、GDA、SimNL、positive-only、S、Q、V、Q+V，输出 220 个逐种子分类结果／44 个方法均值。SimNL 需要拟合小型参数，不能将全部比较称为无需训练。

**缓存衔接：** `cache_reuse_sources_v1.json` 和相关协议默认指向历史 `full_20261024_v1`；新稳定性目录不会被自动发现。使用新缓存前，需生成对应的新来源清单、校验特征／原生预测并更新消费入口，保留原清单和哈希。

TaskRes／SS-Text+ 准备与运行：

```bash
CUDA_VISIBLE_DEVICES='' bash experiments/fewshot_taskres_sstext_20261109/python_local.sh \
  experiments/fewshot_taskres_sstext_20261109/prepare_experiment.py

bash experiments/fewshot_taskres_sstext_20261109/run.sh \
  --output "$REPRO_RUN/taskres_sstext" --device cuda --gpus 0 1 2 3 --workers 4
```

无可用 GPU 时，第二条可改为 `--device cpu --workers 4` 并省略 `--gpus`，使用新输出目录；不要在同一运行中混合 CPU／GPU 拟合。

`prepare_experiment.py` 认证模型权重、上游提交、源码、特征和协议，默认读取 `data_interface.py` 的 `DEFAULT_SOURCE_RUN = .../full_20261024_v2`。若使用本次新主比较结果，需要先在新工作副本中更新该来源和对应 catalog，再重新准备 `READY.json`；不能把旧 READY 用于修改后的源文件。AlphaRAD 两个新增比较方法使用全局池化特征，Q 保留原生读出，二者不是完全相同内部表征的比较。

单独统计新主比较结果：

```bash
CUDA_VISIBLE_DEVICES='' python experiments/fewshot_multiseed_20261024/summarize_multiseed.py \
  --run "$REPRO_RUN/fewshot_multiseed" \
  --protocol experiments/fewshot_multiseed_20261024/protocol_v1/catalog.json \
  --replicates 2000 --seed 20261024

CUDA_VISIBLE_DEVICES='' bash experiments/fewshot_taskres_sstext_20261109/python_local.sh \
  experiments/fewshot_taskres_sstext_20261109/evaluate.py \
  --run "$REPRO_RUN/taskres_sstext" --bootstrap-replicates 2000 --bootstrap-seed 20261024
```

## 8. 查询路径机制实验

对应正文表 4、附录 B.1–B.18。B.7 是干预接口说明表，没有单独的结果生成实验。

### 8.1 AlphaRAD／RSNA、ChestXDet10

```bash
bash experiments/query_path_intervention_20261024/run.sh \
  "$REPRO_RUN/alpha_paths" --gpus 4 5

# 控制器默认执行统计；需要从预测重新统计时使用：
CUDA_VISIBLE_DEVICES='' python experiments/query_path_intervention_20261024/evaluate.py \
  --run "$REPRO_RUN/alpha_paths"
```

四个路径为 Original `(q,q)`、R `(q,q')`、A `(q',q)`、Q `(q',q')`。结果位于运行目录的 **`evaluation/`** 子目录；Q−R 只改变聚合查询。

### 8.2 SIIM 新留出集

```bash
python experiments/query_path_intervention_20261024/run_siim.py \
  --protocol experiments/query_path_intervention_20261024/protocol_siim_v1/catalog.json \
  --output "$REPRO_RUN/siim_paths" --gpus 4 5 6 7

CUDA_VISIBLE_DEVICES='' python experiments/query_path_intervention_20261024/evaluate_siim.py \
  --run "$REPRO_RUN/siim_paths" \
  --protocol experiments/query_path_intervention_20261024/protocol_siim_v1/catalog.json
```

### 8.3 六模型心脏路径

```bash
bash experiments/query_path_multibackbone_20261025/run.sh "$REPRO_RUN/cardiac_paths"

CUDA_VISIBLE_DEVICES='' python experiments/query_path_multibackbone_20261025/summarize.py \
  --run "$REPRO_RUN/cardiac_paths" \
  --protocol experiments/query_path_multibackbone_20261025/protocol_v1/catalog.json \
  --models AlphaRAD RadZero CARZero MedKLIP MAVL KAD \
  --run-plan "$REPRO_RUN/cardiac_paths/FROZEN_RUN_PLAN.json"
```

AlphaRAD／RadZero 比较 Q−R；CARZero／MedKLIP／MAVL／KAD 比较 Q−F。F 固定原始视觉 cross-attention 更新，不能统一解释为六模型 Q−R。全部单原图视图，保留负向和区间跨零结果。

## 9. 验证集选阈值的分割实验

对应附录 D.5：

```bash
bash experiments/qata_validation_threshold_20261024/run.sh "$REPRO_RUN/qata_validation"
```

流程：验证集推理 → 仅根据验证掩码选阈值 → 保存 `thresholds.json` → 固定测试集评分和患者 bootstrap。测试热图默认复用已认证历史缓存，需先恢复该缓存。

单独用已选阈值计分到新目录：

```bash
CUDA_VISIBLE_DEVICES='' bash AlphaRAD/local_setup/python_local.sh \
  experiments/qata_validation_threshold_20261024/evaluate_fixed_test.py \
  --protocol experiments/qata_validation_threshold_20261024/protocol_v1/catalog.json \
  --thresholds "$REPRO_RUN/qata_validation/thresholds.json" \
  --maps "$REPRO_RUN/qata_validation/validation_maps/test_cache_manifest.json" \
  --output "$REPRO_RUN/qata_rescore"
```

应复现的显示值：Original／Q／V／Q+V 阈值为 0.64／0.63／0.64／0.64，Dice 为 52.34／53.81／52.59／54.51%。这是阳性图像上的 Dice；不能与附录 A 历史测试集选阈值 Dice 混合。

## 10. 固定网格敏感性分析

对应附录 D.7、D.8。5 模型 × 2 数据集 × 9 个预设切片，共 90 个模型—数据集—配置组合；不是完整笛卡尔网格。

```bash
# 只生成计划到独立目录，不推理。
python experiments/sensitivity_grid/run_sensitivity_campaign.py \
  --output "$REPRO_RUN/sensitivity_plan" --plan-only \
  --models AlphaRAD RadZero BiomedCLIP MedKLIP CARZero \
  --datasets chexpert chexchonet --gpus 0 1 2 3 --max-workers 4

# 正式运行使用另一个新目录。
python experiments/sensitivity_grid/run_sensitivity_campaign.py \
  --output "$REPRO_RUN/sensitivity" \
  --models AlphaRAD RadZero BiomedCLIP MedKLIP CARZero \
  --datasets chexpert chexchonet \
  --etas 0 0.1 0.2 0.3 0.5 0.7 --crop-ratios 0.6 0.7 0.8 0.9 \
  --view-weight 0.5 --gpus 0 1 2 3 --max-workers 4
```

调度器产生各任务结果。使用本包提供的入口从新批次汇总 90 个切片：

```bash
python "$RELEASE_ROOT/tools/summarize_sensitivity.py" \
  --run "$REPRO_RUN/sensitivity" --output "$REPRO_RUN/sensitivity_summary"
```

该命令生成逐切片、模型—数据集及参数切片三份 CSV 和来源哈希审计。RadZero／CheXpert 的参照必须为 **η=0 的单视图 Original**，不能误用当前 η 的单视图 Q。该汇总已与历史修正网格的全部 90 行数值比较一致（容差 1e-9 pp）。两张 LaTeX 表仍需按汇总维护，`build_supplement.py` 不会重新生成它们。历史正确汇总为：

```text
experiments/sensitivity_grid/campaign_20261108_v3/sensitivity_grid_summary_corrected.csv
```

本包数值核验工具从这个修正后的 90 行网格计算均值；旧 `sensitivity_parameter_summary.csv` 部分值已过时。新运行汇总时，先把 CheXchoNet 三任务在每个模型／参数切片内平均，再跨十个模型—数据集设置平均；不能直接将三任务当作三个独立数据集。该分析不重新选择正文的 η=0.3 和 crop=0.8。

## 11. 从结果到论文表格

### 11.1 已归档结果的原始重建链

下面命令只适用于**恢复完整历史资产**的工作区，并将更新该工作副本的汇总／表格文件。不要在唯一的原始档案上运行。

```bash
python experiments/table2_completion_20261103/verify_completion.py
python experiments/query_resource_completion_20261103/verify_completion.py
python baselines/paper_draft_zh/build_verified_results.py
python latex_ours/scripts/build_supplement.py
python latex_ours/scripts/build_classification_table.py
python latex_ours/scripts/export_taskres_sstext_csv.py
```

RSNA 修正结果须在 `build_verified_results.py` 之前已准备好。`build_supplement.py` 还读取当前附录已注释掉的 cardiac few-shot 数据；为保持旧生成器行为，需要原 `experiments/fewshot_neighbors_20261025/runs/full_20261025_v1` 的汇总 CSV。若需要重新生成这个生成器依赖：

```bash
bash experiments/fewshot_neighbors_20261025/run.sh "$REPRO_RUN/fewshot_neighbors"
```

它不是当前论文新增一组活跃表格，仍需要把新目录接入汇总器才能被读取。

### 11.2 接入新运行结果

**上述旧生成器固定读取历史路径，运行完 `$REPRO_RUN` 并不会自动更新论文。** 修改新工作副本中的以下来源配置，并生成新的审计记录：

| 文件 | 新结果需要接入的位置 |
|---|---|
| `baselines/paper_draft_zh/build_verified_results.py` | 历史聚合、心脏认证来源、`COMPLETION_PATH`、`RESOURCE_COMPLETION_PATH`、`RSNA_CLEANUP_PATH` |
| 两个 `verify_completion.py` | 结果目录、完整批次范围及来源文件；保留对烟雾运行／缺失／重复的拒绝检查 |
| `latex_ours/scripts/build_supplement.py` | `diag`、`siim` 以及 cardiac paths、five-seed、fewshot_neighbors 的 `base` |
| `latex_ours/scripts/export_taskres_sstext_csv.py` | `SOURCE` 指向新的 TaskRes／SS-Text+ 运行 |
| `latex_ours/main.tex` | 正文表 2、表 4 的手工数值与排名标记 |
| `latex_ours/sec/taskres_sstext_comparison.tex` | C.9 配对差值及区间 |
| `latex_ours/sec/validation_segmentation.tex` | D.5 阈值、Dice 与区间 |
| `latex_ours/sec/supplement_tables/e_sensitivity.tex` | D.7、D.8 的切片均值 |

手工表的自动一致性检查见 `tools/verify_manual_tables.py`。它默认核对随包快照，不能直接声称验证了 `$REPRO_RUN`；核验新表时应使用新导出的 CSV 和表格文件建立对应索引。当前还没有一个自动注册全部新批次并更新所有手工表的统一 CLI。

逐表映射见 [TABLE_INDEX.csv](TABLE_INDEX.csv)。B.7、D.2–D.4、D.6 是接口／支持来源／提示／开发范围说明，不对应独立训练指标，需要核对源码和协议文本。

### 11.3 指标与预期计数

- 固定支持：**138 项完成评价 = 120 分类 + 12 定位 + 6 分割**，四分支／指标共 1,032 个数值。分类 Q+V 相对 Original 的双升／双降／混合计数为 95／15／10。
- 分类使用 AUROC 和 **PR 曲线梯形积分 AUPRC**，不能用 average precision 替换；多标签先按有效类别宏平均。
- 表 1 的 Avg. 跨 10 个数据集／协议组；表 3 每个模型跨 9 个数据集，心脏三任务先合并、SIIM 去重。
- 五支持比较先计算每个种子的指标再平均，不能平均五组预测形成集成。
- 配对 bootstrap 通常为 2,000 次、seed=20261024；RSNA／CheXchoNet 按患者，ChestXDet10 按图像／精确重复组，SIIM 按其协议标识组。区间条件于已固定支持集合，未做多重比较校正。
- 模型原生分数域、checkpoint、负查询、query 范数策略、TF32／精度配置和预处理必须保留；显示两位小数一致不等于逐图预测一致。

### 11.4 编译论文

需要完整 LaTeX 工程（包括模板、bibliography、图片等）。本包选取的源码不是完整可编译稿件；恢复原 `latex_ours/` 后：

```bash
cd "$PAPER_ROOT/latex_ours"
latexmk -pdf -interaction=nonstopmode -halt-on-error main_with_supplement.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

优先编译合并稿以生成完整表号，再编译独立附录和正文。表号由交叉引用管理，不手工更改。

## 12. 日志、来源和发布范围

每次保留：命令／环境、运行状态、协议与源码哈希、支持种子、查询拟合、逐图预测、每类／每种子指标、bootstrap 配对结果，以及 `RESULTS.json`／CSV。不同批次日志位置见 [TABLE_LOG_INDEX.csv](TABLE_LOG_INDEX.csv)。

本地审查发现主要实验批次日志均保留；固定支持 147 条来源记录（138 完成，其余为范围内未完成／不适用记录）全部存在且来源哈希一致。RSNA 重新计分目录没有独立文本日志，但保留 `AUDIT.json`、修正结果和分数变化记录。存在日志不代表已逐份确认正常结束。

已归档的日志按原实验目录保存，内容未经改写；空日志、失败及恢复记录也保留，不能把文件数量视为成功实验数量。`PUBLISHED_LOGS.json` 可用于校验归档完整性。`.gitignore` 仅允许 `logs/` 内的文本日志进入版本控制。

不要为了上传日志把整个数据目录、患者级 manifest、预测缓存或 checkpoint 当作普通 Git 文件加入。受数据访问条款限制的资产应通过相应授权渠道获得；模型／数据沿用上游许可证，本仓库尚未替作者选择新的代码许可证。

**当前验证范围：** 表格重建、227 个手工表数值、来源文件哈希，90 个敏感性切片从结果文件的重新汇总，以及本文新增工具的命令生成／语法检查。未重新进行 GPU 推理、未在新机器实测环境安装、未声称全部冻结资产已公开。完整从原图复现的最后发布依赖是：可获取的协议／缓存及重建方式、原生兼容补丁、可迁移路径，以及新结果自动注册流程。
