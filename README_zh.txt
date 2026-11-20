论文表格实验代码整理说明
======================

对应稿件：latex_ours/main_with_supplement.tex。
清点范围：正文 4 张表、附录 47 张表，共 51 张当前启用的表。
原论文、实验程序和原始结果未修改；本目录是独立整理的发布草稿。

先看这些文件：
1. TABLE_INDEX.csv：逐表编号、标签、LaTeX 文件、实验类别、运行入口及生成方式。
2. README.txt：英文复现说明，按实验类别说明代码调用链。
3. source/：按原相对目录组织的推理、评估、统计与表格生成代码。
4. tables/：51 张表的独立 LaTeX 环境，便于逐表审查。
5. ENVIRONMENTS.json：各模型历史实测 Python / PyTorch / 依赖版本。
6. RESULT_PROVENANCE.json、SOURCE_MANIFEST.json：原结果来源与整理文件 SHA-256。
7. PORTABILITY_AUDIT.json：本机绝对路径和本地运行限制的候选位置，含测试样例等误报。

可直接验证的部分
----------------
在本目录运行（Python 3.10+，只需标准库）：

  python tools/rebuild_tables.py --output /tmp/paper_tables_check
  python tools/verify_manual_tables.py

输出目录须不存在。表格重建在临时目录中完成，不覆盖论文或实验结果。
第一个命令使用已保存的汇总数据重建 Python 生成的表格，并对已有论文
快照逐字节比较；额外生成的隐藏表和中文表单独记录为无对应快照。
第二个命令核对手工表中的 227 个数值：正文表 2、表 4，附录 C.9、D.5、D.7、D.8。
这些检查不等于从原图重新推理，也不重新计算原始预测的指标。

发布前仍需配置的部分
--------------------
完整推理还需要模型仓库、权重、原始数据、冻结的支持／测试清单，以及各模型
独立环境。本包保留原实验代码，不包含这些大文件或患者／图像级清单。
部分历史启动器指定 GPU 4–7 或其他本地设备范围，使用本机虚拟环境路径；
相关位置见 PORTABILITY_AUDIT.json。该扫描是候选问题清单，并非全部依赖已闭合
的证明。暂不能宣称本包在新机器上可一键重跑全部实验。

source 保留适配器目录及其辅助脚本，以便追溯交叉目录导入；其中部分历史辅助
程序不是当前论文入口，以 TABLE_INDEX.csv 和 README.txt 指定的入口为准。
上游第三方模型代码、权重和数据应按各自许可获取，本次未替作者选择开源许可证。
本次未运行 GPU 实验，未推送到 GitHub。

容易误用的来源
--------------
- 敏感性表使用 sensitivity_grid_summary_corrected.csv。历史目录中的
  sensitivity_parameter_summary.csv 是旧汇总，部分值与当前论文不一致，未纳入本包。
- 固定 RSNA 使用患者排除后的 5,297 张；五支持抽样／路径实验沿用已隔离的
  5,327 张，不能合并成同一协议。
- 表 1 的 10 个数据集／协议组均值与表 3 的每模型 9 数据集均值定义不同。
- AUPRC 为梯形积分，不能用 average precision 替换。
- 固定支持集分割表的历史测试集选阈值，与 D.5 的验证集选阈值是两个协议。
- build_supplement.py 不生成 e_sensitivity.tex；本包对其数值另行核对。
- 注释掉的表、token-MLP 稿件及后续其他方法不属于这份论文的活跃表格。
  fewshot_neighbors 因原 build_supplement.py 仍读取它而作为重建依赖保留。

工具说明
--------
tools/organize.py、collect_evidence.py 是维护者从原 medical 工作区收集代码的工具，
需要原工作区；读者只运行 rebuild_tables.py 和 verify_manual_tables.py 即可核验表格。
