# TGRB 复现与接入对照清单（NSPGNN → 文本图扩展用）

> 用途：课题组把统一鲁棒性方法（NSPGNN，保邻域相似性的鲁棒结构学习）扩展到文本属性图（TAG）。
> TGRB 是直接对标 + 可复用的评测框架。本文档把论文细节与官方仓库（github.com/Leirunlin/TGRB，arXiv:2510.17185）核对后的可执行清单整理如下。
> 文档日期：2026-09-05。

---

## 1. 一句话定位

**TGRB（ICLR 2026，人大 Zhewei Wei 组 + 蚂蚁）= TAG 鲁棒性的第一个统一评测框架**：经典 GNN / 鲁棒 GNN(RGNN) / GraphLLM 三大族 × 10 数据集 / 4 领域 × 文本/结构/混合扰动 × 毒化(poisoning)/逃逸(evasion)。
它挖出了坑（文本-结构鲁棒性 trade-off、GNN 性能依赖文本编码器、GraphLLM 怕数据污染），并给了唯一的防御 SFT-auto——**只有评测+经验防御，没有理论**。这正是 NSPGNN 类"定理 + 机制"方法切入的位置。

---

## 2. 评测框架速览

| 维度 | 内容 |
|---|---|
| 被评模型 | GCN、GAT、APPNP、GPRGNN（基础）；GNNGuard、ElasticGNN、RobustGCN、SoftMedianGDC、RUNG、EvenNet、GCORN、GRAND、NoisyGCN、Jaccard-GCN、Cosine-GCN、ProGNN、Stable、GPRGNN-AT、GOOD-AT、GPR-GAE、Guardual（RGNN）；GraphGPT、SFT-neighbor、LLaGA、**SFT-auto**（GraphLLM） |
| 模型筛选原则 | 只保留 clean 精度可比的模型（避免"强主干=假鲁棒"）；InstructGLM/GPT zero-shot/GraphPrompt 被排除 |
| 数据集 | Cora、CiteSeer、PubMed、ArXiv（学术）；WikiCS（网页）；Instagram、Reddit（社交）；History、Photo、Computer（电商） |
| 文本编码器 | 默认 **RoBERTa**（GNN 系，仿 LLMNodeBed）；全套附录另报 BoW / Mistral-7B(4096d) / MiniLM；GraphLLM 基座 **Mistral-7B**（LoRA r8/α16） |
| 评测指标 | accuracy；3 个随机 split 报 mean±std（ArXiv 用官方单 split）；**正文报 average rank**（跨数据集可比，`–`=OOM 不计） |
| 协议对齐 | **poisoning ↔ transductive**（10/10/80 半监督），**evasion ↔ inductive**（60/20/20）；无向图；默认 transfer 攻击（白盒 surrogate 生成、黑盒转移 defender），自适应在附录 |

## 3. 攻击规格速查（复现/接对照时必须对齐）

| 攻击 | 类型 | 设定 | 预算 | 生成用 embedding | 入口 |
|---|---|---|---|---|---|
| **PGD** | 结构·白盒梯度 | inductive/evasion（小图） | **0.20** 边 | BoW（surrogate=GCN+BoW） | attacks/gen_attacks_inductive.py |
| **GRBCD** | 结构·贪心随机块坐标下降 | inductive/evasion（大图） | 0.20 边 | BoW | 同上 `--attack grbcd` |
| **HeuristicAttack** | 结构·DICE 式灰盒 | transductive/poisoning | **0.30** 边（排除 Computer/ArXiv） | BoW | gen_attacks_transductive.py |
| **PGD-Guard** | 结构·限相似度对的自适应 PGD | 专打 GNNGuard 类过滤 | 0.20 边 | **RoBERTa**；cos 阈值 {0,0.3,0.5,0.7} | gen_attacks_inductive_guard.py |
| **LLM 文本攻击** | 文本·GPT-4o-mini 改写（neighborhood-aware prompt） | poisoning **80% 训练节点** / evasion **40% 测试节点** | 高 | 不限（跨 encoder 普适） | gen_text_attacks_*_llm.py `--attack gpt` |
| **TextFooler 等** | 文本·word-level | 仅附录分析（过拟合 surrogate+embedding，正文弃用） | 0.40 | MiniLM | gen_text_attacks_*.py |
| **WTGIA** | 混合·文本级节点注入（Llama-3.1-8B 生成） | 仅 Cora/CiteSeer/PubMed | 0.2/0.4 | 攻击嵌入与 def 可不同 | gen_wtgia_inductive.py |

> 关键提醒：正文把 Mettack 弃了——它在 validation-based 防御面前攻击力不足（Cora：Mettack 只降到 77.86，HeuristicAttack 降到 70.33，clean 82.91）。**做"我们的防御 vs TGRB 攻击"对比时别用弱攻击给自己注水。**

## 4. 接入 NSPGNN 的代码位置

- 仓库结构：`Embedding/`（4 编码器×10 数据集的嵌入生成）、`attacks/`、`defenses/`（GNN 评测）、`LLMPredictor/`（GraphLLM + **SFT-auto**，`InstructionTuning/auto_utils.py`）、`common/`、`LLM_scripts/`、内置修改版 **GreatX** 图攻防库。
- GNN 模型 zoo 实现在 `GreatX/greatx/nn/models/supervised/`（gcn/gat/gnnguard/guarddual/purification/rung/evennet/prognn/stable…），评测 CLI 模型名由 `defenses/config.yaml` 的 `hyperparams:` 定义。
- **NSPGNN 接入 = 在 GreatX supervised models 下新增 NSPGNN 类**（双 kNN 图构造 + 邻域相似性保持传播），并在 `defenses/config.yaml` 注册后即可复用全部攻击与评测脚本。
- **算力注意**：NSPGNN 双 kNN 是 O(N²) 相似度计算——TGRB 的 Computer(87k 节点)/ArXiv(169k) 直接跑会爆；论文建议 Recursive Lanczos Bisection/MapReduce 加速（O(N^1.14)），或先在 Cora/CiteSeer/PubMed/WikiCS 上接入验证。

## 5. SFT-auto 配方（正文核心防御，理论防御要超的对象）

训练（Mistral-7B + LoRA，SFT）三类样本：
1. **Normal**：原样 node–neighbor 对（保标准分类能力）；
2. **Attack**：中心文本替换为异类文本，标签扩展成 (|C|+1) 类里的 **"text attacked"**（教模型识别攻击）；注入比例 r = min(1/|C|, 0.15)；
3. **Recovery**：整段删中心文本只给邻居，迫使"用邻居恢复预测"。

推理三阶段：① 检测——文本攻击 = LLM 判到 "text attacked" 类；结构攻击 = **embedding 余弦相似度 < 0.5 的邻居过半**则判 attacked（τ=0.5，非学习式）；②③ 恢复——文本攻击节点丢弃自身文本只用邻居；结构攻击节点用自身文本+过滤后邻居；正常节点仅滤掉 attacked 邻居。
AutoGCN 消融证明：把 LLM 换成 GCN 后文本异常检测率差 **6.2–17.4×** → 语言级文本攻击检测是 LLM 的护城河，纯 GNN 路线打不过它做检测，但可以走"相似性保持传播"路线在表示层防御。

## 6. 复现最小步骤（H100 就绪清单）

```bash
# 依赖（仓库无 requirements.txt，按 README）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric transformers accelerate sentence-transformers textattack openai
pip install scikit-learn numpy pandas tqdm pyyaml
# GreatX/ 目录内置，需按官方方式编译可选 CUDA op

# 配置（两处硬编码，必须改）
#  common/model_path.py  → HF 模型路径：MiniLM/RoBERTa(embedding)、Mistral-7B/Qwen2.5-7B/Llama3.1-8B(LLM)
#  common/dataloader.py  → PATH 与 ATKG_PATH（数据根目录，无环境变量，纯硬编码）

# 数据：GitHub README 给 Google Drive 单档 / HF: xxwu/LLMNodeBed
#      （图 .pt 含 raw_texts 字段；embedding 子目录名即 --emb_type）

# 嵌入 → 攻击 → GNN 评测 → LLM 训练（最小序列）
cd Embedding && bash gen_all.sh
cd ../attacks && python gen_attacks_inductive.py --dataset cora --ptb_rate 0.20 --attack pgd --emb_type bow --re_split 2
cd ../defenses && python eval_inductive.py --dataset cora --model gcn --attack pgd \
    --atk_emb_type roberta --def_emb_type roberta --ptb_rate 0.20 --device 0
cd ../LLM_scripts && bash run_sft_ind.sh cora 0 Mistral-7B auto   # ★ SFT-auto
```

## 7. 关键数值备忘（引用/写 related work 用）

- **发现 1（trade-off）**：文本攻击(evasion 0.4)下，结构导向模型崩——GNNGuard：Cora 83.64→51.91、CiteSeer 74.03→38.82；RUNG：CiteSeer 73.30→38.56；而 **naive GCN/GAT 反而是文本攻击下最稳的**（GCN Cora 87.76→81.12）。反过来结构攻击下 GNNGuard/NoisyGCN/SFT-neighbor 强。
- **发现 2（encoder 决定论）**：GNNGuard 配 RoBERTa 在结构攻击下第一梯队（此前只在 BoW 上测过所以被低估）；embedding 的边区分质量（intra/inter AUC）：Cora 上 BoW 0.565 / Mistral-7B 0.579 / RoBERTa 0.650 / MiniLM 0.662 → **contextual embedding 区分力远超稀疏表示**。
- **发现 3（GraphLLM 怕投毒）**：文本 poisoning(0.8) 下 SFT-neighbor 在 CiteSeer 掉 ~25–35%（附录表推算 ~35.5%），多数 GNN 只掉 5–10%（靠 transductive 聚合邻居扛）。
- **SFT-auto 效果**：结构 evasion 下 Cora 82.59 / WikiCS 84.05（几乎不掉）；文本 evasion 下 WikiCS 77.83 vs SFT-neighbor 51.53。是唯一两类都稳的。
- **附录 K 结论**：对抗训练(GPRGNN-AT/GPR-GAE/GOOD-AT)也逃不开 trade-off → 更坐实"统一理论防御"缺位。

## 8. 论文/仓库的坑（照做前先看）

1. 仓库**无 License**、单次快照提交，README 部分沿用了上游 LLMNodeBed 模板（如提到的 main.py 实际是 train.py），落地以目录内脚本为准。
2. `common/dataloader.py` 路径**硬编码**无环境变量；re_split 语义在两处文档不一致（README 说 1=supervised，API 摘录说 2=60/20/20 inductive），落地时以代码为准。
3. 论文内部不一致：结构 poisoning 表题注 ptb=0.2 vs 正文 0.30；正文"SFT-neighbor 掉 25%" vs 附录表推算 ~35%；正文 Figure 4/6/10 的 pdftotext 列错位（需查原 PDF/原图）。引用数字时以附录数值表为准并注明口径。
4. 10 个数据集的**文本字段来源未在论文写明**（标题+摘要？简介？评论？）——**已核对解决**：LLMNodeBed 源码 `common/descriptions.py` 确认 raw_texts 来源（学术图=标题+摘要、社交图=profile/subreddit 内容、电商图=评论/描述），详见 `docs/llmnodebed_text_fields.md`。做文本攻击实验前仍建议抽样质检语义质量。
5. GraphLLM 端到端评测（预测→acc）脚本未能从仓库核对到唯一入口（inference_vllm.py 推测），跑 LLM 前先摸清其产出格式。

## 9. 对 NSPGNN → TAG 扩展的启示（本清单的核心结论）

1. **威胁模型要对齐**：inductive+evasion / transductive+poisoning 两套协议分开；结构 PGD 0.2 / Heuristic 0.3 / 文本 GPT 改写 0.4(evasion) 与 0.8(poisoning) 是"公平对比"的标准档位。
2. **文本攻击是 NSPGNN 的命门**：NSPGNN 的相似性监督在"结构攻击偏好=连接低相似对"上成立；但 TGRB 证明**文本攻击会直接把节点的嵌入语义改写**——相似度过滤器（GNNGuard 类）在文本攻击下全崩。NSPGNN 若只防结构攻击在 TAG 上会显得偏科，必须把"文本改写导致的嵌入偏移"纳入相似性理论。
3. **encoder 是新的自由度，也是新的理论变量**：同样一个 GNNGuard，BoW 下平庸、RoBERTa 下第一梯队。NSPGNN 扩展到 TAG 时，"对 encoder 鲁棒"或"把 encoder 纳入邻域相似性保持训练"比换数据集更本质。
4. **量化诊断工具现成**：TGRB 附录 G 的 intra/inter 边区分质量指标（Separation/Cohen's d、AUC、Discriminability、Threshold Gap、1-Overlap）可直接用来诊断"双 kNN 图在文本攻击后还能不能分善恶边"。
5. **纯 GNN 检测打不过 LLM，但表示层防御是另一条路**：SFT-auto 用 LLM 的 (|C|+1) 类检测文本攻击，AutoGCN 证明 GNN 学不会；NSPGNN 的价值不在"检测"而在"传播时保相似性、抑制被污染信息扩散"——这与 SFT-auto 的结构攻击恢复（滤低相似边）同思路但更早更理论化。**把 Theorem 1 推广到文本+结构混合扰动，恰好补 TGRB "只有评测+经验防御、无理论"的缺口。**

## 10. 与 TAGLAS 的重叠备忘（供下一步数据集选型）

TGRB 10 数据集 ↔ TAGLAS 25 数据集重叠：**Cora、PubMed、ArXiv（学术）、WikiCS（网页）、Photo/Computer/History（电商，TAGLAS 侧为 Product-subset 等）**。
- 想"复现 TGRB + 换自家防御" → 直接用 TGRB/LLMNodeBed 那套数据（文本字段已备好）。
- 想"跨库验证泛化 + 图语言模型" → TAGLAS 的 cora_node/cora_link/pubmed/arxiv/wikics/products 与 QA 任务型可做扩展舞台。
- 文本扰动实验优先选文本语义丰富的：Cora/PubMed/ArXiv/WikiCS；分子/推荐类文本偏形式化，不适合文本级攻击。
