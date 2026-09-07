# TAGLAS 数据集选型 + embed→GNN 基线实验方案（第 c 步）

> 目标：为"NSPGNN 扩展到文本属性图"搭起评测底座——选数据集、定基线矩阵、给 H100 可执行脚手架。
> 所有攻击档位与评测协议对齐 TGRB（见 docs/tgrb_inventory.md），保证后续论文数字可与 ICLR'26 那篇直接对照。
> 文档日期：2026-09-05

---

## 1. 选型原则

1. **文本语义要足够丰富**：文本级攻击（LLM 改写）才有意义。分子/推荐类 TAGLAS 数据集的"文本"偏形式化（SMILES/属性串），排除在文本攻击主阵外。
2. **覆盖同配/异配与规模梯度**：既要复现 TGRB 结论（好对照），又要能讲"NSPGNN 跨同配/异配统一"的故事。
3. **结构攻击要算得动**：PGD 类白盒攻击是 O(E) 迭代，先在中小图验证，再上大图。
4. **同一数据契约**：不管数据来自 TAGLAS 还是 TGRB/LLMNodeBed，落地成统一 .pt（`node_texts / edge_index / y / masks`），实验代码只认这一种格式（见 exp/ 脚手架）。

## 2. 主阵数据集（TAGLAS）

| 数据集 | 规模(N/E) | 类别 | 文本 | 同配性 | 用途 |
|---|---|---|---|---|---|
| **cora_node** | 2.7k / 5.4k | 7 | 标题+摘要 | 高 | 冒烟测试、超参、理论小实验 |
| **pubmed_node** | ~19.7k / 44k | 3 | 标题+摘要 | 高 | 中等规模结构/文本攻击主战场 |
| **wikics** | 11.7k / 216k | 10 | 维基正文 | 中高 | 文本攻击+结构攻击均衡验证 |
| **arxiv** | 169k / 1.17M | 40 | 标题+摘要（官方 split） | 高 | 规模论证（GRBCD 式采样或只跑 clean/文本攻击） |

补充建议：电商系 products-subset（描述文本）可作跨领域验证；**TAGLAS 缺低同配性的文本图**——异配性论证需借 TGRB/LLMNodeBed 的 History/Photo/Computer/Instagram/Reddit（文本字段已备，直接可用），作为"跨库一致性实验"，顺带证明不是只在 TAGLAS 上过拟合。

## 3. 编码器 × 模型 × 攻击 × 协议矩阵

### 3.1 编码器（对齐 TGRB 命名）
- **BoW**（TF-IDF，稀疏）：结构攻击 surrogate 专用 + 低配对照。
- **MiniLM**（all-MiniLM-L6-v2，384d，快）：默认防御编码器，迭代主力。
- **RoBERTa**（all-roberta-large-v1，1024d，慢）：高配对照（TGRB 主 encoder）。
- 每个数据集只编码一次，缓存到 `cache/{dataset}/{encoder}.pt`，攻击改写后只重算被攻击节点子集。

### 3.2 模型
| 族 | 模型 | 说明 |
|---|---|---|
| 基础 | MLP / GCN / GAT / GPRGNN | 无防御下界 + 结构鲁棒上界参考 |
| 相似度过滤族 | **CosGCN**（GNNGuard 式余弦门控的轻量实现） | 与 NSPGNN 同思路的对照；预期"结构强、文本攻击崩"（复现 TGRB 发现 1） |
| 本组方法 | **NSPGNN**（插件占位） | 接口已在 models.py 预留，接课题组源码 |

### 3.3 攻击档位（照 TGRB）
| 攻击 | 协议 | 预算 | 生成编码器 |
|---|---|---|---|
| 结构 PGD（surrogate=GCN+BoW） | evasion / inductive(60/20/20) | 0.20 | BoW |
| 结构 Heuristic（DICE 式） | poisoning / transductive(10/10/80) | 0.30 | BoW |
| 文本 LLM 改写（neighborhood-aware prompt，**本地 Qwen via vLLM**，不开 OpenAI 也能跑） | evasion 0.40 / poisoning 0.80 | — | 任意 |
| Clean | 全模型必测 | — | 每编码器 |

### 3.4 分阶段执行（H100 预算估算：A100/H100 单卡 80G）
| 阶段 | 内容 | 说明 |
|---|---|---|
| S0 smoke（~30 min） | cora_node × {bow,minilm} × {MLP,GCN,CosGCN} × clean+PGD0.1，1 seed | 验证脚手架/数据契约/缓存逻辑 |
| S1 clean 表（~2–4 h） | 主阵 4 数据集 × {BoW,MiniLM,RoBERTa} × 5 模型 × 3 seeds | 产出 clean 基准，校准模型实现是否正常（对照 TGRB clean 数量级） |
| S2 结构攻击（~4–8 h） | evasion PGD 0.2 + poisoning Heuristic 0.3，主阵中小图 | 复现"CosGCN 结构强"、看 GCN/GAT 崩幅 |
| S3 文本攻击（~6–12 h，Qwen3-32B 本地改写） | evasion 0.4 / poisoning 0.8，RoBERTa/MiniLM 编码下全模型 | 复现"相似度过滤族文本崩"——NSPGNN-T 要超越的曲线 |
| S4 跨库/异配补充 | History/Photo/Computer/Instagram/Reddit（LLMNodeBed .pt） | 异配性统一主张的证据 |

## 4. 验收标准（跑完怎么算通过）

1. S0 全绿：数据契约 OK、编码缓存命中、攻击产物能被评测脚本消费。
2. Clean 数字与 TGRB 同数据集同编码器同模型的数量级一致（±3pp 内），例如 Cora 上 GCN+RoBERTa 应在 ~86–88 区间；不对就查实现而非调数。
3. 复现两个"已知现象"才继续：① CosGCN 结构攻击下优于 GCN、文本攻击下反而不如 GCN（trade-off）；② 文本 poisoning 0.8 下 GNN 系掉点 ≤10pp（靠 transductive 聚合扛）。
4. 报告格式固定：3 seeds mean±std + 每张表附 average rank（和 TGRB 同口径）。

## 5. 输出物清单（本次已交付）
- `exp/README.md`：部署到 H100 的分步说明 + 冒烟验收清单。
- `exp/config.yaml`：默认配置（编码器/模型/攻击/协议/种子）。
- `exp/requirements.txt`：依赖。
- `exp/taglas_bench.py`：核心管线（数据契约加载→编码缓存→模型 zoo→攻击→训练评测→报告），单文件 CLI。
- `exp/prepare_data.py`：TAGLAS / TGRB(.pt) 两种源 → 统一 .pt 的转换脚手架（TAGLAS 侧需在装好 TAGLAS 后按报错逐步补齐，一次性的活）。

> 诚实声明：脚手架在本机（Mac，无 GPU/未下载数据）只做了静态检查，未实际跑通；到 H100 后严格按 README 的 S0 冒烟清单先校准，再铺量。
