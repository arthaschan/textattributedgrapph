# 主研究计划：NSPGNN → 文本属性图（TAG）的理论化统一鲁棒防御

> 本文档是把仓库里三份材料（`theory_extension_roadmap.md` 第 b 步、`tgrb_inventory.md` 清单、
> `taglas_baseline_plan.md` 第 c 步）整合成的**单一执行入口**：一份计划，一条关键路径，
> 明确的 go/no-go 门禁与本周行动。细节以各分文档为准，本文只做优先级、依赖与决策收敛。
> 文档日期：2026-09-07。

---

## 0. 一句话主张（论文立意）

在 TAG 上，文本扰动与结构扰动对分类损失的影响可分解为「各自独立项 + 一个交叉项」；
结构项正是 NSPGNN 已证的 A^τ X 相似度（攻击偏好=连低相似对，Theorem 1），
文本项是节点嵌入沿「远离本类质心、靠近邻居优势类」方向的偏移。
统一鲁棒性 = 同时抑制这两类偏好 —— 给出超越 TGRB SFT-auto（纯经验）的第一个**理论化**防御框架 NSPGNN-T。

**为什么现在做、为什么是我们**：TGRB（ICLR'26）证明了「文本-结构鲁棒性 trade-off」与
「相似度过滤族（GNNGuard/CosGCN）结构强、文本崩」，但只给了评测 + 经验防御（SFT-auto），
明确留白「有评测、有经验防御、无理论」。我们已有 NSPGNN 的定理（Theorem 1）与
双 kNN 相似性保持传播（RGSL）源码资产，把 Theorem 1 从普通图推广到 X=E(S) 的文本图，
是成本最低、卖点最清晰的理论切入。

---

## 1. 现状盘点（已有什么，缺什么）

| 资产 | 状态 | 位置 |
|---|---|---|
| NSPGNN 定理 Theorem 1 + RGSL 防御 | 已发表（arXiv:2401.09754），源码待确认可复用 | `neighbor similarity.pdf` |
| TGRB 评测/攻击/防御清单 | 已核对成可执行清单 | `docs/tgrb_inventory.md` |
| 理论推广路线（P1–P4） | 已成形 | `docs/theory_extension_roadmap.md` |
| 数据集选型 + 基线矩阵 + 脚手架 | 初稿（未 GPU 实测） | `docs/taglas_baseline_plan.md` + `exp/` |

**缺（=本计划的阻塞项）**：
1. NSPGNN 源码访问与许可（接入 `exp/taglas_bench.py` 的 `NSPGNN` 占位类）。
2. 数据：TAGLAS 或 TGRB/LLMNodeBed 的统一 .pt（`prepare_data.py` 的 TAGLAS 接线是 TODO）。
3. H100 环境 + 本地 vLLM（Qwen3-32B）用于文本改写攻击。
4. 三件需要与老师/师兄拍板的决策（见 §6）。

---

## 2. 核心研究问题（RQ）

- **RQ1（可迁移性）**：X=E(S) 由预训练编码器给出时，Theorem 1 的结构攻击偏好是否仍成立？成立条件？
  —— 预期成立：Theorem 1 证明只依赖 surrogate 结构与损失形式，与 X 来源无关。需在 TAG 上复刻 NSPGNN Fig.2 密度图（恶意 vs 良性边在 Ω(X)/Ω(AX)/Ω(A²X) 的 KL 距离）实证。
- **RQ2（文本攻击的几何刻画）**：文本扰动 ΔX 如何影响攻击损失更新？能否证明「文本攻击偏好 = 把嵌入推向类间边界的异类方向」（对齐 TGRB 的 neighborhood-aware prompt 目标）？
- **RQ3（统一防御的保证）**：怎样的邻居相似性保持传播对「结构偏好 + 文本偏移」同时免疫？能否给出可验证条件（对 E 的 Lipschitz 常数、文本扰动幅度的依赖）？

---

## 3. 理论贡献（P1–P4，按价值/风险排序）

| 命题 | 内容 | 价值 | 风险 | 定位 |
|---|---|---|---|---|
| **P2** | 文本攻击=结构攻击在嵌入空间的替身：对 L_NLL 做 (ΔA,ΔX) 联合二阶展开，得三项（结构项·K + 文本项·M + 交叉项·C） | 核心（解释器：为何 GNNGuard 文本崩） | 中（展开不封闭→退到定性） | **主卖点** |
| **P3** | 文本攻击的「类方向」偏好：CE 下单节点改写使损失下降最快的方向 = (s_v − 邻居优势类质心)，neighborhood-aware 攻击近似最优 | 把 TGRB prompt 攻击规范化为嵌入梯度攻击 | 中 | 可单独成命题 |
| **P4** | NSPGNN-T 的可验证鲁棒性上界 O(ε√Δ_struct + L·Δ_text√logN) | 收尾的理论保证 | 中高（上界松→弱化单调性定理） | 保证章节 |
| **P1** | 结构偏好迁移（X→E(S) 逐字迁移 Theorem 1） | 低（立论第一块砖） | 低 | 基础，非卖点 |

执行顺序：**先 P1（便宜、立地基）→ P2/P3 推导 → P4**。P2 卡住走退路 1（定性刻画 + 强实验）。

---

## 4. 方法设计：NSPGNN-T（统一防御蓝图）

自底向上五层，每层有可测中间指标：

1. **编码层 E**：冻结预训练 encoder（默认 RoBERTa/MiniLM），可选 LoRA 微调（放附录，正文冻结）。
2. **净化层**：语义一致性正则 R=E[‖E(S)−E(T(S))‖]（T=同义/释义扰动集，TGRB 弃用 TextFooler 不等于训练期不能用）+ 轻量邻居聚合去噪（1 轮特征传播到 k 近邻均值）。
3. **相似性层**：在净化特征 X̂ 上算 Ω(Â^τ X̂)，构造正/负双 kNN 图（原 RGSL）。
4. **传播层**：低通沿正 kNN 平滑、高通沿负 kNN 判别 + 自适应门控 α/β（原 NSPGNN）。
5. **训练目标**：分类损失 + 语义一致性正则 +（可选）结构/文本对抗样本 consistency。

与 SFT-auto 的叙事分工：SFT-auto 用 LLM 做**检测+恢复**（推理期、靠语言理解）；
NSPGNN-T 用**相似性保持传播**（训练期、靠几何结构）让被污染文本在传播中自然失效 ——
回应 TGRB future work 第一条「GNN 结构鲁棒 + LLM 语义理解」混合管线。

---

## 5. 实验计划（阶段 S0–S4 + 理论验证实验）

### 5.1 理论验证实验（对应 P1–P4）
| 验证对象 | 实验 | 数据 | 对标 |
|---|---|---|---|
| P1 | 密度图：恶意/良性边在 Ω(X)/Ω(AX)/Ω(A²X) 的 KL 距离 | cora/pubmed/wikics(+History 异配) | NSPGNN Table I |
| P2 | 三类攻击损失相对贡献：纯结构/纯文本/混合（同预算） | 同上 | TGRB hybrid |
| P3 | 文本攻击方向 vs (s_v−质心) 余弦一致度；改写嵌入偏移分布 | cora_node | TGRB Fig.7/G.2 的区分度指标 |
| P4 | NSPGNN-T vs {GCN,GAT,CosGCN,GNNGuard*,ProGNN*} × {PGD 0.2, Heur 0.3, Text 0.4/0.8} | 主阵 | TGRB Tables（同档位 rank） |
| 消融 | 净化层有无 / 一致性正则系数 / τ / 双 kNN k | cora_node | — |

*GNNGuard/ProGNN 用 TGRB 仓库（GreatX vendored）直接复现，不自己重写。

### 5.2 工程阶段（H100，估算单卡 80G）
| 阶段 | 内容 | 预算 | 产出/验收 |
|---|---|---|---|
| S0 smoke | cora_node × {bow,minilm} × {MLP,GCN,CosGCN} × clean+PGD0.1 | ~30 min | 数据契约/缓存/攻击产物全绿 |
| S1 clean 表 | 4 数据集 × 3 encoder × 5 模型 × 3 seeds | ~2–4 h | clean 基准，与 TGRB 数量级对齐（±3pp） |
| S2 结构攻击 | evasion PGD 0.2 + poisoning Heur 0.3 | ~4–8 h | 复现「CosGCN 结构强」 |
| S3 文本攻击 | evasion 0.4 / poisoning 0.8（本地 Qwen3-32B） | ~6–12 h | 复现「相似度过滤族文本崩」——要超越的曲线 |
| S4 跨库/异配 | History/Photo/Computer/Instagram/Reddit | — | 异配性统一主张证据 |

### 5.3 验收门禁（go/no-go）
1. S0 全绿（数据契约、编码缓存、攻击产物可消费）。
2. Clean 与 TGRB 同配置数量级一致（如 Cora+GCN+RoBERTa ≈ 86–88），不对就查实现，不调数。
3. 复现两个已知现象才继续铺量：① CosGCN 结构强、文本弱（trade-off）；② 文本 poisoning 0.8 下 GNN 掉点 ≤10pp。
4. 报告格式固定：3 seeds mean±std + average rank（TGRB 同口径）。

---

## 6. 关键路径与决策点（阻塞项 + 谁拍板）

### 6.1 必须先解锁的阻塞项（按依赖顺序）
1. **NSPGNN 源码**：拿到 `forward(x, edge_index)->logits` 实现 → 填回 `exp/taglas_bench.py` 的 `NSPGNN` 类。**没有它 P4 与主阵实验全停摆。**
2. **数据统一 .pt**：跑通 `prepare_data.py`（优先 `--source tgrb`，TAGLAS 侧按报错一次性补接线）。确认文本字段语义质量（TGRB 论文未写明来源，需回 LLMNodeBed 核对）。
3. **H100 + vLLM**：本地 Qwen3-32B（不开 OpenAI 也能跑文本改写），`config.yaml` 填 `vllm.url`。
4. **理论实验 harness**：P1 密度图 + P2 损失分解需要比 `taglas_bench.py` 更细的钩子（输出 Ω(·) 分布、损失项分解）——在 S0 后加。

### 6.2 与老师/师兄确认的三件事（影响路线走向，尽早）
1. **NSPGNN 源码可否复用/引用**（版本、许可、作者署名）。
2. **「统一性」卖点口径**：是「同配+异配」统一（小），还是「结构+文本+毒化+逃逸」四维统一（大，工作量 3 倍）——建议分两篇打。
3. **编码器自由度**：正文冻结 RoBERTa/MiniLM（默认），「对编码器微调」是否纳入（会引入与 LLM 系竞争风险，建议放附录）。

---

## 7. 里程碑（12 周，含 gate）

| 周 | 里程碑 | 产出 | gate |
|---|---|---|---|
| W1–2 | 立论与实证地基 | P1 密度图实验；E 的 Lipschitz 实测表；Intro+Related work 初稿 | 阻塞项 1–3 全解锁 |
| W3–5 | P2/P3 推导 | 联合展开推导 + 交叉项数值验证 | 若 P2 不封闭→走退路 1 |
| W6–8 | NSPGNN-T 实现与消融 | 净化层 + 一致性正则 + RGSL 传播；cora/pubmed 全消融 | 消融支持设计 |
| W9–11 | 全量对比 | 主阵 4 数据集 × TGRB 档位 × 3 seeds rank 表；跨库异配补充 | 数字对齐 TGRB |
| W12 | 成稿 | 图表/附录（密度图、Lipschitz 表、损失分解、rank 散点） | 投稿 |

投稿目标：时间紧先 AAAI'27，赶不上则 ICLR'27，或先组内短文练手（与老师确认 venue 与署名后定）。

---

## 8. 风险与退路

1. P2 展开不封闭 → 退到「单调性/偏好排序」式弱定理（不定量、只定性），配合强实验。
2. 文本→嵌入 Lipschitz 界难给 → 对 MiniLM/RoBERTa 实测 synonym-edit 的经验 Lipschitz（有限集上界），论文作为假设+实证表，不做理论紧界。
3. P4 上界太松 → 弱化为「防御在两类攻击下鲁棒精度不低于 X 的充分条件（k、τ 选择准则）」，转成可操作设计准则。
4. TAG 异配数据稀缺 → 用 LLMNodeBed 的 History/Photo/Computer/Instagram/Reddit 补，论文诚实说明是跨库迁移验证。
5. 审稿人问「为何不用 GraphLLM」→ 正文明确 scope=embed→GNN 理论化（与 SFT-auto 互补），GraphLLM 混合管线列入 future work（引用 TGRB L680-683）。

---

## 9. 本周立即行动（阻塞项优先）

- [x] 2. 核对 TGRB/LLMNodeBed 数据集文本字段来源 → **已完成**，见 `docs/llmnodebed_text_fields.md`（学术图=标题+摘要、社交图=profile/subreddit、电商图=评论/描述）。
- [x] 4. 补理论实验钩子（P1 密度/KL、P2 损失分解、P3 文本方向）→ **已完成**，见 `exp/theory_analysis.py` + `exp/smoke_theory.py`（冒烟已跑通，P1 复现 NSPGNN 现象）。顺手修了 `exp/taglas_bench.py` `heuristic_perturb` 重复加边 bug。
- [ ] 1. 向老师/师兄要 NSPGNN 源码 + 确认许可/署名（§6.2 第 1、2 条一并问）。
- [ ] 3. 拿一张 H100，跑 `exp/README.md` 的 S0 冒烟（先用 `--source tgrb` 走通数据契约）。
- [ ] 5. 起 vLLM 服务（Qwen3-32B），验证 10 节点文本改写链路，把真实 ΔX 接进 P2/P3 钩子。

---

## 附：文档导航

- 理论细节 → `docs/theory_extension_roadmap.md`
- TGRB 复现/接入清单 → `docs/tgrb_inventory.md`
- 数据集选型 + 基线方案 → `docs/taglas_baseline_plan.md`
- 脚手架 → `exp/README.md` + `exp/taglas_bench.py` + `exp/prepare_data.py`
