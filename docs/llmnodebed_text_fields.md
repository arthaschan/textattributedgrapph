# LLMNodeBed 数据集文本字段来源核对（raw_texts 到底是什么）

> 背景：`docs/tgrb_inventory.md` §8.4 曾标注「10 个数据集的文本字段来源未在论文写明，需回 LLMNodeBed 确认」。
> 本文档已核对 LLMNodeBed（github.com/WxxShirley/LLMNodeBed，ICML'25，arXiv:2502.00829）源码，
> 确认每个数据集 `.pt` 中 `raw_texts` 字段的语义来源。核对日期：2026-09-07。

---

## 1. 结论速览

| 领域 | 数据集 | 节点类型 | raw_texts 文本字段来源 |
|---|---|---|---|
| 学术·引文 | cora | paper | 论文**标题 + 摘要** |
| 学术·引文 | citeseer | paper | 论文**标题 + 摘要** |
| 学术·引文 | pubmed | paper（糖尿病） | 论文**标题 + 摘要** |
| 学术·引文 | arxiv | paper（CS） | 论文**标题 + 摘要** |
| 学术·百科 | wikics | Wikipedia 条目 | **维基正文**（条目文本） |
| 社交 | reddit | user | 用户**历史发布的 subreddit 内容** |
| 社交 | instagram | user | 用户 **profile 描述** |
| 电商 | computer | item | 商品**评论（reviews）** |
| 电商 | photo | item | 商品**评论（reviews）** |
| 电商 | history | book | 书名 + **描述（titles + descriptions）** |

> 依据：`common/descriptions.py`（`ZeroG_DESC` / `MATCHING_TEMPLATES` / `GraphGPT_DESC` 三处交叉印证）。
> 注意：TGRB 的 10 数据集 = 上表去掉 citeseer（TGRB 用 9 个 + ArXiv 单 split 口径），字段来源同一套。

---

## 2. 源码证据

1. **学术网络 = 标题+摘要**（`common/descriptions.py` 的 `MATCHING_TEMPLATES["academic_network"]`）：
   > "Each graph token contains the title and abstract information of the paper at this node."

2. **GraphGPT `fetch_title()` 佐证格式**（`LLMPredictor/GraphGPT/dataset.py:35`）：
   ```python
   def fetch_title(txt, max_length=512):
       title = txt.split(":")[0]   # 说明 raw_texts 形如 "Title: ... Abstract: ..."
       title = txt.split(".")[0]
       return title[:max_length]
   ```
   即学术图的 raw_texts 是「标题: 摘要」拼接，标题在前。

3. **社交网络 = profile 描述**（`MATCHING_TEMPLATES["social_network"]`）：
   > "Each graph token contains the profile description of the user represented by this node."

   Reddit 特例（`ZeroG_DESC["reddit"]`）：
   > "the node features are the content of users historically published subreddits"

4. **电商 = 评论/描述**（`ZeroG_DESC`）：
   - computer / photo：`"node features consisting of reviews for each item"`（商品评论）
   - history：`"node features consisting of book titles and descriptions"`（书名+描述）
   - `MATCHING_TEMPLATES["ecommerce_network"]` 又称 "comment of the item"（评论）——与 ZeroG 的 "reviews" 是同一字段的两种措辞，无实质冲突。

5. **数据是预构建的 `.pt`**：`common/dataloader.py` 只 `torch.load(...).raw_texts`，不现场生成。
   即 raw_texts 在数据集发布时已从原始数据（MAG 元数据 / Reddit / Amazon 评论）抽好并打包，
   **LLMNodeBed 仓库内无「原始数据 → raw_texts」的构建脚本**，原始出处需看 LLMNodeBed 论文 §Dataset 或 HF 数据集卡。

---

## 3. 对 NSPGNN-T 文本攻击实验的含义（重要）

1. **语义丰富的文本才适合 LLM 改写攻击**（`docs/taglas_baseline_plan.md` §1 已主张，现可落实）：
   - 学术图（cora/citeseer/pubmed/arxiv/wikics）：标题+摘要/百科正文，语义强、可改写 → **文本攻击主阵首选**。
   - 电商图（computer/photo/history）：评论/描述，语义中等，可做**次选**（短文本、噪声多，改写收益有限）。
   - 社交图（reddit/instagram）：profile/subreddit 内容，长度短、信息密度低 → **最不适合文本攻击**，只用于结构攻击/异配性论证。

2. **文本攻击的 prompt 要按字段类型适配**（对齐 TGRB 的 neighborhood-aware prompt）：
   - 学术图：改「标题+摘要」让分类器误判（如把 ML 论文改写得像 Rule Learning）。
   - 电商图：改「评论」改变商品子类语义。
   - 不要在 reddit/instagram 上硬做文本攻击——改写 profile 很难定向误导类别，且与 TGRB 结论可比性差。

3. **编码器选择提醒**：TGRB 发现 RoBERTa/MiniLM 的 intra/inter 边区分力（AUC 0.65/0.66）远超 BoW（0.565），
   前提正是「文本字段语义足够丰富」。上表第 3 列确认学术图满足，电商图（短评论）会拉低区分力，
   这正好是 P4「净化层」要对付的难点（文本偏移在低区分力 embedding 上更难免疫）。

---

## 4. 待办（可选，非阻塞）

- 若论文里要精确引用「文本字段原始出处」（如 ArXiv 用 MAG 的 title+abstract、Amazon 用评论表），
  需再查 LLMNodeBed 论文 §Dataset 或 HF 数据集卡（`xxwu/LLMNodeBed`）的字段说明，确认字段名与抽取口径。
- 做文本攻击实验前，建议抽样打印 10 条 raw_texts 人工质检语义质量（cora/pubmed 是否真的是完整 title+abstract，
  还是被截断的标题）。
