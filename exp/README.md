# exp/ — embed→GNN 鲁棒性基线脚手架（第 c 步交付）

用途：在 H100 上为「NSPGNN 扩展到文本属性图」搭评测底座。
数据统一契约（本目录所有代码只认这一种 .pt）：

    node_texts: List[str]（长度 N；LLM 文本攻击在此上做）
    x:          Tensor[N,d] 可选（预计算特征，与 node_texts 二选一，文本图优先 node_texts）
    edge_index: LongTensor[2,E]
    y:          LongTensor[N]
    split:      dict 可选（'official' 的 train/val/test mask，arxiv 用）

## 1. 部署到 H100（分步）

```bash
# ① 同步代码（本机已生成；H100 上建 venv）
git add/commit 后 clone，或直接 scp 目录；然后：
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ② 造数据（两种源任选其一，落到统一 .pt）
#   A) 用 TGRB/LLMNodeBed 数据（结构攻击后续可直接用 TGRB 仓库的 PGD/GRBCD 产物）
python prepare_data.py --source tgrb --path /path/to/GraphAD_data/datasets/cora.pt --out data/cora_node.pt
#   B) 用 TAGLAS（安装 TAGLAS 并让它先下载好数据集；此路径为一次性接线，跑挂请把报错贴回来）
pip install git+https://github.com/JiaruiFeng/TAGLAS
python prepare_data.py --source taglas --dataset cora_node --out data/cora_node.pt

# ③ 冒烟测试（S0，~30 min）
python taglas_bench.py --dataset cora_node --encoder bow  --model gcn  --attack pgd_evasion --ptb 0.1 --seeds 1
python taglas_bench.py --dataset cora_node --encoder minilm --model cosgcn --attack none --seeds 1
```

## 2. 常用命令

```bash
# clean 表：数据集 × {bow,minilm,roberta} × {mlp,gcn,gat,gprgnn,cosgcn} × 3 seeds
python taglas_bench.py --dataset cora_node --encoder minilm --model gat  --attack none --seeds 3
# 结构攻击（内置轻量结构扰动用于 smoke；正式档位建议改用 TGRB 仓库生成的攻击图，见 §4）
python taglas_bench.py --dataset pubmed_node --encoder minilm --model cosgcn --attack pgd_evasion --ptb 0.2 --seeds 3
# 文本攻击（本地 vLLM，Qwen3-32B；请在 config.yaml 填 vllm.url）
python taglas_bench.py --dataset cora_node --encoder roberta --model gcn --attack text_evasion --ptb 0.4 --seeds 3 \
  --vllm-url http://localhost:8000/v1 --vllm-model Qwen/Qwen3-32B
```

## 3. 接 NSPGNN

`taglas_bench.py` 里 `MODELS` 注册表已有 `"nspgnn"` 占位：import 失败会给出明确报错。
把课题组 NSPGNN 源码做成一个 `def forward(x, edge_index) -> logits` 的 PyG 模型，
按报错提示填回 `models.py` 的 `class NSPGNN`（本脚手架为省文件数放在 taglas_bench.py 顶部）即可。
注意：NSPGNN 双 kNN 构造 O(N²)，大图（arxiv/products）需要论文里的 Lanczos/分块方案或先降采样。

## 4. 与 TGRB 仓库的衔接（正式档位攻击）

脚手架内置的结构攻击是**轻量启发式**（DICE + 低相似对），只用于 smoke/流程验证。
论文正式数字必须用 TGRB 同款攻击档位生成：
- PGD/GRBCD（结构 evasion 0.20，surrogate=GCN+BoW）：装好 TGRB 后跑
  `python attacks/gen_attacks_inductive.py --dataset cora --ptb_rate 0.20 --attack pgd --emb_type bow --re_split 2`
- Heuristic（结构 poisoning 0.30）、GPT-4o-mini 文本改写（0.40/0.80）：同 TGRB 的 gen_attacks_transductive / gen_text_attacks_*_llm
- 把产物（被攻击的 edge_index 差异 / attacked_texts json / 新嵌入缓存）喂回本脚手架的数据契约继续评测。
这样你的表与 TGRB 同口径，可直接写进 related work/对比实验。

## 5. 冒烟验收清单（S0 全绿才算搭好）

- [ ] prepare_data 两种源至少一种能产出统一 .pt
- [ ] bow / minilm 编码有缓存、重复运行不重算
- [ ] clean 跑 3 个模型不崩；Cora+RoBERTa+GCN 精度在 TGRB 数量级（~86-88）
- [ ] pgd_evasion 0.1 能产攻击图并完成评测（预期有掉点）
- [ ] text_evasion 0.2 能调通本地 vLLM（Qwen3-32B）改写 10 个节点并完成重编码评测
- [ ] 输出：out/{dataset}/{encoder}/{model}_{attack}_{ptb}/seed*.json + 汇总 md 表

诚实声明：本脚手架未经 GPU 实测，属"可执行初稿"，S0 阶段可能有少量 bug，跑挂把堆栈贴回来我修。
