# 项目长期记忆 — text-attributed graph 鲁棒性扩展研究

## 项目目标
把普通图上"统一性鲁棒性"方法（组内工作，疑似 NSPGNN/RGSL，arXiv:2401.09754；作者含珠海学院 Yulin Zhu / Wai Lun LO，通讯 Kai Zhou@PolyU）扩展到文本属性图（TAG）。

## 关键参照物
- **NSPGNN（本组基础方法）**：Theorem 1——攻击损失更新幅值与 A^τX 相似度矩阵负相关，攻击偏好连"低相似对"；防御=RGSL 双 kNN 图 + 邻居相似性保持传播，跨同配/异配统一。
- **CS-TAG benchmark**（NeurIPS'23 D&B）：大型 TAG 数据 + PLM/GNN/TPT 评测；结论：文本属性质量/编码方式主导性能。
- **TGRB**（ICLR'26，github.com/Leirunlin/TGRB）：TAG 鲁棒性评测框架；发现文本-结构鲁棒性 trade-off、GNN 依赖编码器、GraphLLM 怕数据污染；防御 SFT-auto。= 我们理论方法要超越的对标。
- **TAGLAS**（arXiv:2406.14683，github.com/JiaruiFeng/TAGLAS）：25 数据集 / 5 任务型 / 统一文本特征结构 / text↔embed 工具 + 评测。拟用实验平台。

## 用户背景备注
- AI 硕士（香港珠海学院），老师 = 珠海学院 Yulin Zhu（朱禹林）方向；用户偏好：理论+机制+实验结合的研究风格；中文交流；解答先纠正误区再讲、给可执行清单。
