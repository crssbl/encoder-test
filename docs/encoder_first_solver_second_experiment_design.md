# Encoder-First, Solver-Second：面向 Routing 的 Giant Tour + Exact Split 详细实验设计与实现要求

## 1. 文档目的

本文档用于指导 Codex 实现一个新的 routing 学习范式，目标是替代当前 `heavy encoder + 长时序 autoregressive decoder` 的训练接口，减少显存消耗，避免 `POMO × n 步 × batch × rollout` 级别的反向图，同时尽量保留甚至增强性能与规模泛化能力。

本文档优先阐述**你的方法本身**，然后再给出实验协议、基线方法、开源代码复用点和工程实现要求。

本方法不依赖外部最优标签，不需要 imitation of expert tours，不把跨实例的 raw route length 直接作为监督目标。核心训练范式是：

- **Encoder 学习全局解结构 / compatibility field**；
- **Projector 负责把 field 投影成 giant tour**；
- **Exact Split 负责把 giant tour 切成合法 CVRP routes**；
- **训练只在同一实例内部做相对优劣学习（instance-relative listwise / ranking）**。

---

## 2. 核心主张

### 2.1 现有 autoregressive routing 训练接口的问题

对 POMO / PolyNet 一类构造式模型，现有训练图包含：

1. 每一步 decoder 状态更新；
2. 每一步 mask/load/selected node 相关 attention；
3. 多 rollout / 多策略采样；
4. 长时程 credit assignment。

这会导致：

- 显存复杂度随 `rollout_size × n × batch` 线性或更差增长；
- decoder 成为训练瓶颈，和“主要知识在 encoder”这一实证结论不匹配；
- 训练信号长、噪声大、方差高；
- 容易学到特定构造轨迹，而不是问题本身的全局结构。

### 2.2 新范式的核心思想

对 CVRP，真正需要 learned 的部分更像是：

- 哪些 customer 应该彼此靠近；
- 哪些边/顺序在全局上更相容；
- 在容量约束下，哪些节点更适合同一路线。

而以下部分更适合交给确定性算法：

- 容量约束切分；
- depot 插入与 route 切割；
- 可行性保证；
- 轻量修复与投影。

因此方法设计为：

> **先学 giant tour / order field，再用 exact split 生成 CVRP 解。**

这是一个 route-first, split-second 的学习版本，但学习对象不是完整构造轨迹，而是**全局 giant tour 场**。

---

## 3. 方法总览

给定一个 CVRP 实例：

- depot 坐标 `d ∈ R^2`
- customers 坐标 `x_i ∈ R^2`
- demand `q_i`
- 车辆容量标准化为 1

模型输出不是逐步动作，而是：

1. **edge compatibility field**：表示哪个 customer 更适合接在谁后面；
2. **optional order / boundary prior**：表示 giant tour 上的大致顺序与潜在切分位置；
3. **candidate-specific energy / score**：用于训练时在同一实例内对候选解排序。

然后通过：

1. **Cycle-cover / Hamiltonian projector**：把 edge field 投影成 giant tour；
2. **Exact Split DP**：把 giant tour 切成合法 CVRP routes；
3. **Optional repair / local refinement**：只在评测或后期增强中使用；
4. **Instance-relative listwise objective**：不依赖 ground-truth tour，只在同一实例内比较候选优劣。

---

## 4. 第一版方法：Sparse Edge Field + Cycle-Cover Projector + Exact Split

这是建议优先实现的最小可行版本（MVP）。

### 4.1 输入表示

每个 customer 节点输入：

- `coord_x`
- `coord_y`
- `demand`
- `distance_to_depot`
- `polar_angle_wrt_depot`（可选）
- `local_density`（可选，运行前预计算）

注意：第一版不要引入动态 `load` 或 rollout 级状态。

### 4.2 Encoder

优先复用你当前已经有机制分析基础的 encoder 骨架：

- Transformer encoder / graph Transformer encoder；
- 输入是全局节点集合；
- 输出每个 customer 的 node embedding `h_i`；
- 另保留一个 depot embedding `h_0`。

#### 推荐版本

- `embedding_dim = 128 or 192`
- `num_layers = 6`
- `num_heads = 8`
- `ff_dim = 4 * embedding_dim`
- 归一化优先用 pre-norm 或 RMSNorm

#### 设计原则

- Encoder 是唯一重模块；
- 不要引入长序列 decoder；
- 不要在训练图里出现 `n` 步 autoregressive unfold。

### 4.3 Edge Field Head

对每对 customer `(i, j)` 输出一个 successor compatibility score：

\[
s_{ij} = f_{edge}(h_i, h_j, h_i \odot h_j, \phi_{ij})
\]

其中 `φ_ij` 包含：

- 欧氏距离 `dist(i,j)`
- 相对位置 `(x_j - x_i, y_j - y_i)`
- demand pair `(q_i, q_j)`
- depot-relative features（可选）

#### 推荐实现

- 2-layer MLP 或 bilinear + MLP；
- 输出有向 score `s_ij`；
- 对角线置 `-inf`；
- 第一版默认 dense score for `n <= 200`；
- 对 `n > 200`，先 dense 计算块状 score，再保留每个节点 top-m 的 learned outgoing edges，**不是 hand-crafted kNN**。

这里强调：

- **不使用 hand-crafted kNN local policy 作为方法核心**；
- 稀疏化是工程优化，不是 inductive bias 主体；
- top-m 由 learned score 决定，而不是距离最近邻。

### 4.4 Optional Boundary / Split Prior Head（第二优先级，可先留接口）

虽然第一版 giant tour + split 已可运行，但建议预留一个 `split_prior_head`，输入为 giant-tour 上的节点 embedding 序列，输出每个相邻边是否更适合作为 depot cut 的 prior。

这一 head 不是第一版必需，但后面很有价值：

- 它可以帮助 exact split 更快收敛；
- 可以作为 auxiliary analysis target；
- 可以让 encoder 输出和 CVRP route boundary 更直接对齐。

第一版可以只留接口，先不训练。

---

## 5. Giant Tour Projector

### 5.1 目标

从 edge field `s_ij` 生成一个覆盖所有 customers 的单一 giant tour（不含中间 depot）。

### 5.2 推荐投影流程

#### Step 1：Cycle Cover

通过 assignment / matching 得到每个 customer：

- 恰好一个 successor；
- 恰好一个 predecessor；

形成一个 cycle cover（可能有多个 subtours）。

#### Step 2：Subtour Patching

将多个 subtours 合并为一个 giant tour。推荐实现：

- 贪心 subtour merge；
- 使用 edge field score 与距离的组合分数选择 merge edges；
- 或参考你之前 one-shot TSP reconstruction 的 subtour patching 思路。

#### Step 3：Canonical tour extraction

从合并后的单一大环抽取线性 giant tour 序列 `π = (π_1, ..., π_n)`。

### 5.3 为什么不用 autoregressive decoding

因为这里 giant tour 的生成是：

- 一次 encoder forward；
- 一次 edge score 计算；
- 一次 cycle cover + merge 投影；

训练图长度不随 `n` 步构造线性展开。

### 5.4 可选简化版（仅作对照，不作为主方法）

实现一个 scalar position / phase head：

\[
u_i = f_{ord}(h_i)
\]

按 `ν_i` 排序得到 giant tour。

这个版本可以作为轻量 ablation：

- 优点：实现最简单；
- 缺点：通常比 edge field + projector 弱；
- 用来证明“非 AR 但过于简化的排序头不够”。

---

## 6. Exact Split：把 Giant Tour 变成合法 CVRP 解

### 6.1 核心思想

对 giant tour `π`，求最优 depot cut，把它划分为多个容量可行的 routes，最小化总 route cost。

这一步是 route-first split-second 的经典 Split 子问题。

### 6.2 推荐实现

优先复用或参考以下开源实现：

1. **PyVRP / PyVRP**
   - 开源、高性能、支持多种 VRP 变体；
   - 其求解器建立在现代 HGS-CVRP 思路之上；
   - 适合作为 baseline，也适合作为组件参考。

2. **vidalt / HGS-CVRP**
   - 公开实现的 HGS-CVRP；
   - 论文明确说明内部使用 Vidal 2016 的线性时间 Split 算法；
   - 是 route-first split-second / HGS 系的重要实现参考。

3. **Vidal 2016 Split algorithm**
   - giant tour 切分为合法 routes 的线性时间算法；
   - 非常适合直接作为本方法的核心后处理器。

### 6.3 工程建议

第一版实现顺序：

- 先自己写一个纯 Python / PyTorch-friendly 的 `O(n^2)` 或 `O(nB)` 版 Split，保证逻辑正确；
- 再替换成 Vidal 式高性能版本；
- 再封装一个统一接口：

```python
split_solution = split_dp(
    giant_tour,
    demands,
    depot_xy,
    coords,
    capacity=1.0,
)
```

### 6.4 梯度处理

- `split_dp` 默认 **detach / no-grad**；
- 只返回 candidate solution 与 objective；
- 训练时不尝试对 split 本身反传。

这是第一版非常重要的设计原则。

---

## 7. Candidate Generation：非监督训练的关键

因为不用 expert labels，所以必须在同一实例上产生多个 candidate giant tours，形成自比较信号。

### 7.1 候选生成方式

对同一实例，从同一个 edge field 生成 `M` 个 candidate：

\[
S^{(m)} = S + G^{(m)}
\]

其中 `G^(m)` 是 Gumbel noise 或 logistic noise。

然后每个 candidate 依次经过：

- cycle cover projector
- subtour patching
- split DP

得到 CVRP 解 `x^(m)` 和成本 `C^(m)`。

### 7.2 推荐候选数

- 第一版训练：`M = 8 or 16`
- 验证：`M = 32`
- 测试：`M = 64 or 128`

这样既保留多样性，也不会像 POMO 那样把 rollout 图拉得太长。

### 7.3 Candidate score / energy

定义候选 giant tour 的模型能量：

\[
E_\theta(x^{(m)}|s) = - \frac{1}{n} \sum_{(i,j) \in x^{(m)}} s_{ij}
\]

可选再加 split prior 项：

\[
E_\theta = E_{edge} + \lambda_{split} E_{cut}
\]

第一版不加 `split_prior_head` 时，直接使用 `E_edge` 即可。

---

## 8. 训练目标：不要直接把 raw length 当监督

### 8.1 原则

不直接用跨实例 raw route length 做统一回归，因为：

- 不同实例尺度不同；
- 不同分布的绝对值不可比；
- 这会把数据分布本身混进训练目标。

训练目标应该是：

> **只在同一实例内部比较候选解谁更好。**

### 8.2 主损失：Instance-Relative Listwise KL

对同一实例的 `M` 个 candidates，成本为 `C_1, ..., C_M`。

先做归一化：

\[
\tilde C_m = \frac{C_m - \min_j C_j}{\mathrm{MAD}(C_1, ..., C_M) + \epsilon}
\]

其中 `MAD` 是 median absolute deviation，增强鲁棒性。

定义目标分布：

\[
q_m = \frac{\exp(-\tilde C_m / \tau)}{\sum_j \exp(-\tilde C_j / \tau)}
\]

模型分布：

\[
p_m = \frac{\exp(-E_\theta(x^{(m)}|s) / \tau_p)}{\sum_j \exp(-E_\theta(x^{(j)}|s) / \tau_p)}
\]

主损失：

\[
L_{list} = \mathrm{KL}(q \| p)
\]

### 8.3 可选辅助损失：Incumbent-Relative Improvement

给每个实例一个便宜 baseline `B(s)`：

- sweep + split
- Clarke-Wright + split
- 当前模型 greedy candidate

定义相对改善：

\[
r_m = \frac{B(s) - C_m}{B(s) + \epsilon}
\]

再用 `r_m` 构建 soft target 或 pairwise ranking。

### 8.4 Diversity Regularization

如果不用这个项，模型容易 collapse 到单一 giant tour 模式。

定义候选间差异：

- edge overlap
- cut overlap
- route partition overlap

只对同一实例内部的 candidates 加轻量 diversity 正则：

\[
L_{div} = - \mathbb{E}_{m \neq m'} [\text{edge-disagreement}(x^{(m)}, x^{(m')})]
\]

注意：

- 第一版只加很小权重；
- 不要让 diversity 压倒质量。

### 8.5 Augmentation / Symmetry Consistency（可选但推荐）

对同一实例做对称变换（如 8-fold Euclidean augmentation）后，要求 edge field / energy 排序尽量一致。

这个正则可以帮助规模和分布泛化。

### 8.6 总损失

\[
L = L_{list} + \lambda_{imp} L_{imp} + \lambda_{div} L_{div} + \lambda_{cons} L_{cons}
\]

第一版默认：

- `lambda_imp = 0.2`
- `lambda_div = 0.02`
- `lambda_cons = 0.05`

如果不稳定，可只保留 `L_list`。

---

## 9. 梯度设计与为什么它会更稳

### 9.1 梯度路径

训练时的梯度只经过：

- encoder
- edge head
- optional energy head

不经过：

- split DP
- local repair
- route feasibility checker

### 9.2 为什么比 autoregressive RL 稳

当前 autoregressive RL 的难点是：

- n 步 credit assignment；
- 每一步 action 都会影响后续全部可行集合；
- 方差随 rollout 深度大幅增长；
- 对 decoder 状态图的依赖极强。

本方法改成：

- candidate-level sampling，不是 step-level sampling；
- reward 是 instance-relative normalized cost，不是 raw cost；
- 图长度固定，不随 n 步展开；
- 可行性由 split / projector 保证，不需要 policy 自己背全套约束。

### 9.3 如果需要 score-function estimator

第一版优先使用 listwise energy matching，不需要 REINFORCE。

只有在 projector 中必须采样 discrete decisions 且无法稳定训练时，才引入 score-function：

\[
L_{pg} = - A_m \log p_\theta(x^{(m)}|s)
\]

其中 advantage 使用 instance-normalized cost：

\[
A_m = -\tilde C_m - \frac{1}{M} \sum_j (-\tilde C_j)
\]

但这不是主训练路径。

---

## 10. 训练流程伪代码

```python
for batch in train_loader:
    # 1) Encode instance once
    h = encoder(batch.nodes, batch.depot)

    # 2) Predict dense/sparse edge field
    S = edge_head(h)  # shape [B, N, N]
    mask_diagonal(S)

    candidate_solutions = []
    candidate_costs = []
    candidate_energies = []

    for m in range(M):
        # 3) Perturb-and-project
        G = sample_gumbel_like_noise(S.shape)
        S_m = S + noise_scale * G

        cycle_cover = project_cycle_cover(S_m)
        giant_tour = patch_subtours(cycle_cover, S_m, batch.coords)

        # 4) Exact split (detached)
        with torch.no_grad():
            routes = split_dp(giant_tour, batch.demands, batch.coords, batch.depot)
            cost = compute_route_cost(routes, batch.coords, batch.depot)

        # 5) Candidate energy
        energy = compute_candidate_energy(S, giant_tour)

        candidate_solutions.append(routes)
        candidate_costs.append(cost)
        candidate_energies.append(energy)

    # 6) Build instance-relative targets
    q = build_listwise_target(candidate_costs)
    p = softmax(-stack(candidate_energies) / tau_p)

    loss = kl_div(q, p)
    loss += lambda_div * diversity_regularizer(candidate_solutions)
    loss += lambda_cons * augmentation_consistency(...)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 11. 训练与评测协议

## 11.1 训练分布

建议至少做 3 个训练分布：

1. **Uniform Euclidean CVRP**
2. **Clustered CVRP**
3. **Mixed distribution**（uniform + clustered + demand-skew）

### 推荐训练规模

- 主训练：`n = 100`
- 泛化测试：`n = 100, 200, 500, 1000`

### 需求分布

至少包含：

- 标准 Uchoa-style CVRP scaling
- 高 demand skew
- cluster-level demand concentration

## 11.2 测试协议

必须区分：

### In-distribution
- same generator, same n

### Size generalization
- train on 100, test on 200 / 500 / 1000

### Distribution shift
- train on uniform, test on clustered / mixed

### Real benchmark
- CVRPLIB / Uchoa benchmark

---

## 12. 需要比较的 baseline（不要 local kNN policy）

下面这些 baseline 都应该纳入：

## 12.1 经典 OR / solver baseline

### B1. HGS-CVRP

- 开源实现：`vidalt/HGS-CVRP`
- 作用：强 OR baseline
- 备注：内部使用现代 HGS 框架和高效 Split，是 CVRP 强基线之一

### B2. PyVRP

- 开源实现：`PyVRP/PyVRP`
- 作用：强工程化 solver baseline
- 备注：适合直接调用、易复现、对多种 CVRP 变体支持好

### B3. OR-Tools CVRP

- 开源实现：`google/or-tools`
- 作用：通用工业基线
- 备注：不一定最强，但很重要，能体现实用性和可落地性

## 12.2 学习增强 / neural baseline

### B4. POMO / PolyNet（你现有实现）

- 作用：直接对比当前 heavy decoder 范式
- 重点比较：
  - 显存
  - wall-clock
  - size generalization
  - distribution shift

### B5. NeuroLKH

- 开源实现：`liangxinedu/NeuroLKH`
- 作用：学习增强 LKH / solver 类强基线
- 备注：代表“learned priors + classical solver”路线

### B6. Learning to Delegate

- 开源实现：`mit-wu-lab/learning-to-delegate`
- 作用：大规模 routing 的 learning-augmented search baseline
- 备注：尤其适合规模泛化场景

## 12.3 本方法内部 ablation baseline

### A1. Scalar position sort + split
- 非 AR，但太弱的简单版本

### A2. Edge field + projector + split（主方法）

### A3. Edge field + projector + split + diversity loss

### A4. Edge field + projector + split + augmentation consistency

### A5. 主方法 + optional repair（仅测试时）

---

## 13. 最重要的评测指标

### 13.1 质量

- average gap to PyVRP / HGS best known
- average gap to best baseline in this experiment
- best-of-M candidate gap
- greedy candidate gap

### 13.2 训练与推理效率

- peak GPU memory
- train tokens / sec 或 instances / sec
- inference wall-clock
- candidates / second

### 13.3 泛化

- train-100 -> test-200/500/1000 gap
- uniform -> clustered shift gap
- demand-shift robustness

### 13.4 多样性

- candidate edge disagreement
- route partition disagreement
- best-vs-greedy improvement

### 13.5 可行性

- split feasibility rate
- projector validity rate
- repair usage rate

---

## 14. 关键 ablation

这些 ablation 必须做，因为它们决定这条方法线是否成立。

### 14.1 不同 giant tour 生成器

- scalar order sort
- cycle cover + patch
- cycle cover + patch + learned sparsification

### 14.2 不同训练目标

- raw cost regression（应作为反例，不建议主用）
- instance-relative pairwise ranking
- instance-relative listwise KL（主方法）

### 14.3 是否使用 split

- no split：直接用 giant tour 伪装 route（反例）
- exact split（主方法）
- exact split + repair

### 14.4 是否使用 diversity

- no diversity
- light diversity reg

### 14.5 是否使用 augmentation consistency

- no consistency
- + symmetry consistency

---

## 15. 开源代码复用建议

## 15.1 优先复用

### PyVRP

优先用于：

- benchmark comparison
- route cost computation validation
- split / repair reference
- optional local search evaluation

### HGS-CVRP

优先用于：

- 强基线比较
- Split 实现参考
- 可选 repair / local improvement 参考

### OR-Tools

优先用于：

- 通用工程 baseline
- feasibility verification
- 作为部署可解释 baseline

### NeuroLKH

优先用于：

- learning-augmented solver baseline
- 对比“神经先验 + 强 solver”路线

### Learning to Delegate

优先用于：

- 大规模场景 baseline
- 说明本方法是否真正更适合 scale generalization

## 15.2 不建议第一版依赖的东西

- 复杂 diffusion training
- 长时序 RL decoder
- 需要 expert labels 的 imitation 数据生成管线
- 黑盒不可控的大型 local search 作为训练主循环的一部分

---

## 16. 代码实现要求（Codex 需要完成的工程清单）

## 16.1 目录结构

建议新建：

```text
routing_encoder_first/
├── configs/
│   ├── cvrp100_gianttour_split_main.yaml
│   ├── cvrp100_gianttour_split_smoke.yaml
│   ├── cvrp200_eval.yaml
│   └── ablations.yaml
├── src/
│   ├── data/
│   │   ├── generators.py
│   │   ├── cvrplib_loader.py
│   │   └── augment.py
│   ├── models/
│   │   ├── encoder.py
│   │   ├── edge_head.py
│   │   ├── energy.py
│   │   └── order_head.py   # optional baseline
│   ├── projectors/
│   │   ├── cycle_cover.py
│   │   ├── subtour_patch.py
│   │   ├── giant_tour.py
│   │   └── split_dp.py
│   ├── training/
│   │   ├── candidate_sampler.py
│   │   ├── listwise_loss.py
│   │   ├── diversity.py
│   │   ├── consistency.py
│   │   └── trainer.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── compare_pyvrp.py
│   │   ├── compare_hgs.py
│   │   ├── compare_ortools.py
│   │   └── profile_memory.py
│   ├── utils/
│   │   ├── geometry.py
│   │   ├── logging.py
│   │   └── reproducibility.py
│   └── run_main.py
├── scripts/
│   ├── run_smoke.sh
│   ├── run_main.sh
│   ├── run_eval_cvrplib.sh
│   └── run_ablation.sh
└── README.md
```

## 16.2 必须实现的模块

### Module 1. Encoder

接口：

```python
h_nodes, h_depot = encoder(coords, demands, depot)
```

### Module 2. Edge Field Head

接口：

```python
edge_scores = edge_head(h_nodes, h_depot, coords, demands)
# [B, N, N]
```

### Module 3. Candidate Sampler

接口：

```python
perturbed_scores = sample_candidate_scores(edge_scores, num_candidates=M, noise='gumbel')
```

### Module 4. Giant Tour Projector

接口：

```python
giant_tours = project_giant_tour(perturbed_scores, coords)
# list of permutations
```

### Module 5. Exact Split

接口：

```python
routes = split_dp(giant_tour, coords, demands, depot, capacity=1.0)
```

### Module 6. Candidate Energy

接口：

```python
energy = score_candidate(edge_scores, giant_tour)
```

### Module 7. Listwise Loss

接口：

```python
loss = instance_relative_listwise_loss(costs, energies)
```

### Module 8. Baseline Runner

接口：

```python
result = run_baseline(instance, method='pyvrp' | 'hgs' | 'ortools' | 'pomo' | 'polynet' | 'neurolkh')
```

## 16.3 工程要求

- 第一版默认单卡可跑；
- 训练时必须记录 peak GPU memory；
- 所有 candidate generation 和 split 过程都要可复现；
- 所有结果输出到：
  - `raw/`
  - `tables/`
  - `figs/`
  - `summary.json`
  - `summary.md`

---

## 17. 结果表与图表要求

### 必须输出的表

- `main_results.csv`
- `efficiency_profile.csv`
- `size_generalization.csv`
- `distribution_shift.csv`
- `candidate_diversity.csv`
- `ablation_projector.csv`
- `ablation_loss.csv`

### 必须输出的图

- `quality_vs_time.png`
- `peak_memory_comparison.png`
- `size_generalization_curve.png`
- `distribution_shift_curve.png`
- `candidate_diversity_vs_quality.png`
- `giant_tour_examples.png`
- `split_examples.png`

---

## 18. 第一阶段的明确目标

Codex 第一阶段不要追求“完全打穿所有 SOTA”。先完成以下里程碑：

### Milestone 1

在 `CVRP100 uniform` 上跑通：

- edge field
- cycle cover + patch
- split DP
- listwise loss
- greedy / best-of-M 推理

### Milestone 2

在 `CVRP100 -> CVRP200` 上做 size generalization，和：

- POMO
n- PolyNet
- PyVRP
- HGS-CVRP

进行初步比较。

### Milestone 3

完成 3 个关键 ablation：

- scalar order sort vs edge field
- raw cost vs instance-relative listwise
- no split vs exact split

如果这 3 个 ablation 成立，这条方法线就已经非常有价值。

---

## 19. 我对 Codex 的最后要求

请 Codex 在实现时遵守以下原则：

1. **先把 giant tour + split 主链跑通，再加复杂增强。**
2. **不要把训练主循环重新做成长 autoregressive decoder。**
3. **不要依赖 expert labels 或 solver tours 做 imitation。**
4. **优先保证 instance-relative listwise training 稳定。**
5. **所有复杂 repair / local search 默认为 detached，仅用于 candidate evaluation 或 test-time enhancement。**
6. **baseline 必须包含强 OR solver 和强 learning-augmented solver，而不只是本领域常见的小规模 NCO 方法。**

---

## 20. 参考实现与基线来源（供 Codex 对照接入）

- `PyVRP/PyVRP`
- `vidalt/HGS-CVRP`
- `google/or-tools`
- `liangxinedu/NeuroLKH`
- `mit-wu-lab/learning-to-delegate`
- 你当前已有的 `POMO / PolyNet` 代码库

这些不是都要作为训练依赖，但都应该纳入 baseline 或实现参考。

---

## 21. 最后的方法定位

这个方法不是“再做一个更轻的 decoder trick”，而是把 routing 的学习接口重新定义为：

> **Learn a global solution field, then solve constraints with a short deterministic projector.**

如果第一版验证成立，这条线后续可以自然扩展到：

- richer VRP variants（时间窗、多仓、多 trip）
- primal-dual projector
- discrete flow matching over route fields

但第一阶段一定要把最简版本做干净。
