# 独立审查报告：embodied-offpolicy-study（SAC reward scaling on Hopper-v4）

**审查员身份**：独立RL方向审查员，首次接触该项目，未参与开发。
**审查日期**：2026-10-02
**审查对象**：`D:\github项目\embodied-offpolicy-study`（commit 05f2027）
**审查方法**：通读全部源码 + 从原始CSV独立重算统计量 + 实际运行pytest和plot.py + git历史时间线核查

---

## 一、总体判断

这是一件**工程质量合格、研究诚实但科学力度不足**的作品。SAC实现经逐行核对**无算法错误**，预登记流程真实存在且未被事后篡改，统计表述诚实（"方向性信号、不显著、CI大幅重叠"）。但60k步、n=3单环境的设计远不足以支撑"能独立完成RL研究"的强结论——它证明了**工程实现能力**和**研究伦理意识**，但尚未证明**实验设计能力**和**科学发现能力**。

---

## 二、问题清单（按严重度排序）

### [P1-01] reward scaling与alpha动力学混杂——效应归因不成立

**证据**：
- `src/sac.py:81`：`target = self.reward_scale * rew + (1.0 - done) * self.gamma * tq`
- 从CSV独立重算的最终alpha值：
  - Baseline (rs=1.0)：s0=0.047, s1=0.075, s2=0.061（均值≈0.061）
  - RewardScaled (rs=0.1)：s0=0.0087, s1=0.0078, s2=0.0071（均值≈0.008）
- alpha轨迹（`artifacts/check_dynamics.py`输出）：两臂alpha均从初始0.2开始下降，但scaled臂在20k步即降至~0.008并稳定，baseline臂到60k仍在~0.05。
- Q值：baseline q1_mean≈98-153，scaled q1_mean≈16-19（约6-8倍差距，与reward_scale=0.1方向一致但不完全成比例）。

**影响**：reward_scale=0.1不仅缩小了Q目标幅值，还通过自动熵温度机制使alpha下降了约7倍。这意味着"reward scaling 0.1"臂实际上同时运行了**更低的熵正则化强度**。观察到的性能差异无法归因于"减少critic过估计偏差"——它同样可能是"alpha更低→策略更确定性→在Hopper早期阶段表现更好"导致的。PRE_REGISTRATION.md第22-25行提出的机理假设（"smaller Q targets should reduce critic over-estimation bias"）没有控制这个混杂因素。

**建议**：增加一个**固定alpha**臂（`automatic_entropy_tuning: false`，alpha设为相同值），或在报告中明确承认reward scaling通过两条路径影响性能（Q幅值 + alpha自适应）。

---

### [P1-02] 60k步远未收敛——"渐近回报"名不副实

**证据**：
- 独立重算的episode长度分布：
  - baseline_s0: mean_len=102, max_len=1000, 68%的episode不足100步
  - baseline_s1: mean_len=130, max_len=501
  - baseline_s2: mean_len=102, max_len=277
  - rewardscale_s0: mean_len=113, max_len=432
  - rewardscale_s1: mean_len=124, max_len=303
  - rewardscale_s2: mean_len=150, max_len=1000
- 标准SAC在Hopper-v4上（SB3参考）通常需要100k-300k步才开始稳定行走，1M步达到~3000-4000回报。当前所有臂的最终eval回报在250-1375之间，mean episode length仅100-150步（上限1000），说明agent仍在"刚学会不立刻摔倒"的阶段。
- `RESEARCH_RETRO.md`第55行承认"60k env steps per run (I planned 120k...cut to 60k)"。

**影响**：在远未收敛的阶段比较"渐近回报"，本质上是在比较**早期学习速度**而非渐近性能。PRE_REGISTRATION.md第27行预注册的主要终点是"asymptotic (final-10k-step) deterministic eval return"——但60k步时的回报远不是渐近值。如果继续训练到120k或200k，两臂的排序可能完全改变。

**建议**：要么补跑到至少150k步（即使CPU慢，Pendulum已验证代码正确），要么将研究问题重新表述为"早期样本效率"而非"渐近性能"。

---

### [P1-03] "逃逸种子"rewardscale_s2不稳定，单点驱动结论

**证据**：
- rewardscale_s2的eval轨迹（独立重算）：
  - step 50k: 2250.1
  - step 55k: 2895.7
  - step 60k: 1375.5
- 这意味着该种子在50-55k出现了一个尖峰，然后在60k回落了52%。
- 剔除该种子后，rewardscale臂的两个种子最终eval为591.4和606.4，与baseline臂的最优种子(612.8)几乎相同。
- 独立重算：rewardscale臂的tail mean（plot.py方法）per-seed为[529.0, 595.4, 2135.6]——第三个种子是前两个的3.6倍。

**影响**：README和RESEARCH_RETRO中"scaled mean is carried by one seed that exploded to ~1375"的表述是诚实的，但问题在于：这个1375本身是否代表真实性能？eval在50k=2250、55k=2895、60k=1375的剧烈波动表明该策略**不稳定**，而非收敛到高性能。将不稳定尖峰纳入"渐近回报"均值会人为放大效应量。

**建议**：报告时应同时给出"剔除最差/最好种子后的敏感性分析"，或使用中位数而非均值作为汇总统计。

---

### [P2-01] n=3种子统计功效极低，结论虽诚实但信息量有限

**证据**：
- 独立Welch's t-test（pre-reg方法）：t=1.165, p=0.357
- 独立Welch's t-test（plot.py方法）：t=1.351, p=0.300
- 95%CI：baseline ±210（均值363），rewardscaled ±1029（均值1087）——后者CI宽度是均值的95%。
- 两臂CI大幅重叠（baseline CI [153, 573] vs scaled CI [58, 2116]）。

**影响**：p≈0.30意味着即使存在真实效应，n=3也几乎不可能检测到。作者诚实地报告了"directional signal, not a confirmed effect"，但从招生委员会视角看，这个实验**没有产生可发表的科学结论**——它只证明了作者知道如何做预登记和多种子实验。

**建议**：至少5-6个种子（CPU可行），或换一个效应更大的研究问题。

---

### [P2-02] 预登记tail定义与实际plot.py计算不一致

**证据**：
- PRE_REGISTRATION.md第34行："Mean deterministic eval return over the last 20% of training steps"
- 60k步的最后20% = ≥48k步 = eval点在50k, 55k, 60k（3个点）
- plot.py第84行：`tail = arr[:, -max(1, int(0.2 * arr.shape[1])):]`——12个grid点的最后20% = 最后2个点（55k, 60k）
- 独立重算：pre-reg方法（3点）baseline=372.8±217.1, scaled=1043.6±1107.6；plot.py方法（2点）baseline=362.9±210.0, scaled=1086.7±1028.6

**影响**：差异不大（<3%），但严格来说plot.py的tail窗口与预登记定义不一致。这是一个微小的"研究自由度"——如果选3点，scaled臂的均值从1087降到1044，CI从±1029变到±1108。

**建议**：在plot.py中注释清楚tail窗口的选择依据，或严格按预登记用3个eval点。

---

### [P2-03] requirements.txt未锁版本，可复现性不足

**证据**：
```
torch>=2.0
gymnasium[mujoco]==1.3.0
numpy          # 无版本号
pandas         # 无版本号
matplotlib     # 无版本号
pyyaml         # 无版本号
pytest         # 无版本号
```

**影响**：`torch>=2.0`意味着未来安装可能拉到torch 3.x，其数值精度/默认行为可能不同。numpy/pandas未锁版本在跨年份复现时可能产生细微差异。gymnasium[mujoco]==1.3.0锁了版本，这是好的。

**建议**：使用`pip freeze > requirements-lock.txt`或至少给numpy/pandas/pytest加上下限。

---

### [P2-04] 无观察值归一化（observation normalization）

**证据**：`src/`中无VecNormalize或running mean/std实现；`train.py`直接使用原始obs。

**影响**：标准MuJoCo SAC（Haarnoja 2018原文、SB3默认）通常对观察值做running mean/std归一化。缺少归一化会降低样本效率，这可能也是60k步远未收敛的原因之一。这不是bug，但意味着实现偏离了"标准SAC"配置。

**建议**：在后续实验中加入观察归一化作为对照臂，或在README中明确说明未归一化是有意为之。

---

### [P2-05] 无checkpoint/resume机制，OOM导致步数缩减无恢复方案

**证据**：
- `RESEARCH_RETRO.md`第55-57行："the first 6-wide parallel batch got OOM-killed...so I cut to 60k and ran 3-wide"
- `train.py`无模型保存/加载逻辑；`run_all.ps1`无resume逻辑。

**影响**：步数从120k砍到60k是诚实披露的，但根本原因（无checkpoint）意味着任何中途崩溃都意味着完全重跑。这限制了可复现性实验的规模。

**建议**：在train.py中加入定期torch.save + --resume参数。

---

### [P3-01] evaluate()每次调用新建env，浪费资源

**证据**：`train.py:28`：`env = gym.make(env_id)`在evaluate函数内部，每5000步调用一次。

**影响**：不影响正确性，但每次创建新MuJoCo env有开销。eval seed设计（`seed+1000+ep`）是好的。

---

### [P3-02] tanh修正项用epsilon而非数值稳定形式

**证据**：`src/networks.py:70`：`torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)`

**影响**：当pre_tanh很大（|x|>5）时，`1-tanh²(x)`趋近于0，加1e-6防止log(0)。更数值稳定的形式是`2*(log(2) - x - F.softplus(-2*x))`。当前实现引入微小偏差但在实践中可接受。

---

### [P3-03] 单元测试仅验证形状，不验证学习动力学

**证据**：`tests/test_core.py`：3个测试分别验证buffer形状/循环覆盖、actor输出范围、critic双Q形状。无测试验证：(a) logprob梯度方向正确，(b) critic loss下降，(c) 目标网络确实在软更新。

**影响**：CI中的`ci_check.py`（Pendulum学习门控）部分弥补了这个缺口，但单元测试层面的保护不足。

---

### [P3-04] Hopper-v4已被v5取代

**证据**：`logs/*.err.txt`中的DeprecationWarning；README第88行承认。

**影响**：作者选择v4是为了匹配经典SAC基准，这是合理的。但v5的物理参数略有不同，与文献对比时需注意。

---

## 三、算法正确性核对表

逐条核对SAC (Haarnoja et al. 2018)核心组件：

| 组件 | 标准做法 | 本项目实现 | 文件:行 | 结论 |
|---|---|---|---|---|
| Twin critic | 两个独立Q网络，取min | `TwinCritic`含q1/q2两个MLP | networks.py:80-81 | ✅ 正确 |
| Target Q取min | `min(Q1_t, Q2_t) - alpha*logp` | `tq = torch.min(tq1,tq2) - self.alpha*next_logp` | sac.py:80 | ✅ 正确 |
| Bellman target | `r + gamma*(1-d)*V_target` | `reward_scale*rew + (1.0-done)*gamma*tq` | sac.py:81 | ✅ 正确 |
| terminated vs truncated | bootstrap on truncation, not termination | `buf.add(..., terminated)`（仅存terminated） | train.py:107 | ✅ 正确 |
| Tanh squashing | `action = tanh(pre_tanh)` + Jacobian修正 | `action = torch.tanh(pre_tanh)`; logp减去`log(1-tanh²)` | networks.py:51,69-71 | ✅ 正确 |
| Gaussian log-prob中心化 | `(x-mu)²/sigma²`而非`x²/sigma²` | `((pre_tanh - mu) / std) ** 2` | networks.py:63 | ✅ 正确（RETRO记录了曾犯此bug并修复） |
| Soft target update | `theta_t <- tau*theta + (1-tau)*theta_t` | `pt.data.mul_(1-tau); pt.data.add_(tau*p.data)` | sac.py:109-111 | ✅ 正确 |
| Auto entropy temperature | `target_entropy = -act_dim`; 优化log_alpha | `self.target_entropy = -float(act_dim)`; loss=-log_alpha*(logp+H_target).mean() | sac.py:56,102 | ✅ 正确 |
| Actor loss | `alpha*logp - Q`的均值最小化 | `(self.alpha.detach()*logp - q_pi).mean()` | sac.py:93 | ✅ 正确 |
| 动作范围映射 | tanh输出[-1,1]映射到[low,high] | `act_scale=(high-low)/2; act_bias=(high+low)/2` | networks.py:36-37,57 | ✅ 正确 |
| Replay buffer | 均匀采样，环形覆盖 | `np.random.randint(0, self.size, ...)`; ptr循环 | replay_buffer.py:34-39 | ✅ 正确 |
| Warmup阶段 | 前N步随机动作，不更新 | `if step < start_steps: a=env.action_space.sample()` | train.py:98-99 | ✅ 正确 |

**算法结论**：未发现会导致结论不成立的实现错误。RETRO中记录的"log_prob中心化bug"已修复（networks.py:63正确使用了`pre_tanh - mu`）。

---

## 四、统计重算对照表

### 4.1 渐近回报（最后20% eval）

| 指标 | README声称 | plot.py实际输出 | 审查员独立重算(plot.py法) | 审查员独立重算(预登记法) |
|---|---|---|---|---|
| Baseline均值 | 363 | 362.9 | 362.9 | 372.8 |
| Baseline 95%CI | ±210 | ±210.0 | ±210.0 | ±217.1 |
| Scaled均值 | 1087 | 1086.7 | 1086.7 | 1043.6 |
| Scaled 95%CI | ±1029 | ±1028.6 | ±1028.6 | ±1107.6 |

**结论**：README数字与plot.py输出完全一致，与审查员独立重算（plot.py方法）完全一致。无数据篡改。

### 4.2 单种子最终eval点

| 种子 | README声称(baseline) | 重算值 | README声称(scaled) | 重算值 |
|---|---|---|---|---|
| s0 | 250 | 250.4 | 591 | 591.4 |
| s1 | 613 | 612.8 | 606 | 606.4 |
| s2 | 336 | 336.4 | 1375 | 1375.5 |

**结论**：完全一致。

### 4.3 显著性检验（描述性，非功效分析）

| 方法 | Welch's t | p值 |
|---|---|---|
| 预登记tail法（≥48k, 3点/种子） | 1.165 | 0.357 |
| plot.py tail法（最后2 grid点） | 1.351 | 0.300 |

**结论**：p>0.3，远不显著。作者"方向性信号、不显著、CI大幅重叠"的表述**完全诚实**。

---

## 五、实验设计审查

### 5.1 预登记时间线核查

通过git历史：
- commit 82f910b (2026-10-01 23:30:15)：首次提交，包含PRE_REGISTRATION.md + 部分baseline日志
- commit 6e3af23 (2026-10-02 00:21:01)：结果提交，PRE_REGISTRATION.md的diff仅为BOM字符+末尾换行（无实质内容修改）

**判断**：预登记在开跑前撰写（文件注明"Written 2026-10-01"），且在看到scaled臂结果后未修改假设。这是良好的科研实践。但baseline日志在首次提交时已部分存在，无法从git单独证明预登记严格在baseline开跑前——不过文件内容和诚实叙述可信。

### 5.2 步数缩减披露

- 预登记写120k步，实际跑60k。
- README第14行和RETRO第55行均明确披露了OOM原因和缩减。
- **判断**：披露诚实。但缩减损害了结论的一般性（见P1-02）。

### 5.3 评估协议

- 确定性策略（`deterministic=True`），5个eval episodes，固定seed（`seed+1000+ep`）。
- eval回报是无偏的（不参与训练梯度）。
- **判断**：协议无偏。但5个episode在Hopper高方差下不够稳定。

---

## 六、可复现性审查

| 项目 | 状态 | 证据 |
|---|---|---|
| 单元测试 | ✅ 通过 | `pytest tests/ -v` → 3 passed in 3.44s |
| plot.py可重画 | ✅ 通过 | 实际运行，输出数字与README一致 |
| CI学习门控 | ✅ 存在 | ci_check.py在Pendulum上assert回报提升>150 |
| 随机种子 | ✅ 三种子独立 | seeds 0,1,2；set_seed覆盖random/np/torch |
| requirements锁版本 | ⚠️ 部分 | gymnasium锁了，其余未锁（见P2-03） |
| Checkpoint/resume | ❌ 无 | train.py无保存/加载逻辑 |
| git提交纪律 | ✅ 好 | 3个commit，消息清晰，日志文件已提交 |

---

## 七、覆盖清单

- [x] 通读src/sac.py全部120行
- [x] 通读src/networks.py全部86行
- [x] 通读src/replay_buffer.py全部51行
- [x] 通读src/utils.py全部18行
- [x] 通读train.py全部149行
- [x] 核对所有SAC核心组件（见第三节核对表）
- [x] 读取全部6个progress.csv并独立重算统计
- [x] 读取PRE_REGISTRATION.md、README.md、RESEARCH_RETRO.md
- [x] git历史时间线核查（预登记 vs 结果）
- [x] 实际运行pytest（3 passed）
- [x] 实际运行plot.py（数字与README一致）
- [x] 检查figures/下两张图
- [x] 检查CI workflow
- [x] 检查requirements.txt
- [x] 检查Pendulum验证日志

---

## 八、无法验证项

1. **OOM事件的具体情况**：作者称6路并行OOM，但无法从仓库中验证当时内存状态。
2. **rewardscale_s2的2895峰值是否可复现**：未用新种子重跑该条件。
3. **与标准SB3 SAC的对比**：仓库中无SB3基线对照，无法判断本实现的性能是否与标准SAC一致。
4. **GPU/跨平台可复现性**：所有结果在Windows CPU上产生，Linux GPU上的行为未验证。

---

## 九、综合判断：能否向RL招生委员会证明"能独立完成RL研究"？

### 能证明的（强项）：
- ✅ **工程能力**：从零实现SAC，代码干净、结构清晰、无算法错误。这本身就是很大的加分项——很多申请者只会调SB3。
- ✅ **研究伦理意识**：预登记、多种子、诚实报告阴性/不显著结果、主动披露局限。这比很多已发表论文做得好。
- ✅ **调试能力**：RETRO中记录了logprob中心化bug的发现过程（Q符号异常→诊断→定位→修复），展示了正确的调试直觉。
- ✅ **工具链完整**：CI、pytest、学习门控、可复现plot.py、git纪律。

### 不能证明的（短板）：
- ❌ **实验设计能力**：n=3、60k步、单环境、效应被alpha混杂——这个实验设计即使结论正确，也不足以支撑任何论文级claim。
- ❌ **科学发现能力**：reward scaling是已知现象，没有新insight。"方向性信号"不是发现。
- ❌ **与标准方法的对照**：没有跑SB3 SAC或原文超参作为baseline，无法判断本实现是否达到标准性能。

### 结论：
这件作品证明了**"能正确实现RL算法并做诚实的小实验"**，但尚未证明**"能设计有统计功效的实验并产生可发表的科学结论"**。对于Embodied AI/PhD申请，它是一个**合格的writing sample起点**，但需要补强才能成为有力证据。

---

## 十、补强清单（按性价比排序）

| 优先级 | 补强项 | 预期收益 | 成本 |
|---|---|---|---|
| 1 | **补跑到150k步**（即使CPU慢，Pendulum已验证正确，夜间跑即可） | 直接解决P1-02"未收敛"问题，让"渐近"名副其实 | 每臂+~2h×3种子=6h/臂 |
| 2 | **加固定alpha对照臂**（automatic_entropy_tuning=false，alpha=0.2或0.05） | 分离reward scaling和alpha的效应，解决P1-01 | 3种子×60k=~5h CPU |
| 3 | **加到5-6个种子** | 将p值从0.3降到可检测范围，解决P2-01 | 每臂+2-3种子≈10h CPU |
| 4 | **加SB3 SAC基线对照**（同一环境、同样步数） | 证明本实现达到标准性能水平，而非自定义偏差 | 1天（需装SB3，但作者未用是设计选择） |
| 5 | **加观察归一化臂** | 解决P2-04，同时可能大幅提升样本效率 | ~2h实现+运行 |
| 6 | **锁requirements版本**（pip freeze） | 提升可复现性 | 5分钟 |
| 7 | **加checkpoint/resume** | 防止OOM重跑，支撑更长训练 | ~30分钟实现 |
| 8 | **加第二个环境**（如HalfCheetah-v4或Walker2d-v4） | 证明结论不只是Hopper特例 | 每环境~10h CPU |
