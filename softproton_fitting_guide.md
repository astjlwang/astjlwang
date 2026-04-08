# XMM-Newton Residual Soft Proton 拟合指南

## 目录
1. [各成分在不同能段的物理主导性](#1-各成分在不同能段的物理主导性)
2. [高能段低信噪比的处理方法](#2-高能段低信噪比的处理方法)
3. [ESAS proton 命令中 bnorm 的物理含义与合理范围](#3-esas-proton-命令中-bnorm-的物理含义与合理范围)
4. [如何判断拟合是否正确](#4-如何判断拟合是否正确)
5. [参考文献](#5-参考文献)

---

## 1. 各成分在不同能段的物理主导性

你的完整谱模型为：

```
total = rsp_photon[ LHB + tbabs*(powerlaw + MWhalo) + Gaussian_1 + Gaussian_2 ]
      + rsp_softproton[ bknpower ]
```

各成分在物理上的能段主导性如下：

### 1.1 Local Hot Bubble (LHB) — kT ~ 0.1 keV

- **主导能段**: < 0.3 keV（极软 X 射线）
- **物理**: LHB 是太阳系周围的低密度热等离子体，温度约 ~10^6 K (kT ~ 0.1 keV)。它**不受银河系吸收**，因此以最软的 X 射线波段发射为主。
- **在 0.5-6 keV 能段的贡献**: 非常微弱。当你的 emin = 0.5 keV 时，LHB 的贡献已经很小；如果 emin > 1 keV，LHB 基本无法被拟合约束。
- **合理拟合表现**: LHB 的 norm 应该很小（典型 ~10^{-6} 量级），如果 LHB norm 远大于预期或驱动了 > 1 keV 的发射，说明拟合有问题。

### 1.2 Milky Way Halo / Hot Gas — kT ~ 0.2-0.3 keV

- **主导能段**: 0.3 - 1.0 keV
- **物理**: 银河系晕的热气体，温度约 ~2-3 × 10^6 K。被银河系前景 NH 吸收，主要在 0.5-1.0 keV 贡献显著（O VII ~0.57 keV, O VIII ~0.65 keV 等发射线）。
- **合理拟合表现**: kT 应在 0.1-0.6 keV 范围，norm 典型 ~10^{-6}。如果 kT 跑到很高的值（> 0.6 keV），可能在与 soft proton 或 CXB 发生简并。

### 1.3 Cosmic X-ray Background (CXB) — Gamma ~ 1.46

- **主导能段**: > 2 keV
- **物理**: 主要来自未分辨的 AGN，谱形为幂律 Gamma ~ 1.4-1.5。在 2-8 keV 范围内是天空 X 射线背景的绝对主导成分。
- **合理拟合表现**:
  - 光子指数通常固定在 1.46（De Luca & Molendi 2004; Chen et al. 1997）。
  - normalization 取决于点源去除阈值。ESAS cookbook 建议 10.5 keV cm^{-2} s^{-1} sr^{-1} keV^{-1} @ 1 keV 是一个标准值（对应 arcmin^{-2} 单位时约 ~4e-7）。
  - 如果你仔细去除了点源，CXB norm 可以冻结或仅微调。如果 CXB norm 变得不合理地大或小，检查点源去除是否充分。

### 1.4 Instrumental Gaussian Lines — 1.49 keV (Al K) & 1.75 keV (Si K)

- **主导能段**: 极窄的线特征
- **物理**: MOS 探测器的荧光线（Al Kα ~1.49 keV, Si Kα ~1.75 keV）。这些是**仪器特征**，不是天体物理发射。PN 没有明显的 Si K 线。
- **重要注意**: 这些线应通过 diag.rsp（对角响应矩阵）拟合，因为它们是粒子背景的一部分，但你目前在正常响应中拟合它们也可以接受，只是理论上 ESAS 推荐用 diag.rsp。sigma 固定为 0 是合理的起点，拟合好后可尝试解冻（通常宽度很小 ~ 0-30 eV）。

### 1.5 Residual Soft Proton (SP) — Broken Power Law

- **主导能段**: 覆盖整个能段，但最显著影响在 < 5 keV
- **物理**: 低能质子（几百 keV 以下）通过望远镜光学聚焦到探测器上。它们的谱形**类似**平坦的幂律，在 ~3.0-3.2 keV 处有折断（break energy），典型地：
  - **低能指数 (PhoIndx1)**: MOS 典型 0.1-1.4；PN 可能更陡
  - **高能指数 (PhoIndx2)**: 可以比低能指数更陡
  - **Break energy**: 通常固定在 ~3.0-3.2 keV（Kuntz & Snowden 2008）
- **关键**: SP 不走 ARF（有效面积），只用 diag.rsp（对角响应矩阵），因为质子不是光子，不经历相同的有效面积响应。

### 1.6 能段总结表

| 能段 (keV)  | 主导成分                        | 次要成分              |
|-------------|-------------------------------|-----------------------|
| < 0.3       | LHB                           | —                     |
| 0.3 - 0.7   | LHB + MWhalo + SP             | CXB (弱)             |
| 0.7 - 1.5   | MWhalo + SP                   | CXB                   |
| 1.5 - 2.0   | CXB + SP + Gaussian lines    | MWhalo (弱)           |
| 2.0 - 3.0   | CXB + SP                     | —                     |
| 3.0 - 5.0   | CXB + SP（SP 在 break 后变陡）| —                     |
| 5.0 - 10.0  | CXB (弱) + QPB 残余           | SP (非常弱)           |

### 1.7 什么样的拟合结果是"物理上合理"的

- **LHB**: norm 很小，在 0.5 keV 以上基本"看不见"。如果你的拟合中 LHB 对 > 1 keV 有显著贡献，可能有问题。
- **MWhalo**: 在 0.5-1 keV 有明显贡献，但 > 2 keV 基本消失。kT 应在 0.15-0.5 keV。
- **CXB**: 在 > 2 keV 应该是稳定的、平滑的幂律贡献。
- **SP**: 在低能端和 CXB/thermal 成分有简并。SP 在 break 以上应该陡降。在图中 SP 成分不应该在 > 5 keV 还很强（除非确实有严重的 soft proton 污染）。
- **Gaussian lines**: 只在 ~1.49 和 ~1.75 keV 处有窄线贡献。

**判断方法**: 画出各成分分开的图（你的代码已经做了），检查：
1. 各成分是否在其物理上应该主导的能段主导
2. SP 成分是否"合理"——不应比天空背景亮太多倍（除非观测确实受 SP 严重污染）
3. 残差 (delchi) 是否有系统性结构（连续正或负偏）

---

## 2. 高能段低信噪比的处理方法

你描述的现象——"高能段总是有很多直接误差棒下限过大的长线"——是 X 射线谱分析中非常常见的问题。

### 2.1 原因

高能段（> 3-4 keV 对于弥漫源）计数率通常很低，原因：
- 天体物理背景（CXB）本身在高能段就弱
- 有效面积在高能下降
- 弥漫源/extended emission 在高能通常更弱

当每个 bin 的计数数很少（< 5-10 counts）时，chi-square 统计量不再合适，Gehrels 近似会给出很大的误差棒。

### 2.2 解决方案

#### 方案 A: 改用 C-statistic（推荐）

最根本的解决方案是**放弃 chi-square 统计，改用 cstat 或 wstat**。

**在 Sherpa 中**:
```python
ui.set_stat("wstat")  # 有背景谱时用 wstat
# 或
ui.set_stat("cstat")  # 没有背景谱或需要自己建模背景时用 cstat
```

**在 XSPEC 中**:
```
statistic cstat
```

C-stat / W-stat 基于 Poisson 似然函数，在低计数情况下也能正确工作，不需要大量的 binning。

**重要**：使用 cstat/wstat 时**不要做背景相减**（subtract）。让统计量自动处理背景，或者同时建模源和背景。

#### 方案 B: 更激进的 binning / 自适应分组

如果你仍然想用 chi-square（比如为了画图更好看），可以增加 binning：

**在 Sherpa 中**:
```python
# 按最少计数分组
ui.group_counts(cam, 25)   # 每 bin 至少 25 counts

# 或者按信噪比分组
ui.group_snr(cam, 3)       # 每 bin 信噪比至少 3

# 或者自适应分组
ui.group_adapt(cam, 20)    # 自适应，至少 20 counts
```

**用 grppha 工具（命令行）**:
```bash
grppha infile.pi outfile.pi "group min 25" "exit"
```

#### 方案 C: 限制拟合能段

如果高能段确实信噪比太低且对你的科学目标不关键，可以降低 emax：
- MOS: emax = 5.0 keV（而非 6.0 或更高）
- PN: emax = 4.0-5.0 keV

这样丢掉了一些信息，但避免了低计数 bin 对拟合的干扰。

#### 方案 D: 混合策略（推荐的实际操作流程）

1. **先用 cstat + 较少的 binning** 做拟合，获得最佳拟合参数
2. **用 chi2gehrels + group_counts(25)** 重新加载数据并画图，只用于可视化
3. **或者**：用 cstat 拟合，然后画图时单独对数据做 adaptive rebinning

### 2.3 代码中的修改建议

你的代码中设置了 `STAT = "chi2gehrels"`。建议改为：

```python
STAT = "wstat"        # 如果有背景谱且不做 subtract
USE_SUBTRACT = False  # 使用 wstat 时不要相减
```

或者：

```python
STAT = "cstat"        # 如果你 subtract 了背景或没有背景
```

同时建议增加分组：
```python
ui.group_counts(cam, 1)  # wstat 可以用 1 count/bin（不分组也行）
# 或者画图时临时分组
ui.group_counts(cam, 20)  # 用于 chi2 画图
```

---

## 3. ESAS proton 命令中 bnorm 的物理含义与合理范围

### 3.1 bnorm 的物理含义

`bnorm` 是 broken power law 模型的归一化系数，它控制 soft proton 成分的**整体强度**。

在 XSPEC/ESAS 的 `bknpower` 模型中，模型形式为：

```
A(E) = K * E^{-PhoIndx1}                    for E < BreakE
A(E) = K * BreakE^{PhoIndx2 - PhoIndx1} * E^{-PhoIndx2}   for E > BreakE
```

其中 K 就是 `bnorm`（归一化），单位是 photons/cm^2/s/keV（在 1 keV 处）。

但是，因为 soft proton 使用的是 **diag.rsp**（单位对角响应矩阵），而非真实的 ARF+RMF，所以 bnorm 的数值含义与普通光子模型不同——它实际上直接对应 **counts/s/keV**（在探测器上的计数率）。

### 3.2 合理的 bnorm 范围

根据 Snowden & Kuntz (ESAS cookbook) 和 Kuntz & Snowden (2008, A&A 478, 575)：

| 污染程度     | bnorm 范围 (arcmin^{-2})  | 说明                             |
|-------------|--------------------------|----------------------------------|
| 无/极微弱    | < 10^{-5}               | 可以忽略，与 0 无实际差别          |
| 轻微         | 10^{-5} ~ 10^{-3}       | 正常的残余 SP 污染                |
| 中等         | 10^{-3} ~ 10^{-2}       | 需要仔细建模                      |
| 严重         | > 10^{-2}               | 观测可能受到严重污染，需谨慎       |

**你的例子**: `bnorm=0.0019533` ≈ 2 × 10^{-3}，属于**轻微到中等**的残余 SP 污染，这是完全正常的值。

### 3.3 实际注意事项

1. **bnorm 初始值**: ESAS cookbook 中 Snowden et al. (2008) 用 10^{-5} 作为初始猜测值。
2. **归一化与面积的关系**: 如果你使用了前面的 `const*const*(bknpower)` 形式（ESAS 标准），第一个 const 是提取区域的面积（arcmin^2），第二个 const 是仪器间的校准因子。如果你没有用 const，那 bnorm 就是整个提取区域的总计数率归一化。
3. **区域转换**: 如果你在一个子区域拟合了 SP，然后需要用 `proton` 命令做全 FOV 的图像，需要用 `sppartial`/`protonscale` 工具将 bnorm 从子区域 scale 到全 FOV。
4. **你的命令**: `proton imagefile=... speccontrol=2 bindl=1.4317 bbreak=3 bindh=2.3909 bnorm=0.0019533` 中，bnorm 就是在你拟合时得到的全 FOV 归一化。这个值 ~2×10^{-3} 是合理的。

### 3.4 判断 bnorm 是否合理的方法

1. **检查 SP 贡献相对 CXB 的比值**: 在 2-5 keV，SP 的贡献不应该远超 CXB（除非观测确实受到严重污染）。
2. **与 espfilt 的诊断一致**: espfilt 的 IN/OUT ratio（特别是 8-12 keV 的 Sigma_IN/Sigma_OUT）可以独立评估 SP 污染程度。如果 ratio > 1.15，说明有显著 SP 污染，bnorm 应该较大。
3. **比较不同探测器**: MOS1、MOS2 的 bnorm 应该在同一量级（但不必相等）；PN 的 bnorm 可能不同，因为 PN 对 SP 的响应不同。

---

## 4. 如何判断拟合是否正确

### 4.1 统计检验

- **chi2/dof**: 对于 chi-square 拟合，reduced chi^2 (chi^2/dof) 应接近 1.0。
  - > 1.5: 拟合不佳，模型可能缺少成分或参数约束不当
  - < 0.7: 可能过拟合或误差估计过大
  - 1.0 ± 0.2: 通常可接受

- **cstat/wstat**: 对于 C-stat，没有简单的 "reduced cstat" 标准。可以用 goodness 命令：
  ```python
  # Sherpa
  from sherpa.astro import ui
  ui.goodness(cam, nsim=1000)
  ```
  ```
  # XSPEC
  goodness 1000
  ```
  这会通过 Monte Carlo 模拟告诉你有多少百分比的模拟比你的拟合更差。理想情况下应在 20%-80% 之间。

### 4.2 残差检查

- **delchi 图**: 残差应随机分布在 0 附近，大多数在 ±2σ 以内。
- **需要警惕的模式**:
  - 某个能段连续多个 bin 偏正或偏负（系统偏差）→ 缺少模型成分
  - 在某个能量处有明显的"台阶" → 可能缺少吸收边
  - Gaussian 线位置处残差异常 → 线能量或宽度需调整

### 4.3 物理合理性检查

- LHB 和 MWhalo 的参数是否与 ROSAT 全天调查 (RASS) 数据一致？
  - 可以从 HEASARC 的 X-ray background tool 获取你的天区的预期值
- CXB 的 norm 是否与文献值一致（考虑到点源去除阈值）？
- SP 的指数是否在合理范围（MOS: 0.1-1.4 低能, 0.5-2.5 高能; Kuntz & Snowden 2008）？

### 4.4 多种拟合策略交叉验证

1. **固定 vs 解冻**: 先固定 CXB index 和 NH，确认其他参数合理，再逐步解冻
2. **不同统计量**: 用 chi2 和 cstat 分别拟合，看结果是否一致
3. **不同能段**: 在较窄的能段（如 2-5 keV）拟合 SP+CXB，然后扩展到全能段看是否一致
4. **RASS 约束**: 如 ESAS cookbook 强烈建议的，加入 ROSAT 全天调查谱作为低能约束，可以大幅改善 LHB/MWhalo/SP 的简并

### 4.5 增强版脚本提供的诊断功能

我们在改进版脚本 (`softproton_fit_enhanced.py`) 中增加了以下诊断：

1. **自动 goodness-of-fit 评估**: 输出 reduced chi2 或 cstat goodness
2. **支持 wstat/cstat**: 避免高能低计数问题
3. **自适应 binning**: 提供 `group_counts` 和 `group_snr` 选项
4. **参数合理性检查**: 自动检查拟合参数是否在文献推荐范围内
5. **成分比值图**: 画出各成分对总模型的贡献百分比

---

## 5. 参考文献

1. **Kuntz, K. D. & Snowden, S. L.** (2008). "The EPIC-MOS particle-induced background spectra." A&A 478, 575. — soft proton 谱形的系统研究，broken power law 参数范围
2. **Snowden, S. L. et al.** (2008). "A catalog of galaxy clusters observed by XMM-Newton." A&A 478, 615. — ESAS 拟合方法示范，各成分初始值
3. **De Luca, A. & Molendi, S.** (2004). "The 2–8 keV cosmic X-ray background spectrum." A&A 421, 1065. — CXB 谱形和 SP 残余的诊断
4. **Snowden, S. L. & Kuntz, K. D.** (2025). "ESAS Cookbook, SAS V22." — 拟合策略和模型设置的权威指南
5. **Henley, D. B. & Shelton, R. L.** (2013). "An XMM-Newton Survey of the SXRB. III." ApJ 773, 92. — 银河晕 X 射线发射的系统测量
