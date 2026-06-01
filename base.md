# 基于 DSRL 的 Actor 双输出改进方案讨论

## 1. 初始想法

用户希望以 **DSRL** 为基础，参考图像生成领域的方法，让 actor 策略网络不仅输出噪声，还输出另一个额外信息，从而提高机器人任务成功率。

原始 DSRL 可以理解为：

\[
w_t = \pi_\theta(o_t)
\]

\[
a_t = \pi_{\text{diff}}^{\text{frozen}}(o_t, w_t)
\]

其中：

- \(o_t\)：当前观测；
- \(w_t\)：actor 输出的 latent noise；
- \(\pi_{\text{diff}}^{\text{frozen}}\)：冻结的 diffusion policy；
- \(a_t\)：最终执行动作。

DSRL 的核心思想是：

> 冻结原始 diffusion policy，只通过 RL 训练 actor 在 latent-noise space 中选择更好的噪声，从而提升任务成功率。

---

## 2. 已有方案：Noise + Residual Action

一开始提出的方案是：

\[
\pi_\theta(o_t) \rightarrow (w_t, r_t)
\]

其中：

- \(w_t\)：latent noise；
- \(r_t\)：residual action。

最终动作：

\[
a_t = \pi_{\text{diff}}^{\text{frozen}}(o_t, w_t) + \lambda r_t
\]

这个方案的直觉是：

> noise 负责选择更好的行为模式，residual action 负责修正最终动作细节。

但是用户指出：

> 同时输出噪声和残差动作已经有人做了，需要想一个不同的方向。

因此后续方案需要避免直接输出 residual action。

---

## 3. 重新思考方向

如果不走 residual action，那么 actor 的第二个输出不应该作用在最终 action space，而应该作用在：

1. condition space；
2. adapter space；
3. attention space；
4. prompt/token space；
5. denoising process。

也就是说，第二个输出不再是：

\[
r_t
\]

而可以是：

- task prompt token；
- adapter control code；
- grounding token；
- attention bias；
- LoRA mixture weight；
- denoising-level control signal。

---

# 4. 可参考的图像生成方法

用户指出，之前提到的 PAG、SAG、FreeU、CFG schedule 等方法多数是 **train-free** 方法，而 DSRL 中 actor 是需要通过 RL 训练的。

因此更合适参考的是图像生成中这些 **冻结主模型、训练外部模块** 的方法。

## 4.1 ControlNet

ControlNet 的思想是：

> 冻结原始 diffusion model，同时训练一个额外的控制分支，将外部条件注入主模型。

对应到 DSRL 中，可以理解为：

> 冻结原 diffusion policy，训练一个额外 adapter/control module 来调控动作生成过程。

---

## 4.2 T2I-Adapter

T2I-Adapter 的思想是：

> 冻结原始 text-to-image diffusion model，只训练轻量 adapter，把额外条件转化为 diffusion model 可用的控制特征。

对应到 DSRL 中：

> 训练一个 adapter，把 actor 输出的 control code 转换成 action denoiser 的中间特征。

---

## 4.3 IP-Adapter

IP-Adapter 的思想是：

> 训练一个 image prompt adapter，通过 decoupled cross-attention 向 frozen diffusion model 注入图像提示信息。

对应到 DSRL 中：

> actor 可以输出 task-specific 或 state-specific prompt embedding，用来指导 frozen VLA/diffusion policy。

---

## 4.4 GLIGEN

GLIGEN 的思想是：

> 冻结原始 diffusion model，加入可训练的 gated attention layer，将 grounding token 注入生成过程。

对应到 DSRL 中：

> actor 可以输出 grounding token 和 gate，让 frozen policy 在特定状态下关注关键任务信息。

---

## 4.5 Textual Inversion / Prompt Tuning

Textual Inversion 的思想是：

> 冻结大模型，只学习新的 token embedding，让模型表示新的概念。

对应到 DSRL 中：

> actor 可以输出 state-conditioned task prompt token，用作当前状态下的行为提示。

---

# 5. 几个候选方案

## 5.1 Noise + Task Prompt Token

形式为：

\[
\pi_\theta(o_t) \rightarrow (w_t, e_t)
\]

其中：

- \(w_t\)：latent noise；
- \(e_t\)：task prompt token 或 behavior token。

然后将 \(e_t\) 作为额外 token 注入 frozen diffusion policy：

\[
a_t = \pi_{\text{diff}}^{\text{frozen}}(o_t, w_t, e_t)
\]

这个方案的直觉是：

> actor 不直接修改动作，而是告诉 frozen policy 当前应该采用哪种行为模式。

例如：

- approach mode；
- contact mode；
- grasp mode；
- pull mode；
- recovery mode。

优点：

- 实现相对简单；
- 不直接修改最终动作；
- 适合 VLA 的 token-based architecture。

缺点：

- 控制能力可能不够强；
- 更偏高层行为提示，可能难以修正精细动作。

---

## 5.2 Noise + Gated Grounding Token

形式为：

\[
\pi_\theta(o_t) \rightarrow (w_t, q_t, g_t)
\]

其中：

- \(w_t\)：latent noise；
- \(q_t\)：grounding token；
- \(g_t\)：gate，控制 token 注入强度。

注入方式：

\[
h = h + \sigma(g_t) \cdot \text{Attn}(h, q_t)
\]

这个方案的直觉是：

> actor 学习一个 RL grounding token，通过 gate 控制其对 frozen policy 的影响。

优点：

- 适合 Transformer-based VLA；
- gate 具有解释性；
- 比 task prompt token 控制力更强。

缺点：

- 依赖模型内部 attention 结构；
- 实现复杂度高于 prompt token。

---

## 5.3 Noise + Attention Bias

形式为：

\[
\pi_\theta(o_t) \rightarrow (w_t, b_t)
\]

其中 \(b_t\) 是 attention bias。

在 cross-attention 中：

\[
A =
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d}} + b_t
\right)
\]

这个方案的直觉是：

> actor 通过 attention bias 控制 frozen VLA 更关注哪些视觉 token、语言 token 或 robot state token。

优点：

- 不直接修改动作；
- 可以增强任务相关 token 的注意力。

缺点：

- 改变注意力不一定能显著改善最终动作；
- 控制强度中等。

---

## 5.4 Noise + LoRA Mixture Weight

形式为：

\[
\pi_\theta(o_t) \rightarrow (w_t, \lambda_{1:M})
\]

其中：

- \(w_t\)：latent noise；
- \(\lambda_{1:M}\)：多个 LoRA adapter 的混合权重。

模型权重变为：

\[
W' = W + \sum_{i=1}^{M} \lambda_i \Delta W_i
\]

其中：

\[
\Delta W_i = A_i B_i
\]

这个方案的直觉是：

> 不同 LoRA adapter 对应不同技能模式，actor 根据当前状态选择不同 adapter 的混合比例。

例如：

- approach adapter；
- grasp adapter；
- contact adapter；
- recovery adapter。

优点：

- 控制能力较强；
- 可以表达不同技能模式。

缺点：

- 更像微调 base model；
- 可能不够符合纯 DSRL 的思想。

---

## 5.5 Noise + Denoising Action

形式为：

\[
\pi_\theta(o_t) \rightarrow (w_t, u_{1:K})
\]

其中：

- \(w_t\)：初始 latent noise；
- \(u_{1:K}\)：每个 denoising step 的 latent control action。

每一步去噪：

\[
z_{k-1}
=
D_\phi^{\text{frozen}}(z_k, o_t, k)
+
G_\psi(z_k, o_t, k, u_k)
\]

这个方案的直觉是：

> actor 不只控制初始噪声，还控制每一步 denoising 过程。

优点：

- 理论上上限很高；
- 和 DDPO、DPOK 等 diffusion RL 方法有关系。

缺点：

- RL 维度过高；
- 训练难度大；
- 样本效率可能较差。

---

# 6. 最推荐方案：Noise + Adapter Feature

经过比较，最推荐的方案是：

\[
\boxed{
\text{Noise + Adapter Feature}
}
\]

也可以称为：

\[
\boxed{
\text{Adapter-Conditioned DSRL}
}
\]

---

## 6.1 核心思想

让 actor 同时输出：

\[
(w_t, c_t, g_t) = \pi_\theta(o_t)
\]

其中：

- \(w_t\)：DSRL 原来的 latent noise；
- \(c_t\)：低维 control code；
- \(g_t\)：adapter gate，控制 adapter 注入强度。

然后用一个可训练 adapter：

\[
F^{adapter}_{1:L} = A_\psi(c_t, o_t)
\]

将 control code 转换成多层 adapter feature。

再把这些 adapter feature 注入 frozen diffusion policy 的 denoising network：

\[
h_l \leftarrow h_l + \sigma(g_l) \cdot P_l(F^{adapter}_l)
\]

最终动作由 frozen diffusion policy 生成：

\[
a_t =
\pi_{\text{diff}}^{\text{frozen}}
(o_t, w_t; F^{adapter}_{1:L})
\]

---

## 6.2 和原始 DSRL 的区别

原始 DSRL：

\[
\pi_\theta(o_t) \rightarrow w_t
\]

只控制扩散采样的初始噪声。

Adapter-Conditioned DSRL：

\[
\pi_\theta(o_t) \rightarrow (w_t, c_t, g_t)
\]

不仅控制初始噪声，还通过 adapter feature 控制 denoising network 的中间特征。

对比：

| 方法 | 控制对象 | 控制强度 | 是否直接改最终动作 |
|---|---|---:|---|
| DSRL | 初始 latent noise | 中等 | 否 |
| Noise + Residual Action | 最终 action | 强 | 是 |
| Noise + Adapter Feature | denoising 中间特征 | 强 | 否 |

因此该方法可以表述为：

> DSRL 控制采样起点，而 Adapter Feature 控制采样过程。

---

# 7. Noise + Adapter Feature 的直觉

可以理解成两级控制：

## 7.1 Noise 负责选择行为模式

\[
w_t
\]

它决定 frozen diffusion policy 从哪个 latent noise 开始采样。

直觉是：

> 从原始策略可能产生的动作分布中，选择一个更可能成功的轨迹模式。

---

## 7.2 Adapter Feature 负责调节生成过程

\[
F^{adapter}_{1:L}
\]

它不是最终动作残差，而是在 denoising network 的中间层改变 hidden representation。

直觉是：

> 在动作生成过程中，提醒模型当前应该更关注接触、更关注末端姿态、更关注目标区域，或者更保守地执行。

因此：

- DSRL 的 noise 改变采样起点；
- adapter feature 改变采样过程；
- residual action 改变最终结果。

---

# 8. Adapter Feature 的注入方式

## 8.1 Additive Adapter

最简单的方式是：

\[
h_l \leftarrow h_l + \alpha_l F_l
\]

其中：

\[
F_l = P_l(A_\psi(c_t, o_t))
\]

为了训练稳定，可以使用 zero initialization：

\[
h_l \leftarrow h_l + \sigma(g_l) \cdot \text{ZeroLinear}_l(F_l)
\]

这样初始时 adapter 输出接近 0，模型退化为普通 DSRL，不会破坏原始策略。

---

## 8.2 FiLM Adapter

FiLM 形式为：

\[
h_l \leftarrow (1 + s_l) \odot h_l + b_l
\]

adapter 输出：

\[
(s_l, b_l) = A_\psi(c_t, o_t, l)
\]

其中：

- \(s_l\)：scale；
- \(b_l\)：bias。

这个方式比 additive 更强，因为它不仅能加信息，还能重新标定 hidden state。

---

## 8.3 Cross-Attention Adapter

如果 diffusion policy 是 Transformer-based action head，可以用 cross-attention 注入：

\[
h_l \leftarrow h_l + \sigma(g_l) \cdot \text{CrossAttn}(h_l, F_l)
\]

其中 \(F_l\) 可以看作一组控制 token。

优点：

- 表达能力强；
- 适合 VLA / Transformer 结构。

缺点：

- 实现复杂；
- 训练不如 additive 或 FiLM 稳定。

---

# 9. 最小可实现版本

建议先实现最简单版本：

\[
(w_t, c_t, g_t) = \pi_\theta(o_t)
\]

\[
F = A_\psi(c_t)
\]

只在 denoising network 的中间层注入：

\[
h_{mid}
\leftarrow
h_{mid}
+
\sigma(g_t)
\cdot
\text{ZeroLinear}(F)
\]

最后继续正常 denoising，得到 action chunk：

\[
a_{t:t+H}
=
\pi_{\text{diff}}^{\text{frozen}}
(o_t, w_t; F)
\]

该版本特点：

1. 不直接修改最终动作；
2. 不训练整个 VLA；
3. adapter 很小；
4. 初始扰动接近 0；
5. 和 DSRL 代码结构容易结合。

---

# 10. 训练方式

## 10.1 冻结内容

冻结原始 diffusion policy：

\[
\phi \text{ frozen}
\]

训练：

\[
\theta, \psi
\]

其中：

- \(\theta\)：actor 参数；
- \(\psi\)：adapter 参数；
- \(\phi\)：原 diffusion policy 参数。

---

## 10.2 执行过程

actor 输出：

\[
(w_t, c_t, g_t) = \pi_\theta(o_t)
\]

adapter 输出：

\[
F^{adapter}_{1:L} = A_\psi(c_t, o_t)
\]

frozen diffusion policy 生成动作：

\[
a_t =
\pi_\phi^{\text{frozen}}
(o_t, w_t; F^{adapter}_{1:L}, g_t)
\]

环境返回 reward：

\[
r_t
\]

训练目标：

\[
\max_{\theta,\psi}
\mathbb{E}
\left[
\sum_t \gamma^t r_t
\right]
\]

---

## 10.3 Actor-Critic 形式

critic 可以输入最终动作：

\[
Q_\eta(o_t, a_t)
\]

actor loss：

\[
\mathcal{L}_{actor}
=
-
Q_\eta(o_t, a_t)
+
\lambda_c \|c_t\|^2
+
\lambda_g \|g_t\|_1
\]

其中：

\[
a_t =
\pi_\phi^{\text{frozen}}
(o_t, w_t; A_\psi(c_t), g_t)
\]

这样梯度可以从 critic 经过 action，反传到 actor 和 adapter。

---

# 11. 推荐训练流程

## 阶段一：初始化为普通 DSRL

让 adapter 输出接近 0：

\[
h_l \leftarrow h_l + 0
\]

因此初始时：

\[
a_t \approx \pi_\phi^{\text{frozen}}(o_t, w_t)
\]

也就是退化为普通 DSRL。

具体做法：

- adapter 最后一层 zero initialization；
- gate 初始值较小；
- 对 adapter feature 加 L2 正则；
- 对 gate 加 L1 正则。

---

## 阶段二：训练 actor 输出 noise + control code

actor 输出：

\[
(w_t, c_t, g_t)
\]

其中 \(c_t\) 建议低维：

\[
d_c = 16 \text{ or } 32
\]

不要让 actor 直接输出大规模 feature，否则 RL 维度太高，训练不稳定。

可以使用：

\[
c_t = \tanh(f_\theta(o_t))
\]

将范围限制在：

\[
[-1, 1]
\]

gate 使用：

\[
g_t = \sigma(f^g_\theta(o_t))
\]

将范围限制在：

\[
[0, 1]
\]

---

## 阶段三：逐渐开放 adapter 影响

可以设计 warm-up 系数：

\[
h_l \leftarrow h_l + \beta_{train} \cdot \sigma(g_t) \cdot F_l
\]

其中：

\[
\beta_{train}
\]

从 0 逐渐增加到 1。

这样可以避免 adapter 在训练初期对 frozen policy 造成过大扰动。

---

# 12. 为什么该方案可能提高成功率？

因为它解决了 DSRL 的限制：

> DSRL 只能选择初始 noise，不能改变 frozen diffusion policy 的内部生成过程。

Adapter feature 可以改变 denoising hidden state，因此能影响：

1. action chunk 的整体轨迹；
2. 中间时间步的运动模式；
3. 夹爪开合时机；
4. 接触阶段动作稳定性；
5. 视觉、语言和本体信息的融合方式；
6. failure recovery 行为。

例如，原 DSRL 可能生成：

\[
\text{靠近目标} \rightarrow \text{稍微偏一点} \rightarrow \text{抓取失败}
\]

而 adapter feature 可以让动作变成：

\[
\text{靠近目标} \rightarrow \text{对齐目标} \rightarrow \text{稳定闭合夹爪}
\]

---

# 13. 建议的 Ablation

为了证明 adapter feature 有用，可以比较：

| 方法 | 说明 |
|---|---|
| Frozen Diffusion Policy | 原始 BC policy |
| DSRL | 只输出 noise |
| DSRL + Random Adapter | adapter 随机但不训练 |
| DSRL + Control Code Only | actor 输出 \(c_t\)，但不注入中间层 |
| DSRL + Adapter Feature | 完整方法 |
| DSRL + Adapter Feature w/o Gate | 去掉 gate |
| DSRL + Adapter Feature w/o Zero Init | 去掉 zero initialization |
| DSRL + Adapter Feature at Early Layers | 只注入前层 |
| DSRL + Adapter Feature at Mid Layers | 只注入中层 |
| DSRL + Adapter Feature at Late Layers | 只注入后层 |

重点比较：

\[
\text{DSRL}
\quad \text{vs} \quad
\text{DSRL + Adapter Feature}
\]

以及：

\[
\text{with gate}
\quad \text{vs} \quad
\text{without gate}
\]

如果结果是：

\[
\text{DSRL + Adapter Feature} > \text{DSRL}
\]

并且：

\[
\text{with gate} > \text{without gate}
\]

那么方法故事就比较完整。

---

# 14. 实现注意点

## 14.1 Adapter feature 不要太大

actor 只输出低维 \(c_t\)，例如：

\[
d_c = 16, 32, 64
\]

不要让 actor 直接输出和 hidden state 一样大的 feature。

---

## 14.2 使用 zero initialization

adapter 最后一层建议使用 zero initialization。

这样初始时：

\[
F^{adapter} \approx 0
\]

模型退化为普通 DSRL，训练更稳定。

---

## 14.3 必须使用 gate

gate 控制 adapter 的注入强度：

\[
h_l \leftarrow h_l + \sigma(g_l) F_l
\]

没有 gate 时，adapter 可能在原策略本来能成功的状态下过度干预。

---

## 14.4 优先注入中后层

前层通常更影响整体动作结构，扰动过大可能破坏原策略。

中后层更适合调节：

- 接触细节；
- 夹爪动作；
- 末端姿态；
- action chunk 平滑性。

---

## 14.5 加正则

建议加入：

\[
\lambda_c \|c_t\|^2
+
\lambda_F \|F^{adapter}\|^2
+
\lambda_g \|g_t\|_1
\]

防止 adapter 过强。

---

# 15. 可以写成论文贡献

英文表述：

> Existing DSRL methods improve frozen diffusion policies by learning latent noise steering. However, latent steering only controls the initial sampling point and may be insufficient for fine-grained manipulation. Inspired by trainable adapters in controllable image diffusion models, we propose Adapter-Conditioned DSRL, where the RL actor jointly outputs a latent noise and a compact control code. The control code is transformed by a lightweight adapter into intermediate denoising features, enabling state-conditioned modulation of a frozen diffusion policy without directly modifying the final action or updating the base model weights.

中文表述：

> 现有 DSRL 只通过 latent noise 控制 frozen diffusion policy 的采样起点，难以精细调节动作生成过程。受图像扩散模型中可训练 adapter 的启发，我们提出 Adapter-Conditioned DSRL，让 RL actor 同时输出 latent noise 和低维 control code。control code 经过轻量 adapter 转换为中间层 denoising feature，并注入 frozen diffusion policy，从而在不修改 base policy 参数、不直接添加动作残差的情况下，提高动作生成质量。

---

# 16. 最终推荐版本

最推荐的最小可行版本是：

\[
(w_t, c_t, g_t)=\pi_\theta(o_t)
\]

\[
F = A_\psi(c_t)
\]

\[
h_{mid}
\leftarrow
h_{mid}
+
\sigma(g_t)
\cdot
\text{ZeroLinear}(F)
\]

\[
a_t =
\pi_\phi^{\text{frozen}}
(o_t, w_t; F)
\]

其中：

- actor 输出 noise、control code 和 gate；
- adapter 是一个小 MLP；
- 只注入 action denoiser 的中间层；
- frozen diffusion policy 不更新；
- 初始时 adapter 输出为 0；
- 用 RL 训练 actor 和 adapter。

最终可以概括为：

> 以 DSRL 为基础，不走 residual action，而是参考图像生成中可训练 adapter 的思想，让 actor 输出 noise + control code，通过 adapter feature 调控 frozen diffusion policy 的中间去噪过程，从而提高成功率。