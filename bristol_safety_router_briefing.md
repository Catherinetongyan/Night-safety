# Bristol 安全步行路径规划系统 — 项目说明文档

> 本文档供 Claude Code 阅读，包含项目背景、当前代码状态、待完成任务及实现方案。

---

## 1. 项目概述

这是一个基于布里斯托真实地图数据的**步行路径规划工具**，能够在"最快路线"和"最安全路线"之间让用户选择。

系统已完成基础框架，当前阶段的核心任务是：**将 2023–2025 年布里斯托犯罪数据整合进 A\* 算法的安全权重中**。

---

## 2. 现有代码结构

### 2.1 主脚本：`Route_finding_software.py`

**功能：**
- 从本地 `.pkl` 文件加载预构建的路网图 `G`（osmnx + networkx 格式）
- 加载 CCTV 点位（`cctv.gpkg`）和路灯点位（`lights.gpkg`）
- 用 `nx.shortest_path` 计算最快路线（`weight="length"`）
- 用 `nx.astar_path` 计算最安全路线（`weight="safety_weight"`）
- 并排绘制两条路线的地图对比图

**当前文件路径（本地 Windows 机器）：**
```
C:\Users\Oliver\Documents\bristol_safety_graph.pkl   ← 路网图
C:\Users\Oliver\Documents\cctv.gpkg                  ← CCTV 数据
C:\Users\Oliver\Documents\lights.gpkg                ← 路灯数据
```

**示例起终点：**
- 起点：Wills Memorial Building（51.4576, -2.6053）
- 终点：Thekla（51.4493, -2.5980）

### 2.2 现有安全权重

图中每条边已有 `safety_weight` 字段，目前由**路段长度 + 路灯覆盖 + CCTV 覆盖**计算得出。

具体构成方式尚未在脚本中体现（已预构建在 `.pkl` 图中），但字段已存在于每条边的 `data` 中：

```python
data["length"]       # 路段长度（米）
data["light_score"]  # 路灯覆盖得分，范围 [0, 1]
data["cctv_score"]   # CCTV 覆盖得分，范围 [0, 1]
data["safety_weight"] # 综合安全权重（当前版本，待更新）
```

---

## 3. 新增数据：犯罪记录 CSV

### 3.1 文件信息

| 属性 | 内容 |
|------|------|
| 文件名 | `bristol_2023-01_to_2025-12.csv` |
| 记录数 | 201,040 条 |
| 时间范围 | 2023 年 1 月 — 2025 年 12 月 |
| 空值 | 无（lon/lat/crime_type 均完整） |

### 3.2 字段说明

```
month        - 月份，格式如 "Jan-23"
lon          - 经度（WGS84）
lat          - 纬度（WGS84）
location     - 文字描述，如 "On or near Thornfield Road"
lsoa_name    - 统计区域名称，如 "Bristol 001A"
crime_type   - 犯罪类型（见下表）
```

### 3.3 犯罪类型分布

| 犯罪类型 | 记录数 |
|---------|--------|
| Violence and sexual offences | 68,574 |
| Anti-social behaviour | 28,751 |
| Public order | 20,195 |
| Shoplifting | 17,672 |
| Other theft | 14,655 |
| Criminal damage and arson | 14,583 |
| Vehicle crime | 13,509 |
| Burglary | 7,498 |
| Robbery | 3,618 |
| Drugs | 3,449 |
| Bicycle theft | 3,262 |
| Other crime | 2,865 |
| Possession of weapons | 1,233 |
| Theft from the person | 1,176 |

---

## 4. 本次任务：将犯罪数据整合进安全权重

### 4.1 总体思路

犯罪数据是离散的点位，路网是线段图。需要将点位"扩散"到路段上，方法是：

1. 对犯罪点做**加权核密度估计（KDE）**，生成连续的危险密度面
2. 对图中每个**节点**采样密度值
3. 将路段两端节点密度的均值作为该**路段**的犯罪评分
4. 将犯罪评分乘入现有的 `safety_weight` 公式

### 4.2 犯罪严重度权重

不同犯罪类型对步行者的威胁程度不同，需按以下权重加权：

```python
SEVERITY = {
    "Violence and sexual offences": 3.0,
    "Robbery":                       2.5,
    "Possession of weapons":         2.0,
    "Public order":                  1.5,
    "Criminal damage and arson":     1.2,
    "Drugs":                         1.0,
    "Theft from the person":         1.0,
    "Anti-social behaviour":         0.8,
    "Other theft":                   0.6,
    "Burglary":                       0.5,
    "Other crime":                   0.5,
    "Vehicle crime":                 0.4,
    "Bicycle theft":                 0.3,
    "Shoplifting":                   0.2,
}
```

### 4.3 完整实现代码

在现有 `Route_finding_software.py` 中，在加载图之后、运行 A\* 之前，插入以下代码块：

```python
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────
# STEP A：加载犯罪数据并赋予严重度权重
# ─────────────────────────────────────────
SEVERITY = {
    "Violence and sexual offences": 3.0,
    "Robbery":                       2.5,
    "Possession of weapons":         2.0,
    "Public order":                  1.5,
    "Criminal damage and arson":     1.2,
    "Drugs":                         1.0,
    "Theft from the person":         1.0,
    "Anti-social behaviour":         0.8,
    "Other theft":                   0.6,
    "Burglary":                      0.5,
    "Other crime":                   0.5,
    "Vehicle crime":                 0.4,
    "Bicycle theft":                 0.3,
    "Shoplifting":                   0.2,
}

df = pd.read_csv(r"C:\Users\Oliver\Documents\bristol_2023-01_to_2025-12.csv")
df["severity"] = df["crime_type"].map(SEVERITY).fillna(0.5)

coords  = df[["lat", "lon"]].values   # shape (N, 2)
weights = df["severity"].values

# ─────────────────────────────────────────
# STEP B：加权 KDE
#   bandwidth = 0.0015 ≈ 150 米（haversine 单位为弧度）
#   用重复行近似加权（scikit-learn KDE 不原生支持权重）
# ─────────────────────────────────────────
BW = 0.0015   # 可调：0.001 ≈ 100m，0.003 ≈ 300m

repeat_counts   = np.round(weights * 10).astype(int).clip(1, 30)
coords_weighted = np.repeat(coords, repeat_counts, axis=0)

kde = KernelDensity(bandwidth=BW, metric="haversine", kernel="gaussian")
kde.fit(np.radians(coords_weighted))

# ─────────────────────────────────────────
# STEP C：对图中每个节点计算 KDE 密度值
# ─────────────────────────────────────────
node_ids    = list(G.nodes)
node_coords = np.array([[G.nodes[n]["y"], G.nodes[n]["x"]] for n in node_ids])  # lat, lon

log_density  = kde.score_samples(np.radians(node_coords))
density      = np.exp(log_density)

scaler       = MinMaxScaler()
crime_scores = scaler.fit_transform(density.reshape(-1, 1)).flatten()
node_crime   = dict(zip(node_ids, crime_scores))   # 节点 → 归一化犯罪评分 [0,1]

# ─────────────────────────────────────────
# STEP D：更新每条路段的 safety_weight
# ─────────────────────────────────────────
ALPHA = 2.0   # 犯罪惩罚强度（越大，越绕开高犯罪区）
BETA  = 0.3   # 路灯奖励系数
GAMMA = 0.2   # CCTV 奖励系数

for u, v, data in G.edges(data=True):
    edge_crime  = (node_crime.get(u, 0) + node_crime.get(v, 0)) / 2.0
    light_score = data.get("light_score", 0.0)
    cctv_score  = data.get("cctv_score",  0.0)
    length      = data.get("length", 1.0)

    data["safety_weight"] = (
        length
        * (1 + ALPHA * edge_crime)
        * (1 - BETA  * light_score)
        * (1 - GAMMA * cctv_score)
    )
```

**公式含义：**
```
safety_weight = 路段长度
              × (犯罪越多 → 权重越大 → A* 越不愿走这条路)
              × (路灯越亮 → 权重折扣 → A* 更愿走有灯的路)
              × (监控越多 → 权重折扣 → A* 更愿走有监控的路)
```

---

## 5. 可调参数说明

| 参数 | 默认值 | 作用 | 调大效果 |
|------|--------|------|----------|
| `BW` | 0.0015（≈150m）| KDE 扩散半径 | 犯罪影响范围更广，路线更保守 |
| `ALPHA` | 2.0 | 犯罪惩罚强度 | 更大幅度绕开高犯罪区，路线更长 |
| `BETA` | 0.3 | 路灯奖励系数 | 更倾向选有路灯的街道 |
| `GAMMA` | 0.2 | CCTV 奖励系数 | 更倾向选有监控的街道 |

**调参建议：** 先用默认值运行 Wills Memorial → Thekla 这条路线，对比最快和最安全路线的差异是否符合直觉，再微调 `ALPHA` 和 `BW`。

---

## 6. 依赖库

以下库需要安装（如尚未安装）：

```bash
pip install scikit-learn pandas numpy osmnx geopandas networkx matplotlib
```

现有代码已用到：`osmnx`, `geopandas`, `networkx`, `matplotlib`
新增需要：`scikit-learn`（KDE）, `pandas`, `numpy`

---

## 7. 性能说明

- KDE 拟合（20万点）在普通笔记本约需 **15–30 秒**
- 节点评分（图中约 5–10 万节点）约需 **10–20 秒**
- **优化方案：** 如觉得太慢，可以只保留夜间犯罪记录（对步行者威胁更大，且数据量约减少 60%）：

```python
# 只保留傍晚/夜间数据（月份+时间过滤，若有时间字段）
# 或只保留高严重度犯罪（severity >= 1.5）
df = df[df["severity"] >= 1.5]   # 约减少至 9.5 万条
```

---

## 8. 后续可扩展方向（低优先级）

- [ ] 加入时间维度：夜间/白天分别计算 KDE，根据出行时间切换权重
- [ ] 加入 UI 界面：让用户在地图上点击起终点
- [ ] 将图保存为新的 `.pkl` 文件，避免每次运行都重新计算 KDE
- [ ] 把 `ALPHA/BETA/GAMMA` 做成命令行参数或滑块，方便调参

---

*文档生成时间：2025 年 3 月 | 项目：Bristol Safe Walking Router*
