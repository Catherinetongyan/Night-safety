# Bristol Safe Walking Router — 开发记录
*记录时间：2026 年 3 月 24 日*

---

## 1. 本次会话做了什么

### 环境配置
- 检测到系统中安装了 Anaconda（路径：`C:\Users\helen\anaconda3`）
- 将 conda 初始化写入 `~/.bashrc`，使 bash 终端中永久可用
- 在 Claude Code 的 `~/.claude/settings.json` 中添加 `SessionStart` hook，每次启动时自动激活 conda

### 依赖安装（环境：`phaseC`）
在已有 `matplotlib / numpy / pandas` 的基础上，新增安装：

| 库 | 版本 | 用途 |
|----|------|------|
| osmnx | 2.0.7 | 下载/操作 OSM 路网 |
| geopandas | 1.1.3 | 地理数据处理 |
| shapely | 2.1.2 | 几何运算 |
| networkx | 3.4.2 | 图结构与寻路算法 |
| pyproj | 3.7.1 | 坐标系转换 |
| scikit-learn | 1.7.1 | 核密度估计（KDE） |
| pyyaml | 6.0.3 | 读取 YAML 配置文件 |

### 新建文件
- `config.yaml` — 参数配置文件
- `safe_route.py` — 新主程序（整合犯罪数据的路径规划）

### 检查 pkl 图的字段
发现 `bristol_safety_graph.pkl` 中每条边只有 `safety_weight` 字段（路灯/CCTV 分已合并计算），**不含** `light_score` / `cctv_score` 单独字段。`safe_route.py` 已通过 `.get()` 默认值优雅处理，不影响运行。

---

## 2. 项目整体架构

### 2.1 数据流

```
CSV（CCTV / 路灯）
        +
OSM 路网（Internet）
        ↓
  Mapping_software.py          ← 一次性预处理，生成带权重的图
        ↓
bristol_safety_graph.pkl       ← 存储路网 + safety_weight
cctv.gpkg / lights.gpkg        ← 存储点位地理数据
        ↓
  safe_route.py                ← 主程序（加载图 + 犯罪 KDE + 寻路 + 可视化）
  config.yaml                  ← 控制所有路径和参数
        ↓
  并排路线可视化图
```

### 2.2 安全权重公式

```
safety_weight = length
              × (1 + ALPHA × crime_score)    犯罪越多 → 权重越大 → A* 越不走
              × (1 - BETA  × light_score)    路灯越多 → 权重折扣 → A* 更愿走
              × (1 - GAMMA × cctv_score)     监控越多 → 权重折扣 → A* 更愿走
```

> 当前 pkl 中无 `light_score` / `cctv_score` 字段，beta/gamma 静默无效，公式退化为：
> `safety_weight = length × (1 + ALPHA × crime_score)`

### 2.3 犯罪数据处理流程

```
bristol_2023-01_to_2025-12.csv（20 万条）
        ↓
按 crime_type 映射严重度权重（Violence×3.0 ... Shoplifting×0.2）
        ↓
可选：过滤低严重度（min_severity，减少数据量提升速度）
        ↓
加权 KDE：将离散犯罪点"扩散"为连续危险密度面
  - 用重复行近似权重（sklearn KDE 不原生支持权重）
  - metric="haversine"，bandwidth≈150m
        ↓
对图中每个节点采样密度值，MinMaxScaler 归一化到 [0,1]
        ↓
路段犯罪分 = (u节点分 + v节点分) / 2
        ↓
代入 safety_weight 公式
```

---

## 3. 新建文件详解

### 3.1 `config.yaml`

所有可调参数集中在此，**不需要改 Python 代码**。

```yaml
paths:          # 数据文件路径（支持相对路径和绝对路径）
route:          # 起终点坐标和步行速度
kde:            # KDE 带宽、核函数、严重度过滤阈值
weights:        # alpha / beta / gamma 三个系数
severity:       # 各犯罪类型的严重度权重
```

**关键参数说明：**

| 参数 | 默认值 | 效果 |
|------|--------|------|
| `kde.bandwidth` | 0.0015（≈150m） | 犯罪影响扩散范围，调大则路线更保守 |
| `kde.min_severity` | 0.0（全部数据） | 设为 1.5 可过滤至约 9.5 万条，大幅提速 |
| `weights.alpha` | 2.0 | 犯罪惩罚强度，调大则最安全路线绕行更远 |
| `weights.beta` | 0.3 | 路灯奖励（当前 pkl 无效，留待将来） |
| `weights.gamma` | 0.2 | CCTV 奖励（当前 pkl 无效，留待将来） |

### 3.2 `safe_route.py`

程序分为 6 个函数模块：

| 函数 | 职责 |
|------|------|
| `load_config` | 读取 YAML 配置 |
| `resolve_path` | 处理相对/绝对路径 |
| `build_crime_scores` | 读 CSV → 加权 KDE → 节点评分字典 |
| `update_safety_weights` | 将犯罪分写入图的每条边 |
| `find_routes` | 调用 nx 计算最快/最安全路线 |
| `plot_routes` | matplotlib 并排可视化 + 打印摘要 |

---

## 4. 使用方法

### 4.1 运行

```bash
# 激活环境
conda activate phaseC

# 进入项目目录
cd "C:\Users\helen\OneDrive\大三\MDM\phaseC\Night-safety"

# 运行（使用默认 config.yaml）
python safe_route.py

# 或指定其他配置文件
python safe_route.py --config my_other_config.yaml
```

### 4.2 修改起终点

编辑 `config.yaml`：

```yaml
route:
  start:
    name: "我的出发地"
    lat:  51.xxxx
    lon:  -2.xxxx
  end:
    name: "我的目的地"
    lat:  51.xxxx
    lon:  -2.xxxx
```

### 4.3 性能优化（数据量大时较慢）

```yaml
kde:
  min_severity: 1.5   # 只保留高严重度犯罪，约减少至 9.5 万条
```

### 4.4 调整安全偏好

```yaml
weights:
  alpha: 3.0   # 调大：最安全路线更激进地绕开犯罪区
  alpha: 1.0   # 调小：最安全路线更接近最快路线
```

---

## 5. 已知问题 / 待办

- [ ] `bristol_safety_graph.pkl` 中无 `light_score` / `cctv_score` 字段，beta/gamma 当前无效
  - 修复方式：重新运行 `Mapping_software.py`，修改使其在保存时同时保留这两个字段
- [ ] `Mapping_software.py` 和 `Route_finding_software.py` 中路径硬编码为 `C:\Users\Oliver\Documents\...`，若需运行需手动修改
- [ ] README.md 目前几乎为空

## 6. 可扩展方向（来自 briefing 文档）

- 加入时间维度：夜间/白天分别计算 KDE，根据出行时间切换权重
- 加入 UI 界面：让用户在地图上点击起终点
- 将加权后的图保存为新 pkl，避免每次运行重新计算 KDE
- 把 ALPHA/BETA/GAMMA 做成命令行参数或滑块
