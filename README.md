# Unitree G1 · MuJoCo WASM 篮球 / 乒乓球演示

在浏览器中运行 [MuJoCo 官方 WebAssembly 绑定](https://github.com/google-deepmind/mujoco/tree/main/wasm)（npm 包 `@mujoco/mujoco`），
驱动 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 的
**Unitree G1（29 自由度，位置执行器）** 官方模型（场景内共两台，可双机对打），演示三种运动模式：

| 模式 | 内容 |
|------|------|
| 🏀 篮球 | G1 原地运球（接球—下压—反弹循环，物理碰撞），周期性自动投篮；球以"掌系托球"贴在掌心随手掌转动，出手时旋腕托球、弹道由当前手位实时解算，飞行段有温和引导保证过筐，篮筐/篮板为真实碰撞体，进球会计分 |
| 🏓 乒乓球 | G1 持拍站在球台一端与对面"发球机"连续对打；球拍按正手握拍贴合手掌（拍面即掌面延伸），乒乓球按分段抛物线脚本运动，挥拍在触球前 0.34 s 自动触发，击球瞬间球吸附在拍面胶面上被击出，出球方向与拍面朝向一致 |
| 🏓🏓 双机对打 | 两台 G1 隔台正手斜线对拉：二号机站在球台另一端（与一号机关于台心点对称，因此两人动作互为镜像、都用正手），发球机隐藏；球从二号机拍上发出，双方各自完成引拍—击球—随挥，击球均贴合拍面甜区 |

两种模式共用一个 MJCF 场景（篮球架 + 球台 + 发球机），通过 UI 按钮或快捷键切换。
二号机在非对打模式下站在场边待机（平衡辅助保持直立）。
物理以 500 Hz（timestep=0.002，implicitfast 积分器）在 WASM 中步进，three.js 直接读取
`geom_xpos / geom_xmat` 实时视图渲染（mesh 网格从 `mesh_vert/face` 构建）。

## 运行

```bash
cd g1-sports-demo
python3 serve.py            # 或: python3 -m http.server 8000
# 浏览器打开 http://localhost:8000/
```

> 需要通过 HTTP 访问（ES 模块 + WASM），不能直接双击 index.html。
> 首次加载需下载约 20 MB 的机器人网格文件。

## 操作

- **🏀 篮球 / 🏓 乒乓球 / 🏓🏓 双机对打**：切换模式（快捷键 `1` / `2` / `3`）
- **投篮 / 重新发球**：立即触发动作（快捷键 `空格`）
- **暂停** / 速度选择（0.25×–1.5×）
- 鼠标拖拽旋转视角，滚轮缩放，右键平移

## 实现说明

- `src/main.js` — WASM 加载、`MjVFS` 资源注入（35 个 STL）、主循环与 UI、二号机待机
- `src/scene_merge.js` — 模型组装：合并 g1.xml 并复制出二号机（`r2_` 前缀重命名，浏览器/node 共用）
- `src/visualizer.js` — three.js 渲染器：sphere/capsule/cylinder/box/mesh 各类型几何、球场木地板 Canvas 纹理、篮球筋线
- `src/basketball.js` — 运球/投篮状态机（carry → free → windup → flight → recover）
- `src/pingpong.js` — `RallyBot` 单机控制单元（关节平滑 + 挥拍 + 球拍跟随 + 平衡）、`PingpongController` 单人对发球机、`DuelController` 双机对打
- `src/util.js` — 关节索引表、四元数工具、直立/锚点平衡辅助（支持按机器人偏移）、待机姿态
- `tools/gen_scene.py` — 生成 `model/scene_gym.xml`（篮筐圆环、篮网、球台、发球机炮口朝向、第二支球拍、含双机器人关键帧）
- `tools/sim_check.mjs` — **无头自检**：在 node 中用同一套 WASM 与控制器代码跑三种模式并输出指标
- `tools/probe_contact.mjs` — **接触几何探针**：量化手-拍-球贴合度、标定挥拍/出手腕角（调参用）

### 演示性外挂（诚实声明）

为了让站立展示稳定可靠，代码中加入了轻微的"平衡辅助"（对骨盆施加小的水平回位力与
直立力矩，见 `util.js` 的 `applyBalanceAssist`），并非纯控制学习；乒乓球为脚本轨迹，
篮球飞行段有弹道引导。其余接触（运球弹跳、篮筐碰撞、地板反弹）均为物理仿真。

为让"手—拍—球"的接触经得起细看，另有三处演示级处理：

- **乒乓球击球吸附**：球到达击球点前 120 ms 平滑吸附到拍面胶面（`pingpong.js` 的
  `paddleFacePoint`），保证球贴着拍面正中被击出，不会穿模或悬空；出球弧线从贴合点起算
- **篮球掌系托球**：持球/举球段球心按掌面法向放置（`basketball.js` 的 `SEAT_LOCAL`），
  随手掌转动保持贴合；接球要求球落在掌心 0.28 m 内，避免远处瞬移
- **腕角标定**：乒乓球击球时刻与篮球出手时刻的手腕角均由无头仿真采样标定
  （`tools/probe_contact.mjs`），使拍面/掌面对准球的实际去向

双机对打为同一套脚本的两个镜像实例（按台心点对称），因此两人的击球质量一致。

### 已验证指标（`node tools/sim_check.mjs`）

- 篮球 30 s：3 次投篮全部命中，运球周期 ~0.9 s，骨盆高度始终 ≥ 0.78 m（站立稳定）；
  举球段球-掌滑移 ≤ 0.006 m，出手瞬间掌面法向与出球方向夹角 ~36°
- 乒乓球：击球瞬间球心距拍心 0.028 m = 拍半厚 + 球半径（恰好贴在胶面甜区，径向 0.003 m），
  拍面法向与出球方向夹角 ~15°，回球过网高度 1.01 m（净高 0.91 m），
  机器人 6 s 内完成 2 个回合且不跌倒
- 双机对打 10 s：9 次击球（两个方向），两台机器人击球均贴合拍面（0.028 / 0.022 m），
  双向过网高度 0.96 / 0.98 m，两台机器人骨盆高度均 ~0.77 m（都稳定）

## 目录

```
g1-sports-demo/
├── index.html            # 页面与 UI
├── serve.py              # 本地服务器（正确 .wasm MIME）
├── model/
│   ├── scene_gym.xml     # 场景（由 tools/gen_scene.py 生成）
│   ├── g1.xml            # Unitree G1 官方 MJCF（Menagerie）
│   └── assets/*.STL      # G1 网格
├── src/                  # 应用代码
├── vendor/
│   ├── mujoco/           # @mujoco/mujoco 3.13.0 官方 WASM 绑定
│   └── three/            # three.js r180
└── tools/                # 场景生成器与无头自检
```

## 许可

- 模型与代码分别遵循 MuJoCo Menagerie（BSD-3）与各自上游许可
- 本项目代码仅用于演示
