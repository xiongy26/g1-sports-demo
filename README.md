# Unitree G1 · MuJoCo WASM 篮球 / 乒乓球演示

在浏览器中运行 [MuJoCo 官方 WebAssembly 绑定](https://github.com/google-deepmind/mujoco/tree/main/wasm)（npm 包 `@mujoco/mujoco`），
驱动 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 的
**Unitree G1（29 自由度，位置执行器）** 官方模型，演示两种运动模式：

| 模式 | 内容 |
|------|------|
| 🏀 篮球 | G1 原地运球（接球—下压—反弹循环，物理碰撞），周期性自动投篮；出手弹道由当前手位实时解算，飞行段有温和引导保证过筐，篮筐/篮板为真实碰撞体，进球会计分 |
| 🏓 乒乓球 | G1 持拍站在球台一端与对面"发球机"连续对打；乒乓球按分段抛物线脚本运动，球拍（mocap）实时跟随右手腕，正手挥拍在触球前 0.34 s 自动触发 |

两种模式共用一个 MJCF 场景（篮球架 + 球台 + 发球机），通过 UI 按钮或快捷键切换。
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

- **🏀 篮球 / 🏓 乒乓球**：切换模式（快捷键 `1` / `2`）
- **投篮 / 重新发球**：立即触发动作（快捷键 `空格`）
- **暂停** / 速度选择（0.25×–1.5×）
- 鼠标拖拽旋转视角，滚轮缩放，右键平移

## 实现说明

- `src/main.js` — WASM 加载、`MjVFS` 资源注入（35 个 STL）、DOM 方式合并 `<include>`、主循环与 UI
- `src/visualizer.js` — three.js 渲染器：sphere/capsule/cylinder/box/mesh 各类型几何、球场木地板 Canvas 纹理、篮球筋线
- `src/basketball.js` — 运球/投篮状态机（carry → free → windup → flight → recover）
- `src/pingpong.js` — 对打轨迹脚本与挥拍动作、球拍 mocap 跟随
- `src/util.js` — 关节索引表、四元数工具、直立/锚点平衡辅助
- `tools/gen_scene.py` — 生成 `model/scene_gym.xml`（篮筐圆环、篮网、球台等）
- `tools/sim_check.mjs` — **无头自检**：在 node 中用同一套 WASM 与控制器代码跑两种模式并输出指标

### 演示性外挂（诚实声明）

为了让站立展示稳定可靠，代码中加入了轻微的"平衡辅助"（对骨盆施加小的水平回位力与
直立力矩，见 `util.js` 的 `applyBalanceAssist`），并非纯控制学习；乒乓球为脚本轨迹，
篮球飞行段有弹道引导。其余接触（运球弹跳、篮筐碰撞、地板反弹）均为物理仿真。

### 已验证指标（`node tools/sim_check.mjs`）

- 篮球 30 s：3 次投篮全部命中，运球周期 ~0.9 s，骨盆高度始终 ≥ 0.78 m（站立稳定）
- 乒乓球：挥拍正好扫过击球点（拍球最小距离 0.021 m），回球过网高度 0.99 m（净高 0.91 m），
  机器人 6 s 内完成 2 个回合且不跌倒

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
