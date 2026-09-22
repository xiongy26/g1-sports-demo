// 主程序：加载 MuJoCo WASM 与 G1 模型 → three.js 渲染 → 双模式运动控制器
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '../vendor/mujoco/mujoco.js';
import { MujocoVisualizer } from './visualizer.js';
import { BasketballController } from './basketball.js';
import { PingpongController, DuelController } from './pingpong.js';
import { buildJointMap, bodyId, jointId, keyId, setBasePose, applyBalanceAssist, REST_POSE } from './util.js';
import { buildModelXml } from './scene_merge.js';

const TIMESTEP = 0.002;

const JOINT_NAMES = [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
];

const $ = (id) => document.getElementById(id);

// ---------------- 加载 ----------------
function showLoad(msg, frac) {
  $('loading-text').textContent = msg;
  if (frac !== undefined) $('loading-bar').style.width = `${Math.round(frac * 100)}%`;
}

async function fetchAssets() {
  const [sceneXml, g1Xml] = await Promise.all([
    fetch('model/scene_gym.xml').then((r) => r.text()),
    fetch('model/g1.xml').then((r) => r.text()),
  ]);
  const meshNames = [...g1Xml.matchAll(/<mesh(?:\s+name="[^"]*")?\s+file="([^"]+)"/g)].map((m) => m[1]);
  let done = 0;
  const bufs = await Promise.all(meshNames.map(async (name) => {
    const buf = await (await fetch(`model/assets/${name}`)).arrayBuffer();
    done++;
    showLoad(`下载机器人网格 ${done}/${meshNames.length}`, done / meshNames.length);
    return [name, new Uint8Array(buf)];
  }));
  return { sceneXml, g1Xml, bufs };
}

// ---------------- 场景 / 灯光 ----------------
function buildThreeScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x171c26);
  scene.fog = new THREE.Fog(0x171c26, 16, 34);

  const hemi = new THREE.HemisphereLight(0x9fb4cc, 0x44392c, 0.75);
  scene.add(hemi);

  const sun = new THREE.DirectionalLight(0xfff2df, 2.2);
  sun.position.set(4, -6, 9);
  sun.target.position.set(-1.2, 0, 0.6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  const cam = sun.shadow.camera;
  cam.left = -7; cam.right = 7; cam.top = 7; cam.bottom = -7; cam.near = 1; cam.far = 30;
  sun.shadow.bias = -0.0004;
  scene.add(sun, sun.target);

  const fill = new THREE.DirectionalLight(0xbcd0ff, 0.5);
  fill.position.set(-5, 5, 6);
  scene.add(fill);

  return scene;
}

// ---------------- 主程序 ----------------
async function main() {
  showLoad('初始化 MuJoCo WASM（约 10 MB）…', 0.03);
  const mujoco = await loadMujoco();

  const { sceneXml, g1Xml, bufs } = await fetchAssets();
  showLoad('编译 MJCF 模型…', 1);
  const mergedXml = buildModelXml(sceneXml, g1Xml);

  const vfs = new mujoco.MjVFS();
  for (const [name, buf] of bufs) vfs.addBuffer('assets/' + name, buf);
  showLoad('编译:VFS 就绪', 1);

  let model;
  try {
    showLoad('编译:解析 MJCF…', 1);
    model = mujoco.MjModel.from_xml_string(mergedXml, vfs);
    showLoad('编译:完成', 1);
  } catch (e) {
    throw new Error('模型编译失败: ' + (e && e.message ? e.message : e));
  }
  const data = new mujoco.MjData(model);
  if (model.nq !== 79 || model.nu !== 58) throw new Error(`意外的模型规模 nq=${model.nq} nu=${model.nu}`);

  // 名字 -> 索引
  const jmap = buildJointMap(mujoco, model, JOINT_NAMES);
  const env = {
    mujoco, model, data, jmap,
    pelvisBid: bodyId(mujoco, model, 'pelvis'),
    palmBid: bodyId(mujoco, model, 'right_wrist_yaw_link'),
    wristBid: bodyId(mujoco, model, 'right_wrist_yaw_link'),
    ballQadr: model.jnt_qposadr[jointId(mujoco, model, 'basketball_joint')],
    ballDadr: model.jnt_dofadr[jointId(mujoco, model, 'basketball_joint')],
    ballGid: (() => {
      const gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, 'basketball_geom');
      if (gid < 0) throw new Error('找不到 basketball_geom');
      return gid;
    })(),
    ttBallMid: model.body_mocapid[bodyId(mujoco, model, 'tt_ball')],
    paddleMid: model.body_mocapid[bodyId(mujoco, model, 'paddle')],
    paddleMid2: model.body_mocapid[bodyId(mujoco, model, 'paddle_r2')],
    machineMid: model.body_mocapid[bodyId(mujoco, model, 'tt_machine')],
    // 二号机（r2_ 前缀，双机对打用）
    r2WristBid: bodyId(mujoco, model, 'r2_right_wrist_yaw_link'),
    r2PelvisBid: bodyId(mujoco, model, 'r2_pelvis'),
    r2Qadr: model.jnt_qposadr[jointId(mujoco, model, 'r2_floating_base_joint')],
    r2Dof: model.jnt_dofadr[jointId(mujoco, model, 'r2_floating_base_joint')],
    onScore: () => { score++; $('score').textContent = score; flash('投进了！🏀'); },
    onShot: () => {},
    onRally: (n) => { $('score').textContent = n; },
  };

  // three.js
  const scene = buildThreeScene();
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  $('app').appendChild(renderer.domElement);
  const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.05, 120);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI / 2 - 0.02;
  controls.minDistance = 0.8;
  controls.maxDistance = 18;

  const viz = new MujocoVisualizer(mujoco, model, scene);
  mjForward();

  const controllers = {
    basketball: new BasketballController(env),
    pingpong: new PingpongController(env),
    duel: new DuelController(env),
  };
  // 二号机待机位（非对打模式：站在场地边观看）
  const R2_REST = { x: -5.7, y: -1.9, yaw: 0.6 };
  let mode = 'basketball';
  let paused = false;
  let speed = 1;
  let score = 0;

  function mjForward() { mujoco.mj_forward(model, data); }

  function setMode(m) {
    mode = m;
    const key = keyId(mujoco, model, 'home');
    mujoco.mj_resetDataKeyframe(model, data, key);
    if (m === 'basketball') setBasePose(data, 0, 0, 0.783675, 0);
    else setBasePose(data, -1.15, 0, 0.783675, Math.PI); // 乒乓球/对打：一号机站球台右端
    // 二号机：对打站球台另一端；其它模式在场边待机
    if (m === 'duel') setBasePose(data, -4.65, 0, 0.783675, 0, env.r2Qadr, env.r2Dof);
    else setBasePose(data, R2_REST.x, R2_REST.y, 0.783675, R2_REST.yaw, env.r2Qadr, env.r2Dof);
    // 隐藏不相关的道具：TT 球和球拍移到地下；篮球放回手中/角落
    const hidden = [0, 0, -5];
    const parkTT = m !== 'pingpong' && m !== 'duel';
    if (parkTT) {
      data.mocap_pos.set(hidden, 3 * env.ttBallMid);
      data.mocap_pos.set(hidden, 3 * env.paddleMid);
      data.mocap_pos.set(hidden, 3 * env.paddleMid2);
      data.qpos[env.ballQadr] = 3.4; data.qpos[env.ballQadr + 1] = 1.8; data.qpos[env.ballQadr + 2] = 0.123;
      data.qvel[env.ballDadr] = 0; data.qvel[env.ballDadr + 1] = 0; data.qvel[env.ballDadr + 2] = 0;
      model.geom_contype[env.ballGid] = 1; model.geom_conaffinity[env.ballGid] = 1;
    }
    // 发球机：单人乒乓球就位，双机对打藏到地下（把位置让给二号机）
    if (m === 'pingpong') data.mocap_pos.set([-4.72, 0.25, 0], 3 * env.machineMid);
    else if (m === 'duel') data.mocap_pos.set(hidden, 3 * env.machineMid);
    controllers[m].reset();
    mjForward();
    const c = controllers[m].cameraPreset();
    camera.position.set(...c.pos);
    controls.target.set(...c.target);
    controls.update();
    score = 0; $('score').textContent = '0';
    $('mode-basketball').classList.toggle('active', m === 'basketball');
    $('mode-pingpong').classList.toggle('active', m === 'pingpong');
    $('mode-duel').classList.toggle('active', m === 'duel');
    $('action-btn').textContent = m === 'basketball' ? '投篮 🏀' : '重新发球 🏓';
    $('hint').textContent = m === 'basketball'
      ? 'G1 原地运球，周期性起跳投篮（弹道由 WASM 物理仿真）'
      : m === 'duel'
        ? '两台 G1 隔台正手斜线对拉，击球贴合拍面（快捷键 1/2/3 切换）'
        : 'G1 与对面发球机连续对打，球拍实时跟随手腕（旁边是等待上场的二号机）';
  }

  // UI
  $('mode-basketball').onclick = () => setMode('basketball');
  $('mode-pingpong').onclick = () => setMode('pingpong');
  $('mode-duel').onclick = () => setMode('duel');
  $('action-btn').onclick = () => controllers[mode].action();
  $('pause-btn').onclick = () => { paused = !paused; $('pause-btn').textContent = paused ? '继续' : '暂停'; };
  $('speed-sel').onchange = (e) => { speed = parseFloat(e.target.value); };
  window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); controllers[mode].action(); }
    if (e.key === '1') setMode('basketball');
    if (e.key === '2') setMode('pingpong');
    if (e.key === '3') setMode('duel');
  });

  function flash(msg) {
    const el = $('toast');
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove('show'), 1600);
  }

  // ---------------- 主循环 ----------------
  let acc = 0, last = -1, fpsT = 0, fpsN = 0;
  // 非对打模式下，二号机在场边待机站立（平衡辅助保持直立）
  function idleR2() {
    for (let i = 0; i < 29; i++) data.ctrl[29 + i] = REST_POSE[i];
    applyBalanceAssist(data, env.r2PelvisBid, R2_REST.x, R2_REST.y, 1.0, env.r2Dof);
  }
  function stepFrame(dt) {
    if (!paused) {
      acc += dt * speed;
      const maxN = Math.ceil(dt / TIMESTEP) + 8;
      let n = 0;
      while (acc >= TIMESTEP && n < maxN) {
        if (mode !== 'duel') idleR2();
        controllers[mode].step(TIMESTEP);
        mujoco.mj_step(model, data);
        acc -= TIMESTEP;
        n++;
      }
      if (n >= maxN) acc = 0;
    }
    viz.update(data);
    controls.update();
    renderer.render(scene, camera);
  }
  function frame(now) {
    try {
      if (last < 0) last = now;
      const dtRaw = Math.max(0, (now - last) / 1000);
      last = now;
      // 部分环境（后台/无头）只以 ~1Hz 派发帧：允许大步长补偿，保持接近实时
      stepFrame(Math.min(0.5, dtRaw));
      fpsT += dtRaw; fpsN++;
      if (fpsT > 0.5) {
        $('fps').textContent = `${Math.round(fpsN / fpsT)} FPS · t=${data.time.toFixed(1)}s`;
        fpsT = 0; fpsN = 0;
      }
    } catch (err) {
      console.error('frame error:', err);
      $('fps').textContent = '错误: ' + (err && err.message ? err.message : err);
    }
  }

  // 主循环：优先 requestAnimationFrame；若环境不再派发 rAF（后台/无头），定时器自动接管
  let lastRaf = -1e9;
  function driverRaf(now) {
    lastRaf = performance.now();
    frame(now);
    requestAnimationFrame(driverRaf);
  }
  requestAnimationFrame(driverRaf);
  setInterval(() => {
    if (performance.now() - lastRaf > 250) frame(performance.now());
  }, 15);

  setMode('basketball');
  $('loading').classList.add('done');
  // 同步驱动（调试/测试用）：__sim.drive(秒) 按帧步进仿真与渲染
  window.__sim = {
    mujoco, model, data, env, controllers, viz, camera, controls, scene, renderer,
    setMode,
    get mode() { return mode; },
    drive(secs = 0.1, dt = 1 / 60, render = true) {
      const n = Math.max(1, Math.round(secs / dt));
      for (let i = 0; i < n; i++) {
        if (!paused) {
          acc += dt * speed;
          const maxN = Math.ceil(dt / TIMESTEP) + 8;
          let k = 0;
          while (acc >= TIMESTEP && k < maxN) {
            if (mode !== 'duel') idleR2();
            controllers[mode].step(TIMESTEP);
            mujoco.mj_step(model, data);
            acc -= TIMESTEP;
            k++;
          }
          if (k >= maxN) acc = 0;
        }
        if (render) viz.update(data);
      }
      if (render) { controls.update(); renderer.render(scene, camera); }
      viz.update(data);
      return data.time;
    },
  };
}

main().catch((err) => {
  console.error(err);
  $('loading-text').textContent = '加载失败：' + (err && err.message ? err.message : err);
  $('loading-text').style.color = '#ff8080';
  document.querySelector('.spinner').style.display = 'none';
});
