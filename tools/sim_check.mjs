// 无头仿真自检：在 node 中加载与浏览器完全相同的模型与控制器，验证两种模式的运动逻辑。
// 用法: node tools/sim_check.mjs
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import loadMujoco from '../vendor/mujoco/mujoco.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

// ---- 与 main.js 相同的 include 合并 ----
function mergeInclude(sceneXml, g1Xml) {
  const inner = g1Xml.replace(/^[\s\S]*?<mujoco[^>]*>/, '').replace(/<\/mujoco>\s*$/, '');
  return sceneXml.replace('<include file="g1.xml"/>', inner);
}

// ---- 与 main.js 相同的关节表 ----
const JOINT_NAMES = [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
];

const TIMESTEP = 0.002;
const { BasketballController } = await import('../src/basketball.js');
const { PingpongController } = await import('../src/pingpong.js');
const { buildJointMap, bodyId, jointId, keyId, setBasePose } = await import('../src/util.js');

const mujoco = await loadMujoco();

const sceneXml = readFileSync(join(ROOT, 'model/scene_gym.xml'), 'utf8');
const g1Xml = readFileSync(join(ROOT, 'model/g1.xml'), 'utf8');
const merged = mergeInclude(sceneXml, g1Xml);

const vfs = new mujoco.MjVFS();
for (const f of readdirSync(join(ROOT, 'model/assets'))) {
  const buf = readFileSync(join(ROOT, 'model/assets', f));
  vfs.addBuffer('assets/' + f, new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength));
}

let model;
try {
  model = mujoco.MjModel.from_xml_string(merged, vfs);
} catch (e) {
  console.error('模型编译失败:', e.message);
  process.exit(1);
}
console.log(`模型编译成功: nq=${model.nq} nv=${model.nv} nu=${model.nu} ngeom=${model.ngeom} nbody=${model.nbody}`);
const data = new mujoco.MjData(model);

const jmap = buildJointMap(mujoco, model, JOINT_NAMES);
const env = {
  mujoco, model, data, jmap,
  pelvisBid: bodyId(mujoco, model, 'pelvis'),
  palmBid: bodyId(mujoco, model, 'right_wrist_yaw_link'),
  wristBid: bodyId(mujoco, model, 'right_wrist_yaw_link'),
  ballQadr: model.jnt_qposadr[jointId(mujoco, model, 'basketball_joint')],
  ballDadr: model.jnt_dofadr[jointId(mujoco, model, 'basketball_joint')],
  ballGid: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, 'basketball_geom'),
  ttBallMid: model.body_mocapid[bodyId(mujoco, model, 'tt_ball')],
  paddleMid: model.body_mocapid[bodyId(mujoco, model, 'paddle')],
  onScore: () => { score++; },
  onShot: () => {},
  onRally: (n) => { rallies = n; },
};

function keyIdOf(name) { return keyId(mujoco, model, name); }
function resetMode(m) {
  mujoco.mj_resetDataKeyframe(model, data, keyIdOf('home'));
  if (m === 'pingpong') setBasePose(data, -1.15, 0, 0.783675, Math.PI);
  else setBasePose(data, 0, 0, 0.783675, 0);
}

// ================= 篮球 =================
console.log('\n--- 篮球模式 22s ---');
let score = 0, rallies = 0;
const bb = new BasketballController(env);resetMode('basketball');
bb.reset();
mujoco.mj_forward(model, data);
const states = new Map();
let minPelZ = 9, maxBallZ = 0, minBallZ = 9;
const T_BB = 30;
for (let i = 0; i < T_BB / TIMESTEP; i++) {
  bb.step(TIMESTEP);
  mujoco.mj_step(model, data);
  states.set(bb.state, (states.get(bb.state) || 0) + TIMESTEP);
  const pz = data.qpos[2];
  if (pz < minPelZ) minPelZ = pz;
  const bz = data.qpos[env.ballQadr + 2];
  if (bz > maxBallZ) maxBallZ = bz;
  if (bz < minBallZ) minBallZ = bz;
}
console.log('状态时长(s):', Object.fromEntries([...states.entries()].map(([k, v]) => [k, +v.toFixed(2)])));
console.log(`进球数: ${score}  髋部最低高度: ${minPelZ.toFixed(3)} m (站立${minPelZ > 0.5 ? '稳定' : '失败'})`);
console.log(`篮球高度范围: ${minBallZ.toFixed(2)} ~ ${maxBallZ.toFixed(2)} m (运球${maxBallZ > 0.9 && minBallZ < 0.3 ? '正常' : '异常'})`);

// ================= 乒乓球 =================
console.log('\n--- 乒乓球模式 6s ---');
score = 0; rallies = 0;
const pp = new PingpongController(env);
resetMode('pingpong');
pp.reset();
mujoco.mj_forward(model, data);
let swingFrames = 0, minD = 9;
let netCrossZ = null, prevBx = null;
const T_TT = 6;
const ballSamples = [];
for (let i = 0; i < T_TT / TIMESTEP; i++) {
  pp.step(TIMESTEP);
  mujoco.mj_step(model, data);
  if (pp.swinging) swingFrames++;
  const mp = data.mocap_pos;
  const d = Math.hypot(mp[3 * env.paddleMid] - mp[3 * env.ttBallMid], mp[3 * env.paddleMid + 1] - mp[3 * env.ttBallMid + 1], mp[3 * env.paddleMid + 2] - mp[3 * env.ttBallMid + 2]);
  if (d < minD) minD = d;
  const bx = mp[3 * env.ttBallMid], bz = mp[3 * env.ttBallMid + 2];
  // 机器人击球方向 (C->D) 过网 (x=-2.9) 时的高度
  if (prevBx !== null && prevBx > -2.9 && bx <= -2.9 && pp.leg === 2 && netCrossZ === null) netCrossZ = bz;
  prevBx = bx;
  if (i % 100 === 0) ballSamples.push(bz);
}
console.log(`挥拍帧数: ${swingFrames} (~${(swingFrames * TIMESTEP).toFixed(2)}s)`);
console.log(`击球瞬间拍球最小距离: ${minD.toFixed(3)} m ${minD < 0.3 ? '(挥拍覆盖击球点)' : '(未覆盖, 需调整)'}`);
console.log(`回球过网高度: ${netCrossZ === null ? '未捕捉' : netCrossZ.toFixed(2) + ' m (净高0.9125) ' + (netCrossZ > 0.92 ? '✓' : '✗')}`);
console.log(`机器人高度: ${data.qpos[2].toFixed(3)} m ${data.qpos[2] > 0.5 ? '(稳定)' : '(跌倒)'}`);
console.log(`完成回合数: ${rallies}`);

console.log('\n自检完成。');
