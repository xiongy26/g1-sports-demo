// 无头仿真自检：在 node 中加载与浏览器完全相同的模型与控制器，验证三种模式的运动逻辑。
// 用法: node tools/sim_check.mjs
import { readFileSync, readdirSync } from 'node:fs';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import loadMujoco from '../vendor/mujoco/mujoco.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

// ---- 与 main.js 相同的模型组装（含二号机） ----
import { buildModelXml } from '../src/scene_merge.js';

// ---- 与 main.js 相同的关节表 ----
const JOINT_NAMES = [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
];

const TIMESTEP = 0.002;
const { BasketballController, PALM_NORMAL_LOCAL } = await import('../src/basketball.js');
const { PingpongController, DuelController } = await import('../src/pingpong.js');
const { buildJointMap, bodyId, jointId, keyId, setBasePose, quatRotVec, quatConj, REST_POSE, applyBalanceAssist } = await import('../src/util.js');

const mujoco = await loadMujoco();

const sceneXml = readFileSync(join(ROOT, 'model/scene_gym.xml'), 'utf8');
const g1Xml = readFileSync(join(ROOT, 'model/g1.xml'), 'utf8');
const merged = buildModelXml(sceneXml, g1Xml);

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
  paddleMid2: model.body_mocapid[bodyId(mujoco, model, 'paddle_r2')],
  machineMid: model.body_mocapid[bodyId(mujoco, model, 'tt_machine')],
  r2WristBid: bodyId(mujoco, model, 'r2_right_wrist_yaw_link'),
  r2PelvisBid: bodyId(mujoco, model, 'r2_pelvis'),
  r2Qadr: model.jnt_qposadr[jointId(mujoco, model, 'r2_floating_base_joint')],
  r2Dof: model.jnt_dofadr[jointId(mujoco, model, 'r2_floating_base_joint')],
  onScore: () => { score++; },
  onShot: () => {},
  onRally: (n) => { rallies = n; },
};

function keyIdOf(name) { return keyId(mujoco, model, name); }
// 二号机场边待机位（与 main.js 的 R2_REST 一致）
const R2_REST = { x: -5.7, y: -1.9, yaw: 0.6 };
function resetMode(m) {
  mujoco.mj_resetDataKeyframe(model, data, keyIdOf('home'));
  if (m === 'basketball') setBasePose(data, 0, 0, 0.783675, 0);
  else setBasePose(data, -1.15, 0, 0.783675, Math.PI);
  if (m !== 'duel') setBasePose(data, R2_REST.x, R2_REST.y, 0.783675, R2_REST.yaw, env.r2Qadr, env.r2Dof);
  // 道具停放与 main.js setMode 一致：对打时发球机藏到地下（让位给二号机）
  const hidden = [0, 0, -5];
  if (m === 'pingpong' || m === 'basketball') data.mocap_pos.set([-4.72, 0.25, 0], 3 * env.machineMid);
  else if (m === 'duel') data.mocap_pos.set(hidden, 3 * env.machineMid);
  if (m === 'pingpong' || m === 'duel') data.mocap_pos.set([-4.42, 0.25, 1.10], 3 * env.ttBallMid);
}
// 非对打模式下二号机待机（每个物理子步调用）
function idleR2() {
  for (let i = 0; i < 29; i++) data.ctrl[29 + i] = REST_POSE[i];
  applyBalanceAssist(data, env.r2PelvisBid, R2_REST.x, R2_REST.y, 1.0, env.r2Dof);
}

function wristFrame() {
  const o = 3 * env.wristBid, q = 4 * env.wristBid;
  return {
    p: [data.xpos[o], data.xpos[o + 1], data.xpos[o + 2]],
    q: [data.xquat[q], data.xquat[q + 1], data.xquat[q + 2], data.xquat[q + 3]],
  };
}
function toWristLocal(f, pt) {
  const d = [pt[0] - f.p[0], pt[1] - f.p[1], pt[2] - f.p[2]];
  return quatRotVec(quatConj(f.q), d);
}
function angleDeg(a, b) {
  const la = Math.hypot(...a), lb = Math.hypot(...b);
  return Math.acos(Math.max(-1, Math.min(1, (a[0]*b[0]+a[1]*b[1]+a[2]*b[2]) / (la*lb)))) * 180 / Math.PI;
}

// 拍柄必须在拍面内延伸，不能像旧模型沿拍面法线伸出。
for (const name of ['paddle', 'paddle_r2']) {
  const gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, name === 'paddle' ? 'paddle_handle' : 'paddle_r2_handle');
  const sizes = Array.from(model.geom_size.slice(3 * gid, 3 * gid + 3));
  const axis = sizes.indexOf(Math.max(...sizes));
  const unit = [0, 0, 0]; unit[axis] = 1;
  const direction = quatRotVec(Array.from(model.geom_quat.slice(4 * gid, 4 * gid + 4)), unit);
  assert.ok(Math.abs(direction[2]) < 0.01, `${name} 拍柄沿拍面法线伸出`);
  assert.ok(Math.abs(model.geom_pos[3 * gid + 2]) < 0.01, `${name} 拍柄偏离拍面`);
}

// ================= 篮球 =================
console.log('\n--- 篮球模式 30s ---');
let score = 0, rallies = 0;
const bb = new BasketballController(env); resetMode('basketball');
bb.reset();
mujoco.mj_forward(model, data);
const states = new Map();
let minPelZ = 9, maxBallZ = 0, minBallZ = 9;
let dribblePalmMaxZ = -1, dribbleSamples = 0;
// 球-掌贴合指标：windup 全程球心在腕系的滑移；出手瞬间掌面法向与出球方向夹角
let windSlide = 0, windFirst = null;
let releaseAngle = null, prevState = null, prevBall = null;
const T_BB = 30;
for (let i = 0; i < T_BB / TIMESTEP; i++) {
  idleR2();
  bb.step(TIMESTEP);
  mujoco.mj_step(model, data);
  states.set(bb.state, (states.get(bb.state) || 0) + TIMESTEP);
  const pz = data.qpos[2];
  if (pz < minPelZ) minPelZ = pz;
  const ball = [data.qpos[env.ballQadr], data.qpos[env.ballQadr + 1], data.qpos[env.ballQadr + 2]];
  const bz = ball[2];
  if (bz > maxBallZ) maxBallZ = bz;
  if (bz < minBallZ) minBallZ = bz;
  if ((bb.state === 'carry' || bb.state === 'free') && bb.t > 0.3) {
    // +y 是手指弯曲一侧的真实掌面；运球全周期应朝下。
    const normal = quatRotVec(wristFrame().q, [0, 1, 0]);
    dribblePalmMaxZ = Math.max(dribblePalmMaxZ, normal[2]);
    dribbleSamples++;
  }
  if (bb.state === 'windup') {
    const loc = toWristLocal(wristFrame(), ball);
    if (!windFirst) windFirst = loc;
    else windSlide = Math.max(windSlide, Math.hypot(loc[0]-windFirst[0], loc[1]-windFirst[1], loc[2]-windFirst[2]));
  }
  if (prevState === 'windup' && bb.state === 'flight' && prevBall) {
    // 出手瞬间：球速 vs 掌面法向(-y 手系)
    const v = [(ball[0]-prevBall[0])/TIMESTEP, (ball[1]-prevBall[1])/TIMESTEP, (ball[2]-prevBall[2])/TIMESTEP];
    const nrm = quatRotVec(wristFrame().q, PALM_NORMAL_LOCAL);
    releaseAngle = angleDeg(nrm, v);
  }
  prevState = bb.state; prevBall = ball.slice();
}
console.log('状态时长(s):', Object.fromEntries([...states.entries()].map(([k, v]) => [k, +v.toFixed(2)])));
console.log(`进球数: ${score}  髋部最低高度: ${minPelZ.toFixed(3)} m (站立${minPelZ > 0.5 ? '稳定' : '失败'})`);
console.log(`篮球高度范围: ${minBallZ.toFixed(2)} ~ ${maxBallZ.toFixed(2)} m (运球${maxBallZ > 0.9 && minBallZ < 0.3 ? '正常' : '异常'})`);
console.log(`windup 球-掌滑移: ${windSlide.toFixed(3)} m ${windSlide < 0.03 ? '✓ (托在掌心)' : '✗ (球在手上漂移)'}`);
console.log(`出手瞬间掌面-出球夹角: ${releaseAngle === null ? '未捕捉' : releaseAngle.toFixed(1) + '°' + (releaseAngle < 45 ? ' ✓ (掌心托球出手)' : ' ✗ (掌面朝向不对)')}`);

console.log(`运球掌面最大向上分量: ${dribblePalmMaxZ.toFixed(3)} (应 < -0.75)`);
assert.ok(dribbleSamples > 500 && dribblePalmMaxZ < -0.75, '运球时掌心未持续朝下');

// 验证实际动力学结果，而非仅检查状态机是否运行。
assert.ok(minPelZ > 0.70, '篮球动作导致机器人失去站立高度');
assert.ok(Number.isFinite(maxBallZ) && minBallZ > 0.07, '篮球出现无效状态或严重穿地');
assert.ok(states.has('gather') && states.has('windup') && states.has('flight'), '投篮阶段未完成');
assert.ok(releaseAngle !== null && releaseAngle < 15, '实际出手掌面未对准球速');
assert.ok(windSlide < 0.02, '举球阶段球在掌上滑移');
assert.ok(score >= 2, '自由飞行投篮未正常完成');

// ================= 乒乓球 =================
console.log('\n--- 乒乓球模式 6s ---');
score = 0; rallies = 0;
resetMode('pingpong');
const pp = new PingpongController(env);
pp.reset();
mujoco.mj_forward(model, data);
let swingFrames = 0, minD = 9, minAt = "";
let minDAxial = 0, minDRadial = 0, minDNrm = null;
let netCrossZ = null, prevBx = null;
const T_TT = 6;
const CONTACT = 0.028; // 拍半厚 0.008 + 球半径 0.02 = 贴合时球心-拍心距离
for (let i = 0; i < T_TT / TIMESTEP; i++) {
  idleR2();
  pp.step(TIMESTEP);
  mujoco.mj_step(model, data);
  if (pp.bot.swinging) swingFrames++;
  const mp = data.mocap_pos, mq = data.mocap_quat;
  const px = mp[3*env.paddleMid], py = mp[3*env.paddleMid+1], pz = mp[3*env.paddleMid+2];
  const bx = mp[3*env.ttBallMid], by = mp[3*env.ttBallMid+1], bz = mp[3*env.ttBallMid+2];
  const d = Math.hypot(px-bx, py-by, pz-bz);
  if (d < minD) {
    minD = d;
    minAt = `leg=${pp.leg} legT=${pp.legT.toFixed(3)} state=${pp.state}`;
    // 球心在拍系下坐标 → 轴向/径向
    const q = [mq[4*env.paddleMid], mq[4*env.paddleMid+1], mq[4*env.paddleMid+2], mq[4*env.paddleMid+3]];
    const loc = quatRotVec(quatConj(q), [bx-px, by-py, bz-pz]);
    minDAxial = loc[2]; minDRadial = Math.hypot(loc[0], loc[1]);
    minDNrm = quatRotVec(q, [0, 0, 1]);
  }
  // 机器人击球方向 (C->D) 过网 (x=-2.9) 时的高度
  if (prevBx !== null && prevBx > -2.9 && bx <= -2.9 && pp.leg === 2 && netCrossZ === null) netCrossZ = bz;
  prevBx = bx;
}
console.log(`挥拍帧数: ${swingFrames} (~${(swingFrames * TIMESTEP).toFixed(2)}s)`);
console.log(`击球瞬间拍球最小距离: ${minD.toFixed(3)} m (贴合=${CONTACT.toFixed(3)}) ${Math.abs(minD - CONTACT) < 0.006 ? "✓ 贴在拍面" : (minD < CONTACT ? "✗ 穿透" : "✗ 悬空")} [${minAt}]`);
console.log(`  轴向=${minDAxial.toFixed(3)} (应在 +${CONTACT})  径向=${minDRadial.toFixed(3)} (拍半径 0.085${minDRadial < 0.05 ? ', 甜区' : ', 偏拍缘'})`);
{
  const outDir = [-3.30 - (-1.43), 0.05 - 0.47, 0.5 * 9.81 * 0.5 * 0.5];
  const ang = minDNrm ? angleDeg(minDNrm, outDir) : NaN;
  console.log(`  拍面法向 vs 出球方向夹角: ${ang.toFixed(1)}° ${ang < 30 ? '✓ (拍面对准出球)' : '✗ (拍面偏得太远)'}`);
}
console.log(`回球过网高度: ${netCrossZ === null ? '未捕捉' : netCrossZ.toFixed(2) + ' m (净高0.9125) ' + (netCrossZ > 0.92 ? '✓' : '✗')}`);
console.log(`机器人高度: ${data.qpos[2].toFixed(3)} m ${data.qpos[2] > 0.5 ? '(稳定)' : '(跌倒)'}`);
console.log(`完成回合数: ${rallies}`);

// ================= 双机对打 =================
console.log('\n--- 双机对打模式 10s ---');
score = 0; rallies = 0;
resetMode('duel');
const duel = new DuelController(env);
duel.reset();
mujoco.mj_forward(model, data);
let minD1 = 9, minD2 = 9, minD2Info = '';
let crossLR = null, crossRL = null; // 过网高度（一→二 / 二→一）
let prevBx2 = null;
const T_DUEL = 10;
for (let i = 0; i < T_DUEL / TIMESTEP; i++) {
  duel.step(TIMESTEP);
  mujoco.mj_step(model, data);
  const mp = data.mocap_pos;
  const bx = mp[3*env.ttBallMid], by = mp[3*env.ttBallMid+1], bz = mp[3*env.ttBallMid+2];
  const d1 = Math.hypot(mp[3*env.paddleMid]-bx, mp[3*env.paddleMid+1]-by, mp[3*env.paddleMid+2]-bz);
  const d2 = Math.hypot(mp[3*env.paddleMid2]-bx, mp[3*env.paddleMid2+1]-by, mp[3*env.paddleMid2+2]-bz);
  if (d1 < minD1) minD1 = d1;
  if (d2 < minD2) {
    minD2 = d2;
    const q2 = [data.mocap_quat[4*env.paddleMid2], data.mocap_quat[4*env.paddleMid2+1], data.mocap_quat[4*env.paddleMid2+2], data.mocap_quat[4*env.paddleMid2+3]];
    const loc2 = quatRotVec(quatConj(q2), [bx-mp[3*env.paddleMid2], by-mp[3*env.paddleMid2+1], bz-mp[3*env.paddleMid2+2]]);
    minD2Info = `leg=${duel.leg} legT=${duel.legT.toFixed(3)} 轴向=${loc2[2].toFixed(3)} 径向=${Math.hypot(loc2[0], loc2[1]).toFixed(3)}`;
  }
  if (prevBx2 !== null) {
    if (prevBx2 > -2.9 && bx <= -2.9) crossLR = bz;
    if (prevBx2 < -2.9 && bx >= -2.9) crossRL = bz;
  }
  prevBx2 = bx;
}
console.log(`完成击球次数: ${duel.hits} (10s 内 ${duel.hits >= 6 ? '✓' : '✗'})`);
for (const [name, d, info] of [['一号机', minD1, ''], ['二号机', minD2, minD2Info]]) {
  const ok = d <= CONTACT + 0.004 && d >= CONTACT - 0.008;
  console.log(`${name}击球贴合距离: ${d.toFixed(3)} m ${ok ? '✓ 贴在拍面(含轻微压缩)' : (d < CONTACT ? '✗ 穿透' : '✗ 悬空')} ${info}`);
}
console.log(`过网高度: 一→二 ${crossLR === null ? '未捕捉' : crossLR.toFixed(2) + ' m'} / 二→一 ${crossRL === null ? '未捕捉' : crossRL.toFixed(2) + ' m'} (净高0.9125, ${crossLR > 0.92 && crossRL > 0.92 ? '✓' : '✗'})`);
console.log(`一号机高度: ${data.qpos[2].toFixed(3)} m  二号机高度: ${data.qpos[env.r2Qadr + 2].toFixed(3)} m ${(data.qpos[2] > 0.5 && data.qpos[env.r2Qadr + 2] > 0.5) ? '(都稳定)' : '(有跌倒!)'}`);

console.log('\n自检完成。');
