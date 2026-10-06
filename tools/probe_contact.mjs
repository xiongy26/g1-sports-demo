// 几何探针：量化手-拍-球接触质量（用于调参，不属于运行时）
// 用法: node tools/probe_contact.mjs
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import loadMujoco from '../vendor/mujoco/mujoco.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));


const TIMESTEP = 0.002;
const { BasketballController } = await import('../src/basketball.js');
const { PingpongController } = await import('../src/pingpong.js');
const { buildJointMap, bodyId, jointId, keyId, setBasePose, quatRotVec, quatMul, quatConj } = await import('../src/util.js');

import { buildModelXml } from '../src/scene_merge.js';
const mujoco = await loadMujoco();
const sceneXml = readFileSync(join(ROOT, 'model/scene_gym.xml'), 'utf8');
const g1Xml = readFileSync(join(ROOT, 'model/g1.xml'), 'utf8');
const merged = buildModelXml(sceneXml, g1Xml);
const vfs = new mujoco.MjVFS();
for (const f of readdirSync(join(ROOT, 'model/assets'))) {
  const buf = readFileSync(join(ROOT, 'model/assets', f));
  vfs.addBuffer('assets/' + f, new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength));
}
const model = mujoco.MjModel.from_xml_string(merged, vfs);
const data = new mujoco.MjData(model);
const jmap = buildJointMap(mujoco, model, [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]);
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
  onScore: () => {}, onShot: () => {}, onRally: () => {},
};
const BALL_R = 0.02, BLADE_HALF_T = 0.008, BLADE_R = 0.085;
const BBALL_R = 0.123;

function meshGeomId(meshName) {
  const mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH.value, meshName);
  for (let g = 0; g < model.ngeom; g++) if (model.geom_dataid[g] === mid && model.geom_type[g] === 7) return g;
  return -1;
}
// 手部网格在腕系下的统计（AABB + 质心）
const handGid = meshGeomId('right_rubber_hand');
const meshId = model.geom_dataid[handGid];
const vadr = model.mesh_vertadr[meshId], vnum = model.mesh_vertnum[meshId];
const gp = [model.geom_pos[3*handGid], model.geom_pos[3*handGid+1], model.geom_pos[3*handGid+2]];
const gq = [model.geom_quat[4*handGid], model.geom_quat[4*handGid+1], model.geom_quat[4*handGid+2], model.geom_quat[4*handGid+3]];
let mn = [1e9,1e9,1e9], mx = [-1e9,-1e9,-1e9], sum = [0,0,0], n = 0;
for (let i = 0; i < vnum; i++) {
  let v = [model.mesh_vert[3*(vadr+i)], model.mesh_vert[3*(vadr+i)+1], model.mesh_vert[3*(vadr+i)+2]];
  v = quatRotVec(gq, v);
  for (let a = 0; a < 3; a++) { const c = v[a] + gp[a]; mn[a] = Math.min(mn[a], c); mx[a] = Math.max(mx[a], c); sum[a] += c; n++; }
}
const cen = sum.map(s => s / n);
console.log('== 右手网格 在 right_wrist_yaw_link 系下 ==');
console.log('  AABB min', mn.map(x=>x.toFixed(3)).join(' '), ' max', mx.map(x=>x.toFixed(3)).join(' '));
console.log('  质心', cen.map(x=>x.toFixed(3)).join(' '), ' 顶点数', n);
console.log('  尺寸 x(指向):', (mx[0]-mn[0]).toFixed(3), ' y:', (mx[1]-mn[1]).toFixed(3), ' z:', (mx[2]-mn[2]).toFixed(3));

// ---------- 通用 ----------
function bodyFrame(bid) {
  const o = 3 * bid, q = 4 * bid;
  return {
    p: [data.xpos[o], data.xpos[o+1], data.xpos[o+2]],
    q: [data.xquat[q], data.xquat[q+1], data.xquat[q+2], data.xquat[q+3]],
  };
}
function toWorld(f, v) { const o = quatRotVec(f.q, v); return [f.p[0]+o[0], f.p[1]+o[1], f.p[2]+o[2]]; }
function toLocal(f, pt) { const d = [pt[0]-f.p[0], pt[1]-f.p[1], pt[2]-f.p[2]]; return quatRotVec(quatConj(f.q), d); }
function norm(v) { const l = Math.hypot(...v); return v.map(x=>x/l); }
function angleDeg(a, b) {
  const d = a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  return Math.acos(Math.max(-1, Math.min(1, d))) * 180 / Math.PI;
}
function paddleFrame() {
  const mid = env.paddleMid;
  return {
    p: [data.mocap_pos[3*mid], data.mocap_pos[3*mid+1], data.mocap_pos[3*mid+2]],
    q: [data.mocap_quat[4*mid], data.mocap_quat[4*mid+1], data.mocap_quat[4*mid+2], data.mocap_quat[4*mid+3]],
  };
}
function resetMode(m) {
  mujoco.mj_resetDataKeyframe(model, data, keyId(mujoco, model, 'home'));
  if (m === 'pingpong') setBasePose(data, -1.15, 0, 0.783675, Math.PI);
  else setBasePose(data, 0, 0, 0.783675, 0);
  // 探针只测单人模式：二号机停到地下
  const jq = jointId(mujoco, model, 'r2_floating_base_joint');
  setBasePose(data, 0, 0, -5, 0, model.jnt_qposadr[jq], model.jnt_dofadr[jq]);
}

// ---------- HOME 手系轴；真实掌面为局部 +y（手指弯曲侧） ----------
console.log('\n== HOME(篮球 reset) 姿态下右手轴向（世界系） ==');
resetMode('basketball');
mujoco.mj_forward(model, data);
{
  const f = bodyFrame(env.wristBid);
  console.log('  手 x 轴 →', norm(quatRotVec(f.q, [1,0,0])).map(x=>x.toFixed(2)).join(', '), '(应为前/下)');
  console.log('  手 +y 轴 →', norm(quatRotVec(f.q, [0,1,0])).map(x=>x.toFixed(2)).join(', '));
  console.log('  手 +z 轴 →', norm(quatRotVec(f.q, [0,0,1])).map(x=>x.toFixed(2)).join(', '));
  console.log('  → 掌心朝向是 ±y 之一（自然下垂时贴大腿 ≈ 世界 -y）');
}

// ================= 机器 =================
console.log('\n== 发球机炮口指向 ==');
{
  const gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, 'machine_barrel');
  const q = [model.geom_quat[4*gid], model.geom_quat[4*gid+1], model.geom_quat[4*gid+2], model.geom_quat[4*gid+3]];
  const pos = [model.geom_pos[3*gid], model.geom_pos[3*gid+1], model.geom_pos[3*gid+2]];
  const axis = norm(quatRotVec(q, [0,0,1]));
  const half = model.geom_size[3*gid+2];
  const bid = model.geom_bodyid[gid];
  const bp = [model.body_pos[3*bid], model.body_pos[3*bid+1], model.body_pos[3*bid+2]];
  const mouthLocal = [pos[0]+axis[0]*half, pos[1]+axis[1]*half, pos[2]+axis[2]*half];
  console.log('  炮管轴(世界系)=', axis.map(x=>x.toFixed(2)).join(', '), ' (发球方向应为 +x 偏上)');
  console.log('  炮口1(世界)=', [bp[0]+mouthLocal[0], bp[1]+mouthLocal[1], bp[2]+mouthLocal[2]].map(x=>x.toFixed(2)).join(', '));
  const mouth2 = [pos[0]-axis[0]*half, pos[1]-axis[1]*half, pos[2]-axis[2]*half];
  console.log('  炮口2(世界)=', [bp[0]+mouth2[0], bp[1]+mouth2[1], bp[2]+mouth2[2]].map(x=>x.toFixed(2)).join(', '));
}

// ================= 乒乓球 =================
console.log('\n== 乒乓球：击球瞬间 ==');
resetMode('pingpong');
const pp = new PingpongController(env); pp.reset();
mujoco.mj_forward(model, data);
let best = null;
const ttSamples = [];
for (let i = 0; i < 6 / TIMESTEP; i++) {
  pp.step(TIMESTEP);
  mujoco.mj_step(model, data);
  const f = paddleFrame();
  const mid = env.ttBallMid;
  const ball = [data.mocap_pos[3*mid], data.mocap_pos[3*mid+1], data.mocap_pos[3*mid+2]];
  const loc = toLocal(f, ball);
  const d = Math.hypot(...loc);
  if (!best || d < best.d) {
    best = { d, loc: loc.slice(), t: data.time, leg: pp.leg, swingT: pp.bot.swingT,
      f: { p: f.p.slice(), q: f.q.slice() }, wf: JSON.parse(JSON.stringify(bodyFrame(env.wristBid))) };
  }
  ttSamples.push({ t: data.time, loc, leg: pp.leg, swinging: pp.bot.swinging, fp: f.p.slice() });
}
{
  const b = best;
  console.log(`  最近 t=${b.t.toFixed(3)} leg=${b.leg} swingT=${b.swingT.toFixed(3)} 距离=${b.d.toFixed(3)} 球心(拍系)=[${b.loc.map(x=>x.toFixed(3)).join(', ')}]`);
  const axial = Math.abs(b.loc[2]);
  const state = axial > BLADE_HALF_T + BALL_R + 0.004 ? '未接触(悬空)' : axial < BLADE_HALF_T - 0.002 ? '穿透拍面!' : '贴合拍面 ✓';
  console.log(`  轴向(法向)=${b.loc[2].toFixed(3)} → 球在拍面${b.loc[2] > 0 ? '正面(+z 黑面)' : '背面(-z 红面)'}: ${state}`);
  console.log(`  径向=${Math.hypot(b.loc[0], b.loc[1]).toFixed(3)} (拍半径 ${BLADE_R})`);
  // 拍面法向(世界) vs 出球方向 C→D（含出射仰角）
  const nrm = norm(quatRotVec(b.f.q, [0,0,1]));
  const outDir = norm([-3.30 - (-1.36), 0.05 - 0.50, 0]);
  const outDir3D = norm([-3.30 - (-1.36), 0.05 - 0.50, 0.5*9.81*0.5]);
  console.log(`  击球时拍面法向(世界)= [${nrm.map(x=>x.toFixed(2)).join(', ')}]`);
  console.log(`  出球方向(3D 含仰角)= [${outDir3D.map(x=>x.toFixed(2)).join(', ')}]  夹角=${angleDeg(nrm, outDir3D).toFixed(1)}° (水平向 ${angleDeg(nrm, outDir).toFixed(1)}°)`);
  // 手系三轴（世界）与期望出球方向在手系下的表达
  const wf = b.wf;
  const hx = norm(quatRotVec(wf.q, [1,0,0])), hy = norm(quatRotVec(wf.q, [0,1,0])), hz = norm(quatRotVec(wf.q, [0,0,1]));
  console.log(`  手系轴(世界): x=[${hx.map(x=>x.toFixed(2)).join(', ')}] y=[${hy.map(x=>x.toFixed(2)).join(', ')}] z=[${hz.map(x=>x.toFixed(2)).join(', ')}]`);
  console.log(`  出球方向(手系)= [${toLocal(wf, [b.f.p[0]+outDir3D[0], b.f.p[1]+outDir3D[1], b.f.p[2]+outDir3D[2]]).map(x=>x.toFixed(2)).join(', ')}]  (拍面法向当前在手系= ${toLocal(wf, [b.f.p[0]+nrm[0], b.f.p[1]+nrm[1], b.f.p[2]+nrm[2]]).map(x=>x.toFixed(2)).join(', ')})`);
  console.log(`  击球时 拍心(世界)= [${b.f.p.map(x=>x.toFixed(3)).join(', ')}]`);
  const facePt = [b.f.p[0]+nrm[0]*0.028, b.f.p[1]+nrm[1]*0.028, b.f.p[2]+nrm[2]*0.028];
  console.log(`  击球时 拍面接触点(世界)= [${facePt.map(x=>x.toFixed(3)).join(', ')}]`);
  // 击球前后拍心扫过方向（±60ms 有限差分）
  const around = ttSamples.filter(s => s.leg === 1 && s.swinging && s.t >= b.t - 0.06 && s.t <= b.t + 0.06).sort((a,c)=>a.t-c.t);
  if (around.length >= 5) {
    const a0 = around[0].fp, a1 = around[around.length-1].fp;
    const sweep = norm([a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2]]);
    console.log(`  击球±60ms 拍心扫动方向(世界)= [${sweep.map(x=>x.toFixed(2)).join(', ')}]  vs 出球方向夹角=${angleDeg(sweep, outDir3D).toFixed(1)}°`);
  }
  console.log(`  (手网格 x∈[${mn[0].toFixed(2)},${mx[0].toFixed(2)}] y∈[${mn[1].toFixed(2)},${mx[1].toFixed(2)}] z∈[${mn[2].toFixed(2)},${mx[2].toFixed(2)}])`);
}

// ready 状态拍相对手
resetMode('pingpong');
pp.reset(); mujoco.mj_forward(model, data);
for (let i = 0; i < 1.0 / TIMESTEP; i++) { pp.step(TIMESTEP); mujoco.mj_step(model, data); }
{
  const f = paddleFrame();
  const wf = bodyFrame(env.wristBid);
  const bladeW = toLocal(wf, f.p);
  const zAxisW = quatRotVec(quatConj(wf.q), quatRotVec(f.q, [0,0,1]));
  console.log(`  ready 时 拍心(腕系)= [${bladeW.map(x=>x.toFixed(3)).join(', ')}]  拍轴(腕系)= [${zAxisW.map(x=>x.toFixed(2)).join(', ')}]`);
  const nrm = norm(quatRotVec(f.q, [0,0,1]));
  const inDir = norm([-0.99, 0, 0]); // 来球来向（球从机器一侧飞来，拍面应大致朝它）
  console.log(`  ready 拍面法向(世界)= [${nrm.map(x=>x.toFixed(2)).join(', ')}]  vs 来球来向夹角=${angleDeg(nrm, inDir).toFixed(1)}°`);
  const rwf = bodyFrame(env.wristBid);
  console.log(`  ready 手系轴(世界): x=[${norm(quatRotVec(rwf.q,[1,0,0])).map(x=>x.toFixed(2)).join(', ')}] y=[${norm(quatRotVec(rwf.q,[0,1,0])).map(x=>x.toFixed(2)).join(', ')}] z=[${norm(quatRotVec(rwf.q,[0,0,1])).map(x=>x.toFixed(2)).join(', ')}]`);
  const tgt = norm([-0.95, 0.05, 0.31]);
  console.log(`  ready 目标法向(世界)= [${tgt.map(x=>x.toFixed(2)).join(', ')}] → 目标(手系)= [${toLocal(rwf, [rwf.p[0]+tgt[0], rwf.p[1]+tgt[1], rwf.p[2]+tgt[2]]).map(x=>x.toFixed(2)).join(', ')}]`);
}

// ================= 篮球 =================
console.log('\n== 篮球：球相对掌面 ==');
resetMode('basketball');
const bb = new BasketballController(env); bb.reset();
mujoco.mj_forward(model, data);
// 球心在腕系下的坐标波动（贴合 = 恒定）
function ballStats(N, filter) {
  let mnL = [1e9,1e9,1e9], mxL = [-1e9,-1e9,-1e9];
  for (let i = 0; i < N / TIMESTEP; i++) {
    bb.step(TIMESTEP);
    mujoco.mj_step(model, data);
    if (filter && !filter(bb.state)) continue;
    const f = bodyFrame(env.wristBid);
    const ball = [data.qpos[env.ballQadr], data.qpos[env.ballQadr+1], data.qpos[env.ballQadr+2]];
    const loc = toLocal(f, ball);
    for (let a = 0; a < 3; a++) { mnL[a] = Math.min(mnL[a], loc[a]); mxL[a] = Math.max(mxL[a], loc[a]); }
  }
  const spread = mnL.map((v, a) => (mxL[a] - v).toFixed(3)).join(', ');
  console.log(`    球心(腕系)范围 x[${mnL[0].toFixed(3)},${mxL[0].toFixed(3)}] y[${mnL[1].toFixed(3)},${mxL[1].toFixed(3)}] z[${mnL[2].toFixed(3)},${mxL[2].toFixed(3)}]  滑移=${spread}`);
}
console.log('  [carry 8s]'); ballStats(8, st => st === 'carry');
bb.requestShoot = true;
let prevState = null, prevBall = null;
for (let i = 0; i < 4 / TIMESTEP; i++) {
  bb.step(TIMESTEP);
  mujoco.mj_step(model, data);
  const ball = [data.qpos[env.ballQadr], data.qpos[env.ballQadr+1], data.qpos[env.ballQadr+2]];
  if (prevState === 'windup' && bb.state === 'flight' && prevBall) {
    // 出手瞬间：掌面法向 vs 出球速度方向
    const v = [(ball[0]-prevBall[0])/TIMESTEP, (ball[1]-prevBall[1])/TIMESTEP, (ball[2]-prevBall[2])/TIMESTEP];
    const nrm = norm(quatRotVec(bodyFrame(env.wristBid).q, [0, 1, 0]));
    console.log(`  出手瞬间: 掌面法向(世界)= [${nrm.map(x=>x.toFixed(2)).join(', ')}]  出球速度= [${v.map(x=>x.toFixed(2)).join(', ')}]  夹角=${angleDeg(nrm, norm(v)).toFixed(1)}°`);
  }
  prevState = bb.state; prevBall = ball.slice();
}
console.log('\n探针完成。');
