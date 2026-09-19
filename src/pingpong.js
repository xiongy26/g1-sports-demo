// 乒乓球模式：机器人持拍在球台一端，与对面发球机连续对打。
// 乒乓球为运动学脚本（分段抛物线），球拍 mocap 跟随右手腕。
import { smoothInto, quatMul, quatRotVec, BLADE_QUAT, setBasePose, applyBalanceAssist } from './util.js';

const READY = [
  -0.28, 0, 0, 0.55, -0.25, 0,   // 左腿
  -0.28, 0, 0, 0.55, -0.25, 0,   // 右腿
  0, 0, 0.18,                    // 腰
  -0.6, 0.25, 0, 1.1, 0, 0.3, 0, // 左臂
  -0.5, -0.15, 0.3, 1.3, 0, -0.2, 0, // 右臂
];

// 对打关键点（世界系）：发球机 -> 弹跳 -> 机器人正手 -> 弹跳 -> 发球机
// C 取自球拍实际扫过轨迹的击球窗口中心（挥拍采样标定）
const A = [-4.45, 0.12, 1.02];   // 机器出球
const B = [-3.20, 0.25, 0.788];  // 对手台面弹跳
const C = [-1.36, 0.50, 0.79];   // 机器人击球点
const D = [-3.30, 0.05, 0.788];  // 对手台面弹跳
const E = [-4.40, -0.16, 1.05];  // 机器击球点

const LEGS = [
  { a: A, b: B, T: 0.36 },
  { a: B, b: C, T: 0.55 },   // 到达 C 时挥拍已触球
  { a: C, b: D, T: 0.50 },
  { a: D, b: E, T: 0.50 },
];

const SWING_T = 0.62, CONTACT_FRAC = 0.55;

export class PingpongController {
  constructor(env) {
    this.env = env;
    const j = env.jmap;
    this.ix = {
      lHip: j.left_hip_pitch_joint.c, lKnee: j.left_knee_joint.c, lAnkle: j.left_ankle_pitch_joint.c,
      rHip: j.right_hip_pitch_joint.c, rKnee: j.right_knee_joint.c, rAnkle: j.right_ankle_pitch_joint.c,
      waistYaw: j.waist_yaw_joint.c, waistPitch: j.waist_pitch_joint.c,
      lShP: j.left_shoulder_pitch_joint.c, lShR: j.left_shoulder_roll_joint.c, lEl: j.left_elbow_joint.c,
      rShP: j.right_shoulder_pitch_joint.c, rShR: j.right_shoulder_roll_joint.c,
      rShY: j.right_shoulder_yaw_joint.c, rEl: j.right_elbow_joint.c,
      rWrR: j.right_wrist_roll_joint.c, rWrP: j.right_wrist_pitch_joint.c,
    };
    this.goal = Float64Array.from(READY);
    this.smoothed = Float64Array.from(READY);
  }

  reset() {
    setBasePose(this.env.data, -1.15, 0, 0.783675, Math.PI);
    this.state = 'serve';
    this.t = 0; this.tState = 0;
    this.leg = 0; this.legT = 0;
    this.ballStart = A.slice();
    this.swinging = false; this.swingT = 0;
    this.goal.set(READY); this.smoothed.set(READY);
    this.rallyCount = 0;
    this.writeBall(A);
  }

  action() { // 重新发球
    this.state = 'serve'; this.tState = 0;
    this.leg = 0; this.legT = 0;
    this.writeBall(this.ballStart);
  }

  cameraPreset() {
    return { pos: [-0.1, -2.6, 1.85], target: [-2.7, 0, 0.85] };
  }

  writeBall(p) {
    const mid = this.env.ttBallMid;
    this.env.data.mocap_pos[3 * mid] = p[0];
    this.env.data.mocap_pos[3 * mid + 1] = p[1];
    this.env.data.mocap_pos[3 * mid + 2] = p[2];
  }

  arcVel(a, b, T) {
    return [(b[0] - a[0]) / T, (b[1] - a[1]) / T, (b[2] - a[2]) / T + 0.5 * 9.81 * T];
  }

  step(dt) {
    this.t += dt; this.tState += dt;
    const goal = this.goal;
    goal.set(READY);

    // 呼吸起伏
    const bob = 0.5 + 0.5 * Math.sin(2 * Math.PI * 0.9 * this.t);
    goal[this.ix.lKnee] = 0.55 + 0.03 * bob;
    goal[this.ix.rKnee] = 0.55 + 0.03 * bob;

    // ---------- 球的脚本运动 ----------
    if (this.state === 'serve') {
      if (this.tState > 0.6) {
        this.state = 'rally'; this.tState = 0; this.leg = 0; this.legT = 0;
        this.ballStart = [A[0], A[1] + (Math.random() - 0.5) * 0.08, A[2]];
        this.legs = LEGS.map((L, i) => (i === 0 ? { a: this.ballStart, b: B, T: L.T } : L));
        this.swingArmed = false;
      }
    } else if (this.state === 'rally') {
      const L = this.legs[this.leg];
      this.legT += dt;
      const v0 = this.arcVel(L.a, L.b, L.T);
      const t = Math.min(this.legT, L.T);
      this.writeBall([
        L.a[0] + v0[0] * t,
        L.a[1] + v0[1] * t,
        L.a[2] + v0[2] * t - 4.905 * t * t,
      ]);
      // 准备挥拍：到达击球点前 0.34s 启动
      if (this.leg === 1 && !this.swingArmed) {
        const tSwingStart = L.T - SWING_T * CONTACT_FRAC;
        if (this.legT >= tSwingStart) { this.swinging = true; this.swingT = 0; this.swingArmed = true; }
      }
      if (this.legT >= L.T) {
        this.legT -= L.T;
        this.leg = (this.leg + 1) % 4;
        if (this.leg === 0) { // 一分结束，机器重新发球
          this.rallyCount++;
          this.ballStart = [A[0], A[1] + (Math.random() - 0.5) * 0.08, A[2]];
          this.legs = [{ a: this.ballStart, b: B, T: LEGS[0].T }, LEGS[1], LEGS[2], LEGS[3]];
          this.swingArmed = false;
          if (this.env.onRally) this.env.onRally(this.rallyCount);
        }
        if (this.leg === 2) { this.swinging = false; }
      }
    }

    // ---------- 挥拍动作 ----------
    if (this.swinging) {
      this.swingT += dt;
      const p = Math.min(1, this.swingT / SWING_T);
      const seg = (t0, t1) => Math.max(0, Math.min(1, (p - t0) / (t1 - t0)));
      const ease = (x) => x * x * (3 - 2 * x);
      const w = ease(seg(0, CONTACT_FRAC));        // 引拍
      const s = ease(seg(CONTACT_FRAC, 0.78));      // 击球
      const f = ease(seg(0.78, 1));                 // 随挥
      goal[this.ix.waistYaw] = -0.38 * w + 0.68 * s - 0.30 * f;
      goal[this.ix.rShP] = -0.5 + (-0.7) * w + 0.55 * s + 0.45 * f;
      goal[this.ix.rShR] = -0.15 + (-0.55) * w + 0.35 * s + 0.25 * f;
      goal[this.ix.rShY] = 0.3 + 0.15 * w + (-1.35) * s - 0.15 * f;
      goal[this.ix.rEl] = 1.3 + 0.35 * w + (-0.85) * s + 0.35 * f;
      goal[this.ix.rWrR] = -0.6 * s + 0.5 * f;
      goal[this.ix.rWrP] = -0.3 * s;
      goal[this.ix.lShP] = -0.6 - 0.25 * s;
      goal[this.ix.lShR] = 0.25 + 0.15 * s;
    }

    smoothInto(this.smoothed, goal, dt, this.swinging ? 0.025 : 0.08);
    const ctrl = this.env.data.ctrl;
    for (let i = 0; i < 29; i++) ctrl[i] = this.smoothed[i];

    this.updatePaddle();
    // 平衡辅助锚在站位点
    applyBalanceAssist(this.env.data, this.env.pelvisBid, -1.15, 0, 1.0);
  }

  updatePaddle() {
    const { data, wristBid, paddleMid } = this.env;
    const o = 3 * wristBid, q = 4 * wristBid;
    const wpos = [data.xpos[o], data.xpos[o + 1], data.xpos[o + 2]];
    const wquat = [data.xquat[q], data.xquat[q + 1], data.xquat[q + 2], data.xquat[q + 3]];
    const off = quatRotVec(wquat, [0.16, -0.005, -0.015]);
    data.mocap_pos[3 * paddleMid] = wpos[0] + off[0];
    data.mocap_pos[3 * paddleMid + 1] = wpos[1] + off[1];
    data.mocap_pos[3 * paddleMid + 2] = wpos[2] + off[2];
    const pq = quatMul(wquat, BLADE_QUAT);
    data.mocap_quat[4 * paddleMid] = pq[0];
    data.mocap_quat[4 * paddleMid + 1] = pq[1];
    data.mocap_quat[4 * paddleMid + 2] = pq[2];
    data.mocap_quat[4 * paddleMid + 3] = pq[3];
  }
}
