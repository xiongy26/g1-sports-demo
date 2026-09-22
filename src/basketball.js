// 篮球模式：原地运球（接球-下压-弹起循环）+ 周期性投篮（引导弹道过筐）
import { applyBalanceAssist, smoothInto, quatRotVec, G } from './util.js';

const RIM = [2.125, 0, 3.05];

// "home" 关键帧的执行器目标（29 个）
const HOME = [
  -0.1, 0, 0, 0.3, -0.2, 0,   // 左腿
  -0.1, 0, 0, 0.3, -0.2, 0,   // 右腿
  0, 0, 0,                     // 腰
  0.2, 0.2, 0, 1.28, 0, 0, 0,  // 左臂
  0.2, -0.2, 0, 1.28, 0, 0, 0, // 右臂
];

// 托球位姿（right_wrist_yaw_link 系）：掌面 = -y（右手自然下垂时贴大腿），
// 锚点取掌心表面；球心 = 锚点沿掌面法向外推 (球半径 - 陷入量)，视觉上托在掌心。
const SEAT_LOCAL = [0.10, -0.022 - (0.123 - 0.02), 0.008];
// 出手时腕角（标定，tools/probe_contact.mjs 网格解）：掌心转到球的下方托住出球方向
const RELEASE_WRIST = { r: -1.00, p: -0.60, y: 0.95 };

export class BasketballController {
  constructor(env) {
    this.env = env;
    const j = env.jmap;
    this.ix = {
      lHip: j.left_hip_pitch_joint.c, lKnee: j.left_knee_joint.c, lAnkle: j.left_ankle_pitch_joint.c,
      rHip: j.right_hip_pitch_joint.c, rKnee: j.right_knee_joint.c, rAnkle: j.right_ankle_pitch_joint.c,
      waistYaw: j.waist_yaw_joint.c, waistPitch: j.waist_pitch_joint.c, waistRoll: j.waist_roll_joint.c,
      lShP: j.left_shoulder_pitch_joint.c, lShR: j.left_shoulder_roll_joint.c, lShY: j.left_shoulder_yaw_joint.c,
      lEl: j.left_elbow_joint.c, lWrP: j.left_wrist_pitch_joint.c,
      rShP: j.right_shoulder_pitch_joint.c, rShR: j.right_shoulder_roll_joint.c, rShY: j.right_shoulder_yaw_joint.c,
      rEl: j.right_elbow_joint.c, rWrP: j.right_wrist_pitch_joint.c, rWrR: j.right_wrist_roll_joint.c,
      rWrY: j.right_wrist_yaw_joint.c,
    };
    this.goal = Float64Array.from(HOME);
    this.smoothed = Float64Array.from(HOME);
    this.palmPrev = [0, 0, 0];
    this.palmVel = [0, 0, 0];
  }

  reset() {
    const { data, ballQadr } = this.env;
    this.state = 'carry';
    this.t = 0; this.tState = 0;
    this.shootAt = 5 + 3 * Math.random();
    this.freq = 1.55;
    this.goal.set(HOME); this.smoothed.set(HOME);
    this.placeBallAtSeat();
    const v = data.qvel; v[ballQadr] = 0; v[ballQadr + 1] = 0; v[ballQadr + 2] = 0;
    this.emit = null;
  }

  action() { this.requestShoot = true; }

  cameraPreset() {
    return { pos: [-2.4, -3.6, 2.1], target: [1.9, 0, 1.5] };
  }

  // ---------- 小工具 ----------
  // 托球点（球心应处的世界位置）：随腕系转动，手转到哪里球贴到哪里
  seat() {
    const { data, palmBid } = this.env;
    const o = 3 * palmBid, q = 4 * palmBid;
    const pos = [data.xpos[o], data.xpos[o + 1], data.xpos[o + 2]];
    const quat = [data.xquat[q], data.xquat[q + 1], data.xquat[q + 2], data.xquat[q + 3]];
    const off = quatRotVec(quat, SEAT_LOCAL);
    return [pos[0] + off[0], pos[1] + off[1], pos[2] + off[2]];
  }
  ballPos() {
    const { data, ballQadr } = this.env;
    const o = ballQadr;
    return [data.qpos[o], data.qpos[o + 1], data.qpos[o + 2]];
  }
  setBallPos(p) {
    const { data, ballQadr } = this.env;
    data.qpos[ballQadr] = p[0]; data.qpos[ballQadr + 1] = p[1]; data.qpos[ballQadr + 2] = p[2];
  }
  setBallVel(v) {
    const { data, ballDadr } = this.env;
    data.qvel[ballDadr] = v[0]; data.qvel[ballDadr + 1] = v[1]; data.qvel[ballDadr + 2] = v[2];
  }
  ballVel() {
    const { data, ballDadr } = this.env;
    return [data.qvel[ballDadr], data.qvel[ballDadr + 1], data.qvel[ballDadr + 2]];
  }
  placeBallAtSeat() {
    // 球心贴在掌心托球点（掌系放置，随手掌转动，视觉上托在手掌上）
    this.setBallPos(this.seat());
    this.setBallVel([0, 0, 0]);
  }
  // 持球时关闭球的碰撞（避免与手部碰撞体挤压爆炸），释放后恢复
  setBallCollide(on) {
    const { model, ballGid, ballContype } = this.env;
    model.geom_contype[ballGid] = on ? ballContype : 0;
    model.geom_conaffinity[ballGid] = on ? ballContype : 0;
  }

  // 球体安全钳制：接触求解在极端冲击下可能穿透地面，按状态分级处理
  ballSafety() {
    const p = this.ballPos();
    if (!isFinite(p[2])) {
      this.state = 'carry'; this.tState = 0;
      this.shootAt = this.t + 5.5 + 3.5 * Math.random();
      this.placeBallAtSeat();
      return;
    }
    if (p[2] > 5) {
      this.state = 'recover'; this.tState = 0.4;
      this.placeBallAtSeat();
      return;
    }
    if (p[2] < -0.4) {
      // 严重穿透：直接回到投篮收球流程（recover 完成时会重排 shootAt）
      if (this.state === 'flight' || this.state === 'recover') {
        this.state = 'recover'; this.tState = 0.45;
        this.placeBallAtSeat();
      } else if (this.state === 'free') {
        this.placeBallAtSeat();
        this.state = 'carry'; this.tState = 0;
      }
    } else if (this.state === 'free' && p[2] < -0.02) {
      // 轻微穿透：贴回地面并保留反弹
      const v = this.ballVel();
      this.setBallPos([p[0], p[1], 0.13]);
      this.setBallVel([v[0] * 0.5, v[1] * 0.5, Math.max(1.5, Math.abs(v[2]) * 0.4)]);
    }
  }

  // ---------- 主循环（每个物理子步调用一次） ----------
  step(dt) {
    this.t += dt; this.tState += dt;
    this.ballSafety();
    const P = this.seat();
    this.palmVel = [
      (P[0] - this.palmPrev[0]) / dt,
      (P[1] - this.palmPrev[1]) / dt,
      (P[2] - this.palmPrev[2]) / dt];
    this.palmPrev = P;

    const goal = this.goal; goal.set(HOME);

    let tau = 0.06;
    const dueToShoot = this.t > this.shootAt;
    const wantShoot = this.requestShoot || dueToShoot;

    if (this.state === 'carry') {
      // 运球：手臂下压 + 球黏在手掌上方跟随下压
      const φ = 2 * Math.PI * this.freq * this.tState;
      const s = Math.sin(φ);
      goal[this.ix.rShP] = -0.62 + 0.40 * s;
      goal[this.ix.rEl] = 0.85 - 0.32 * s;
      goal[this.ix.rWrP] = 0.45 - 0.30 * s;
      goal[this.ix.rShR] = -0.18;
      goal[this.ix.waistYaw] = -0.14 * s;
      goal[this.ix.waistPitch] = 0.12 + 0.05 * s;
      const knee = 0.3 + 0.11 * (0.5 - 0.5 * s);
      goal[this.ix.lHip] = -0.1 - 0.06 * (0.5 - 0.5 * s); goal[this.ix.lKnee] = knee; goal[this.ix.lAnkle] = -0.2 + 0.055 * (0.5 - 0.5 * s);
      goal[this.ix.rHip] = -0.1 - 0.06 * (0.5 - 0.5 * s); goal[this.ix.rKnee] = knee; goal[this.ix.rAnkle] = -0.2 + 0.055 * (0.5 - 0.5 * s);
      goal[this.ix.lShP] = -0.32 + 0.07 * s;
      goal[this.ix.lEl] = 0.9;
      tau = 0.07;
      // 球黏在手掌上，随手上抬/下压（无碰撞）
      this.setBallCollide(false);
      this.placeBallAtSeat(); this.setBallVel(this.palmVel);
      if (this.tState > 0.45) { // 手压到低位后放球
        const hv = this.palmVel;
        this.setBallCollide(true);
        this.setBallVel([hv[0] * 0.4 + (Math.random() - 0.5) * 0.06, hv[1] * 0.4 + (Math.random() - 0.5) * 0.06, -3.0]);
        this.state = 'free'; this.tState = 0; this.bounces = 0;
      }
      if (wantShoot && this.tState > 0.2) {
        this.state = 'windup'; this.tState = 0; this.requestShoot = false;
      }
    } else if (this.state === 'free') {
      // 球自由飞行/弹地，等待弹回到手的高度
      goal[this.ix.rShP] = -0.55; goal[this.ix.rEl] = 0.75;
      goal[this.ix.waistPitch] = 0.12;
      tau = 0.09;
      const p = this.ballPos(), v = this.ballVel();
      // 轻微水平归中，防止漂走
      const k = 2.2 * dt;
      this.setBallVel([v[0] - k * (v[0] * 0.8 + (p[0] - P[0])), v[1] - k * (v[1] * 0.8 + (p[1] - P[1])), v[2]]);
      // 落地助攻：弹起瞬间补足能量，保证节奏稳定（弹跳高度 ≈ 手高）
      if (p[2] < 0.3 && v[2] > 0.2 && v[2] < 3.2) {
        this.setBallVel([v[0] * 0.5, v[1] * 0.5, 3.35 + 0.25 * Math.random()]);
      }
      const p2 = this.ballPos(), v2 = this.ballVel();
      // 需在手掌附近才接球（水平 0.28m 内），避免远处瞬移
      const reached = v2[2] > 0.5 && p2[2] > P[2] - 0.30 &&
        Math.hypot(p2[0] - P[0], p2[1] - P[1]) < 0.28;
      if (reached || this.tState > 2.2) {
        this.state = 'carry'; this.tState = 0;
        this.freq = 1.45 + 0.25 * Math.random();
      }
    } else if (this.state === 'windup') {
      // 双手持球上举，同时旋腕让掌心转到球的下方（托球出手）
      const p = Math.min(1, this.tState / 0.45);
      const e = p * p * (3 - 2 * p);
      goal[this.ix.rShP] = -0.62 + (-1.35) * e; goal[this.ix.rEl] = (0.85) + (-0.35) * e;
      goal[this.ix.lShP] = -0.32 + (-0.95) * e; goal[this.ix.lEl] = 0.9 - 0.1 * e;
      goal[this.ix.rShR] = -0.18 - 0.1 * e; goal[this.ix.lShR] = 0.25 * e;
      goal[this.ix.waistPitch] = 0.12 - 0.08 * e;
      goal[this.ix.rKnee] = 0.3 + 0.08 * e; goal[this.ix.lKnee] = 0.3 + 0.08 * e;
      goal[this.ix.rWrR] = RELEASE_WRIST.r * e;
      goal[this.ix.rWrP] = 0.3 * (1 - e) + RELEASE_WRIST.p * e;
      goal[this.ix.rWrY] = RELEASE_WRIST.y * e;
      tau = 0.05;
      this.setBallCollide(false);
      this.placeBallAtSeat();
      if (this.tState >= 0.45) {
        // 出手：从当前位置解算一条过筐的抛物线
        const T = 0.82;
        const v0 = [
          (RIM[0] - P[0]) / T,
          (RIM[1] - P[1]) / T,
          (RIM[2] - P[2]) / T + 0.5 * G * T,
        ];
        this.setBallCollide(true);
        this.setBallVel(v0);
        this.relP = P.slice(); this.relV = v0; this.tF = 0; this.scored = false;
        this.state = 'flight'; this.tState = 0;
        if (this.env.onShot) this.env.onShot();
      }
    } else if (this.state === 'flight') {
      // 出手跟随动作（保持托球腕位，腕再前压一点作随挥）
      goal[this.ix.rShP] = -2.0; goal[this.ix.rEl] = 0.45;
      goal[this.ix.lShP] = -1.35; goal[this.ix.lEl] = 0.75;
      goal[this.ix.waistPitch] = 0.04;
      goal[this.ix.rWrR] = RELEASE_WRIST.r;
      goal[this.ix.rWrP] = RELEASE_WRIST.p + 0.2;
      goal[this.ix.rWrY] = RELEASE_WRIST.y;
      tau = 0.06;
      this.tF += dt;
      const T = 0.82;
      if (this.tF < T * 1.05) {
        // 温和引导：沿理想抛物线修正（演示用，保证过筐）
        const p = this.ballPos(), v = this.ballVel();
        for (let a = 0; a < 3; a++) {
          const ideal = this.relP[a] + this.relV[a] * this.tF - (a === 2 ? 0.5 * G * this.tF * this.tF : 0);
          const idealV = this.relV[a] - (a === 2 ? G * this.tF : 0);
          let acc = 5.0 * (ideal - p[a]) + 1.2 * (idealV - v[a]);
          acc = Math.max(-6, Math.min(6, acc));
          v[a] += acc * dt;
        }
        this.setBallVel(v);
      }
      // 计分：从上往下穿过筐心附近
      const p2 = this.ballPos();
      if (!this.scored && this.tF < 1.5 && p2[2] < RIM[2] && this.ballVel()[2] < 0 &&
          Math.hypot(p2[0] - RIM[0], p2[1] - RIM[1]) < 0.20) {
        this.scored = true;
        if (this.env.onScore) this.env.onScore();
      }
      // 进球且已落到较低处即收球，避免长时间高速下砸
      if (this.scored && this.tF > 1.3 && this.ballPos()[2] < 1.0) {
        this.state = 'recover'; this.tState = 0;
      } else if (this.tF > 1.5 && this.ballPos()[2] < 0.35) {
        this.state = 'recover'; this.tState = 0;
      } else if (this.tF > 3.0) {
        this.state = 'recover'; this.tState = 0;
      }
    } else if (this.state === 'recover') {
      goal[this.ix.rShP] = -0.4; goal[this.ix.rEl] = 1.0;
      tau = 0.09;
      if (this.tState > 0.5) {
        this.state = 'carry'; this.tState = 0;
        this.shootAt = this.t + 5.5 + 3.5 * Math.random();
        this.placeBallAtSeat();
      }
    }

    smoothInto(this.smoothed, goal, dt, tau);
    const ctrl = this.env.data.ctrl;
    for (let i = 0; i < 29; i++) ctrl[i] = this.smoothed[i];

    applyBalanceAssist(this.env.data, this.env.pelvisBid, 0, 0, 1.0);
  }
}
