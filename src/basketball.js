// 篮球演示：短促下压、自由反弹、收球蓄力、蹬伸出手与随挥。
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

// 托球位姿（right_wrist_yaw_link 系）：掌面 = +y（由手指向 +y 弯曲的网格确认），
// 锚点取掌心表面；球心 = 锚点沿掌面法向外推 (球半径 - 陷入量)，视觉上托在掌心。
export const PALM_NORMAL_LOCAL = [0, 1, 0];
const SEAT_LOCAL = [0.10, 0.015 + (0.123 - 0.005), 0.008];
const DRIBBLE_WRIST = { r: -1.35, p: 0.05, y: -0.35 };
// 蹬伸结束时的腕角：按实际出手肩肘姿势标定，让掌面朝向出球方向。
const RELEASE_WRIST = { r: -1.15, p: 0.45, y: -1.35 };

export class BasketballController {
  constructor(env) {
    this.env = env;
    // 持球是掌系约束；释放后球只碰场地/篮架，避免刚释放时
    // 球与机器人手臂网格重叠产生冲击。保留原有场地碰撞掩码。
    const { model } = env;
    for (let g = 0; g < model.ngeom; g++) {
      if (model.body_mass[model.geom_bodyid[g]] === 0 && model.geom_contype[g] !== 0) {
        model.geom_conaffinity[g] |= 2;
      }
    }
    const j = env.jmap;
    this.ix = {
      lHip: j.left_hip_pitch_joint.c, lKnee: j.left_knee_joint.c, lAnkle: j.left_ankle_pitch_joint.c,
      rHip: j.right_hip_pitch_joint.c, rKnee: j.right_knee_joint.c, rAnkle: j.right_ankle_pitch_joint.c,
      waistYaw: j.waist_yaw_joint.c, waistPitch: j.waist_pitch_joint.c, waistRoll: j.waist_roll_joint.c,
      lShP: j.left_shoulder_pitch_joint.c, lShR: j.left_shoulder_roll_joint.c, lShY: j.left_shoulder_yaw_joint.c,
      lEl: j.left_elbow_joint.c, lWrP: j.left_wrist_pitch_joint.c,
      lWrR: j.left_wrist_roll_joint.c, lWrY: j.left_wrist_yaw_joint.c,
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
    this.state = 'carry';
    this.t = 0; this.tState = 0;
    this.shootAt = 6.5;
    this.requestShoot = false;
    this.palmPrev = this.seat();
    this.goal.set(HOME);
    this.goal[this.ix.rShP] = -0.85; this.goal[this.ix.rShR] = -0.18;
    this.goal[this.ix.rEl] = 0.90;
    this.goal[this.ix.rWrR] = DRIBBLE_WRIST.r;
    this.goal[this.ix.rWrP] = DRIBBLE_WRIST.p;
    this.goal[this.ix.rWrY] = DRIBBLE_WRIST.y;
    this.smoothed.set(this.goal);
    for (const joint of Object.values(this.env.jmap)) {
      this.env.data.qpos[joint.q] = this.goal[joint.c];
      this.env.data.ctrl[joint.c] = this.goal[joint.c];
    }
    this.env.mujoco.mj_forward(this.env.model, this.env.data);
    this.palmPrev = this.seat();
    this.setBallCollide(false);
    this.placeBallAtSeat();
    this.setBallVel([0, 0, 0]);
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
    const { model, ballGid } = this.env;
    model.geom_contype[ballGid] = on ? 2 : 0;
    model.geom_conaffinity[ballGid] = 0;
  }

  // 球体安全钳制：接触求解在极端冲击下可能穿透地面，按状态分级处理
  ballSafety() {
    const p = this.ballPos();
    if (!isFinite(p[2])) {
      this.state = 'carry'; this.tState = 0;
      this.shootAt = this.t + 7.5;
      this.placeBallAtSeat();
      return;
    }
    if (p[2] > 5) {
      this.state = 'recover'; this.tState = 0.4;
      this.placeBallAtSeat();
      return;
    }
    if ((this.state === 'free' || this.state === 'flight' || this.state === 'recover') && p[2] < 0.12) {
      // 仅修复离散碰撞漏检，不在空中改轨迹。
      const v = this.ballVel();
      this.setBallPos([p[0], p[1], 0.123]);
      this.setBallVel([v[0] * 0.85, v[1] * 0.85, v[2] < 0 ? -v[2] * 0.72 : v[2]]);
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
      // 短接触下压，掌面朝下；随后释放，让球自由落地反弹。
      const phase = Math.min(1, this.tState / 0.16);
      const s = Math.cos(Math.PI * phase);
      goal[this.ix.rShP] = -0.85 + 0.50 * (0.5 - 0.5 * s);
      goal[this.ix.rEl] = 0.90 - 0.30 * (0.5 - 0.5 * s);
      goal[this.ix.rWrP] = DRIBBLE_WRIST.p;
      goal[this.ix.rWrR] = DRIBBLE_WRIST.r; goal[this.ix.rWrY] = DRIBBLE_WRIST.y;
      goal[this.ix.rShR] = -0.18;
      goal[this.ix.waistYaw] = -0.04;
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
      if (this.tState > 0.16) { // 手压到低位后放球
        const hv = this.palmVel;
        this.setBallCollide(true);
        this.setBallVel([hv[0] * 0.4, hv[1] * 0.4, -2.8]);
        this.state = 'free'; this.tState = 0; this.bounces = 0;
      }
      if (wantShoot && this.state === 'carry') {
        this.state = 'gather'; this.tState = 0; this.requestShoot = false;
        this.shotStart = this.smoothed.slice();
      }
    } else if (this.state === 'free') {
      // 球自由飞行/弹地，等待弹回到手的高度
      const rise = Math.max(0, Math.min(1, this.ballVel()[2] / 3.5));
      goal[this.ix.rShP] = -0.35 - 0.50 * rise; goal[this.ix.rEl] = 0.60 + 0.30 * rise;
      goal[this.ix.rShR] = -0.18; goal[this.ix.rWrP] = DRIBBLE_WRIST.p;
      goal[this.ix.rWrR] = DRIBBLE_WRIST.r; goal[this.ix.rWrY] = DRIBBLE_WRIST.y;
      goal[this.ix.lShP] = -0.32; goal[this.ix.lEl] = 0.9;
      goal[this.ix.waistPitch] = 0.12;
      tau = 0.09;
      const p = this.ballPos(), v = this.ballVel();
      // 轻微水平归中，防止漂走
      const k = 2.2 * dt;
      this.setBallVel([v[0] - k * (v[0] * 0.8 + (p[0] - P[0])), v[1] - k * (v[1] * 0.8 + (p[1] - P[1])), v[2]]);
      // 落地助攻：弹起瞬间补足能量，保证节奏稳定（弹跳高度 ≈ 手高）
      if (p[2] < 0.3 && v[2] > 0.2 && v[2] < 3.2) {
        this.setBallVel([v[0] * 0.5, v[1] * 0.5, 3.35]);
      }
      const p2 = this.ballPos(), v2 = this.ballVel();
      // 需在手掌附近才接球（水平 0.28m 内），避免远处瞬移
      const reached = v2[2] > 0.5 && p2[2] > P[2] - 0.08 &&
        Math.hypot(p2[0] - P[0], p2[1] - P[1]) < 0.28;
      if (reached) {
        this.state = 'carry'; this.tState = 0;
      }
      if (this.tState > 2.2) { this.state = 'recover'; this.tState = 0; }
    } else if (this.state === 'gather' || this.state === 'windup') {
      // 收球停顿后从屈膝到蹬伸：避免肩肘同时甩直。
      const gathering = this.state === 'gather';
      const duration = gathering ? 0.48 : 0.32;
      const u = Math.min(1, this.tState / duration);
      const e = u * u * (3 - 2 * u);
      const setPose = { rShP: -1.28, rShR: -0.18, rEl: 1.15,
        lShP: -1.27, lShR: 0.11, lShY: -1.02, lEl: 1.34,
        lWrR: 1.89, lWrP: 0.08, lWrY: -0.29,
        rWrR: RELEASE_WRIST.r, rWrP: RELEASE_WRIST.p, rWrY: RELEASE_WRIST.y,
        waistPitch: 0.08, waistYaw: 0, lHip: -0.24, rHip: -0.24,
        lKnee: 0.52, rKnee: 0.52, lAnkle: -0.28, rAnkle: -0.28 };
      const releasePose = { ...setPose, rShP: -1.98, rEl: 0.32,
        lShP: -1.45, lEl: 0.95, lShR: 0.38, lShY: -0.65, lWrR: 1.10,
        lHip: -0.08, rHip: -0.08, lKnee: 0.22, rKnee: 0.22,
        lAnkle: -0.14, rAnkle: -0.14, waistPitch: 0.015 };
      for (const name of Object.keys(setPose)) {
        const ix = this.ix[name];
        const start = gathering ? this.shotStart[ix] : setPose[name];
        goal[ix] = start + ((gathering ? setPose[name] : releasePose[name]) - start) * e;
      }
      tau = 0.035;
      this.setBallCollide(false);
      this.placeBallAtSeat();
      this.setBallVel(this.palmVel);
      if (gathering && u === 1) { this.state = 'windup'; this.tState = 0; }
      if (!gathering && this.tState >= duration + 0.08) {
        // 出手：从当前位置解算一条过筐的抛物线
        const T = 0.90;
        const v0 = [
          (RIM[0] - P[0]) / T,
          (RIM[1] - P[1]) / T,
          (RIM[2] - P[2]) / T + 0.5 * G * T,
        ];
        this.setBallCollide(true);
        this.setBallVel(v0);
        this.tF = 0; this.scored = false;
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
      // 引导手先离球，投篮手维持伸展，腕部压下，然后自然收臂。
      const lower = Math.min(1, Math.max(0, (this.tState - 0.55) / 0.7));
      goal[this.ix.rShP] = -2.0 + 1.55 * lower;
      goal[this.ix.rEl] = 0.25 + 0.70 * lower;
      goal[this.ix.lShP] = -1.35 + 1.03 * Math.min(1, this.tState / 0.35);
      goal[this.ix.lEl] = 0.95;
      goal[this.ix.rWrP] = RELEASE_WRIST.p + 0.55 * Math.min(1, this.tState / 0.12);
      this.tF += dt;
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
      goal[this.ix.rShP] = -0.85; goal[this.ix.rEl] = 0.90;
      goal[this.ix.rShR] = -0.18;
      goal[this.ix.rWrR] = DRIBBLE_WRIST.r;
      goal[this.ix.rWrP] = DRIBBLE_WRIST.p;
      goal[this.ix.rWrY] = DRIBBLE_WRIST.y;
      tau = 0.09;
      if (this.tState > 0.5) {
        this.state = 'carry'; this.tState = 0;
        this.shootAt = this.t + 7.5;
        this.placeBallAtSeat();
      }
    }

    smoothInto(this.smoothed, goal, dt, tau);
    const ctrl = this.env.data.ctrl;
    for (let i = 0; i < 29; i++) ctrl[i] = this.smoothed[i];

    applyBalanceAssist(this.env.data, this.env.pelvisBid, 0, 0, 1.0);
  }
}
