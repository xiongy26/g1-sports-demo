// 乒乓球模式：机器人持拍与对手对打。
// 单人模式：对面是发球机，球为运动学脚本（分段抛物线），球拍 mocap 跟随右手腕。
// 双人模式（DuelController）：两台 G1 隔台正手斜线对拉，轨迹按台心点对称，
//   每次击球都有独立的"贴合吸附"与拍面标定，发球机隐藏。
import { smoothInto, quatMul, quatRotVec, PADDLE_GRIP_QUAT, setBasePose, applyBalanceAssist } from './util.js';

const READY = [
  -0.28, 0, 0, 0.55, -0.25, 0,   // 左腿
  -0.28, 0, 0, 0.55, -0.25, 0,   // 右腿
  0, 0, 0.18,                    // 腰
  -0.6, 0.25, 0, 1.1, 0, 0.3, 0, // 左臂
  -0.5, -0.15, 0.3, 1.3, -1.06, 0.45, 0, // 右臂（腕角标定：拍面正对来球）
];

const SWING_T = 0.62, CONTACT_FRAC = 0.55;
const CONTACT_BLEND_T = 0.12;   // 击球前把球吸附到拍面的时间窗

// 拍柄沿拍面内的 -y 延伸，对应腕系 -z；握柄横过收拢的手指。
// 真实掌面是腕系 +y；黑色击球面朝 -y，红面朝掌心。
export const PADDLE_OFFSET = [0.105, 0.025, 0.140]; // 拍柄中段在腕系 [0.105, 0.025, 0.015]
const BALL_R = 0.02, BLADE_HALF_T = 0.008;

// 单人模式对打关键点（世界系）：发球机 -> 弹跳 -> 机器人正手 -> 弹跳 -> 发球机
// A/E 取自发球机炮口（scene_gym.xml 中 machine_barrel 的出口）；
// C 取自球拍实际扫过轨迹的击球窗口中心（挥拍采样标定，运行时另有贴合吸附兜底）
const A = [-4.42, 0.25, 1.10];   // 机器出球（炮口）
const B = [-3.20, 0.25, 0.788];  // 对手台面弹跳
const C = [-1.43, 0.47, 0.85];   // 机器人击球点（拍面黑胶面上）
const D = [-3.30, 0.05, 0.788];  // 对手台面弹跳
const E = [-4.40, 0.23, 1.09];   // 机器击球点（回到炮口）

const LEGS = [
  { a: A, b: B, T: 0.36 },
  { a: B, b: C, T: 0.55 },   // 到达 C 时挥拍已触球
  { a: C, b: D, T: 0.50 },
  { a: D, b: E, T: 0.50 },
];

// 双人模式对打关键点：与单人模式关于台心 (-2.9, 0) 点对称（两人都用正手斜线）
// hit: 该段末端击球的机器人下标（-1 为弹跳段）
const DUEL_T = [0.52, 0.50, 0.50, 0.50];

function arcVel(a, b, T) {
  return [(b[0] - a[0]) / T, (b[1] - a[1]) / T, (b[2] - a[2]) / T + 0.5 * 9.81 * T];
}

// 单机的执行器索引（ctrl 下标 + 偏移；二号机执行器排在一号机之后）
function botIx(jmap, off) {
  const j = jmap;
  return {
    lKnee: j.left_knee_joint.c + off,
    rKnee: j.right_knee_joint.c + off,
    waistYaw: j.waist_yaw_joint.c + off,
    lShP: j.left_shoulder_pitch_joint.c + off, lShR: j.left_shoulder_roll_joint.c + off,
    rShP: j.right_shoulder_pitch_joint.c + off, rShR: j.right_shoulder_roll_joint.c + off,
    rShY: j.right_shoulder_yaw_joint.c + off, rEl: j.right_elbow_joint.c + off,
    rWrR: j.right_wrist_roll_joint.c + off, rWrP: j.right_wrist_pitch_joint.c + off,
  };
}

// 一台持拍机器人：关节目标平滑、正手挥拍、球拍跟随、平衡辅助
// 与具体对打脚本无关——球脚本由外层控制器驱动，通过 startSwing() 触发挥拍
class RallyBot {
  constructor(env, opts) {
    this.env = env;
    this.ctrlOff = opts.ctrlOff || 0;
    this.dofOff = opts.dofOff || 0;
    this.anchorX = opts.anchorX; this.anchorY = opts.anchorY;
    this.wristBid = opts.wristBid; this.paddleMid = opts.paddleMid;
    this.pelvisBid = opts.pelvisBid;
    this.ix = botIx(env.jmap, this.ctrlOff);
    this.goal = Float64Array.from(READY);
    this.smoothed = Float64Array.from(READY);
    this.t = 0; this.swingT = 0; this.swinging = false;
  }

  reset() {
    this.t = 0; this.swingT = 0; this.swinging = false;
    this.goal.set(READY); this.smoothed.set(READY);
  }

  startSwing() {
    if (!this.swinging) { this.swinging = true; this.swingT = 0; }
  }

  step(dt) {
    this.t += dt;
    const goal = this.goal; goal.set(READY);
    // 呼吸起伏
    const bob = 0.5 + 0.5 * Math.sin(2 * Math.PI * 0.9 * this.t);
    goal[this.ix.lKnee] = 0.55 + 0.03 * bob;
    goal[this.ix.rKnee] = 0.55 + 0.03 * bob;

    // 挥拍：w=引拍(至击球姿态, p=CONTACT_FRAC 时球到位) s=击球穿过 f=随挥
    // 击球姿态的手腕角经标定：拍面法向对准出球方向
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
      goal[this.ix.rWrR] = -1.16 - 0.25 * s + 0.2 * f; // 击球瞬间拍面对准出球方向（标定值）
      goal[this.ix.rWrP] = -0.15 * s;
      goal[this.ix.lShP] = -0.6 - 0.25 * s;
      goal[this.ix.lShR] = 0.25 + 0.15 * s;
      if (this.swingT >= SWING_T) this.swinging = false; // 随挥完成后回位
    }

    smoothInto(this.smoothed, goal, dt, this.swinging ? 0.025 : 0.08);
    const ctrl = this.env.data.ctrl;
    for (let i = 0; i < 29; i++) ctrl[this.ctrlOff + i] = this.smoothed[i];

    this.updatePaddle();
    applyBalanceAssist(this.env.data, this.pelvisBid, this.anchorX, this.anchorY, 1.0, this.dofOff);
  }

  // 拍面黑胶面上的贴合点（球心位置）：拍心 + 法向 * (拍半厚 + 球半径)
  paddleFacePoint() {
    const { data } = this.env;
    const m = this.paddleMid;
    const p = [data.mocap_pos[3 * m], data.mocap_pos[3 * m + 1], data.mocap_pos[3 * m + 2]];
    const q = [data.mocap_quat[4 * m], data.mocap_quat[4 * m + 1], data.mocap_quat[4 * m + 2], data.mocap_quat[4 * m + 3]];
    const n = quatRotVec(q, [0, 0, 1]);
    const d = BLADE_HALF_T + BALL_R;
    return [p[0] + n[0] * d, p[1] + n[1] * d, p[2] + n[2] * d];
  }

  updatePaddle() {
    const { data } = this.env;
    const o = 3 * this.wristBid, q = 4 * this.wristBid;
    const wpos = [data.xpos[o], data.xpos[o + 1], data.xpos[o + 2]];
    const wquat = [data.xquat[q], data.xquat[q + 1], data.xquat[q + 2], data.xquat[q + 3]];
    const off = quatRotVec(wquat, PADDLE_OFFSET);
    data.mocap_pos[3 * this.paddleMid] = wpos[0] + off[0];
    data.mocap_pos[3 * this.paddleMid + 1] = wpos[1] + off[1];
    data.mocap_pos[3 * this.paddleMid + 2] = wpos[2] + off[2];
    const pq = quatMul(wquat, PADDLE_GRIP_QUAT);
    data.mocap_quat[4 * this.paddleMid] = pq[0];
    data.mocap_quat[4 * this.paddleMid + 1] = pq[1];
    data.mocap_quat[4 * this.paddleMid + 2] = pq[2];
    data.mocap_quat[4 * this.paddleMid + 3] = pq[3];
  }
}

// ================= 单人模式：vs 发球机 =================
export class PingpongController {
  constructor(env) {
    this.env = env;
    this.bot = new RallyBot(env, {
      ctrlOff: 0, dofOff: 0, anchorX: -1.15, anchorY: 0,
      wristBid: env.wristBid, paddleMid: env.paddleMid, pelvisBid: env.pelvisBid,
    });
    this.goal = this.bot.goal; this.smoothed = this.bot.smoothed; // 兼容旧引用
  }

  reset() {
    setBasePose(this.env.data, -1.15, 0, 0.783675, Math.PI);
    this.state = 'serve';
    this.t = 0; this.tState = 0;
    this.leg = 0; this.legT = 0;
    this.ballStart = A.slice();
    this.swingArmed = false;
    this.rallyCount = 0;
    this.bot.reset();
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

  step(dt) {
    this.t += dt; this.tState += dt;

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
      const v0 = arcVel(L.a, L.b, L.T);
      const t = Math.min(this.legT, L.T);
      const bp = [
        L.a[0] + v0[0] * t,
        L.a[1] + v0[1] * t,
        L.a[2] + v0[2] * t - 4.905 * t * t,
      ];
      // 击球贴合：leg1 尾段把球平滑吸附到拍面黑胶面上（消除标定残差，杜绝穿模/悬空）
      if (this.leg === 1 && this.legT > L.T - CONTACT_BLEND_T) {
        const k = (this.legT - (L.T - CONTACT_BLEND_T)) / CONTACT_BLEND_T;
        const e = k * k * (3 - 2 * k);
        const face = this.bot.paddleFacePoint();
        for (let a = 0; a < 3; a++) bp[a] += (face[a] - bp[a]) * e;
      }
      this.writeBall(bp);
      // 准备挥拍：到达击球点前 0.34s 启动
      if (this.leg === 1 && !this.swingArmed) {
        const tSwingStart = L.T - SWING_T * CONTACT_FRAC;
        if (this.legT >= tSwingStart) { this.bot.startSwing(); this.swingArmed = true; }
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
        if (this.leg === 2) { // 击球完成：出球弧线从拍面实际位置起算（无跳变）
          this.legs[2] = { a: this.bot.paddleFacePoint(), b: D, T: LEGS[2].T };
        }
      }
    }

    this.bot.step(dt);
  }
}

// ================= 双人模式：两台 G1 隔台对拉 =================
export class DuelController {
  constructor(env) {
    this.env = env;
    this.bots = [
      new RallyBot(env, {
        ctrlOff: 0, dofOff: 0, anchorX: -1.15, anchorY: 0,
        wristBid: env.wristBid, paddleMid: env.paddleMid, pelvisBid: env.pelvisBid,
      }),
      new RallyBot(env, {
        ctrlOff: 29, dofOff: env.r2Dof, anchorX: -4.65, anchorY: 0,
        wristBid: env.r2WristBid, paddleMid: env.paddleMid2, pelvisBid: env.r2PelvisBid,
      }),
    ];
  }

  reset() {
    setBasePose(this.env.data, -1.15, 0, 0.783675, Math.PI); // 一号机
    setBasePose(this.env.data, -4.65, 0, 0.783675, 0, this.env.r2Qadr, this.env.r2Dof); // 二号机
    this.state = 'serve';
    this.tState = 0; this.leg = 0; this.legT = 0;
    this.hits = 0;
    this.legs = this.makeLegs();
    this.bots.forEach((b) => b.reset());
    this.writeBall([-4.37, -0.47, 0.85]);
  }

  action() { // 重新发球
    this.state = 'serve'; this.tState = 0;
    this.leg = 0; this.legT = 0;
    this.legs = this.makeLegs();
  }

  cameraPreset() {
    return { pos: [-2.9, -5.4, 2.5], target: [-2.9, 0, 0.9] };
  }

  // 一分的多段轨迹：二号机击球 -> 一号机半台弹跳 -> 一号机击球 -> 二号机半台弹跳 -> 循环
  makeLegs() {
    const jy = () => (Math.random() - 0.5) * 0.06;
    const c = [-1.43, 0.47 + jy(), 0.85];   // 一号机击球点
    const c2 = [-4.37, -0.47 + jy(), 0.85]; // 二号机击球点
    return [
      { a: c2, b: [-2.50, -0.05, 0.788], T: DUEL_T[0], hit: -1 },
      { a: [-2.50, -0.05, 0.788], b: c, T: DUEL_T[1], hit: 0 },
      { a: c, b: [-3.30, 0.05, 0.788], T: DUEL_T[2], hit: -1 },
      { a: [-3.30, 0.05, 0.788], b: c2, T: DUEL_T[3], hit: 1 },
    ];
  }

  writeBall(p) {
    const mid = this.env.ttBallMid;
    this.env.data.mocap_pos[3 * mid] = p[0];
    this.env.data.mocap_pos[3 * mid + 1] = p[1];
    this.env.data.mocap_pos[3 * mid + 2] = p[2];
  }

  step(dt) {
    this.tState += dt;

    if (this.state === 'serve') {
      // 发球：球托在二号机拍面上，随后从拍面击出
      this.bots.forEach((b) => b.step(dt));
      this.writeBall(this.bots[1].paddleFacePoint());
      if (this.tState > 0.7) {
        this.state = 'rally';
        this.leg = 0; this.legT = 0;
        this.legs = this.makeLegs();
        this.legs[0].a = this.bots[1].paddleFacePoint();
      }
      return;
    }

    const L = this.legs[this.leg];
    this.legT += dt;
    const v0 = arcVel(L.a, L.b, L.T);
    const t = Math.min(this.legT, L.T);
    const bp = [
      L.a[0] + v0[0] * t,
      L.a[1] + v0[1] * t,
      L.a[2] + v0[2] * t - 4.905 * t * t,
    ];
    // 击球贴合：击球段尾段把球吸附到击球者的拍面上
    if (L.hit >= 0 && this.legT > L.T - CONTACT_BLEND_T) {
      const k = (this.legT - (L.T - CONTACT_BLEND_T)) / CONTACT_BLEND_T;
      const e = k * k * (3 - 2 * k);
      const face = this.bots[L.hit].paddleFacePoint();
      for (let a = 0; a < 3; a++) bp[a] += (face[a] - bp[a]) * e;
    }
    this.writeBall(bp);
    // 挥拍触发：到达击球点前 0.34s
    if (L.hit >= 0 && !L.armed && this.legT >= L.T - SWING_T * CONTACT_FRAC) {
      L.armed = true;
      this.bots[L.hit].startSwing();
    }
    if (this.legT >= L.T) {
      this.legT -= L.T;
      if (L.hit >= 0) {
        // 击球完成：下一段从拍面实际位置起算（无跳变）
        this.hits++;
        if (this.env.onRally) this.env.onRally(this.hits);
        this.legs[(this.leg + 1) % 4].a = this.bots[L.hit].paddleFacePoint();
      }
      this.leg = (this.leg + 1) % 4;
    }
    this.bots.forEach((b) => b.step(dt));
  }
}
