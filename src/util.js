// 共享工具：关节索引表、四元数数学、平衡辅助、基座位姿设置
export function mjName2id(mujoco, model, objType, name) {
  return mujoco.mj_name2id(model, objType, name);
}

// name -> {q: qposadr, d: dofadr, c: actuator adr}
export function buildJointMap(mujoco, model, jointNames) {
  const mjOBJ_JOINT = mujoco.mjtObj.mjOBJ_JOINT.value;
  const mjOBJ_ACTUATOR = mujoco.mjtObj.mjOBJ_ACTUATOR.value;
  const map = {};
  for (const name of jointNames) {
    const jid = mjName2id(mujoco, model, mjOBJ_JOINT, name);
    const aid = mjName2id(mujoco, model, mjOBJ_ACTUATOR, name);
    if (jid < 0 || aid < 0) throw new Error('joint/actuator not found: ' + name);
    map[name] = {
      q: model.jnt_qposadr[jid],
      d: model.jnt_dofadr[jid],
      c: aid,
    };
  }
  return map;
}

export function bodyId(mujoco, model, name) {
  return mjName2id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY.value, name);
}
export function jointId(mujoco, model, name) {
  return mjName2id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT.value, name);
}
export function keyId(mujoco, model, name) {
  return mjName2id(mujoco, model, mujoco.mjtObj.mjOBJ_KEY.value, name);
}
export function geomId(mujoco, model, name) {
  return mjName2id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM.value, name);
}

// ---------- 四元数（mujoco wxyz 约定） ----------
export function quatRotVec(q, v) {
  // 返回 q*v*q^-1
  const [w, x, y, z] = q;
  const [vx, vy, vz] = v;
  // t = 2 q_vec × v
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}
export function quatConj(q) { return [q[0], -q[1], -q[2], -q[3]]; }
export function quatMul(a, b) {
  const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}
export function yawQuat(yaw) { return [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)]; }
// 正手握拍：把球拍圆柱轴(拍面法向 +z，+z 为黑胶击球面)转到腕系 -y（掌面法向），
// 拍面与掌面共面、拍柄穿过拳头——外观即"握手式持拍"，击球面与掌心同侧。
export const PADDLE_GRIP_QUAT = [0.7071068, 0.7071068, 0, 0];

// ---------- 平衡辅助（演示外挂：把躯干轻轻拉回锚点并保持直立） ----------
// dofOff：机器人基座的 dof 偏移（一号机 0，二号机由 main.js/sim_check 按模型查询）
export function applyBalanceAssist(data, pelvisBid, anchorX, anchorY, strength = 1.0, dofOff = 0) {
  const px = data.xpos[3 * pelvisBid], py = data.xpos[3 * pelvisBid + 1];
  const vx = data.qvel[dofOff], vy = data.qvel[dofOff + 1];
  let fx = 140 * (anchorX - px) - 45 * vx;
  let fy = 140 * (anchorY - py) - 45 * vy;
  const cap = 55 * strength;
  fx = Math.max(-cap, Math.min(cap, fx));
  fy = Math.max(-cap, Math.min(cap, fy));
  data.qfrc_applied[dofOff] = fx;
  data.qfrc_applied[dofOff + 1] = fy;

  // 直立力矩：用重力在机体系的投影作为倾斜误差（世界系力矩换算到机体系施加）
  const q = [data.xquat[4 * pelvisBid], data.xquat[4 * pelvisBid + 1], data.xquat[4 * pelvisBid + 2], data.xquat[4 * pelvisBid + 3]];
  const w = [data.qvel[dofOff + 3], data.qvel[dofOff + 4], data.qvel[dofOff + 5]]; // 机体系角速度
  const gravBody = quatRotVec(quatConj(q), [0, 0, -1]);
  const kTilt = 90 * strength, kDamp = 14 * strength, capT = 22 * strength;
  let tx = kTilt * gravBody[1] - kDamp * w[0];
  let ty = -kTilt * gravBody[0] - kDamp * w[1];
  let tz = -kDamp * w[2] * 0.5;
  tx = Math.max(-capT, Math.min(capT, tx));
  ty = Math.max(-capT, Math.min(capT, ty));
  tz = Math.max(-capT, Math.min(capT, tz));
  data.qfrc_applied[dofOff + 3] = tx;
  data.qfrc_applied[dofOff + 4] = ty;
  data.qfrc_applied[dofOff + 5] = tz;
}

// 把基座（自由关节）放到 world 位姿，其余关节保持
// qadr/dadr：基座的 qpos/dof 偏移（一号机 0/0，二号机由 main.js/sim_check 按模型查询）
export function setBasePose(data, x, y, z, yaw, qadr = 0, dadr = 0) {
  data.qpos[qadr] = x; data.qpos[qadr + 1] = y; data.qpos[qadr + 2] = z;
  const q = yawQuat(yaw);
  data.qpos[qadr + 3] = q[0]; data.qpos[qadr + 4] = q[1]; data.qpos[qadr + 5] = q[2]; data.qpos[qadr + 6] = q[3];
  for (let i = 0; i < 6; i++) data.qvel[dadr + i] = 0;
}

// 场边待机姿态（29 执行器目标）：双臂略前伸的放松站立，非对打模式下二号机用它待机
export const REST_POSE = [
  -0.1, 0, 0, 0.3, -0.2, 0,   // 左腿
  -0.1, 0, 0, 0.3, -0.2, 0,   // 右腿
  0, 0, 0,                     // 腰
  0.2, 0.2, 0, 1.28, 0, 0, 0,  // 左臂
  0.2, -0.2, 0, 1.28, 0, 0, 0, // 右臂
];

// 指数平滑目标：cur += (goal-cur) * min(1, dt/tau)
export function smoothInto(cur, goal, dt, tau) {
  const a = Math.min(1, dt / tau);
  for (let i = 0; i < goal.length; i++) cur[i] += (goal[i] - cur[i]) * a;
}

export const G = 9.81;
