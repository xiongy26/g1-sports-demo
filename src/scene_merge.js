// 模型组装：把 g1.xml 合并进 scene_gym.xml，并复制出一台"二号机"（r2_ 前缀）供双机对打使用。
// 纯字符串变换，浏览器（src/main.js）与 node（tools/sim_check.mjs、tools/probe_contact.mjs）共用。
// 二号机与一号机共享网格资产与 <default> 类；传感器/关键帧不复制。

// 二号机默认站位：球台另一端（与一号机关于台心点对称）
export const R2_DUEL_POS = [-4.65, 0, 0.783675];

function prefixNames(text) {
  return text.replace(/name="/g, 'name="r2_').replace(/joint="/g, 'joint="r2_');
}

export function buildModelXml(sceneXml, g1Xml) {
  // 1) 常规 include 合并（一号机）
  const g1Inner = g1Xml.replace(/^[\s\S]*?<mujoco[^>]*>/, '').replace(/<\/mujoco>\s*$/, '');
  let merged = sceneXml.replace('<include file="g1.xml"/>', g1Inner);

  // 2) 提取一号机 pelvis 子树与 actuator 列表，加前缀后作为二号机插入
  const pelvisStart = g1Xml.indexOf('<body name="pelvis"');
  const worldEnd = g1Xml.indexOf('</worldbody>');
  const actStart = g1Xml.indexOf('<actuator>');
  const actEnd = g1Xml.indexOf('</actuator>');
  if (pelvisStart < 0 || worldEnd < 0 || actStart < 0) throw new Error('g1.xml 结构不符合预期');

  let r2Body = prefixNames(g1Xml.slice(pelvisStart, worldEnd));
  r2Body = r2Body.replace(
    '<body name="r2_pelvis" pos="0 0 0.793"',
    `<body name="r2_pelvis" pos="${R2_DUEL_POS[0]} ${R2_DUEL_POS[1]} ${R2_DUEL_POS[2]}"`);
  const r2Act = prefixNames(g1Xml.slice(actStart + '<actuator>'.length, actEnd));

  merged = merged.replace('</worldbody>', r2Body + '</worldbody>');
  merged = merged.replace('</mujoco>', `  <actuator>\n${r2Act}\n  </actuator>\n</mujoco>`);
  return merged;
}
