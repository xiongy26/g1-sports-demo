// three.js 渲染器：直接从 MjModel/MjData 构建 geom 网格并逐帧同步位姿。
// 不经过 mjvScene，直接使用 geom_xpos / geom_xmat 实时视图（零拷贝）。
import * as THREE from 'three';

const mjGEOM = { PLANE: 0, HFIELD: 1, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6, MESH: 7 };

export class MujocoVisualizer {
  constructor(mujoco, model, scene) {
    this.mujoco = mujoco;
    this.model = model;
    this.scene = scene;
    this.meshes = [];       // geom index -> THREE.Mesh | null
    this.mats = [];
    this.ballMesh = null;   // 篮球（加筋线用）

    this._m4 = new THREE.Matrix4();
    this.build(model, scene);
  }

  build(model, scene) {
    const n = model.ngeom;
    const gtype = this.view(model.geom_type);
    const gsize = this.view(model.geom_size, 3);
    const grgba = this.view(model.geom_rgba, 4);
    const ggroup = this.view(model.geom_group);
    const gdataid = this.view(model.geom_dataid);
    const gbody = this.view(model.geom_bodyid);
    const bpos = this.view(model.body_pos, 3);
    const bquat = this.view(model.body_quat, 4);

    for (let i = 0; i < n; i++) {
      // 跳过碰撞凸包 (group 3) 与辅助组：只渲染视觉网格 (group 2) 和场景道具 (group 0)
      if (ggroup[i] === 3 || ggroup[i] === 4 || ggroup[i] === 5) { this.meshes[i] = null; continue; }
      const t = gtype[i];
      const s = [gsize[3 * i], gsize[3 * i + 1], gsize[3 * i + 2]];
      let geo = null;
      if (t === mjGEOM.PLANE) {
        this.meshes[i] = null;
        this.buildFloor(scene);
        continue;
      } else if (t === mjGEOM.SPHERE) geo = new THREE.SphereGeometry(s[0], 32, 20);
      else if (t === mjGEOM.CAPSULE) {
        geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 14);
        geo.rotateX(Math.PI / 2); // mujoco capsule 轴为 z
      } else if (t === mjGEOM.CYLINDER) {
        geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[2], 24);
        geo.rotateX(Math.PI / 2);
      } else if (t === mjGEOM.BOX) geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
      else if (t === mjGEOM.MESH) geo = this.meshGeometry(model, gdataid[i]);
      else { this.meshes[i] = null; continue; }

      const rgba = [grgba[4 * i], grgba[4 * i + 1], grgba[4 * i + 2], grgba[4 * i + 3]];
      const dark = rgba[0] + rgba[1] + rgba[2] < 0.6;
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
        roughness: dark ? 0.55 : 0.7,
        metalness: dark ? 0.15 : 0.05,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      scene.add(mesh);
      this.meshes[i] = mesh;
      this.mats[i] = mat;
      if (gbody[i] > 0 && this.bodyName(model, gbody[i]) === 'basketball') {
        this.ballMesh = mesh;
        this.addBallSeams(mesh);
      }
    }
    void bpos; void bquat;
  }

  view(v, stride = 1) { return v; } // 绑定返回的就是实时 TypedArray 视图
  bodyName(model, id) {
    try { return model.body(id).name; } catch { return ''; }
  }

  meshGeometry(model, meshId) {
    const vadr = model.mesh_vertadr[meshId], vnum = model.mesh_vertnum[meshId];
    const fadr = model.mesh_faceadr[meshId], fnum = model.mesh_facenum[meshId];
    const verts = model.mesh_vert, faces = model.mesh_face;
    const pos = new Float32Array(vnum * 3);
    for (let i = 0; i < vnum * 3; i++) pos[i] = verts[vadr * 3 + i];
    const idx = new Uint32Array(fnum * 3);
    for (let i = 0; i < fnum * 3; i++) idx[i] = faces[fadr * 3 + i];
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    return geo;
  }

  addBallSeams(ballMesh) {
    // 篮球筋线：两个正交圆环
    const mat = new THREE.MeshBasicMaterial({ color: 0x201005 });
    for (const rot of [0, Math.PI / 2]) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.1225, 0.004, 6, 40), mat);
      ring.rotation.y = rot;
      ballMesh.add(ring);
    }
    const ring3 = new THREE.Mesh(new THREE.TorusGeometry(0.1225, 0.004, 6, 40), mat);
    ring3.rotation.x = Math.PI / 2;
    ballMesh.add(ring3);
  }

  buildFloor(scene) {
    const W = 34, D = 24, S = 60; // 米 / 像素
    const cv = document.createElement('canvas');
    cv.width = W * S; cv.height = D * S;
    const g = cv.getContext('2d');
    // 木地板
    g.fillStyle = '#c9a06b'; g.fillRect(0, 0, cv.width, cv.height);
    for (let y = 0; y < cv.height; y += 14) {
      g.fillStyle = `rgba(120,80,40,${0.04 + 0.05 * Math.random()})`;
      g.fillRect(0, y, cv.width, 14);
      g.fillStyle = 'rgba(60,35,15,0.18)';
      g.fillRect(0, y, cv.width, 1);
    }
    for (let i = 0; i < 500; i++) { // 木纹噪点
      g.fillStyle = `rgba(90,55,25,${0.03 * Math.random()})`;
      g.fillRect(Math.random() * cv.width, Math.random() * cv.height, 60 + Math.random() * 120, 2);
    }
    const X = (wx) => (wx + W / 2) * S;
    const Y = (wy) => (D / 2 - wy) * S;
    // 篮球场地画线（以篮筐 x=2.125 为基准）
    g.strokeStyle = '#f5f5f2'; g.lineWidth = 4.5;
    g.beginPath(); g.arc(X(0), Y(0), 1.0 * S, 0, Math.PI * 2); g.stroke(); // 中圈
    g.beginPath(); g.moveTo(X(0), Y(-3.2)); g.lineTo(X(0), Y(3.2)); g.stroke(); // 中线
    g.strokeRect(X(1.0), Y(-0.9), 1.6 * S, 1.8 * S); // 罚球区
    g.beginPath(); g.arc(X(2.125), Y(0), 2.7 * S, Math.PI / 2, 3 * Math.PI / 2, false); g.stroke(); // 三分弧
    // 乒乓球区域
    g.strokeStyle = 'rgba(40,120,190,0.85)'; g.lineWidth = 4;
    g.strokeRect(X(-2.9 - 3.2), Y(-1.9), 6.4 * S, 3.8 * S);
    g.fillStyle = 'rgba(30,30,40,0.5)';
    g.font = `${0.5 * S}px bold sans-serif`;
    g.textAlign = 'center';
    g.fillText('G1 × MuJoCo WASM', X(-2.9), Y(2.35));

    const tex = new THREE.CanvasTexture(cv);
    tex.anisotropy = 8;
    tex.colorSpace = THREE.SRGBColorSpace;
    const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85, metalness: 0.0 });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(W, D), mat);
    mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    mesh.matrix.identity();
    scene.add(mesh);
  }

  update(data) {
    const xp = data.geom_xpos, xr = data.geom_xmat;
    for (let i = 0; i < this.meshes.length; i++) {
      const mesh = this.meshes[i];
      if (!mesh) continue;
      const o = 3 * i, q = 9 * i;
      this._m4.set(
        xr[q], xr[q + 1], xr[q + 2], xp[o],
        xr[q + 3], xr[q + 4], xr[q + 5], xp[o + 1],
        xr[q + 6], xr[q + 7], xr[q + 8], xp[o + 2],
        0, 0, 0, 1);
      mesh.matrix.copy(this._m4);
      mesh.matrixWorldNeedsUpdate = true;
    }
  }
}
