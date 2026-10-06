// 仅改变可视手指；G1 的固定橡胶手没有可驱动的手指关节。
import * as THREE from 'three';

export function paddleGripGeometry(source, geomPosition, geomQuat) {
  const q = new THREE.Quaternion(geomQuat[1], geomQuat[2], geomQuat[3], geomQuat[0]);
  const inverse = q.clone().invert(), origin = new THREE.Vector3(...geomPosition);
  const positions = [];
  const input = source.index ? source.toNonIndexed() : source.clone();
  const vertices = input.attributes.position;
  const triangle = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
  // 保留原模型的掌部。不要逐顶点强扭固定手指，避免跨指缝三角形被拉长。
  for (let i = 0; i < vertices.count; i += 3) {
    for (let k = 0; k < 3; k++) triangle[k].fromBufferAttribute(vertices, i + k).applyQuaternion(q).add(origin);
    if (triangle.every(p => p.x <= 0.095)) {
      for (const p of triangle) positions.push(p.x, p.y, p.z);
    }
  }
  input.dispose();
  function segment(a, b, radius) {
    const start = new THREE.Vector3(...a), end = new THREE.Vector3(...b);
    const direction = end.clone().sub(start), center = start.clone().add(end).multiplyScalar(0.5);
    const rotation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
    const indexed = new THREE.CapsuleGeometry(radius, direction.length(), 5, 10);
    const geo = indexed.toNonIndexed();
    const attr = geo.attributes.position, p = new THREE.Vector3();
    for (let i = 0; i < attr.count; i++) {
      p.fromBufferAttribute(attr, i).applyQuaternion(rotation).add(center);
      positions.push(p.x, p.y, p.z);
    }
    geo.dispose(); indexed.dispose();
  }
  function finger(points, radius) {
    for (let i = 1; i < points.length; i++) segment(points[i - 1], points[i], radius);
  }
  // 中指、无名指、小指包住柄的两侧与外缘，指端回到掌内。
  for (const [z, r] of [[-0.034, 0.0052], [-0.015, 0.006], [0.004, 0.006]]) {
    finger([[0.090, 0, z], [0.125, 0.016, z], [0.122, 0.044, z], [0.101, 0.046, z]], r);
  }
  // 食指沿红面下缘，拇指压在另一侧黑面拍肩。
  finger([[0.089, 0, 0.024], [0.103, 0.033, 0.044], [0.123, 0.039, 0.065], [0.140, 0.039, 0.075]], 0.006);
  finger([[0.075, 0, 0.041], [0.086, 0.004, 0.060], [0.113, 0.007, 0.075]], 0.008);
  // 转回该 visual geom 的网格坐标，继续使用原有 geom_xmat 同步。
  const p = new THREE.Vector3();
  for (let i = 0; i < positions.length; i += 3) {
    p.set(positions[i], positions[i + 1], positions[i + 2]).sub(origin).applyQuaternion(inverse);
    positions[i] = p.x; positions[i + 1] = p.y; positions[i + 2] = p.z;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals(); geometry.computeBoundingSphere();
  return geometry;
}
