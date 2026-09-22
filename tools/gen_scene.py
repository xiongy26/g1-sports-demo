#!/usr/bin/env python3
"""Generate model/scene_gym.xml — gym scene: basketball hoop + table-tennis table."""
import math

RIM_X, RIM_Z, RIM_R = 2.125, 3.05, 0.226      # rim center / height / ring radius
POLE_X = 3.15
TABLE_X = -2.9                                 # table center
TABLE_H = 0.76

def q_axis_z_to(dx, dy, dz):
    """Quaternion rotating +z onto (dx,dy,dz) (normalized)."""
    n = math.sqrt(dx*dx + dy*dy + dz*dz)
    dx, dy, dz = dx/n, dy/n, dz/n
    ax, ay, az = -dy, dx, 0.0
    s = math.sqrt(ax*ax + ay*ay)
    if s < 1e-9:
        return "1 0 0 0" if dz > 0 else "0 1 0 0"
    ax, ay = ax/s*0.70710678, ay/s*0.70710678
    return f"{0.70710678:.6f} {ax:.6f} {ay:.6f} 0"

def capsule(a, b, radius, extra=""):
    return (f'<geom type="capsule" size="{radius}" fromto="{a[0]:.4f} {a[1]:.4f} {a[2]:.4f} '
            f'{b[0]:.4f} {b[1]:.4f} {b[2]:.4f}" {extra}/>')

# rim: 16 capsules forming a ring; rim center is 0.375 m in front of the board face
N = 16
segs = []
for i in range(N):
    a0 = 2 * math.pi * i / N
    p0 = (RIM_X - POLE_X + 0.63 - RIM_R * math.cos(a0), RIM_R * math.sin(a0), RIM_Z)
    a1 = 2 * math.pi * (i + 1.4) / N
    p1 = (RIM_X - POLE_X + 0.63 - RIM_R * math.cos(a1), RIM_R * math.sin(a1), RIM_Z)
    segs.append(f'        <geom type="capsule" size="0.017" fromto="{p0[0]:.4f} {p0[1]:.4f} {p0[2]:.4f} '
                f'{p1[0]:.4f} {p1[1]:.4f} {p1[2]:.4f}" material="hoop_orange" name="rim_{i}"/>')
rim_capsules = "\n".join(segs)

# net: 12 slightly inward-tilted strands
segs = []
N = 12
for i in range(N):
    a = 2 * math.pi * i / N + 0.13
    top = (RIM_X - POLE_X + 0.63 - (RIM_R + 0.005) * math.cos(a), (RIM_R + 0.005) * math.sin(a), RIM_Z - 0.01)
    bot = (RIM_X - POLE_X + 0.63 - 0.13 * math.cos(a), 0.13 * math.sin(a), RIM_Z - 0.35)
    segs.append('        ' + capsule(top, bot, 0.0045, 'material="net_white" contype="0" conaffinity="0"'))
net_capsules = "\n".join(segs)

xml = f'''<mujoco model="g1_sports_gym">
  <!-- Unitree G1 sports gym: basketball hoop + table-tennis table.
       The <include> is spliced by main.js (DOM merge) before compiling. -->
  <include file="g1.xml"/>

  <statistic center="-0.3 0 1.0" extent="3.6"/>

  <visual>
    <headlight diffuse="0.45 0.45 0.5" ambient="0.3 0.3 0.32" specular="0.4 0.4 0.4"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <map stiffness="700" shadowscale="0.5" fogstart="14" fogend="30" zfar="45" haze="0.25"/>
  </visual>

  <option timestep="0.002" integrator="implicitfast"/>

  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.1"/>
    <material name="hoop_orange" rgba="0.85 0.35 0.06 1"/>
    <material name="board_white" rgba="0.93 0.93 0.96 1"/>
    <material name="steel" rgba="0.32 0.34 0.38 1"/>
    <material name="table_blue" rgba="0.05 0.35 0.5 1"/>
    <material name="net_white" rgba="0.95 0.95 0.95 1"/>
    <material name="bball" rgba="0.85 0.4 0.08 1"/>
    <material name="machine" rgba="0.75 0.16 0.14 1"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>

    <!-- ==================== basketball hoop ==================== -->
    <body name="hoop" pos="{POLE_X} 0 0">
      <geom name="hoop_pole" type="cylinder" size="0.06 1.72" pos="0 0 1.72" material="steel"/>
      <geom name="hoop_base" type="cylinder" size="0.35 0.04" pos="0 0 0.04" material="steel"/>
      <geom name="hoop_arm" type="box" size="0.31 0.035 0.035" pos="-0.31 0 3.36" material="steel"
            zaxis="-1 0 0"/>
      <body name="backboard" pos="-0.63 0 0">
        <geom name="board" type="box" size="0.02 0.9 0.55" pos="0 0 3.44" material="board_white"/>
        <!-- rim -->
{rim_capsules}
        <!-- net strands -->
{net_capsules}
      </body>
    </body>

    <!-- ==================== basketball ==================== -->
    <body name="basketball" pos="0.32 -0.3 0.9">
      <freejoint name="basketball_joint"/>
      <geom name="basketball_geom" type="sphere" size="0.123" mass="0.62" material="bball"
            priority="2" solref="0.004 0.2" solimp="0.9 0.95 0.001"
            friction="0.9 0.02 0.001"/>
    </body>

    <!-- ==================== table-tennis table ==================== -->
    <body name="tt_table" pos="{TABLE_X} 0 0">
      <geom name="tt_top" type="box" size="1.37 0.7625 0.025" pos="0 0 {TABLE_H-0.025}" material="table_blue"/>
      <geom name="tt_apron_l" type="box" size="1.37 0.02 0.10" pos="0 0.75 {TABLE_H-0.13}" material="steel"/>
      <geom name="tt_apron_r" type="box" size="1.37 0.02 0.10" pos="0 -0.75 {TABLE_H-0.13}" material="steel"/>
      <geom name="tt_leg_fl" type="box" size="0.035 0.035 0.33" pos="1.17 0.62 0.33" material="steel"/>
      <geom name="tt_leg_fr" type="box" size="0.035 0.035 0.33" pos="1.17 -0.62 0.33" material="steel"/>
      <geom name="tt_leg_bl" type="box" size="0.035 0.035 0.33" pos="-1.17 0.62 0.33" material="steel"/>
      <geom name="tt_leg_br" type="box" size="0.035 0.035 0.33" pos="-1.17 -0.62 0.33" material="steel"/>
      <!-- white lines -->
      <geom name="tt_line_mid" type="box" size="1.37 0.006 0.001" pos="0 0 {TABLE_H+0.0005}" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
      <geom name="tt_line_edge_l" type="box" size="1.37 0.01 0.001" pos="0 0.7525 {TABLE_H+0.0005}" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
      <geom name="tt_line_edge_r" type="box" size="1.37 0.01 0.001" pos="0 -0.7525 {TABLE_H+0.0005}" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
      <geom name="tt_line_end_l" type="box" size="0.01 0.7625 0.001" pos="1.36 0 {TABLE_H+0.0005}" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
      <geom name="tt_line_end_r" type="box" size="0.01 0.7625 0.001" pos="-1.36 0 {TABLE_H+0.0005}" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
      <!-- net + posts -->
      <geom name="tt_net" type="box" size="0.006 0.7825 0.076" pos="0 0 {TABLE_H+0.076}" material="net_white" contype="0" conaffinity="0"/>
      <geom name="tt_post_l" type="cylinder" size="0.01 0.085" pos="0 0.7825 {TABLE_H+0.075}" material="steel"/>
      <geom name="tt_post_r" type="cylinder" size="0.01 0.085" pos="0 -0.7825 {TABLE_H+0.075}" material="steel"/>
    </body>

    <!-- ==================== ball machine (opponent) ====================
         炮管沿实际出球仰角(约15°)指向机器人一侧；出球点 A 在炮口处（见 src/pingpong.js）。
         mocap 体：双机对打模式下移到地下隐藏，给二号机让位 -->
    <body name="tt_machine" mocap="true" pos="-4.72 0.25 0">
      <geom name="machine_body" type="box" size="0.16 0.16 0.19" pos="0 0 1.0" material="machine"/>
      <geom name="machine_barrel" type="cylinder" size="0.045 0.16" pos="0.14 0 1.06" material="steel"
            euler="0 1.31 0"/>
      <geom name="machine_wheel_fl" type="cylinder" size="0.045 0.02" pos="0.06 0.17 0.045" material="steel" euler="1.5708 0 0"/>
      <geom name="machine_wheel_fr" type="cylinder" size="0.045 0.02" pos="0.06 -0.17 0.045" material="steel" euler="1.5708 0 0"/>
      <geom name="machine_wheel_bl" type="cylinder" size="0.045 0.02" pos="-0.06 0.17 0.045" material="steel" euler="1.5708 0 0"/>
      <geom name="machine_wheel_br" type="cylinder" size="0.045 0.02" pos="-0.06 -0.17 0.045" material="steel" euler="1.5708 0 0"/>
    </body>

    <!-- ==================== kinematic props ==================== -->
    <!-- table-tennis ball (scripted rally) -->
    <body name="tt_ball" mocap="true" pos="-1.7 0.15 1.05">
      <geom name="tt_ball_geom" type="sphere" size="0.02" rgba="1 1 1 1" contype="0" conaffinity="0"/>
    </body>

    <!-- paddle attached to the right hand at runtime -->
    <body name="paddle" mocap="true" pos="-1.3 0.2 1.0">
      <geom name="paddle_blade" type="cylinder" size="0.085 0.008" rgba="0.8 0.25 0.15 1" contype="0" conaffinity="0"/>
      <geom name="paddle_blade_face" type="cylinder" size="0.084 0.004" pos="0 0 0.009" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>
      <geom name="paddle_handle" type="box" size="0.013 0.018 0.05" pos="0 0 -0.062" rgba="0.7 0.6 0.4 1" contype="0" conaffinity="0"/>
    </body>

    <!-- second paddle for the duel mode (二号机，默认停放在地下) -->
    <body name="paddle_r2" mocap="true" pos="0 0 -5">
      <geom name="paddle_r2_blade" type="cylinder" size="0.085 0.008" rgba="0.8 0.25 0.15 1" contype="0" conaffinity="0"/>
      <geom name="paddle_r2_blade_face" type="cylinder" size="0.084 0.004" pos="0 0 0.009" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>
      <geom name="paddle_r2_handle" type="box" size="0.013 0.018 0.05" pos="0 0 -0.062" rgba="0.7 0.6 0.4 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <keyframe>
    <!-- qpos 顺序 = 模型内自由体顺序: 一号机(36) + 二号机(36, 对打站位) + 篮球(7, 场边停放) -->
    <key name="home"
      qpos="0 0 0.783675 1 0 0 0 -0.1 0 0 0.3 -0.2 0 -0.1 0 0 0.3 -0.2 0 0 0 0 0.2 0.2 0 1.28 0 0 0 0.2 -0.2 0 1.28 0 0 0 -4.65 0 0.783675 1 0 0 0 -0.1 0 0 0.3 -0.2 0 -0.1 0 0 0.3 -0.2 0 0 0 0 0.2 0.2 0 1.28 0 0 0 0.2 -0.2 0 1.28 0 0 0 3.4 1.8 0.123 1 0 0 0"
      ctrl="-0.1 0 0 0.3 -0.2 0 -0.1 0 0 0.3 -0.2 0 0 0 0 0.2 0.2 0 1.28 0 0 0 0.2 -0.2 0 1.28 0 0 0 -0.1 0 0 0.3 -0.2 0 -0.1 0 0 0.3 -0.2 0 0 0 0 0.2 0.2 0 1.28 0 0 0 0.2 -0.2 0 1.28 0 0 0"/>
  </keyframe>
</mujoco>
'''

out = __file__.rsplit("/", 2)[0] + "/model/scene_gym.xml"
open(out, "w").write(xml)
print("wrote", out, len(xml), "bytes")
