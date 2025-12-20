"""Visualize gripperframe vs graspframe site positions with in-simulation spheres."""
import mujoco
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# Create a modified XML with marker spheres
base_xml = """
<mujoco model="site_visualization">
    <include file="so101_new_calib.xml"/>

    <visual>
        <global offwidth="640" offheight="480"/>
    </visual>

    <asset>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
        <!-- Marker materials - high emission for visibility -->
        <material name="red_marker" rgba="1 0.2 0.2 1" emission="1"/>
        <material name="green_marker" rgba="0.2 1 0.2 1" emission="1"/>
        <material name="blue_marker" rgba="0.2 0.2 1 1" emission="1"/>
    </asset>

    <worldbody>
        <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
        <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" />

        <!-- Marker spheres - positions will be updated in code -->
        <body name="gripperframe_marker" pos="0 0 0" mocap="true">
            <geom type="sphere" size="0.008" material="red_marker" contype="0" conaffinity="0"/>
        </body>
        <body name="graspframe_marker" pos="0 0 0" mocap="true">
            <geom type="sphere" size="0.008" material="green_marker" contype="0" conaffinity="0"/>
        </body>
        <body name="fingermid_marker" pos="0 0 0" mocap="true">
            <geom type="sphere" size="0.008" material="blue_marker" contype="0" conaffinity="0"/>
        </body>
    </worldbody>
</mujoco>
"""

# Write temp XML in SO-ARM100 directory so include paths work
temp_xml_path = Path("SO-ARM100/Simulation/SO101/site_viz_temp.xml")
temp_xml_path.write_text(base_xml)

model = mujoco.MjModel.from_xml_path(str(temp_xml_path))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

# Set wrist to top-down grasp pose
data.qpos[3] = 1.65  # wrist_flex points down
data.qpos[4] = np.pi / 2  # wrist_roll horizontal
data.ctrl[3] = 1.65
data.ctrl[4] = np.pi / 2
mujoco.mj_forward(model, data)

# Get site positions
gripperframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
graspframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "graspframe")

gripperframe_pos = data.site_xpos[gripperframe_id].copy()
graspframe_pos = data.site_xpos[graspframe_id].copy()

# Get finger positions
finger_28 = data.geom_xpos[28].copy()
finger_30 = data.geom_xpos[30].copy()
finger_mid = (finger_28 + finger_30) / 2

print("Site Positions (wrist pointing down):")
print(f"  gripperframe (RED):   {gripperframe_pos}")
print(f"  graspframe (GREEN):   {graspframe_pos}")
print(f"  finger_mid (BLUE):    {finger_mid}")

# Update mocap body positions to show markers at site locations
gripperframe_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripperframe_marker")
graspframe_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "graspframe_marker")
fingermid_marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingermid_marker")

# Get mocap indices
mocap_ids = {
    'gripperframe': model.body_mocapid[gripperframe_marker_id],
    'graspframe': model.body_mocapid[graspframe_marker_id],
    'fingermid': model.body_mocapid[fingermid_marker_id],
}

data.mocap_pos[mocap_ids['gripperframe']] = gripperframe_pos
data.mocap_pos[mocap_ids['graspframe']] = graspframe_pos
data.mocap_pos[mocap_ids['fingermid']] = finger_mid

mujoco.mj_forward(model, data)

# Render scene
renderer = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.22, 0.0, 0.10]
cam.distance = 0.35
cam.azimuth = 180  # Front view
cam.elevation = -15

renderer.update_scene(data, camera=cam)
frame = renderer.render().copy()

# Convert to PIL Image and add legend
img = Image.fromarray(frame)
draw = ImageDraw.Draw(img)

# Add legend
legend_y = 20
legend_x = 20
legend_items = [
    ('gripperframe (RED)', gripperframe_pos, 'red'),
    ('graspframe (GREEN)', graspframe_pos, 'lime'),
    ('finger_mid (BLUE)', finger_mid, 'cyan'),
]

for name, pos_3d, color in legend_items:
    draw.rectangle([legend_x, legend_y, legend_x + 20, legend_y + 15], fill=color)
    pos_str = f"{name}: Z={pos_3d[2]:.3f}"
    draw.text((legend_x + 25, legend_y), pos_str, fill='white')
    legend_y += 25

# Add distance info
legend_y += 10
grip_to_finger = np.linalg.norm(gripperframe_pos - finger_mid)
grasp_to_finger = np.linalg.norm(graspframe_pos - finger_mid)
draw.text((legend_x, legend_y), f"gripperframe to finger_mid: {grip_to_finger*1000:.1f}mm", fill='red')
legend_y += 20
draw.text((legend_x, legend_y), f"graspframe to finger_mid: {grasp_to_finger*1000:.1f}mm", fill='lime')

# Save
output_path = Path("runs/site_positions_annotated.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
img.save(output_path)
print(f"\nSaved annotated image to {output_path}")

renderer.close()
