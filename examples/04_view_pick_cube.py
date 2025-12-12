"""View the pick-cube scene with interactive control."""
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("SO-ARM100/Simulation/SO101/pick_cube_scene.xml")
data = mujoco.MjData(model)

print("Pick-cube scene loaded!")
print("  - Red cube: object to pick")
print("  - Green square: target location")
print("\nControls:")
print("  - Left-click + drag: rotate view")
print("  - Right-click + drag: pan view")
print("  - Scroll: zoom")
print("  - Double-click body + drag: apply force")
print("  - ESC: quit")

mujoco.viewer.launch(model, data)
