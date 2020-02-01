import numpy as np
import vispy.scene
from vispy.scene import visuals

pos = np.load("../data/Point Clouds/MEC1.npy")

canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# generate data
scatter = visuals.Markers()
scatter.set_data(pos.T, edge_color=None, face_color=(1, 1, 1, .5), size=5)
view.add(scatter)
view.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)
import sys
if sys.flags.interactive != 1:
    vispy.app.run()
