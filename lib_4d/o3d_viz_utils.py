import numpy as np
import open3d as o3d
import logging
import time


def _vis_update_thread(__vis, __vis_dict, fps=30):
    # for unknown reason, mu workstation always pop GLX current context warning, shut it up
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    __vis.create_window("Super4D")
    __vis.get_render_option().background_color = np.array([0, 0, 0])

    __vis.add_geometry(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    )
    ctr = __vis.get_view_control()
    ctr.set_front([-0.3, -0.15, -0.95])
    ctr.set_lookat([-0.07, -0.35, 1.7])
    ctr.set_up([0.05, -0.99, 0.12])
    ctr.set_zoom(0.34)
    ctr.set_constant_z_far(1000)
    logging.info(f"start vis thread loop")

    while True:
        time.sleep(1 / float(fps))
        __vis.poll_events()
        __vis.update_renderer()
        for k, v in __vis_dict.items():
            __vis.update_geometry(v)
