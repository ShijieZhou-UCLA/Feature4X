import logging
import sys, os, os.path as osp

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))

from gs_optim_helpers import update_learning_rate, get_expon_lr_func


class GSControlCFG:
    def __init__(
        self,
        densify_steps=300,
        reset_steps=900,
        prune_steps=300,
        densify_max_grad=0.0002,
        densify_percent_dense=0.01,
        prune_opacity_th=0.012,
        reset_opacity=0.01,
    ):
        if isinstance(densify_steps, int):
            densify_steps = [densify_steps * i for i in range(100000)]
        if isinstance(reset_steps, int):
            reset_steps = [reset_steps * i for i in range(100000)]
        if isinstance(prune_steps, int):
            prune_steps = [prune_steps * i for i in range(100000)]
        self.densify_steps = densify_steps
        self.reset_steps = reset_steps
        self.prune_steps = prune_steps
        self.densify_max_grad = densify_max_grad
        self.densify_percent_dense = densify_percent_dense
        self.prune_opacity_th = prune_opacity_th
        self.reset_opacity = reset_opacity
        self.summary()
        return

    def summary(self):
        logging.info("GSControlCFG: Summary")
        logging.info(
            f"GSControlCFG: densify_steps={self.densify_steps[:min(5, len(self.densify_steps))]}..."
        )
        logging.info(
            f"GSControlCFG: reset_steps={self.reset_steps[:min(5, len(self.densify_steps))]}..."
        )
        logging.info(
            f"GSControlCFG: prune_steps={self.prune_steps[:min(5, len(self.densify_steps))]}..."
        )
        logging.info(f"GSControlCFG: densify_max_grad={self.densify_max_grad}")
        logging.info(
            f"GSControlCFG: densify_percent_dense={self.densify_percent_dense}"
        )
        logging.info(f"GSControlCFG: prune_opacity_th={self.prune_opacity_th}")
        logging.info(f"GSControlCFG: reset_opacity={self.reset_opacity}")
        return


class OptimCFG:
    def __init__(
        self,
        # GS
        lr_p=0.00016,
        lr_q=0.001,
        lr_s=0.005,
        lr_o=0.05,
        lr_sph=0.0025,
        lr_sph_rest_factor=20.0,
        lr_semantic_feature=0.0005,
        lr_semantic_heads=0.0005,
        lr_p_final=None,
        lr_cam_q=0.0001,
        lr_cam_t=0.0001,
        lr_cam_f=0.00,
        lr_cam_q_final=None,
        lr_cam_t_final=None,
        lr_cam_f_final=None,
        # # dyn
        lr_np=0.00016,
        lr_nq=0.001,
        lr_nsig=0.00001,
        lr_sk_q=0.001,
        lr_w=0.0,  # ! use 0.0
        lr_dyn=0.01,
        lr_np_final=None,
        lr_nq_final=None,
        lr_w_final=None,
    ) -> None:
        # gs
        self.lr_p = lr_p
        self.lr_q = lr_q
        self.lr_s = lr_s
        self.lr_o = lr_o
        self.lr_sph = lr_sph
        self.lr_sph_rest = lr_sph / lr_sph_rest_factor
        self.lr_semantic_feature = lr_semantic_feature
        self.lr_semantic_heads = lr_semantic_heads
        # cam
        self.lr_cam_q = lr_cam_q
        self.lr_cam_t = lr_cam_t
        self.lr_cam_f = lr_cam_f
        # # dyn
        self.lr_np = lr_np
        self.lr_nq = lr_nq
        self.lr_w = lr_w
        self.lr_dyn = lr_dyn
        self.lr_nsig = lr_nsig
        self.lr_sk_q = lr_sk_q

        # gs scheduler
        self.lr_p_final = lr_p_final if lr_p_final is not None else lr_p / 100.0
        self.lr_cam_q_final = (
            lr_cam_q_final if lr_cam_q_final is not None else lr_cam_q / 10.0
        )
        self.lr_cam_t_final = (
            lr_cam_t_final if lr_cam_t_final is not None else lr_cam_t / 10.0
        )
        self.lr_cam_f_final = (
            lr_cam_f_final if lr_cam_f_final is not None else lr_cam_f / 10.0
        )
        self.lr_np_final = lr_np_final if lr_np_final is not None else lr_np / 100.0
        self.lr_nq_final = lr_nq_final if lr_nq_final is not None else lr_nq / 10.0
        if lr_w is not None:
            self.lr_w_final = lr_w_final if lr_w_final is not None else lr_w / 10.0
        else:
            self.lr_w_final = 0.0
        return

    def summary(self):
        logging.info("OptimCFG: Summary")
        logging.info(f"OptimCFG: lr_p={self.lr_p}")
        logging.info(f"OptimCFG: lr_q={self.lr_q}")
        logging.info(f"OptimCFG: lr_s={self.lr_s}")
        logging.info(f"OptimCFG: lr_o={self.lr_o}")
        logging.info(f"OptimCFG: lr_sph={self.lr_sph}")
        logging.info(f"OptimCFG: lr_sph_rest={self.lr_sph_rest}")
        logging.info(f"OptimCFG: lr_semantic_feature={self.lr_semantic_feature}")
        logging.info(f"OptimCFG: lr_semantic_heads={self.lr_semantic_heads}")
        logging.info(f"OptimCFG: lr_cam_q={self.lr_cam_q}")
        logging.info(f"OptimCFG: lr_cam_t={self.lr_cam_t}")
        logging.info(f"OptimCFG: lr_cam_f={self.lr_cam_f}")
        logging.info(f"OptimCFG: lr_p_final={self.lr_p_final}")
        logging.info(f"OptimCFG: lr_cam_q_final={self.lr_cam_q_final}")
        logging.info(f"OptimCFG: lr_cam_t_final={self.lr_cam_t_final}")
        logging.info(f"OptimCFG: lr_cam_f_final={self.lr_cam_f_final}")
        logging.info(f"OptimCFG: lr_np={self.lr_np}")
        logging.info(f"OptimCFG: lr_nq={self.lr_nq}")
        logging.info(f"OptimCFG: lr_w={self.lr_w}")
        logging.info(f"OptimCFG: lr_dyn={self.lr_dyn}")
        logging.info(f"OptimCFG: lr_nsig={self.lr_nsig}")
        logging.infp(f"OptimCFG: lr_sk_q={self.lr_sk_q}")
        logging.info(f"OptimCFG: lr_np_final={self.lr_np_final}")
        logging.info(f"OptimCFG: lr_nq_final={self.lr_nq_final}")
        logging.info(f"OptimCFG: lr_w_final={self.lr_w_final}")
        return

    @property
    def get_static_lr_dict(self):
        return {
            "lr_p": self.lr_p,
            "lr_q": self.lr_q,
            "lr_s": self.lr_s,
            "lr_o": self.lr_o,
            "lr_sph": self.lr_sph,
            "lr_sph_rest": self.lr_sph_rest,
            "lr_semantic_feature": self.lr_semantic_feature,
        }

    @property
    def get_dynamic_lr_dict(self):
        return {
            "lr_p": self.lr_p,
            "lr_q": self.lr_q,
            "lr_s": self.lr_s,
            "lr_o": self.lr_o,
            "lr_sph": self.lr_sph,
            "lr_sph_rest": self.lr_sph_rest,
            "lr_semantic_feature": self.lr_semantic_feature,
            "lr_np": self.lr_np,
            "lr_nq": self.lr_nq,
            "lr_w": self.lr_w,
            "lr_dyn": self.lr_dyn,
            "lr_nsig": self.lr_nsig,
            "lr_sk_q": self.lr_sk_q,
        }

    @property
    def get_dynamic_node_lr_dict(self):
        return {
            "lr_p": 0.0,
            "lr_q": 0.0,
            "lr_s": 0.0,
            "lr_o": 0.0,
            "lr_sph": 0.0,
            "lr_sph_rest": 0.0,
            "lr_semantic_feature": self.lr_semantic_feature,
            "lr_np": self.lr_np,
            "lr_nq": self.lr_nq,
            "lr_w": 0.0,
            "lr_dyn": 0.0,
            "lr_nsig": self.lr_nsig,
            "lr_sk_q": self.lr_sk_q,
        }

    @property
    def get_cam_lr_dict(self):
        return {
            "lr_q": self.lr_cam_q,
            "lr_t": self.lr_cam_t,
            "lr_f": self.lr_cam_f,
        }

    def get_scheduler(self, total_steps):
        # todo: decide whether to decay skinning weights
        gs_scheduling_dict = {
            "xyz": get_expon_lr_func(
                lr_init=self.lr_p,
                lr_final=self.lr_p_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "node_xyz": get_expon_lr_func(
                lr_init=self.lr_np,
                lr_final=self.lr_np_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "node_rotation": get_expon_lr_func(
                lr_init=self.lr_nq,
                lr_final=self.lr_nq_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
        }
        cam_scheduling_dict = {
            "R": get_expon_lr_func(
                lr_init=self.lr_cam_q,
                lr_final=self.lr_cam_q_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "t": get_expon_lr_func(
                lr_init=self.lr_cam_t,
                lr_final=self.lr_cam_t_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "f": get_expon_lr_func(
                lr_init=self.lr_cam_f,
                lr_final=self.lr_cam_f_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
        }
        return gs_scheduling_dict, cam_scheduling_dict
