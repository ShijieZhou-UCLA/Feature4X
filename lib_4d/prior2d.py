import numpy as np
import torch, numpy as np
import os, sys, os.path as osp
from torchvision.transforms import Resize
import torchvision
from glob import glob
import cv2

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from graph_utils import *
from prior2d_utils import *
from index_helper import query_image_buffer_by_pix_int_coord
from lib_prior.tracking.cotracker_wrapper import Visualizer
from prior2d_tracking_helpers import (
    connect_optical_flow,
    gather_curve_attr_large_amount,
    viz_track_coverage,
)
from camera import SimpleFovCamerasIndependent
import kornia
from projection_helper import backproject, project
import yaml

class Prior2D:
    # * This function just used as to load the data, the detailed processing is in solver
    @torch.no_grad()
    def __init__(
        self,
        log_dir,
        src_dir,
        cache_device=torch.device("cuda:0"),
        working_device=torch.device("cuda:0"),
        depth_mode="zoe",
        start_frame=0,
        end_frame=-1,
        frame_skip_step=1,
        flow_interval=[1],
        consider_sky_depth_mask=True,
        dino_first_ch=64,
        min_valid_cnt=4,
        epi_error_th_factor=400.0,  # davis: 91.0
        mask_prop_steps=1,  # 6
        mask_consider_track=False,  # use track to make conservative static mask
        mask_consider_track_dilate_radius=7, 
        mask_init_erode=0,  # ! debug
        viz_init_flag=True,  # ! debug
        use_short_track=False,
        semantic_th_quantile=0.95,
        depth_boundary_th=0.5,  # smaller stricter and more removed
        dino_name="dino",
        nerfies_flag=False,
        feature_config="configs/default_config.yaml",
    ):
        self.src_dir = src_dir
        self.log_dir = log_dir
        self.cache_device = cache_device
        self.working_device = working_device
        logging.info(f"Load flow interval {flow_interval}")

        with open(feature_config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.feature_config = config
        self.latent_feature_channel = config["NUM_SEMANTIC_CHANNELS"]

        # * Load from disk
        (
            self.view_list,
            self.view_pairs,
            rgbs,
            semantic_features,
            depths,
            flows_dict,
            sky_dep_masks,
            epi_errs,
        ) = load_prior_dir(
            skip=frame_skip_step,
            src=self.src_dir,
            depth_dir=f"{depth_mode}_depth",
            start=start_frame,
            end=end_frame,
            flow_interval=flow_interval,
            load_sky_flag=consider_sky_depth_mask,
            nerfies_flag=nerfies_flag,
            feature_config=self.feature_config,
        )
        self.T = len(self.view_list)

        # * Globally normalize and filter the depth median
        depth_rescale_median = float(np.median(depths))
        depths = depths / depth_rescale_median
        self.depth_rescale_factor_model_world = (
            1 / depth_rescale_median
        )  # depth_model = depth_world * depth_rescale_model_world
        logging.info(
            f"Normalizing the sensor depth to have median=1 with model depth {self.depth_rescale_factor_model_world:.4f} times of real world depth"
        )

        depth_masks, _ = laplacian_filter_depth(depths, depth_boundary_th, 5) * (
            depths > 0.0
        )
        # # ! debug
        # imageio.mimsave("./debug/depth_boundary.mp4", depth_masks)
        if consider_sky_depth_mask:
            depth_masks = depth_masks * (~sky_dep_masks)

        # * prepare coordinates and multi-resolution
        self.resize_level: float = 1.0
        self._resized_data = {}
        self._H, self._W = rgbs[0].shape[:2]
        self._homo_map = (
            torch.from_numpy(get_homo_coordinate_map(H=self.H, W=self.W))
            .float()
            .to(self.working_device)
        )
        self._pixel_int_map = (
            torch.from_numpy(
                np.stack(np.meshgrid(np.arange(self.W), np.arange(self.H)), -1)
            )
            .int()
            .to(self.working_device)
        )

        # * move
        self._depths = torch.from_numpy(depths).float().to(self.cache_device)
        self._epi_errs = torch.from_numpy(epi_errs).float().to(self.cache_device)
        self._rgbs = (
            torch.from_numpy(rgbs.astype(np.float32) / 255)
            .float()
            .to(self.cache_device)
        )

        self._semantic_features = semantic_features
        # self._cls_features=( # 1, 1, 1408
        #     semantic_features["cls_feat"].to(torch.float32).to(self.cache_device)
        # )
        self._depth_masks = torch.from_numpy(depth_masks).bool().to(self.working_device)

        # prepare flows
        self.flow_name_list = list(flows_dict.keys())
        self.flow_ij_list = [
            (self.view_list.index(name[0]), self.view_list.index(name[1]))
            for name in self.flow_name_list
        ]

        self._flows = torch.from_numpy(
            np.stack([flows_dict[k]["flow"] for k in self.flow_name_list], 0)
        ).float()
        self._flow_masks = torch.from_numpy(
            np.stack([flows_dict[k]["mask"] for k in self.flow_name_list], 0)
        ).bool()

        # * check the flow, round it to integer and update the flow masks
        logging.info("rounding the flow ...")
        self._flows, self._flow_masks = self.__round_flows__(
            self.flow_name_list,
            self._flows,
            self._flow_masks,
            self._pixel_int_map,
            self.H,
            self.W,
        )
        self._flows = self._flows.to(self.cache_device)
        self._flow_masks = self._flow_masks.to(self.cache_device)

        # * load dino source
        dino_fn = osp.join(self.src_dir, f"{dino_name}.npz")
        if osp.exists(dino_fn):
            # todo: dino is not used yet, but still load
            logging.info(f"loading dino from {dino_fn}...")
            dino_data = np.load(dino_fn, allow_pickle=True)
            dino_feat = dino_data["low_res_reduced_featmaps"]
            self.low_res_dino_featmaps = (
                torch.from_numpy(dino_feat).float().to(self.working_device)
            )[
                start_frame : end_frame if end_frame > 0 else len(dino_feat),
                :dino_first_ch,
            ]
            self.low_res_dino_featmaps = self.low_res_dino_featmaps[::frame_skip_step]
            assert (
                len(self.low_res_dino_featmaps) == self.T
            ), "dino feat length mismatch"

        ##############################################################################
        # * load the tracks
        (
            track,
            track_vis,
            track_mask,
            static_track_mask,
            semantic_th,
            feat_mean,
            feat_var,
        ) = self.__prepare_track__(
            *self.__track_load__(
                working_device,
                start_frame,
                end_frame,
                frame_skip_step,
            ),
            min_valid_cnt=min_valid_cnt,
            epi_error_th_factor=epi_error_th_factor,
            viz_flag=viz_init_flag,
            name="cotracker",
            semantic_th_quantile=semantic_th_quantile,
        )
        self._track = torch.as_tensor(track).to(self.cache_device)
        self.track_vis = torch.as_tensor(track_vis).to(self.cache_device)
        self.track_mask = torch.as_tensor(track_mask).to(self.cache_device)
        self.track_static_mask = torch.as_tensor(static_track_mask).to(
            self.cache_device
        )
        self.track_dynamic_mask = (
            ~self.track_static_mask.clone()
        ) * self.hard_dyn_marker
        # ! here hardly set the dyn mask to only be inside the dyn loaded

        # * also save the sem feature
        self.track_feat_mean = torch.as_tensor(feat_mean).to(self.cache_device)
        self.track_feat_var = torch.as_tensor(feat_var).to(self.cache_device)

        self.use_short_track = use_short_track
        if use_short_track:
            raise NotImplementedError()
            logging.info("Use short track")
            (
                short_track,
                short_track_vis,
                short_track_mask,
                static_short_track_mask,
                _,
            ) = self.__prepare_track__(
                *connect_optical_flow(self, verbose=False),
                min_valid_cnt=1,
                epi_error_th_factor=epi_error_th_factor,
                viz_flag=viz_init_flag,
                semantic_th=semantic_th,
                name="optical_connect",
            )
            self._short_track = torch.as_tensor(short_track).to(self.cache_device)
            self.short_track_vis = torch.as_tensor(short_track_vis).to(
                self.cache_device
            )
            self.short_track_mask = torch.as_tensor(short_track_mask).to(
                self.cache_device
            )
            self.short_track_static_mask = torch.as_tensor(static_short_track_mask).to(
                self.cache_device
            )
            self.short_track_dynamic_mask = ~self.short_track_static_mask.clone()
        ##############################################################################

        ################################################################
        # * prepare statc masks
        # ! here use the track to produce conservative mask if set

        epi_th = (self.H * self.W) / (epi_error_th_factor**2)
        dynamic_2d_mask = self.epi_errs > epi_th

        if mask_consider_track:
            # todo: these parameters can be tuned!
            track_dyn_mask = get_dynamic_track_mask(
                self.track[:, ~self.track_static_mask],
                self.track_mask[:, ~self.track_static_mask],
                self.H,
                self.W,
                filter_radius=5,
                filter_cnt=3,
                dilate_radius=mask_consider_track_dilate_radius,  # 3,
            )
            imageio.mimsave(
                osp.join(self.log_dir, "dyn_mask_from_track.mp4"),
                track_dyn_mask.cpu().numpy().astype(np.uint8) * 255,
            )
            dynamic_2d_mask = dynamic_2d_mask | track_dyn_mask

        if mask_prop_steps > 0:
            dynamic_2d_mask = prop_mask_by_flow(
                self,
                dynamic_2d_mask,
                steps=mask_prop_steps,
                operator=torch.logical_or,
            )

        # later this will be updated
        static_2d_mask = ~dynamic_2d_mask
        self._static_masks = (
            torch.as_tensor(static_2d_mask).bool().to(self.working_device)
        )
        self._dynamic_masks = ~self._static_masks.clone()

        if mask_init_erode > 0:
            # erode both dynamic and static mask
            kernel = torch.ones(mask_init_erode, mask_init_erode).to(
                self.working_device
            )
            # chuck wise
            with torch.no_grad():
                chunk, cur = 8, 0
                new_dyn_masks, new_sta_masks = [], []
                while cur < self.T:
                    new_dyn_masks.append(
                        kornia.morphology.erosion(
                            self._dynamic_masks[cur : cur + chunk].float().unsqueeze(1),
                            kernel=kernel,
                        )
                        .squeeze(1)
                        .bool()
                    )
                    new_sta_masks.append(
                        kornia.morphology.erosion(
                            self._static_masks[cur : cur + chunk].float().unsqueeze(1),
                            kernel=kernel,
                        )
                        .squeeze(1)
                        .bool()
                    )
                    cur = cur + chunk
                self._dynamic_masks = torch.cat(new_dyn_masks, 0)
                self._static_masks = torch.cat(new_sta_masks, 0)

        # * viz
        if self.log_dir is not None and viz_init_flag:
            viz_mask_video(
                self.rgbs, dynamic_2d_mask, osp.join(self.log_dir, "dyn_mask.mp4")
            )
            viz_choice = torch.randperm(static_track_mask.sum())[:1024].cpu().numpy()
            if self.T > 100:
                skip = self.T // 50
            else:
                skip = 1
            cotracker_vis = Visualizer(
                save_dir=self.log_dir,
                linewidth=1,
                mode="rainbow",
            )
            cotracker_vis.visualize(
                video=self.rgbs.permute(0, 3, 1, 2)[::skip][None].cpu()
                * 255,  # B,T,C,H,W
                tracks=track[::skip, static_track_mask][:, viz_choice][None],
                visibility=track_mask[::skip, static_track_mask][:, viz_choice][None],
                filename=f"static_init",
            )
        return

    def __track_load__(self, device, start, end, skip, mode="cotracker"):
        long_track_fns = glob(osp.join(self.src_dir, f"*{mode}*.npz"))
        logging.info(f"Prior2D: Loading Long Track from {long_track_fns}...")
        assert len(long_track_fns) > 0, "no cotracker data found!"
        track, track_vis = [], []
        max_epi_list, dep_mask_buffer_list, feat_var_sum_list = [], [], []
        # ! have a hard mark of which track is loaded as dynamic
        self.hard_dyn_marker = torch.zeros(0)
        for track_fn in long_track_fns:
            track_data = np.load(track_fn, allow_pickle=True)
            if end < 0:
                end = len(track_data["visibility"])
            end = end + 1  # include
            track.append(track_data["tracks"][start:end][::skip])
            track_vis.append(track_data["visibility"][start:end][::skip])
            if "dyn" in track_fn:
                # use it t_sub_list to mark which frame is actually tracked by cotracker
                self.dyn_track_subsample_t_list = track_data["sub_t_list"]
                self.hard_dyn_marker = torch.cat(
                    [self.hard_dyn_marker, torch.ones(track_data["tracks"].shape[1])]
                )
            else:
                self.hard_dyn_marker = torch.cat(
                    [self.hard_dyn_marker, torch.zeros(track_data["tracks"].shape[1])]
                )
        # check dyn_track_subsample_t_list, because the old prepare v3.2 does not have subsample support
        if not hasattr(self, "dyn_track_subsample_t_list"):
            logging.warning(
                f"Most likely loading from old prepare func which does not has a dyn_xxxxxxx.npz"
            )
            self.dyn_track_subsample_t_list = np.arange(start, end - 1, skip)
        if not self.hard_dyn_marker.any():
            logging.warning(
                f"Most likely loading from old prepare func which does not has a dyn_xxxxxxx.npz"
            )
            self.hard_dyn_marker = torch.ones_like(self.hard_dyn_marker)
        self.hard_dyn_marker = self.hard_dyn_marker.bool().to(device)
        track = torch.from_numpy(np.concatenate(track, 1)).to(device)  # T,N,2
        track_vis = (
            torch.from_numpy(np.concatenate(track_vis, 1)).bool().to(device)
        )  # T,N
        # check whether inside the image
        track, _track_int_mask = round_int_coordinates(track, self.H, self.W)
        track_vis = track_vis * _track_int_mask
        return track, track_vis

    @torch.no_grad()
    def __prepare_track__(
        self,
        track,
        track_vis,
        min_valid_cnt=4,
        epi_error_th_factor=400.0,
        semantic_th=None,
        semantic_th_quantile=0.95,
        viz_flag=False,
        # max_viz_n=1024,
        max_viz_n=2048,
        name="cotracker",
    ):
        # * do the following:
        # - based on epi, classify the track in to absolute static
        # - update the track mask to remove the invalid depth point
        track, _track_int_mask = round_int_coordinates(track, self.H, self.W)
        track_vis = track_vis * _track_int_mask

        max_epi, _, dep_mask_buf, feat_mean, feat_var = gather_curve_attr_large_amount(
            self, track, track_vis, return_features=True
        )
        feat_var_sum = feat_var[..., 3:].sum(-1)  # ! first three are rgb color

        # * filter track naively, admit that the track is in-accurate
        dep_mask_buf = dep_mask_buf.to(track_vis) * track_vis
        valid_cnt = dep_mask_buf.sum(0)
        filter_track_mask = valid_cnt > min_valid_cnt
        logging.info(
            f"After depth valid check with min cnt={min_valid_cnt} {(~filter_track_mask).sum()} tracks are removed!"
        )
        self.hard_dyn_marker = self.hard_dyn_marker[filter_track_mask]
        track = track[:, filter_track_mask]
        track_vis = track_vis[:, filter_track_mask]
        track_mask = dep_mask_buf[:, filter_track_mask].clone()
        max_epi = max_epi.to(track.device)[filter_track_mask]
        feat_var_sum = feat_var_sum.to(track.device)[filter_track_mask]
        feat_mean = feat_mean.to(track.device)[filter_track_mask]
        feat_var = feat_var.to(track.device)[filter_track_mask]

        # further fileter
        if semantic_th is None:
            semantic_th = torch.quantile(
                feat_var_sum.float(), semantic_th_quantile
            ).item()
        semantic_invalid_mask = feat_var_sum > semantic_th

        # draw historgram of the variance
        fig = plt.figure(figsize=(8, 6))
        H, bins = np.histogram(feat_var_sum.cpu().numpy(), bins=100)
        plt.bar(bins[:-1], H, width=bins[1] - bins[0])
        plt.plot([semantic_th, semantic_th], [0, max(H)], "r--")
        plt.title(
            f"Semantic Filtering quat={semantic_th_quantile:.3f} th={semantic_th:.3f}"
        )
        plt.xlabel("Variance")
        plt.ylabel("Count")
        plt.savefig(osp.join(self.log_dir, f"{name}_feat_var_hist.png"))
        plt.close()

        if self.log_dir is not None and viz_flag:
            cotracker_vis = Visualizer(
                save_dir=self.log_dir,
                linewidth=2,
                mode="rainbow",
                tracks_leave_trace=10,
            )
            # skip = 1
            # viz_choice = torch.argsort(feat_var_sum, descending=True)[:max_viz_n]
            # cotracker_vis.visualize(
            #     video=self.rgbs.permute(0, 3, 1, 2)[::skip][None].cpu()
            #     * 255,  # B,T,C,H,W
            #     tracks=track[::skip][:, viz_choice][None],
            #     visibility=track_mask[::skip][:, viz_choice][None],
            #     filename=f"{name}_large_dino_var",
            # )
        
            skip = 1
            viz_choice = torch.randperm(track.shape[1])[:max_viz_n]
            cotracker_vis.visualize(
                video=self.rgbs.permute(0, 3, 1, 2)[::skip][None].cpu()
                * 255,  # B,T,C,H,W
                tracks=track[::skip][:, viz_choice][None],
                visibility=track_mask[::skip][:, viz_choice][None],
                filename=f"{name}_viz_co-tracker",
            )

        # filter the tracks
        semantic_filter_mask = ~semantic_invalid_mask
        track = track[:, semantic_filter_mask]
        track_vis = track_vis[:, semantic_filter_mask]
        track_mask = track_mask[:, semantic_filter_mask]
        max_epi = max_epi[semantic_filter_mask]
        feat_mean = feat_mean[semantic_filter_mask]
        feat_var = feat_var[semantic_filter_mask]
        self.hard_dyn_marker = self.hard_dyn_marker[semantic_filter_mask]

        # * separate
        # Warning, in the frist stage should set the epi-th to small, i.e. large epi_error_th_factor
        epi_th = (self.H * self.W) / (epi_error_th_factor**2)
        static_track_mask = max_epi < epi_th

        torch.cuda.empty_cache()
        if self.log_dir is not None and viz_flag:
            viz_track_coverage(
                self.H,
                self.W,
                track,
                track_mask,
                osp.join(self.log_dir, f"{name}_coverage.mp4"),
            )
        return (
            track,
            track_vis,
            track_mask,
            static_track_mask,
            semantic_th,
            feat_mean,
            feat_var,
        )

    @torch.no_grad()
    def rescale_depth(self, depth_scale):
        self.rescaled_flag = True
        scale_median = float(depth_scale.median())
        self.depth_rescale_factor_model_world = (
            self.depth_rescale_factor_model_world * scale_median
        )
        assert len(self._depths) == self.T, "depth length mismatch!"
        logging.info(
            f"Prior2D: rescale depth by {depth_scale}, Now the model depth is {self.depth_rescale_factor_model_world:.4f} times of the world depth!"
        )
        self.__reset_depth_all__(
            self._depths * depth_scale[:, None, None].to(self._depths)
        )
        return

    @torch.no_grad()
    def __reset_depth_all__(self, new_depths):
        assert len(new_depths) == self.T, "depth length mismatch!"
        logging.info(f"Prior2D: replacing depth ...")
        self._depths = new_depths.clone()
        for level in self._resized_data.keys():
            resizer = Resize(
                (int(self._H * level), int(self._W * level)),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            )
            self._resized_data[level]["depths"] = resizer(self._depths)

    def __round_flows__(self, flow_name_list, flows, flow_masks, pixel_int_map, H, W):
        for flow_name in tqdm(flow_name_list):
            flow_ind = flow_name_list.index(flow_name)
            flow_i2j = flows[flow_ind]
            int_uv_i = pixel_int_map.to(flows.device)
            int_uv_j = torch.round(int_uv_i + flow_i2j).int()
            valid_mask = (
                (int_uv_j[..., 0] >= 0)
                * (int_uv_j[..., 0] < W)
                * (int_uv_j[..., 1] >= 0)
                * (int_uv_j[..., 1] < H)
            )
            valid_mask = valid_mask * flow_masks[flow_ind]
            flow_masks[flow_ind] = valid_mask.clone()
            int_flow = int_uv_j - int_uv_i
            flows[flow_ind] = int_flow.float()
        flows = flows.int()
        flow_masks = flow_masks.bool()
        return flows, flow_masks

    #####################################################################################
    # Handling multi resolution
    #####################################################################################

    def set_resize_level(self, resize_level):
        logging.info(f"2D Prior: set resize level to {resize_level}")
        self.resize_level = float(resize_level)
        if resize_level == 1 or resize_level in self._resized_data.keys():
            return
        data = {}
        resizer = Resize(
            (self.H, self.W),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )
        data["homo_map"] = resizer(self._homo_map.permute(2, 0, 1)).permute(1, 2, 0)
        data["pixel_int_map"] = (
            torch.from_numpy(
                np.stack(np.meshgrid(np.arange(self.W), np.arange(self.H)), -1)
            )
            .int()
            .to(self.cache_device)
        )
        for name, src in [
            ("depths", self._depths),
            ("epi_errs", self._epi_errs),
            ("rgbs", self._rgbs),
            ("static_masks", self._static_masks),
            ("dynamic_masks", self._dynamic_masks),
            ("depth_masks", self._depth_masks),
            ("flows", self._flows),
            ("flow_masks", self._flow_masks),
            # ("fg_masks", self._fg_masks),
        ]:
            if src is None:
                data[name] = None
            elif src.ndim == 4:
                data[name] = resizer(src.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            elif src.ndim == 3:
                data[name] = resizer(src)
            else:
                raise RuntimeError()
            if name == "flows":
                data[name] = data[name] * resize_level

        # round to smaller abs value int
        flows, flow_masks = self.__round_flows__(
            self.flow_name_list,
            data["flows"],
            data["flow_masks"],
            data["pixel_int_map"],
            self.H,
            self.W,
        )
        data["flows"] = flows.to(self.cache_device)
        data["flow_masks"] = flow_masks.to(self.cache_device)
        self._resized_data[resize_level] = data
        return

    @property
    def track(self):
        return self._track * self.resize_level

    @property
    def short_track(self):
        assert self.use_short_track, "short track is not used"
        return self._short_track * self.resize_level

    @property
    def H(self):
        return int(self._H * self.resize_level)

    @property
    def W(self):
        return int(self._W * self.resize_level)

    @property
    def SemCh(self):
        return self.low_res_dino_featmaps.shape[1]

    @property
    def pixel_size(self):
        return 2.0 / min(self.H, self.W)  # the short size is [-1,+1]

    def __safe_multi_res_handler__(self, default, key):
        # ! detach and clone, otherwise may cause unexpected shallow copy and value change
        if self.resize_level == 1:
            return default.to(self.working_device).detach().clone()
        else:
            return (
                self._resized_data[self.resize_level][key]
                .to(self.working_device)
                .detach()
                .clone()
            )

    def __single_step_safe_multi_res_handler__(self, t, default, key):
        # ! detach and clone, otherwise may cause unexpected shallow copy and value change
        if self.resize_level == 1:
            assert t < len(default), "index out of range"
            return default[t].to(self.working_device).detach().clone()
        else:
            assert t < len(
                self._resized_data[self.resize_level][key]
            ), "index out of range"
            return (
                self._resized_data[self.resize_level][key][t]
                .to(self.working_device)
                .detach()
                .clone()
            )

    ########################################################################################
    @property
    def rgbs(self):
        return self.__safe_multi_res_handler__(self._rgbs, "rgbs")
    
    def get_rgb(self, t):
        return self.__single_step_safe_multi_res_handler__(t, self._rgbs, "rgbs")

    @property
    def semantic_features(self):
        return self._semantic_features


    def get_semantic_feature(self, t):
        feature_dict = self._semantic_features(t)
        ret= {
            k: feature_dict[k].to(self.working_device, non_blocking=True).detach().clone()
            for k in feature_dict.keys()
        }
        return ret

    
    # @property
    # def cls_features(self):
    #     return self.__safe_multi_res_handler__(self._cls_features, "semantic_features")

    # def get_cls_feature(self, t):
    #     return self.__single_step_safe_multi_res_handler__(
    #         t, self._cls_features, "cls_features"
    #     )

    @property
    def depths(self):
        return self.__safe_multi_res_handler__(self._depths, "depths")

    def get_depth(self, t):
        return self.__single_step_safe_multi_res_handler__(t, self._depths, "depths")

    @property
    def normals(self):
        if not hasattr(self, "_normals"):
            raise RuntimeError("Should compute normals first!")
        return self.__safe_multi_res_handler__(self._normals, "normals")

    def get_normal(self, t):
        if not hasattr(self, "_normals"):
            raise RuntimeError("Should compute normals first!")
        return self.__single_step_safe_multi_res_handler__(t, self._normals, "normals")

    @property
    def static_masks(self):
        return self.__safe_multi_res_handler__(self._static_masks, "static_masks")

    def get_static_mask(self, t):
        return self.__single_step_safe_multi_res_handler__(
            t, self._static_masks, "static_masks"
        )

    @property
    def dynamic_masks(self):
        return self.__safe_multi_res_handler__(self._dynamic_masks, "dynamic_masks")

    def get_dynamic_mask(self, t):
        return self.__single_step_safe_multi_res_handler__(
            t, self._dynamic_masks, "dynamic_masks"
        )

    @property
    def depth_masks(self):
        return self.__safe_multi_res_handler__(self._depth_masks, "depth_masks")

    def get_depth_mask(self, t):
        return self.__single_step_safe_multi_res_handler__(
            t, self._depth_masks, "depth_masks"
        )

    ########################################################################################

    @property
    def homo_map(self):
        return self.__safe_multi_res_handler__(self._homo_map, "homo_map")

    @property
    def pixel_int_map(self):
        return self.__safe_multi_res_handler__(self._pixel_int_map, "pixel_int_map")

    @property
    def epi_errs(self):
        return self.__safe_multi_res_handler__(self._epi_errs, "epi_errs")

    @property
    def flows(self):
        return self.__safe_multi_res_handler__(self._flows, "flows")

    @property
    def flow_masks(self):
        return self.__safe_multi_res_handler__(self._flow_masks, "flow_masks")

    #####################################################################################
    # Masks
    #####################################################################################

    @torch.no_grad()
    def get_flow_ind(self, i_ind, j_ind):
        flow_id = self.flow_name_list.index(
            (self.view_list[i_ind], self.view_list[j_ind])
        )
        return flow_id

    @torch.no_grad()
    def __get_bind_prev_valid_mask__(self, key, cur_ind, pre_ind):
        # stricter means: 1. consider the previous validity 2. use depth filtering mask
        mask_j = self.get_mask_by_key(key, cur_ind)
        if cur_ind >= 1:
            # ! considering the previous valid mask as well
            j2i_flow_id = self.flow_name_list.index(
                (self.view_list[cur_ind], self.view_list[pre_ind])
            )
            flow_j2i_mask = self.flow_masks[j2i_flow_id]
            flow_j2i = self.flows[j2i_flow_id]
            i_pixel_index = flow_j2i + self.pixel_int_map
            flow_j2i_query_valid = query_image_buffer_by_pix_int_coord(
                self.get_mask_by_key("dep", pre_ind), i_pixel_index[flow_j2i_mask]
            )
            flow_j2i_mask[flow_j2i_mask.detach().clone()] = flow_j2i_query_valid
            mask_j = mask_j * flow_j2i_mask
        return mask_j

    @torch.no_grad()
    def get_mask_by_key(self, key, ind, prev_ind=None):
        key = key.lower().split("_")
        key.sort()
        key = set(key)
        # ! remove the "all" if something else is in the key
        if "all" in key and len(key) > 1:
            key.remove("all")
        ############################################################
        if "all" in key:
            mask = torch.ones(self.H, self.W).bool().to(self.working_device)
        elif key == set("dep_sta_pre".split("_")):
            # this is for on-line direct
            mask = self.__get_bind_prev_valid_mask__("dep_sta", ind, prev_ind)
        elif key == set("pre".split("_")):
            assert prev_ind is not None
            flow_ind = self.get_flow_ind(i_ind=ind, j_ind=prev_ind)
            mask = self.flow_masks[flow_ind]
        elif key == set("dep_pre".split("_")):
            # do not sup invalid depth and last frame invalid pixels
            mask = self.__get_bind_prev_valid_mask__("dep", ind, prev_ind)
        ############################################################
        elif key == set("sta".split("_")):
            mask = self.get_static_mask(ind)
        elif key == set("sta_dep".split("_")):
            mask = self.get_static_mask(ind) * self.get_depth_mask(ind)
        elif key == set("dyn".split("_")):
            mask = self.get_dynamic_mask(ind)
        elif key == set("dyn_dep".split("_")):
            mask = self.get_dynamic_mask(ind) * self.get_depth_mask(ind)
        elif key == set("dep".split("_")):
            mask = self.get_depth_mask(ind)
        else:
            raise NotImplementedError()
        return mask

    #####################################################################################
    # Update
    #####################################################################################

    def update_mask(self, new_mask, mode="dynamic"):
        if mode == "dynamic":
            new_mask = new_mask.to(self._dynamic_masks)
            self._dynamic_masks = new_mask
            key = "dynamic_masks"
        elif mode == "static":
            new_mask = new_mask.to(self._static_masks)
            self._static_masks = new_mask
            key = "static_masks"
        else:
            raise ValueError(f"mode {mode} not supported")
        for level in self._resized_data.keys():
            resizer = Resize(
                (int(self._H * level), int(self._W * level)),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            )
            self._resized_data[level][key] = resizer(new_mask)
        return

    ######################################################################
    ######################################################################

    @torch.no_grad()
    def compute_normal_maps(
        self,
        cams: SimpleFovCamerasIndependent,
        viz_flag=True,
        patch_size=7,
        nn_dist_th=0.03,
        nn_min_cnt=4,
    ):
        logging.info(f"Computing normal maps from depth maps using local SVD...")
        # ! this function also update the depth mask
        # ! the computed normals are always pointing backward, on -z direction
        ray_direction = backproject(
            self._homo_map, torch.ones_like(self._homo_map[..., 0]), cams
        )
        ray_direction = F.normalize(ray_direction, dim=-1)
        normal_map_list = []
        for t in tqdm(range(self.T)):
            dep = self._depths[t]
            dep_mask = self._depth_masks[t]
            normal_map = torch.zeros(*dep.shape, 3).to(dep)
            xyz = backproject(self._homo_map[dep_mask], dep[dep_mask], cams)
            vtx_map = torch.zeros_like(normal_map).float()
            vtx_map[dep_mask] = xyz

            normal_map, mask = estimate_normal_map(
                vtx_map, dep_mask, patch_size, nn_dist_th, nn_min_cnt
            )

            normal = normal_map[mask]
            inner = (normal * ray_direction[mask]).sum(-1)
            correct_orient = inner < 0
            sign = torch.ones_like(normal[..., :1])
            sign[~correct_orient] = -1.0
            normal = normal.clone() * sign
            normal_map[mask] = normal

            self._depth_masks[t] = self._depth_masks[t] * mask
            normal_map_list.append(normal_map)

        self._normals = torch.stack(normal_map_list, 0).clone()

        # update all other res: dep-mask and normal maps
        for level in self._resized_data.keys():
            resizer = Resize(
                (int(self._H * level), int(self._W * level)),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            )
            self._resized_data[level]["depth_masks"] = resizer(self._depth_masks)
            self._resized_data[level]["normals"] = resizer(self._normals)

        if viz_flag:
            viz_fn = osp.join(self.log_dir, "normal.mp4")
            logging.info(f"Viz normal maps to {viz_fn}")
            viz_frames = (
                ((-self.normals + 1) / 2.0 * 255)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            if len(viz_frames) > 50:
                _step = max(1, len(viz_frames) // 50)
            else:
                _step = 1
            imageio.mimsave(viz_fn, viz_frames[::_step])
        return

    ###############################################
    @torch.no_grad()
    def query_low_res_semantic_feat(self, uv_int, tid):
        # _uv_int: N,2
        low_res_feat = self.low_res_dino_featmaps[tid : tid + 1]
        _grid = uv_int.float()[None, None].to(low_res_feat.device)
        _grid[..., 0] = (_grid[..., 0] / self.W) * 2 - 1
        _grid[..., 1] = (_grid[..., 1] / self.H) * 2 - 1
        sampled_feat = F.grid_sample(
            low_res_feat, grid=_grid, mode="bilinear", align_corners=False
        )[0, :, 0].permute(1, 0)
        return sampled_feat
