# the base class of shading function
import torch


class BaseShader:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        gb_pos,  # world space position
        gb_geometric_normal,  # world space mesh normal
        gb_normal,  # actual normal used
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        **kwargs
    ):
        ################################################################################
        #
        raise NotImplementedError()
        return buffers


class NVDiffRecShader(BaseShader):
    def __init__(self) -> None:
        super().__init__()
        pass

    def __call__(
        self,
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        # additional args
        view_pos,
        lgt,
        material,
        bsdf,
    ):
        ################################################################################
        # Texture lookups
        ################################################################################
        perturbed_nrm = None
        if "kd_ks_normal" in material:
            # Combined texture, used for MLPs because lookups are expensive
            all_tex_jitter = material["kd_ks_normal"].sample(
                gb_pos
                + torch.normal(mean=0, std=0.01, size=gb_pos.shape, device="cuda")
            )
            all_tex = material["kd_ks_normal"].sample(gb_pos)
            assert (
                all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10
            ), "Combined kd_ks_normal must be 9 or 10 channels"
            kd, ks, perturbed_nrm = (
                all_tex[..., :-6],
                all_tex[..., -6:-3],
                all_tex[..., -3:],
            )
            # Compute albedo (kd) gradient, used for material regularizer
            kd_grad = (
                torch.sum(
                    torch.abs(all_tex_jitter[..., :-6] - all_tex[..., :-6]),
                    dim=-1,
                    keepdim=True,
                )
                / 3
            )
        else:
            kd_jitter = material["kd"].sample(
                gb_texc
                + torch.normal(mean=0, std=0.005, size=gb_texc.shape, device="cuda"),
                gb_texc_deriv,
            )
            kd = material["kd"].sample(gb_texc, gb_texc_deriv)
            ks = material["ks"].sample(gb_texc, gb_texc_deriv)[..., 0:3]  # skip alpha
            if "normal" in material:
                perturbed_nrm = material["normal"].sample(gb_texc, gb_texc_deriv)
            kd_grad = (
                torch.sum(
                    torch.abs(kd_jitter[..., 0:3] - kd[..., 0:3]), dim=-1, keepdim=True
                )
                / 3
            )

        # Separate kd into alpha and color, default alpha = 1
        alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1])
        kd = kd[..., 0:3]

        ################################################################################
        # Normal perturbation & normal bend
        ################################################################################
        if "no_perturbed_nrm" in material and material["no_perturbed_nrm"]:
            perturbed_nrm = None

        gb_normal = ru.prepare_shading_normal(
            gb_pos,
            view_pos,
            perturbed_nrm,
            gb_normal,
            gb_tangent,
            gb_geometric_normal,
            two_sided_shading=True,
            opengl=True,
        )

        ################################################################################
        # Evaluate BSDF
        ################################################################################

        assert (
            "bsdf" in material or bsdf is not None
        ), "Material must specify a BSDF type"
        bsdf = material["bsdf"] if bsdf is None else bsdf
        if bsdf == "pbr":
            # if isinstance(lgt, light.EnvironmentLight):
            #     shaded_col = lgt.shade(
            #         gb_pos, gb_normal, kd, ks, view_pos, specular=True
            #     )
            # else:
            assert False, "Invalid light type"
        elif bsdf == "diffuse":
            # if isinstance(lgt, light.EnvironmentLight):
            #     shaded_col = lgt.shade(
            #         gb_pos, gb_normal, kd, ks, view_pos, specular=False
            #     )
            # else:
            assert False, "Invalid light type"
        elif bsdf == "normal":
            shaded_col = (gb_normal + 1.0) * 0.5
        elif bsdf == "tangent":
            shaded_col = (gb_tangent + 1.0) * 0.5
        elif bsdf == "kd":
            shaded_col = kd
        elif bsdf == "ks":
            shaded_col = ks
        else:
            assert False, "Invalid BSDF '%s'" % bsdf

        # Return multiple buffers
        buffers = {
            "shaded": torch.cat((shaded_col, alpha), dim=-1),
            "kd_grad": torch.cat((kd_grad, alpha), dim=-1),
            "occlusion": torch.cat((ks[..., :1], alpha), dim=-1),
        }
        return buffers


class NaiveShader(BaseShader):
    def __init__(self) -> None:
        super().__init__()
        pass

    def __call__(
        self,
        gb_pos,  # world space position
        gb_geometric_normal,  # world space mesh normal
        gb_normal,  # actual normal used
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        # opt
        naive_appearance,
        **kwargs
    ):
        color = naive_appearance.sample(gb_pos)
        alpha = torch.ones_like(color[..., :1])
        shade = torch.cat((color, alpha), dim=-1)
        buffers = {
            "shaded": shade,
        }

        return buffers
