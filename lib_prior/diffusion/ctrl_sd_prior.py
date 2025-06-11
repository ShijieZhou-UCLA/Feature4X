# use SD to hallucinate the missing part and enhance the novel views
# there are two way: 1. direct fil the holes 2. use sds to enhance the novel views, which has no missing part!

# if only use one control net inpaint model, check degenerated case, what if the mask is non? what if the image is corrupted??

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers.utils import load_image
from diffusers import DDIMScheduler
import numpy as np
import torch
from PIL import Image


def cvt21pil(img01):
    # input range [0,1]
    assert img01.max() <= 1.0 and img01.min() >= 0.0
    if isinstance(img01, np.ndarray):
        np_img255 = np.clip(img01 * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img255)
    elif isinstance(img01, torch.Tensor):
        np_img255 = np.clip(img01.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img255)
    else:
        raise NotImplementedError
    return pil_img


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert (
        image.shape[0:1] == image_mask.shape[0:1]
    ), "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


class CtrSDPrior:
    def __init__(self) -> None:
        self.res = 512  # use this size

        self.generator = torch.Generator(device="cpu").manual_seed(1)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
        )
        
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()  # help reduce vram by using cpu ram

    def inpaint(self, init_image, mask_image, text=None, diff_steps=20):
        # 512, 512
        assert init_image.ndim == 3 and init_image.shape[2] == 3
        assert init_image.shape[0] == init_image.shape[1]
        assert init_image.shape[:2] == mask_image.shape[:2]

        init_pil = cvt21pil(init_image)
        mask_pil = cvt21pil(mask_image)
        init_pil = init_pil.resize((self.res, self.res))
        mask_pil = mask_pil.resize((self.res, self.res))

        control_pil = make_inpaint_condition(init_pil, mask_pil)

        ret_pil = self.pipe(
            text,
            num_inference_steps=diff_steps,
            generator=self.generator,
            eta=1.0,
            image=init_pil,
            mask_image=mask_pil,
            control_image=control_pil,
        ).images[0]

        ret = np.asarray(ret_pil).astype(np.float32) / 255.0
        return ret

    def distill(self):
        raise NotImplementedError()


if __name__ == "__main__":
    import imageio

    img_fn = "../data/DAVIS/train/images/00000.jpg"

    img_np = np.asarray(imageio.imread(img_fn)).astype(np.float32) / 255.0

    img_pil = cvt21pil(img_np)

    img_pil.save("test.png")
