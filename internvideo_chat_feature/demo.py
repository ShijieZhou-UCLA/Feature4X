import os
try:
  token =os.environ['HF_TOKEN']
except:
  print("paste your hf token here!")
  token = "hf_YourToken"
  os.environ['HF_TOKEN'] = token
os.environ["http_proxy"] = "http://localhost:22333"
os.environ["https_proxy"] = "http://localhost:22333"
import torch
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

from transformers import AutoTokenizer, AutoModel

# ========================================
#             Model Initialization
# ========================================

tokenizer =  AutoTokenizer.from_pretrained('./',
    # trust_remote_code=True,
    use_fast=False,
    # token=token
                                           )
if torch.cuda.is_available():
  model = AutoModel.from_pretrained(
      './',
      torch_dtype=torch.bfloat16,
      trust_remote_code=True
  ).cuda()
else:
  model = AutoModel.from_pretrained(
      './',
      torch_dtype=torch.bfloat16,
      trust_remote_code=True)

from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import numpy as np
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
decord.bridge.set_bridge("torch")

# ========================================
#          Define Utils
# ========================================
def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)

    T_, C, H, W = frames.shape
        
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames

# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state = []
    if img_list is not None:
        img_list = None
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list


def upload_img( gr_video, num_segments, hd_num, padding):
    img_list = []
    if gr_video is None:
        return None, None, gr.update(interactive=True),gr.update(interactive=True, placeholder='Please upload video/image first!'),  None
    if gr_video:
        video_tensor, msg = load_video(gr_video, num_segments=num_segments, return_msg=True)
        video_tensor = video_tensor.to(model.device)
        return gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), video_tensor
    # if gr_img:
    #     llm_message, img_list,chat_state = chat.upload_img(gr_img, chat_state, img_list)
    #     return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False)

def clear_():
    return [], []

def gradio_ask(user_message, chatbot):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chatbot = chatbot + [[user_message, None]]
    return user_message, chatbot


def gradio_answer(chatbot, sys_prompt, user_prompt, video_tensor, chat_state, num_beams, temperature, do_sample=False):
    response, chat_state = model.chat(tokenizer,
                                        sys_prompt,
                                        user_prompt,
                                        media_type='video',
                                        media_tensor=video_tensor,
                                        chat_history= chat_state,
                                        return_history=True,
                                        generation_config={
                                            "num_beams": num_beams,
                                            "temperature": temperature,
                                            "do_sample": do_sample})
    print(response)
    chatbot[-1][1] = response
    return chatbot, chat_state


class OpenGVLab(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )


gvlabtheme = OpenGVLab(primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        )

title = """<h1 align="center"><a href="https://github.com/OpenGVLab/Ask-Anything"><img src="https://s1.ax1x.com/2023/05/07/p9dBMOU.png" alt="Ask-Anything" border="0" style="margin: 0 auto; height: 100px;" /></a> </h1>"""
description ="""
        VideoChat2 powered by InternVideo!<br><p><a href='https://github.com/OpenGVLab/Ask-Anything'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p>
        """
SYS_PROMPT =""

with gr.Blocks(title="InternVideo-VideoChat!",theme=gvlabtheme,css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5, visible=True) as video_upload:
            with gr.Column(elem_id="image", scale=0.5) as img_part:
                # with gr.Tab("Video", elem_id='video_tab'):
                up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload")
                # with gr.Tab("Image", elem_id='image_tab'):
                #     up_image = gr.Image(type="pil", interactive=True, elem_id="image_upload")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            restart = gr.Button("Restart")
            sys_prompt = gr.State(f"{SYS_PROMPT}")

            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                                                                 label="beam search numbers)",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,label="Temperature",
            )

            num_segments = gr.Slider(
                minimum=8,
                maximum=64,
                value=8,
                step=1,
                interactive=True,
                label="Input Frames",
            )

            resolution = gr.Slider(
                minimum=224,
                maximum=224,
                value=224,
                step=1,
                interactive=True,
                label="Vision encoder resolution",
            )

            hd_num = gr.Slider(
                minimum=1,
                maximum=10,
                value=4,
                step=1,
                interactive=True,
                label="HD num",
            )

            padding = gr.Checkbox(
                label="padding",
                info=""
            )

        with gr.Column(visible=True)  as input_raws:
            chat_state = gr.State([])
            img_list = gr.State()
            chatbot = gr.Chatbot(elem_id="chatbot",label='VideoChat')
            with gr.Row():
                with gr.Column(scale=0.7):
                    text_input = gr.Textbox(show_label=False, placeholder='Please upload your video first', interactive=False)
                with gr.Column(scale=0.15, min_width=0):
                    run = gr.Button("💭Send")
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("🔄Clear️")

    upload_button.click(upload_img, [ up_video, num_segments, hd_num, padding], [ up_video, text_input, upload_button, img_list])

    text_input.submit(gradio_ask, [text_input, chatbot], [text_input, chatbot]).then(
        gradio_answer, [chatbot, sys_prompt, text_input, img_list, chat_state, num_beams, temperature], [chatbot, chat_state]
    )
    run.click(gradio_ask, [text_input, chatbot], [text_input, chatbot]).then(
        gradio_answer, [chatbot, sys_prompt, text_input, img_list, chat_state, num_beams, temperature], [chatbot, chat_state]
    )
    run.click(lambda: "", None, text_input)
    clear.click(clear_, None, [chatbot, chat_state])
    restart.click(gradio_reset, [chat_state, img_list], [chatbot,  up_video, text_input, upload_button, chat_state, img_list], queue=False)

demo.launch(server_name="0.0.0.0")