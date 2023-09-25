# AnimateDiff for ComfyUI

[AnimateDiff](https://github.com/guoyww/AnimateDiff/) integration for ComfyUI, adapts from [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff). Please read the original repo README for more information.

## How to Use

1. Clone this repo into `custom_nodes` folder.
2. Download motion modules and put them under `comfyui-animatediff/models/`.

- Original modules: [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836) | [Baidu NetDisk](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y)
- Community modules: [manshoety/AD_Stabilized_Motion](https://huggingface.co/manshoety/AD_Stabilized_Motion) | [CiaraRowles/TemporalDiff](https://huggingface.co/CiaraRowles/TemporalDiff)
- AnimateDiff v2 [mm_sd_v15_v2.ckpt](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt)

## Update 2023/09/25

#### **Motion LoRA** is now supported!

Download [motion LoRAs](https://huggingface.co/guoyww/animatediff/tree/main) and put them under `comfyui-animatediff/loras/` folder.

Note: LoRAs only work with **AnimateDiff v2** [mm_sd_v15_v2.ckpt](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt) module.

#### New node: `AnimateDiffLoraLoader`

<img width="370" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/7a9f62f7-702e-48a4-934c-bbfe1e23aff2">

Example workflow:
<img width="1280" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/93e7550f-4648-4482-9961-6cece5132dc9">

Workflow: [lora.json](https://github.com/ArtVentureX/comfyui-animatediff/blob/main/workflows/lora.json)

Samples:

<table>
<tr>
<td>
<img width="512" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/2c5aa25e-0682-481f-8842-066c5b988864">
</td>
</tr>
<tr>
<td>
<img width="512" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/adfbad45-3ba5-42e3-9bee-d2b83f43989c">
</td>
</tr>
<tr>
<td>
<img width="512" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/8e484c74-c691-4d1c-9514-719dbfe3a0b5">
</td>
</tr>
<tr>
<td>
<img width="512" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/4921a335-9207-4a7b-9d66-61a5d76e3179">
</td>
</tr>
</table>

## Update 2023/09/21

#### **Sliding Window** is now available!

The sliding window feature enables you to generate GIFs without a frame length limit. It divides frames into smaller batches with a slight overlap. This feature is activated automatically when generating more than 16 frames. To modify the trigger number and other settings, utilize the `SlidingWindowOptions` node. See the [sample workflow](#long-duration-with-sliding-window) bellow.

## Nodes

#### AnimateDiffLoader

<img width="370" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/9d756d01-ea45-4d1c-8e48-56f2725c7ca1">

#### AnimateDiffSampler

- Mostly the same with `KSampler`
- `motion_module`: use `AnimateDiffLoader` to load the motion module
- `inject_method`: should left default
- `frame_number`: animation length
- `latent_image`: You can pass an `EmptyLatentImage`
- `sliding_window_opts`: custom sliding window options

<img width="370" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/a352195d-f40c-494d-bd3d-30ee88174b88">

#### AnimateDiffCombine

- Combine GIF frames and produce the GIF image
- `frame_rate`: number of frame per second
- `loop_count`: use 0 for infinite loop
- `save_image`: should GIF be saved to disk
- `format`: supports `image/gif`, `image/webp` (better compression), `video/webm`, `video/h264-mp4`, `video/h265-mp4`. To use video formats, you'll need [ffmpeg](https://ffmpeg.org/download.html) installed and available in **`PATH`**

<img width="370" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/381c5acc-06ef-43da-ada0-3dc76f37a3e4">

#### SlidingWindowOptions

Custom sliding window options

- `context_length`: number of frame per _window_. Use **16** to get the best results. Reduce it if you have low VRAM.
- `context_stride`:
  - 1: sampling every frame
  - 2: sampling every frame then every second frame
  - 3: sampling every frame then every second frame then every third frames
  - ...
- `context_overlap`: overlap frames between each window slice
- `closed_loop`: make the GIF a closed loop, will add more sampling step

<img width="370" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/6679a8dd-bf96-419f-8934-ea2b046dd23c">

#### LoadVideo

Load GIF or video as images. Usefull to load a GIF as ControlNet input.

- `frame_start`: Skip some begining frames and start at `frame_start`
- `frame_limit`: Only take `frame_limit` frames

<img width="370" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/684176d5-6369-4a27-9f33-e721e0fe1876">

## Workflows

### Simple txt2gif

<img width="1280" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/b7164539-bc58-4ef9-b178-d914e833805e">

Workflow: [simple.json](https://github.com/ArtVentureX/comfyui-animatediff/blob/main/workflows/simple.json)

Samples:

![animate_diff_01](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/97efb96f-3d3d-4976-8789-78b88f89b2eb)

![animate_diff_02](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/c39b26f7-a2af-4dc4-902f-c363e2e6f39a)

### Long duration with sliding window

<img width="1280" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/0f8bfb87-83cb-4119-9777-e3948ec0cb5c">

Workflow: [sliding-window.json](https://github.com/ArtVentureX/comfyui-animatediff/blob/main/workflows/sliding-window.json)

Samples:

<table>
<tr>
<td>
<img width="512" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/e1da7a66-e615-475d-9400-41eff484ad49">
</td>
</tr>
<tr>
<td>
<img width="768" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/4faa7e5e-cdaa-49da-8759-46d779c0e0b6">
</td>
</tr>
</table>

### Latent upscale

Upscale latent output using `LatentUpscale` then do a 2nd pass with `AnimateDiffSampler`.

<img width="1280" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/987a1c5a-c1f8-4b24-8c62-f14496261d6c">

Workflow: [latent-upscale.json](https://github.com/ArtVentureX/comfyui-animatediff/blob/main/workflows/latent-upscale.json)

Samples:
![animate_diff_upscale](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/f363f6f8-3117-4fa8-bca9-62f6a6e38ce7)

### Using with ControlNet

You will need following additional nodes:

- [Kosinkadink/ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet): Apply different weight for each latent in batch
- [Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux): ControlNet preprocessors

#### Animate with starting and ending images

- Use `LatentKeyframe` and `TimestampKeyframe` from [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) to apply diffrent weights for each latent index.
- Use 2 controlnet modules for two images with weights reverted.

![image](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/bcca1070-e4a1-4698-a2af-aadf9723d015)

Workflow: [cn-2images.json](https://github.com/ArtVentureX/comfyui-animatediff/blob/main/workflows/cn-2images.json)

Samples:

<table>
<tr>
<td>
<img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/e73fc3cd-a590-40a9-8b33-11358b54f0cd">
</td>
<td>
<img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/96c2ee92-d457-4862-94d3-d675b7fa2d1f">
</td>
</tr>
<tr>
<td>
<img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/46338853-1ae0-433e-925c-2a41e0382e68">
</td>
<td>
<img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/707e4ce3-3594-4ff5-9a5f-f9596eb2bcf4">
</td>
</tr>
</table>

#### Using GIF as ControlNet input

Using a GIF (or video, or a list of images) as ControlNet input.

![image](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/cfeed634-e683-4797-b2fd-dbe0926a449e)

Workflow: [cn-vid2vid.json](https://github.com/ArtVentureX/comfyui-animatediff/blob/main/workflows/cn-vid2vid.json)

Samples:

<table>
<tr>
<td>
<img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/bf926f52-da97-4fb4-b86a-8b26ef5fab04">
</td>
<td>
<img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/f6472c8c-9b92-47c2-8f28-638726f21be7">
</td>
</tr>
</table>

## Known Issues

### CUDA error: invalid configuration argument

It's an `xformers` bug accidentally triggered by the way the original AnimateDiff CrossAttention is passed in. The current workaround is to disable xformers with `--disable-xformers` when booting ComfyUI.

### GIF split into multiple scenes

![AnimateDiff_00007_](https://github.com/ArtVentureX/comfyui-animatediff/assets/8894763/e6cd53cb-9878-45da-a58a-a15851882386)

Work around:

- Shorter your prompt and negative prompt
- Reduce resolution. AnimateDiff is trained on 512x512 images so it works best with 512x512 output.
- Disable xformers with `--disable-xformers`

### GIF has Wartermark (especially when using mm_sd_v15)

See: https://github.com/continue-revolution/sd-webui-animatediff/issues/31

Training data used by the authors of the AnimateDiff paper contained Shutterstock watermarks. Since mm_sd_v15 was finetuned on finer, less drastic movement, the motion module attempts to replicate the transparency of that watermark and does not get blurred away like mm_sd_v14. Try other community finetuned modules.
