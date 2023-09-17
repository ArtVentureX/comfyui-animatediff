# AnimateDiff for ComfyUI

[AnimateDiff](https://github.com/guoyww/AnimateDiff/) integration for ComfyUI, adapts from [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff). Please read the original repo README for more information.

## How to Use

1. Clone this repo into `custom_nodes` folder.
2. Download motion modules and put them under `comfyui-animatediff/models/`.
  * Original modules: [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836) | [Baidu NetDisk](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y)
  * Community modules: [manshoety/AD_Stabilized_Motion](https://huggingface.co/manshoety/AD_Stabilized_Motion) | [CiaraRowles/TemporalDiff](https://huggingface.co/CiaraRowles/TemporalDiff)
  * AnimateDiff v2 [mm_sd_v15_v2.ckpt](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt)

## Nodes

#### AnimateDiffLoader

<img width="370" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/9d756d01-ea45-4d1c-8e48-56f2725c7ca1">

#### AnimateDiffSampler
- Mostly the same with `KSampler`
- Use `AnimateDiffLoader` to load the motion module
- `inject_method`: should left default
- `frame_number`: animation length
- `latent_image`: You can pass an `EmptyLatentImage`

<img width="370" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/f22d6b36-ce36-44cc-80e8-dffe6f77b296">

#### AnimateDiffCombine
- Combine GIF frames and produce the GIF image
- `frame_rate`: number of frame per second
- `loop_count`: use 0 for infinite loop
- `save_image`: should GIF be saved to disk
- `format`: supports `image/gif`, `image/webp` (better compression) or `video/webm` (need `ffmpeg` installed and available in PATH)

<img width="370" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/381c5acc-06ef-43da-ada0-3dc76f37a3e4">

#### Example Workflow

<img width="1311" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/b7164539-bc58-4ef9-b178-d914e833805e">


Workflow file: https://github.com/ArtVentureX/comfyui-animatediff/blob/main/workflow.json

## Samples

![23b44c29-29e8-4f48-ab3c-4df87c90c13f](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/97efb96f-3d3d-4976-8789-78b88f89b2eb)

![25f6c60c-f8ac-4abe-984f-1559c355d7f6](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/c39b26f7-a2af-4dc4-902f-c363e2e6f39a)

## Known Issues

### GIF split into multiple scenes

![AnimateDiff_00007_](https://github.com/ArtVentureX/comfyui-animatediff/assets/8894763/e6cd53cb-9878-45da-a58a-a15851882386)

Work around:

- Shorter your prompt and negative prompt
- Reduce resolution. AnimateDiff is trained on 512x512 images so it works best with 512x512 output.
- Disable xformers with `--disable-xformers`

### GIF has Wartermark (especially when using mm_sd_v15)

See: https://github.com/continue-revolution/sd-webui-animatediff/issues/31

Training data used by the authors of the AnimateDiff paper contained Shutterstock watermarks. Since mm_sd_v15 was finetuned on finer, less drastic movement, the motion module attempts to replicate the transparency of that watermark and does not get blurred away like mm_sd_v14. Try other community finetuned modules.
