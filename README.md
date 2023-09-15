# AnimateDiff for ComfyUI

[AnimateDiff](https://github.com/guoyww/AnimateDiff/) integration for ComfyUI, adapts from [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff). Please read the original repo README for more information.

## How to Use

1. Clone this repo into `custom_nodes` folder.
2. Download motion modules from [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836) | [Baidu NetDisk](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y). You only need to download one of `mm_sd_v14.ckpt` | `mm_sd_v15.ckpt`. Put the model weights under `comfyui-animatediff/models/`. DO NOT change model filename.

#### Update 2023/09/15

- You can now use community models from [manshoety/AD_Stabilized_Motion](https://huggingface.co/manshoety/AD_Stabilized_Motion) or [CiaraRowles/TemporalDiff](https://huggingface.co/CiaraRowles/TemporalDiff)
- Supports AnimateDiff v2 [mm_sd_v15_v2.ckpt](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt) model
- Fix image is grayed out.
- New node: **AnimateDiffSampler** and **AnimateDiffLoader**
  - Mostly the same with `KSampler`
  - Use `AnimateDiffLoader` to load the motion module
  - `inject_method`: should left default. See [this issue](https://github.com/ArtVentureX/comfyui-animatediff#gif-has-wartermark-after-update-to-the-latest-version) for more details.
  - `frame_number`: animation length

<img width="506" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/f22d6b36-ce36-44cc-80e8-dffe6f77b296">

#### Example Workflow

<img width="1311" alt="image" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/b7164539-bc58-4ef9-b178-d914e833805e">

## Samples

![23b44c29-29e8-4f48-ab3c-4df87c90c13f](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/97efb96f-3d3d-4976-8789-78b88f89b2eb)

![25f6c60c-f8ac-4abe-984f-1559c355d7f6](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/c39b26f7-a2af-4dc4-902f-c363e2e6f39a)

## Known Issues

### GIF split into multiple scenes

<<<<<<< Updated upstream
![AnimateDiff_00002](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/c78d64b9-b308-41ec-9804-bbde654d0b47)

## Known Issues

### GIF split into multiple scenes

![AnimateDiff_00007_](https://github.com/ArtVentureX/comfyui-animatediff/assets/8894763/e6cd53cb-9878-45da-a58a-a15851882386)

This is usually due to memory (VRAM) is not enough to process the whole image batch at the same time. Try reduce the image size and frame number.

### GIF has Wartermark after update to the latest version

See https://github.com/continue-revolution/sd-webui-animatediff/issues/31
=======
![AnimateDiff_00007_](https://github.com/ArtVentureX/comfyui-animatediff/assets/8894763/e6cd53cb-9878-45da-a58a-a15851882386)

See: https://github.com/continue-revolution/sd-webui-animatediff/issues/38

Main reasons:

- Promt are too long (more than 75 tokens)
- Resolution are too high
- Number of frame too high

Work around:

- Shorter your prompt and negative prompt
- Reduce resolution. AnimateDiff is trained on 512x512 images so it works best with 512x512 output.
- Shouldn't generate longer than 16 frames. AnimateDiff is trained to output the best results with 16 frames.

### GIF has Wartermark after update to the latest version

See: https://github.com/continue-revolution/sd-webui-animatediff/issues/31
>>>>>>> Stashed changes

As mentioned in the issue thread, it seems to be due to the training dataset. The new version is the correct implementation and produces smoother GIFs compared to the older version.

<table  class="center">
    <tr>
    <td>Old revision</td>
    <td>New revision</td>
    </tr>
    <tr>
    <td><img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/8f1a6233-875f-4f0c-aa60-ba93e73b7d64" /></td>
    <td><img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/a2029eba-f519-437c-a0b5-1f881e099a20" /></td>
    </tr>
    <tr>
    <td><img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/41ec449f-1955-466c-bd38-6f2a55d654f8" /></td>
    <td><img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/766c2891-5d27-4052-99f9-be9862620919" /></td>
    </tr>
</table>

<<<<<<< Updated upstream

I played around with both version and found that the watermark only present in some models, not always. So I've brought back the old method and also created a new node with the new method. You can try both to find the best fit for each model.

![Screenshot 2023-07-28 at 18 14 14](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/25cf6092-3e67-435e-86cc-43614ca7d6aa)
=======
I played around with both version and found that the watermark only present in some models, not always. To use the **old (legacy)** method, change `injection_method` to `legacy` in the `AnimateDiffSampler` node.
>>>>>>> Stashed changes
