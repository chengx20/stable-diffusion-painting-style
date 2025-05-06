import os
import torch
import base64
from io import BytesIO
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
)


def load_custom_pipeline(
    repo_id: str = "xcheng20/stable-diffusion-painting-style-v1",
    subfolder: str = "fine-tuned-model",
    use_mps_if_available: bool = True,
    hf_token: str = None,
    cache_dir: str = None,
):
    """
    1) snapshot_download pulls the HF repo.
    2) We then manually load tokenizer/text_encoder/unet/vae from the `fine-tuned-model` subfolder.
    """
    # ── 1) pull the repo locally ──────────────────────────────────────────
    repo_dir = snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        use_auth_token=hf_token,
    )

    # ── 2) point at the exact subfolder where you stored your fine-tuned bits ──
    model_dir = os.path.join(repo_dir, subfolder)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Could not find '{subfolder}' inside '{repo_dir}'")

    # ── 3) manual component loading, all local only ────────────────────────
    tokenizer = CLIPTokenizer.from_pretrained(
        os.path.join(model_dir, "tokenizer"),
        local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        os.path.join(model_dir, "text_encoder"),
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(model_dir, "unet"),
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(
        os.path.join(model_dir, "vae"),
        torch_dtype=torch.float32,
        local_files_only=True,
    )

    # ── 4) scheduler still comes from the official v1-4 repo ───────────────
    scheduler = DDIMScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="scheduler",
        torch_dtype=torch.float32,
    )

    # ── 5) stitch it all together ──────────────────────────────────────────
    pipe = StableDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )

    # ── 6) device placement & memory tweaks ────────────────────────────────
    device = (
        torch.device("mps")
        if (torch.backends.mps.is_available() and use_mps_if_available)
        else torch.device("cpu")
    )
    pipe.to(device)
    pipe.enable_attention_slicing()

    return pipe


def load_base_pipeline(
    model_id: str = "CompVis/stable-diffusion-v1-4",
    use_mps_if_available: bool = True
):
    """
    Loads the original Stable Diffusion v1.4 model from Hugging Face.
    Returns a pipeline object ready for inference.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        feature_extractor=None
    )
    device = (
        torch.device("mps") if (torch.backends.mps.is_available() and use_mps_if_available)
        else torch.device("cpu")
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = None
):
    """
    Generates a single image from the provided pipeline and prompt.
    Optionally accepts a 'seed' for reproducibility.
    """
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
    return result.images[0]


def pil_image_to_base64_str(img: Image.Image) -> str:
    """
    Converts a PIL Image into a Base64-encoded PNG string.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
