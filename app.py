import os
os.environ["PYTORCH_DISABLE_NNPACK"] = "1"  # (optional) silence NNPACK warning

import streamlit as st
import torch
import diffusers
import transformers
from PIL import Image


def has_accelerate():
    try:
        import accelerate  # noqa: F401
        return True
    except Exception:
        return False


# Device & dtype
@st.cache_resource
def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


# Load models (cached)
@st.cache_resource
def load_models(model_id, device, dtype):
    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer"
    )

    text_encoder = transformers.CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype
    ).to(device).eval()

    vae = diffusers.AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype
    ).to(device).eval()

    unet = diffusers.UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=dtype,
        low_cpu_mem_usage=has_accelerate()
    ).to(device).eval()

    scheduler = diffusers.UniPCMultistepScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )

    return tokenizer, text_encoder, vae, unet, scheduler


# Prompt → embeddings
@torch.no_grad()
def get_embeddings(prompt, negative_prompt, tokenizer, encoder, device):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    neg_inputs = tokenizer(
        negative_prompt,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    text_emb = encoder(text_inputs.input_ids.to(device)).last_hidden_state
    neg_emb = encoder(neg_inputs.input_ids.to(device)).last_hidden_state

    return torch.cat([neg_emb, text_emb], dim=0)


# Latent creation
def create_latents(vae, scheduler, device, dtype, height, width):
    latent_shape = (1, vae.config.latent_channels, height // 8, width // 8)
    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    return latents * scheduler.init_noise_sigma


# Denoising loop
@torch.no_grad()
def denoise(latents, unet, scheduler, embeddings, guidance_scale, steps):
    scheduler.set_timesteps(steps)

    for t in scheduler.timesteps:
        latent_input = torch.cat([latents, latents], dim=0)
        latent_input = scheduler.scale_model_input(latent_input, t)

        noise_pred = unet(
            latent_input,
            t,
            encoder_hidden_states=embeddings
        ).sample

        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


# Latents → Image
@torch.no_grad()
def latents_to_image(latents, vae):
    decoded = vae.decode(latents / vae.config.scaling_factor).sample
    image = (decoded / 2 + 0.5).clamp(0, 1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = (image * 255).astype("uint8")
    return Image.fromarray(image)


# STREAMLIT UI
st.set_page_config(page_title="Stable Diffusion (CPU)", layout="centered")
st.title("Stable Diffusion Prototype Streamlit App")

prompt = st.text_input("Prompt", "A red bird flying over a green forest")
negative_prompt = st.text_input("Negative Prompt", "")

steps = st.slider("Denoising Steps", 10, 50, 30)
guidance = st.slider("Guidance Scale (CFG)", 1.0, 15.0, 7.5)

generate = st.button("Generate Image")

if generate:
    with st.spinner("Generating image (CPU — this may take a few minutes)…"):
        device, dtype = get_device_and_dtype()

        tokenizer, encoder, vae, unet, scheduler = load_models(
            "CompVis/stable-diffusion-v1-4",
            device,
            dtype
        )

        embeddings = get_embeddings(prompt, negative_prompt, tokenizer, encoder, device)

        latents = create_latents(vae, scheduler, device, dtype, height=512, width=512)
        latents = denoise(latents, unet, scheduler, embeddings, guidance, steps)

        image = latents_to_image(latents, vae)

    # FIXED: no deprecated parameter
    st.image(image, caption="Generated Image", width="stretch")
