import time
import gc
import torch
from diffusers import (
    UNet2DConditionModel,
    ZImageTransformer2DModel,
    AutoencoderKL,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    ZImagePipeline,
    DDIMScheduler,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTextConfig,
    CLIPTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    AutoTokenizer,
)


def create_sdxl_pipe():
    # Build components from their configs (random weights)
    device = "cuda"

    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    text_model_id = "openai/clip-vit-large-patch14"

    # 1) Diffusers parts from config (no UNet / VAE weights)
    unet_cfg = UNet2DConditionModel.load_config(base_model_id, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_cfg)

    vae_cfg = AutoencoderKL.load_config(base_model_id, subfolder="vae")
    vae = AutoencoderKL.from_config(vae_cfg)

    # schedulers don't have large checkpoints anyway, but for completeness:
    scheduler_cfg = DDIMScheduler.load_config(base_model_id, subfolder="scheduler")
    scheduler = DDIMScheduler.from_config(scheduler_cfg)

    # 3) Text encoders: config -> randomly initialized models
    # text_encoder (CLIPTextModel)
    text_cfg_1 = CLIPTextConfig.from_pretrained(
        base_model_id,
        subfolder="text_encoder",
    )
    text_encoder = CLIPTextModel(text_cfg_1)

    # text_encoder_2 (CLIPTextModelWithProjection)
    text_cfg_2 = CLIPTextConfig.from_pretrained(
        base_model_id,
        subfolder="text_encoder_2",
    )
    text_encoder_2 = CLIPTextModelWithProjection(text_cfg_2)

    # 4) Tokenizers (small files; OK to download)
    tokenizer = CLIPTokenizerFast.from_pretrained(
        base_model_id,
        subfolder="tokenizer",
    )
    tokenizer_2 = CLIPTokenizerFast.from_pretrained(
        base_model_id,
        subfolder="tokenizer_2",
    )

    # 5) Assemble SDXL pipeline
    pipe = StableDiffusionXLPipeline(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        # SDXL has optional components like:
        #   add_watermark, feature_extractor
        # you can omit them for benchmarking
    )

    pipe = pipe.to(device, torch.float16)

    return pipe
    # Example timing run
    #prompt = "benchmark run"
    #with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    #    _ = pipe(
    #        prompt, 
    #        num_inference_steps=20,
    #        height=1024,   # desired output height in pixels
    #        width=1024,    # desired output width in pixels
    #        )

def create_z_image_turbo_pipe():
    device = "cuda"

    base_model_id = "Tongyi-MAI/Z-Image-Turbo"

    # 1) Diffusers parts from config (no UNet / VAE weights)
    unet_cfg = ZImageTransformer2DModel.load_config(base_model_id, subfolder="transformer")
    unet = ZImageTransformer2DModel.from_config(unet_cfg)


    vae_cfg = AutoencoderKL.load_config(base_model_id, subfolder="vae")
    vae = AutoencoderKL.from_config(vae_cfg)

    # schedulers don't have large checkpoints anyway, but for completeness:
    scheduler_cfg = FlowMatchEulerDiscreteScheduler.load_config(base_model_id, subfolder="scheduler")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_cfg)


    # 2) Tokenizer (small; downloading is fine)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        subfolder="tokenizer",
    )

    # 3) Config for the text encoder (small as well)
    qwen_config = Qwen3Config.from_pretrained(
        base_model_id,
        subfolder="text_encoder",
    )

    # 4) Instantiate the model *from the config* => random/dummy weights
    text_encoder = Qwen3ForCausalLM(qwen_config)


    # 5) Assemble SDXL pipeline
    pipe = ZImagePipeline(
        vae=vae,
        transformer=unet,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )

    pipe = pipe.to(device, torch.float16)
        
    return pipe

    # Example timing run
    #prompt = "benchmark run"
    #with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    #    _ = pipe(
    #        prompt,
    #        num_inference_steps=20,
    #        height=1024,   # desired output height in pixels
    #        width=1024,    # desired output width in pixels
    #        guidance_scale=0.0,
    #        )

def unload_pipe(pipe):
    # Make sure all pending kernels are done
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Drop the reference in this scope
    del pipe

    # Force Python GC to collect already-deleted objects
    gc.collect()

    # Release cached CUDA memory back to the allocator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_pipe(
    pipe,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float = 0.0,
    height: int = 1024,
    width: int = 1024,
    num_iters: int = 10,
    warmup_iters: int = 2,
) -> float:
    """
    Benchmark a Diffusers pipeline and return iterations/sec.

    Assumes:
      - `pipe` is already moved to CUDA and (optionally) cast to fp16.
      - Guidance-free sampling (guidance_scale=0.0).
    """

    # Warmup (not timed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(warmup_iters):
            _ = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed loop
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(num_iters):
            _ = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    it_per_s = num_iters / elapsed
    return it_per_s

###################################################
##      Begin Benchmarking
###################################################

sdxl_prompt = "This is a test prompt" # does not need to be long as CLIP is always 77 tokens
z_image_prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."


sdxl_pipe = create_sdxl_pipe()

num_sdxl_steps=20
sdxl_speed = benchmark_pipe(
    pipe=sdxl_pipe,
    prompt=sdxl_prompt,
    num_inference_steps=num_sdxl_steps,
    guidance_scale=8.0,
    height=1024,
    width=1024,
    num_iters=20,
    warmup_iters=5,
    )*num_sdxl_steps


unload_pipe(sdxl_pipe)
sdxl_pipe = None


z_image_pipe = create_z_image_turbo_pipe()

num_z_steps = 9
z_image_speed = benchmark_pipe(
    pipe=z_image_pipe,
    prompt= z_image_prompt,
    num_inference_steps=num_z_steps,
    guidance_scale=0.0,
    height=1024,
    width=1024,
    num_iters=20,
    warmup_iters=5,
    )*num_z_steps

print(f"SDXL Speed:   {sdxl_speed:.3f} it/s")
print(f"Z-Image Speed: {z_image_speed:.3f} it/s")
