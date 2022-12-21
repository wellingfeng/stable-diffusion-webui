import modules.scripts
from modules import sd_samplers
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
from PIL import Image

from clip_interrogator.clip_interrogator.clip_interrogator import Config, Interrogator
clip_model_name = 'ViT-L-14/openai'
config = Config()
config.blip_num_beams = 64
config.blip_offload = False
config.clip_model_name = clip_model_name
ci = Interrogator(config)

def inference(image, mode, best_max_flavors=32):
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image, max_flavors=int(best_max_flavors))
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)
    
def image2tile(files):
    outStr = ""
    for idx, file in enumerate(files):
            image = Image.open(file.name).convert('RGB')
            outStr += inference(image, 'best') + "\n\n"
    return outStr

def generateTileImgs(prompt_img2tile_result):
    promptItem = prompt_img2tile_result.split("\n\n")[0]
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=promptItem,
        styles=[None, None],
        negative_prompt="",
        seed=-1,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=False,
        sampler_name=sd_samplers.samplers[0].name,
        batch_size=1,
        n_iter=1,
        steps=20,
        cfg_scale=7,
        width=512,
        height=512,
        restore_faces=False,
        tiling=True,
        enable_hr=False,
        denoising_strength=0,
        firstphase_width=0,
        firstphase_height=0.7,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = None

    if cmd_opts.enable_console_prompts:
        print(f"\nimg2tile: {promptItem}", file=shared.progress_print_out)

    processed = modules.scripts.scripts_txt2img.run(p,[None])

    if processed is None:
        processed = process_images(p)

    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info)