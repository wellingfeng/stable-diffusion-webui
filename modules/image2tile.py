import modules.scripts
from modules import sd_samplers
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html

# from clip_interrogator.clip_interrogator.clip_interrogator import Config, Interrogator
# clip_model_name = 'ViT-L-14/openai'
# config = Config()
# config.blip_num_beams = 64
# config.blip_offload = False
# config.clip_model_name = clip_model_name
# ci = Interrogator(config)

# def inference(image, mode, best_max_flavors=32):
#     ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
#     ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
#     image = image.convert('RGB')
#     if mode == 'best':
#         return ci.interrogate(image, max_flavors=int(best_max_flavors))
#     elif mode == 'classic':
#         return ci.interrogate_classic(image)
#     else:
#         return ci.interrogate_fast(image)
    
def image2tile():
    print("image2tile")
    pass