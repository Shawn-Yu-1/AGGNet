from .models import Generator, Discriminator256, Inception, GatedAttentionNet, GatedAttentionNetV2, Discriminator512, GatedAttentionNetV3

gen_list = {"generator": Generator, 
            "gatedattentionnet": GatedAttentionNet, 
            "AGGV2": GatedAttentionNetV2,
            "AGGV3": GatedAttentionNetV3}

def get_generator(opt):
    return gen_list[opt["name"]](**opt["config"])