import torchvision.transforms as T
import pytorch_lightning as pl

from third_party.clip.model import *
from third_party.clip.clip import BICUBIC

clip_transform = T.Compose([
        T.Resize(224, interpolation=BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_structure(state_dict: dict):
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    args = {'vision_width': vision_width,
            'vision_layers': vision_layers,
            'vision_patch_size': vision_patch_size,
            'image_resolution': image_resolution,
            'embed_dim': embed_dim,
            'context_length': context_length,
            'vocab_size': vocab_size,
            'transformer_width': transformer_width,
            'transformer_heads': transformer_heads,
            'transformer_layers': transformer_layers}
    return args

class TextEncoder(pl.LightningModule):
    def __init__(self, ckpt_path, **kargs):
        super().__init__()
        with open(ckpt_path, 'rb') as opened_file:
            model = torch.jit.load(opened_file, map_location="cpu").eval()
        state_dict = model.state_dict()
        args = get_structure(state_dict)
        self.encoder = ClipTextEncoder(**args)
        #model_path = "/home/ma-user/work/lijiacheng/pretrained/clip/ViT-B-16.pt"
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"TextEncoder Loaded, missing:{missing}")

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, text):
        return self.encoder(text)

class VisualEncoder(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        with open(ckpt_path, 'rb') as opened_file:
            model = torch.jit.load(opened_file, map_location="cpu").eval()
        state_dict = model.state_dict()
        args = get_structure(state_dict)
        self.encoder = CLIPVisualEncoder(**args)
        missing, _ = self.encoder.load_state_dict(state_dict, strict=False)                  
        print(f"VisualEncoder Loaded, missing:{missing}")
        
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, image):
        return self.encoder(image)

    def clip_score(self, image, text_features):
        image_features = self(image)
        # cosine similarity as logits
        logit_scale = self.encoder.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

class ClipTextEncoder(nn.Module):
    def __init__(self,
             embed_dim: int,
             # text
             context_length: int,
             vocab_size: int,
             transformer_width: int,
             transformer_heads: int,
             transformer_layers: int,
             **kargs
             ):
        super().__init__()
        self.context_length = context_length
        self.transformer_width = transformer_width
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.vocab_size = vocab_size
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.initialize_parameters()
        
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)    

    @property
    def dtype(self):
        try:
            dtype = next(self.transformer.parameters()).dtype
        except:
            dtype = torch.float32
        return dtype

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_features = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features, x

class CLIPVisualEncoder(pl.LightningModule):
    def __init__(self,
             embed_dim: int,
             # vision
             image_resolution: int,
             vision_patch_size: int,
             vision_width: int,
             vision_layers: int,
             **kargs
             ):
        super().__init__()     
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        try:
            dtype = next(self.parameters()).dtype
        except:
            dtype = torch.float32   
        return dtype

    def forward(self, image):
        image_features =  self.visual(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

def main():
    model_path = "/home/ma-user/work/lijiacheng/pretrained/clip/ViT-B-16.pt"
    text_encoder = TextEncoder(model_path)
    breakpoint()

if __name__ == '__main__':
    main()