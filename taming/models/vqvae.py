import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
import math

def new_model():
    '''Return a New Instance of VQVAE, the same parameters with the pretrained model.
        This is for torch.load().
    '''
    return VQVAE(
        channel=512, n_res_block=0,
        n_res_channel=32, embed_dim=256,
        n_embed=8192, stride=6
    )

def img2code(model, img):
    '''Convert a batch of img to code
    Args:
        model: The tokenizer model.
        img: [b, c, h, w]
    '''
    with torch.no_grad():
        quant_t1, _, id_t1 = model.encode(img)
    return id_t1.view(img.shape[0], -1) 

def code2img(model, code):
    '''Convert a batch of code to imgs
    Args:
        model: ...
        code: [b, h, w] or [b, h*w] LongTensor
    '''
    if len(code.shape) == 2:
        s = int(math.sqrt(len(code.view(-1))) + 1e-5)
        code = code.view(code.shape[0], s, s)
    with torch.no_grad():
        out = model.decode_code(code)
        out = out * torch.tensor([0.30379, 0.32279, 0.32800], device=out.device).view(1, -1, 1, 1) + torch.tensor([0.79093, 0.76271, 0.75340], device=out.device).view(1, -1, 1, 1)
    return out


class VQVAE(pl.LightningModule):
    def __init__(
        self,
        in_channel=3,
        channel=512,
        n_res_block=0,
        n_res_channel=32,
        embed_dim=256,
        n_embed=1024,
        stride=6,
        simple=True,
        decay=0.99,
        ckpt_path=None,
        mean=[0.79093, 0.76271, 0.75340],
        std=[0.30379, 0.32279, 0.32800]
    ):
        super().__init__()
        if channel == 2048:
            n_res_block = 0
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride, embed_dim, n_embed, simple)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec = Decoder(
            in_channel=embed_dim, 
            out_channel=in_channel,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            stride=stride-2,
            simple=simple
        )
        self.criterion = nn.MSELoss()
        self.sched = None
        self.register_buffer('MEAN', torch.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer('STD', torch.tensor(std).reshape(1, 3, 1, 1))
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, input, continuous_relax=False, temperature=1., hard=False, KL=False):
        quant_t, diff, _, = self.encode(input, continuous_relax, temperature, hard, KL)
        dec = self.dec(quant_t)

        return dec, diff

    def encode(self, input, continuous_relax=False, temperature=1., hard=False, KL=False):
        logits = self.enc_b(input)
        quant_t, diff_t, id_t = self.quantize_t.forward_(logits, continuous_relax, temperature, hard)
        quant_t = quant_t.permute(0, 3, 1, 2)
        if not continuous_relax or KL:
            diff_t = diff_t.unsqueeze(0)
        else:
            diff_t = torch.zeros_like(diff_t).unsqueeze(0) # placeholder to return right shape 
        return quant_t, diff_t , id_t
    
    def decode(self, code):
        return self.dec(code)

    def decode_code(self, code_t):
        if len(code_t.shape) == 2:
            s = int(math.sqrt(code_t.shape[1]))
            code_t = code_t.view(code_t.shape[0], s, s)
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        dec = self.dec(quant_t)
        return dec
  
    @torch.no_grad()
    def decode_to_img(self, index):
        img_t = self.decode_code(index)
        img_t = self.de_normalize(img_t)
        return img_t

    def de_normalize(self, img_t):
        return (img_t * self.STD.to(img_t) + self.MEAN.to(img_t)).clamp(0,1) # [B,C,H,W]

    def shared_step(self, batch):
        latent_loss_weight = 0.25
        img = batch['image'].to(self.device)
        out, latent_loss = self(img)
        recon_loss = self.criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        return loss, recon_loss, latent_loss

    def training_step(self, batch, batch_idx):
        loss, recon_loss, latent_loss = self.shared_step(batch)
        log_dict = {"train/rec": recon_loss.item(), 
                    "train/latent": latent_loss.item()}
        self.log_dict(log_dict, prog_bar=True,
            logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, latent_loss = self.shared_step(batch)
        log_dict = {"val/rec": recon_loss.item(), 
                    "val/latent": latent_loss.item()}
        self.log_dict(log_dict, prog_bar=True,
            logger=True, on_step=True, on_epoch=True)
        return self.log_dict

    def configure_optimizers(self):
        from third_party.scheduler.scheduler import CycleScheduler
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = None
        if self.sched == "cycle":
            scheduler = CycleScheduler(
                optimizer,
                self.lr,
                n_iter=self.n_iter,
                momentum=None,
                warmup_proportion=0.05,
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch['image'].to(self.device)
        with torch.no_grad():
            xrec, _ = self(x)
        log[f"inputs"] = x
        log[f"rec"] = xrec
        return log


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        torch.nn.init.xavier_uniform_(embed, gain=torch.nn.init.calculate_gain('tanh'))
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward_(self, input, continuous_relax=False, temperature=1., hard=False):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        ) # dist map, shape=[*, n_embed]

        if not continuous_relax:
            # argmax + lookup
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)
        elif not hard:
            # gumbel softmax weighted sum
            embed_soft, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=False)
            embed_ind = embed_ind.view(*input.shape[:-1])
            embed_soft = embed_soft.view(*input.shape[:-1], self.n_embed)
            quantize = embed_soft @ self.embed.transpose(0, 1)
        else:
            # gumbel softmax hard lookup
            embed_onehot, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=True)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)

        if self.training and ((continuous_relax and hard) or (not continuous_relax)):
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        if not continuous_relax:
            diff = (quantize.detach() - input).pow(2).mean()
            quantize = input + (quantize - input).detach()
        else:
            # maybe need replace a KL term here
            qy = (-dist).softmax(-1)
            diff = torch.sum(qy * torch.log(qy * self.n_embed + 1e-20), dim=-1).mean() # KL
            #diff = (quantize - input).pow(2).mean().detach() # gumbel softmax do not need diff
            quantize = quantize.to(memory_format=torch.channels_last)
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride, embed_dim, n_embed, simple):
        super().__init__()
        if stride == 8:
            if simple:
                blocks = [
                    nn.Conv2d(in_channel, channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                ]
            else:
                blocks = [
                        nn.Conv2d(in_channel, channel//8, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channel//8, channel//4, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channel//4, channel//2, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channel//2, channel, 4, stride=2, padding=1),
                    ]

        if stride == 6:
            if simple:
                blocks = [
                    nn.Conv2d(in_channel, channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                ]
            else:
                blocks = [
                    nn.Conv2d(in_channel, channel // 4, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel // 4, channel //2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel //2, channel, 4, stride=2, padding=1),
                ]

            
        elif stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel, embed_dim, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input).permute(0, 2, 3, 1)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, simple
    ):
        super().__init__()
        blocks = [
            nn.ConvTranspose2d(in_channel, channel, 4, stride=2, padding=1),
        ]
        
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))
        if stride == 6:
            if simple:
                blocks.extend(
                    [
                        nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channel, out_channel, 1)
                    ]
                )
            else:
                blocks.extend(
                    [
                        nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(channel, channel//2, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(channel//2, channel//4, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channel//4, out_channel, 1)
                    ]
                )

        if stride == 4 and simple:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, out_channel, 1)
                ]
            )
        elif stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel, channel // 2, 1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

try:
    from torch.overrides import has_torch_function, handle_torch_function
except ImportError as e:
    from torch._overrides import has_torch_function, handle_torch_function
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if not torch.jit.is_scripting():
        if type(logits) is not torch.Tensor and has_torch_function((logits,)):
            return handle_torch_function(
                gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)

    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret, index
    else:
        # Reparametrization trick.
        ret = y_soft
        index = y_soft.max(dim, keepdim=True)[1]
        return ret, index

class CycleAnnealScheduler:
    def __init__(
        self, optimizer, lr_max, lr_divider, cut_point, step_size, momentum=None
    ):
        self.lr_max = lr_max
        self.lr_divider = lr_divider
        self.cut_point = step_size // cut_point
        self.step_size = step_size
        self.iteration = 0
        self.cycle_step = int(step_size * (1 - cut_point / 100) / 2)
        self.momentum = momentum
        self.optimizer = optimizer

    def get_lr(self):
        if self.iteration > 2 * self.cycle_step:
            cut = (self.iteration - 2 * self.cycle_step) / (
                self.step_size - 2 * self.cycle_step
            )
            lr = self.lr_max * (1 + (cut * (1 - 100) / 100)) / self.lr_divider

        elif self.iteration > self.cycle_step:
            cut = 1 - (self.iteration - self.cycle_step) / self.cycle_step
            lr = self.lr_max * (1 + cut * (self.lr_divider - 1)) / self.lr_divider

        else:
            cut = self.iteration / self.cycle_step
            lr = self.lr_max * (1 + cut * (self.lr_divider - 1)) / self.lr_divider

        return lr

    def get_momentum(self):
        if self.iteration > 2 * self.cycle_step:
            momentum = self.momentum[0]

        elif self.iteration > self.cycle_step:
            cut = 1 - (self.iteration - self.cycle_step) / self.cycle_step
            momentum = self.momentum[0] + cut * (self.momentum[1] - self.momentum[0])

        else:
            cut = self.iteration / self.cycle_step
            momentum = self.momentum[0] + cut * (self.momentum[1] - self.momentum[0])

        return momentum

    def step(self):
        lr = self.get_lr()

        if self.momentum is not None:
            momentum = self.get_momentum()

        self.iteration += 1

        if self.iteration == self.step_size:
            self.iteration = 0

        for group in self.optimizer.param_groups:
            group['lr'] = lr

            if self.momentum is not None:
                group['betas'] = (momentum, group['betas'][1])

        return lr