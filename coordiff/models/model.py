import torch
import torch.nn as nn
import torch.nn.functional as F
from coordiff.models.language_encoder import MLPEncoder
from coordiff.models.transformer import Transformer
from coordiff.models.dit import DiTBlockRms, FinalActionLayerRms, TimestepEmbedder, DiTBlock, FinalActionLayer
from coordiff.models.pos_embed import *
from einops import rearrange


class ProcedureMLP(nn.Module):
    def __init__(self, 
        hist_len, rot_type, hidden_dim, use_language, layers_dim, drop_rate,
        device,
        language_enc_cfg=None,
        num_states=1):
        super().__init__()

        self.hist_len = hist_len
        self.device = device
        self.hidden_dim = hidden_dim
        self.rot_type = rot_type
        self.use_language = use_language
        self.num_states = num_states
        
        input_dim = 3
        if rot_type == 'quat':
            input_dim += 4
        elif rot_type == 'rpy':
            input_dim += 3
        elif rot_type == '6d':
            input_dim += 6
        
        all_input_dim = input_dim * self.hist_len

        if num_states > 1:
            self.state_encoder = nn.Embedding(num_embeddings=num_states, embedding_dim=hidden_dim) 
            all_input_dim += hidden_dim

        if use_language:
            self.lang_encoder = MLPEncoder(
                output_size=hidden_dim,
                input_size=language_enc_cfg.input_dim,
                hidden_size=language_enc_cfg.hidden_dim,
                num_layers=language_enc_cfg.num_layers,
            )
            all_input_dim += hidden_dim

        layers = []
        for i, dim in enumerate(layers_dim):
            if i == 0:
                layers.append(nn.Linear(all_input_dim, dim))
            else:
                layers.append(nn.Linear(layers_dim[i-1], dim))
            
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(drop_rate))

        layers.append(nn.Linear(layers_dim[-1], 2))
        self.decoder = nn.Sequential(*layers)
        print(self.decoder)

        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


    def forward(self, hist, text=None, state=None):
        """
        返回二分类的logits
        """
        mix = rearrange(hist, 'b t c -> b (t c)')
        if self.use_language:
            text = self.lang_encoder(text)
            mix = torch.concat([mix, text], dim=1)
        if self.num_states > 1:
            state = self.state_encoder(state)
            mix = torch.concat([mix, state], dim=1)

        logits = self.decoder(mix)

        return logits

class ProcedureTransformer(nn.Module):
    def __init__(self, 
        hist_len, rot_type, hidden_dim, use_language, drop_rate,
        device,
        mix_encoder_cfg,
        language_enc_cfg=None,
        num_states=1):
        super().__init__()


        self.hist_len = hist_len
        self.device = device
        self.hidden_dim = hidden_dim
        self.rot_type = rot_type
        self.use_language = use_language
        self.num_states = num_states
        
        input_dim = 3
        if rot_type == 'quat':
            input_dim += 4
        elif rot_type == 'rpy':
            input_dim += 3
        elif rot_type == '6d':
            input_dim += 6
        
        self.pose_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, hidden_dim),
        )

        if num_states > 1:
            self.state_encoder = nn.Embedding(num_embeddings=num_states, embedding_dim=hidden_dim) 
            
        if use_language:
            self.lang_encoder = MLPEncoder(
                output_size=hidden_dim,
                input_size=language_enc_cfg.input_dim,
                hidden_size=language_enc_cfg.hidden_dim,
                num_layers=language_enc_cfg.num_layers,
            )
        
        self.mix_encoder = self.setup_mix_encoder(mix_encoder_cfg)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim // 2, 2)  # 输出两个类别的logits
        )

        self.initialize_weights()

    def setup_mix_encoder(self, mix_encoder_cfg):
        decoder_norm = nn.LayerNorm(mix_encoder_cfg.hidden_dim, eps=1e-5)
        mix_encoder = nn.TransformerEncoder( 
                nn.TransformerEncoderLayer(
                    d_model=mix_encoder_cfg.hidden_dim,
                    nhead=mix_encoder_cfg.nhead,
                    dim_feedforward=mix_encoder_cfg.mlp_hidden_dim,
                    dropout=mix_encoder_cfg.drop_rate,
                    batch_first=True
                ),
                num_layers=mix_encoder_cfg.num_layers,
                norm = decoder_norm
        )
        return mix_encoder

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        self.action_cls_token = nn.Parameter(torch.zeros(1, self.hidden_dim))
        nn.init.normal_(self.action_cls_token, std=1e-6)

        self.hist_embed = nn.Parameter(torch.randn(1, self.hist_len, self.hidden_dim), requires_grad=False)
        self.hist_embed.data.copy_(torch.from_numpy(
            get_1d_sincos_pos_embed(
                embed_dim=self.hidden_dim,
                len=self.hist_len)
            ).unsqueeze(0)
        )

        modality_embed_dim = 1
        modality_index = [0] * self.hist_len
        if self.use_language:
            modality_embed_dim += 1
            modality_index += [modality_index[-1] + 1]
        if self.num_states > 1:
            modality_embed_dim += 1
            modality_index += [modality_index[-1] + 1]
        
        self.modality_embed = nn.Parameter(
            torch.randn(1, modality_embed_dim, self.hidden_dim))
        self.modality_index = modality_index


    def forward(self, hist, text=None, state=None):
        """
        返回二分类的logits
        """
        mix = self.pose_proj(hist)
        mix += self.hist_embed
        
        if self.use_language:
            text = self.lang_encoder(text).unsqueeze(1)
            mix = torch.concat([mix, text], dim=1)
        if self.num_states > 1:
            state = self.state_encoder(state).unsqueeze(1)
            mix = torch.concat([mix, state], dim=1)

        mix += self.modality_embed[:, self.modality_index, :]

        action_cls_token = self.action_cls_token.expand(hist.size(0), 1, -1)
        mix = torch.concat([action_cls_token, mix], dim=1)
        mix = self.mix_encoder(mix)

        logits = self.decoder(mix[:, 0, :])  # [batch_size, 2]

        return logits


class CoorDiff(nn.Module):
    def __init__(self, 
                 hist_len, rot_type, hidden_dim, act_trunk, use_language,
                 mix_encoder_cfg,
                 dit_cfg,
                 device,
                 language_enc_cfg=None,
                 num_states=1):
        super().__init__()
        
        input_dim = 3
        if rot_type == 'quat':
            input_dim += 4
        elif rot_type == 'rpy':
            input_dim += 3
        elif rot_type == '6d':
            input_dim += 6
        
        self.pose_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, hidden_dim),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, hidden_dim),
        )
        
        if use_language:
            self.lang_encoder = MLPEncoder(
                output_size=hidden_dim,
                input_size=language_enc_cfg.input_dim,
                hidden_size=language_enc_cfg.hidden_dim,
                num_layers=language_enc_cfg.num_layers,
            )
        if num_states > 1:
            self.state_encoder = nn.Embedding(num_embeddings=num_states, embedding_dim=hidden_dim)

        self.mix_encoder = self.setup_mix_encoder(mix_encoder_cfg)
        self.mask = self.compute_mask(hist_len)
        self.device = device
    
        self.dit_decoder = self.setup_dit_decoder(dit_cfg)
        self.final_layer = eval(dit_cfg.final_layer_cls)(hidden_dim, input_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim)
        
        self.input_dim = input_dim
        self.action_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hist_len = hist_len
        self.num_states = num_states
        self.use_language = use_language
        self.act_trunk = act_trunk

        self.setup_pos_embed()
        self.initialize_weights()


    def setup_pos_embed(self):
        self.hist_embed = nn.Parameter(torch.randn(1, self.hist_len, self.hidden_dim), requires_grad=False)
        self.hist_embed.data.copy_(torch.from_numpy(
            get_1d_sincos_pos_embed(
                embed_dim=self.hidden_dim,
                len=self.hist_len)
            ).unsqueeze(0)
        )
        
        modality_embed_dim = 1
        modality_index = [0] * self.hist_len
        if self.use_language:
            modality_embed_dim += 1
            modality_index += [modality_index[-1] + 1]
        if self.num_states > 1:
            modality_embed_dim += 1
            modality_index += [modality_index[-1] + 1]
        
        self.modality_embed = nn.Parameter(
            torch.randn(1, modality_embed_dim, self.hidden_dim))
        self.modality_index = modality_index
    
        self.action_embed = nn.Parameter(torch.randn(1, self.act_trunk, self.hidden_dim), requires_grad=False)
        self.action_embed.data.copy_(torch.from_numpy(
            get_1d_sincos_pos_embed(
                embed_dim=self.hidden_dim,
                len=self.act_trunk)
            ).unsqueeze(0)
        )
    
    def compute_mask(self, seq_len):
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    
    def setup_mix_encoder(self, mix_encoder_cfg):
        decoder_norm = nn.LayerNorm(mix_encoder_cfg.hidden_dim, eps=1e-5)
        mix_encoder = nn.TransformerEncoder( 
                nn.TransformerEncoderLayer(
                    d_model=mix_encoder_cfg.hidden_dim,
                    nhead=mix_encoder_cfg.nhead,
                    dim_feedforward=mix_encoder_cfg.mlp_hidden_dim,
                    dropout=mix_encoder_cfg.drop_rate,
                    batch_first=True
                ),
                num_layers=mix_encoder_cfg.num_layers,
                norm = decoder_norm
        )
        return mix_encoder
    
    def setup_dit_decoder(self, dit_cfg):
        dit_decoder = nn.ModuleList([
            eval(dit_cfg.model_cls)(
                hidden_size=dit_cfg.hidden_dim,
                num_heads=dit_cfg.num_heads,
                mlp_ratio=dit_cfg.mlp_ratio,
                attn_drop=dit_cfg.attn_drop,
                proj_drop=dit_cfg.proj_drop,
            ) for _ in range(dit_cfg.depth)
        ])

        return dit_decoder
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.dit_decoder:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    
    def class_to_one_hot(self, classes):
        return F.one_hot(classes, self.num_states)
    
    def forward_enc(self, hist, text, state):
        mix = self.pose_proj(hist)
        mix += self.hist_embed
        # mix = self.mix_encoder(mix, self.mask.to(self.device))
        mix = self.mix_encoder(mix)
        
        if self.use_language:
            text = self.lang_encoder(text).unsqueeze(1)
            mix = torch.concat([mix, text], dim=1)
        if self.num_states > 1:
            state = self.state_encoder(state).unsqueeze(1)
            mix = torch.concat([mix, state], dim=1)
        
        mix += self.modality_embed[:, self.modality_index, :]
        
        return mix
    
    def forward_dec(self, x, c, t):
        c = torch.mean(c, dim=1)
        t = self.time_embedder(t)
        c += t

        out = self.action_proj(x)

        out += self.action_embed
        
        for block in self.dit_decoder:
            out = block(out, c)  # b, t+o*np*n, d
        
        out = self.final_layer(out, c)
        return out
    
    def forward(self, x, t, hist, text=None, state=None):
        '''
        Args:
            x: noise_action (action_chunk, action_dim)
            hist: history trajectory
            text: task description
            state: task state
        '''
        c = self.forward_enc(hist, text, state)
        # print(c.shape)
        out = self.forward_dec(x, c, t)
        
        return out
    
    
if __name__ == '__main__':
    from dataclasses import dataclass
   
    @dataclass
    class mix_encoder_cfg:
        hidden_dim = 64
        mlp_hidden_dim = 64 * 4
        drop_rate = 0.1
        num_layers = 2
        nhead = 8
        
    @dataclass
    class dit_cfg:
        hidden_dim = 64
        num_heads = 8
        mlp_ratio = 4
        attn_drop = 0.1
        proj_drop = 0.1
        depth = 2

    @dataclass
    class language_enc_cfg:
        input_dim = 784
        hidden_dim = 128
        num_layers = 2

    coordiff = CoorDiff(
        hist_len=10,
        rot_type='6d',
        hidden_dim=64,
        act_trunk=10,
        use_language=True,
        mix_encoder_cfg = mix_encoder_cfg,
        dit_cfg = dit_cfg,
        language_enc_cfg=language_enc_cfg,
        num_states=10,
        device='cpu',
    )
    
    hist = torch.zeros((2, 10, 9))
    t = torch.randint(10, size=(2,))
    state = torch.randint(10, size=(2,))
    noise_action = torch.zeros((2, 10, 9))
    text = torch.rand(2, 784)
    coordiff(noise_action, t, hist, text=text, state=state)