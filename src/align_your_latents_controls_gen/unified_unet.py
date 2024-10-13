from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange, repeat

from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from diffusers.models.modeling_utils import ModelMixin
from src.utils.Transformer.layers.layer_norm import LayerNorm
from src.utils.Transformer.layers.multi_head_attention import MultiHeadAttention


class ResBlock3D(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(num_channels, num_channels, kernel_size=(3, 1, 1), padding='same')
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out += identity
        out = self.relu(out)
        return out


class TemporalAttention(torch.nn.Module):
    def __init__(self, num_attention_head, attention_head_dim, num_channels_latent, dim_condition_text):
        super().__init__()
        self.cross_attention_1 = torch.nn.MultiheadAttention(embed_dim=attention_head_dim, num_heads=num_attention_head,
                                                             batch_first=True)
        self.cross_attention_2 = torch.nn.MultiheadAttention(embed_dim=attention_head_dim, num_heads=num_attention_head,
                                                             batch_first=True)
        self.norm_1 = torch.nn.LayerNorm(attention_head_dim)
        self.norm_2 = torch.nn.LayerNorm(attention_head_dim)
        self.linear_text = torch.nn.Linear(dim_condition_text, attention_head_dim)
        self.linear_latent_in = torch.nn.Linear(num_channels_latent, attention_head_dim)
        self.linear_latent_out = torch.nn.Linear(attention_head_dim, num_channels_latent)

    def forward(self, x, y):
        y = self.linear_text(y)
        x = self.linear_latent_in(x)
        _x = x
        attention_output_1, _ = self.cross_attention_1(x, y, y)
        output_norm_1 = self.norm_1(attention_output_1 + _x)
        _x = output_norm_1
        attention_output_2, _ = self.cross_attention_2(output_norm_1, y, y)
        output_norm_2 = self.norm_2(attention_output_2 + _x)
        out = self.linear_latent_out(output_norm_2)
        return out


class UnetConditionModelUnified(ModelMixin):

    def __init__(self, unets_list):
        super(ModelMixin, self).__init__()
        self.unets_list = torch.nn.ModuleList(unets_list)
        self.unets_list.requires_grad_(False)
        self.resnets_3d_and_temporal_attns_down = torch.nn.ModuleList([])
        self.resnets_3d_and_temporal_attns_up = torch.nn.ModuleList([])
        self.learnable_params_down = torch.nn.ParameterList([])
        self.learnable_params_up = torch.nn.ParameterList([])
        self.attention_head_dim = self.unets_list[0].config.attention_head_dim
        self.block_out_channels = self.unets_list[0].config.block_out_channels
        self.cross_attention_dim = self.unets_list[0].config.cross_attention_dim
        self.block_out_channels_reversed = self.block_out_channels[::-1]
        for i, down_block in enumerate(unets_list[0].down_blocks):
            if i % 2 == 0:
                self.resnets_3d_and_temporal_attns_down.append(ResBlock3D(num_channels=self.block_out_channels[i]))
            else:
                self.resnets_3d_and_temporal_attns_down.append(
                    TemporalAttention(num_attention_head=8,
                                      attention_head_dim=512,
                                      num_channels_latent=self.block_out_channels[i],
                                      dim_condition_text=1024))
            self.learnable_params_down.append(torch.nn.Parameter(torch.zeros(1, 2, 1, 1, 1),
                                                                 requires_grad=True))

        for i, up_block in enumerate(unets_list[0].up_blocks):
            if i % 2 == 0:
                self.resnets_3d_and_temporal_attns_up.append(
                    ResBlock3D(num_channels=self.block_out_channels_reversed[i]))
            else:
                self.resnets_3d_and_temporal_attns_up.append(
                    TemporalAttention(num_attention_head=8,
                                      attention_head_dim=512,
                                      num_channels_latent=self.block_out_channels_reversed[i],
                                      dim_condition_text=1024))
            self.learnable_params_up.append(torch.nn.Parameter(torch.zeros(1, 2, 1, 1, 1),
                                                               requires_grad=True))

        self.resnets_3d_and_temporal_attns_down.requires_grad_(True)
        self.resnets_3d_and_temporal_attns_up.requires_grad_(True)

    @property
    def dtype(self):
        return self.unets_list[0].dtype

    @property
    def config(self):
        return self.unets_list[0].config

    def prep_and_time(self,
                      unet,
                      sample: torch.FloatTensor,
                      timestep: Union[torch.Tensor, float, int],
                      encoder_hidden_states: torch.Tensor,
                      timestep_cond: Optional[torch.Tensor] = None,
                      attention_mask: Optional[torch.Tensor] = None,
                      cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                      added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                      encoder_attention_mask: Optional[torch.Tensor] = None, ):
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2 ** unet.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if unet.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = unet.get_time_embed(sample=sample, timestep=timestep)
        emb = unet.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        aug_emb = unet.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if unet.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if unet.time_embed_act is not None:
            emb = unet.time_embed_act(emb)

        encoder_hidden_states = unet.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = unet.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": unet.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        # if USE_PEFT_BACKEND:
        #     # weight the lora layers by setting `lora_scale` for each PEFT layer
        #     scale_lora_layers(unet, lora_scale)

        return (sample, emb, encoder_hidden_states, attention_mask, encoder_attention_mask, forward_upsample_size,
                upsample_size)

    def apply_down_block(self,
                         ind,
                         unet,
                         sample: torch.FloatTensor,
                         emb: Union[torch.Tensor, float, int],
                         encoder_hidden_states: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None,
                         cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                         encoder_attention_mask: Optional[torch.Tensor] = None, ):
        downsample_block = unet.down_blocks[ind]
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        return sample, res_samples

    def apply_middle_block(self,
                           unet,
                           sample: torch.FloatTensor,
                           emb: Union[torch.Tensor, float, int],
                           encoder_hidden_states: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None,
                           cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                           encoder_attention_mask: Optional[torch.Tensor] = None,
                           ):
        if hasattr(unet.mid_block, "has_cross_attention") and unet.mid_block.has_cross_attention:
            sample = unet.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = unet.mid_block(sample, emb)
        return sample

    def apply_up_block(self,
                       ind,
                       unet,
                       down_block_res_samples,
                       sample: torch.FloatTensor,
                       emb: Union[torch.Tensor, float, int],
                       encoder_hidden_states: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None,
                       cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                       encoder_attention_mask: Optional[torch.Tensor] = None,
                       forward_upsample_size=None,
                       upsample_size=None):

        is_final_block = ind == len(unet.up_blocks) - 1

        res_samples = down_block_res_samples[-len(unet.up_blocks[ind].resnets):]
        down_block_res_samples = down_block_res_samples[: -len(unet.up_blocks[ind].resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]

        if hasattr(unet.up_blocks[ind], "has_cross_attention") and unet.up_blocks[ind].has_cross_attention:
            sample = unet.up_blocks[ind](
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = unet.up_blocks[ind](
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
            )
        return sample, down_block_res_samples

    def from_none_to_list(self, arg):
        if arg is None:
            arg = [None for _ in range(len(self.unets_list))]
        return arg

    def forward(
            self,
            samples_list,
            timesteps_list,
            encoder_hidden_states_list,
            encoder_hidden_states_unified,
            attention_masks_list=None,
            encoder_attention_masks_list=None,
            timestep_conds_list=None,
            cross_attention_kwargs_list=None,
            added_cond_kwargs_list=None,
            return_dict=True,
    ) -> Union[UNet2DConditionOutput, Tuple]:

        attention_masks_list = self.from_none_to_list(attention_masks_list)
        encoder_attention_masks_list = self.from_none_to_list(encoder_attention_masks_list)
        timestep_conds_list = self.from_none_to_list(timestep_conds_list)
        cross_attention_kwargs_list = self.from_none_to_list(cross_attention_kwargs_list)
        added_cond_kwargs_list = self.from_none_to_list(added_cond_kwargs_list)

        samples_processed = []
        timesteps_processed = []
        encoder_hidden_states_processed = []
        attention_masks_processed = []
        encoder_attention_masks_processed = []
        forward_upsample_sizes_processed = []
        upsample_sizes_processed = []
        for i, unet in enumerate(self.unets_list):
            (sample,
             emb,
             encoder_hidden_states,
             attention_mask,
             encoder_attention_mask,
             forward_up_sample_size,
             upsample_size) = self.prep_and_time(unet,
                                                 sample=samples_list[i],
                                                 timestep=timesteps_list[i],
                                                 encoder_hidden_states=encoder_hidden_states_list[i],
                                                 attention_mask=attention_masks_list[i],
                                                 encoder_attention_mask=encoder_attention_masks_list[i],
                                                 timestep_cond=timestep_conds_list[i],
                                                 cross_attention_kwargs=cross_attention_kwargs_list[i],
                                                 added_cond_kwargs=added_cond_kwargs_list[i])
            samples_processed.append(sample)
            timesteps_processed.append(emb)
            encoder_hidden_states_processed.append(encoder_hidden_states)
            attention_masks_processed.append(attention_mask)
            encoder_attention_masks_processed.append(encoder_attention_mask)
            forward_upsample_sizes_processed.append(forward_up_sample_size)
            upsample_sizes_processed.append(upsample_size)

        down_block_res_samples_processed = [(sample_element,) for sample_element in samples_processed]
        for ind in range(len(self.unets_list[0].down_blocks)):
            for i, unet in enumerate(self.unets_list):
                (sample,
                 down_block_res_samples) = self.apply_down_block(ind,
                                                                 unet,
                                                                 sample=samples_processed[i],
                                                                 emb=timesteps_processed[i],
                                                                 encoder_hidden_states=
                                                                 encoder_hidden_states_processed[i],
                                                                 attention_mask=
                                                                 attention_masks_processed[i],
                                                                 cross_attention_kwargs=
                                                                 cross_attention_kwargs_list[i],
                                                                 encoder_attention_mask=
                                                                 encoder_attention_masks_processed[i])
                samples_processed[i] = sample
                down_block_res_samples_processed[i] += down_block_res_samples

            samples_stacked_res = torch.stack(samples_processed, dim=1)
            temp_layer = self.resnets_3d_and_temporal_attns_down[ind]
            samples_stacked = torch.stack(samples_processed, dim=1)
            # assert samples_stacked.shape[1] == 2
            if isinstance(temp_layer, ResBlock3D):
                samples_re = rearrange(samples_stacked, "b t c h w -> b c t h w", t=2)
                samples_re = temp_layer(samples_re)
                samples_stacked = rearrange(samples_re, "b c t h w -> b t c h w", t=2)
            else:
                h, w = samples_stacked.shape[-2:]
                samples_re = rearrange(samples_stacked, "b t c h w -> (b h w) t c", t=2)
                encoder_hidden_states_rep = repeat(encoder_hidden_states_unified, "b s d -> (b h w) s d", h=h, w=w)
                samples_re = temp_layer(samples_re, encoder_hidden_states_rep)
                samples_stacked = rearrange(samples_re, "(b h w) t c -> b t c h w", t=2, h=h, w=w)
            samples_stacked = (torch.sigmoid(self.learnable_params_down[ind]) * samples_stacked_res +
                               (1 - torch.sigmoid(self.learnable_params_down[ind])) * samples_stacked)
            samples_processed = list(
                [sample_processed.squeeze(1) for sample_processed in samples_stacked.chunk(2, dim=1)])

        for i, unet in enumerate(self.unets_list):
            sample = self.apply_middle_block(unet,
                                             sample=samples_processed[i],
                                             emb=timesteps_processed[i],
                                             encoder_hidden_states=encoder_hidden_states_processed[i],
                                             attention_mask=attention_masks_processed[i],
                                             cross_attention_kwargs=cross_attention_kwargs_list[i],
                                             encoder_attention_mask=encoder_attention_masks_processed[i])
            samples_processed[i] = sample

        for ind in range(len(self.unets_list[0].up_blocks)):
            for i, unet in enumerate(self.unets_list):
                sample, down_block_res_samples_processed[i] = self.apply_up_block(ind,
                                                                                  unet,
                                                                                  down_block_res_samples=
                                                                                  down_block_res_samples_processed[i],
                                                                                  sample=samples_processed[i],
                                                                                  emb=timesteps_processed[i],
                                                                                  encoder_hidden_states=
                                                                                  encoder_hidden_states_processed[i],
                                                                                  attention_mask=
                                                                                  attention_masks_processed[i],
                                                                                  cross_attention_kwargs=
                                                                                  cross_attention_kwargs_list[i],
                                                                                  encoder_attention_mask=
                                                                                  encoder_attention_masks_processed[i],
                                                                                  upsample_size=
                                                                                  upsample_sizes_processed[i],
                                                                                  forward_upsample_size=
                                                                                  forward_upsample_sizes_processed[i])
                samples_processed[i] = sample

            samples_stacked_res = torch.stack(samples_processed, dim=1)
            temp_layer = self.resnets_3d_and_temporal_attns_up[ind]
            samples_stacked = torch.stack(samples_processed, dim=1)
            # assert samples_stacked.shape[1] == 2
            if isinstance(temp_layer, ResBlock3D):
                samples_re = rearrange(samples_stacked, "b t c h w -> b c t h w", t=2)
                samples_re = temp_layer(samples_re)
                samples_stacked = rearrange(samples_re, "b c t h w -> b t c h w", t=2)
            else:
                h, w = samples_stacked.shape[-2:]
                samples_re = rearrange(samples_stacked, "b t c h w -> (b h w) t c", t=2)
                encoder_hidden_states_rep = repeat(encoder_hidden_states_unified, "b s d -> (b h w) s d", h=h, w=w)
                samples_re = temp_layer(samples_re, encoder_hidden_states_rep)
                samples_stacked = rearrange(samples_re, "(b h w) t c -> b t c h w", t=2, h=h, w=w)
            samples_stacked = (torch.sigmoid(self.learnable_params_up[ind]) * samples_stacked_res +
                               (1 - torch.sigmoid(self.learnable_params_up[ind])) * samples_stacked)
            samples_processed = list(
                [sample_processed.squeeze(1) for sample_processed in samples_stacked.chunk(2, dim=1)])

        for i, unet in enumerate(self.unets_list):
            if unet.conv_norm_out:
                sample = unet.conv_norm_out(samples_processed[i])
                sample = unet.conv_act(sample)
                samples_processed[i] = unet.conv_out(sample)

        # if USE_PEFT_BACKEND:
        #     # remove `lora_scale` from each PEFT layer
        #     unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return tuple(samples_processed[i] for i in range(len(samples_processed)))

        return UNet2DConditionOutput(sample=tuple(samples_processed[i] for i in range(len(samples_processed))))
