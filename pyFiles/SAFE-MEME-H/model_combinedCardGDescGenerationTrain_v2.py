'''
Adapted from https://github.com/huggingface/transformers and https://github.com/j-min/VL-T5
'''

from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, __HEAD_MASK_WARNING_MSG, T5Block, T5LayerNorm
import copy
from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import math
import os
import warnings
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from torch.utils.checkpoint import checkpoint
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_file = f"./logFiles/train_partialFT_categoryCard_gDescGeneration_v1.log"
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)


hidden_dim = 768
current_device_id = torch.cuda.current_device()
# projector_card = torch.nn.Linear(hidden_dim, 1, device=current_device_id)

projector_card = torch.load(f'./trained_cards/card_train_fullFT_categoryCard_gDescGeneration_v1_GENERAL').to(current_device_id) # M12
# projector_card = torch.load(f'./trained_cards/card_train_partialFT_categoryCard_gDescGeneration_v1_GENERAL').to(current_device_id) # M13


logger.info(f'*** projector_card: {current_device_id}')

# QA

# path = f'./trained_cards/noGDesc/allQA/card_train_noGDescSingleCardsAllQAGen'
# path = f'./trained_cards/noGDesc/allQA/card_train_withGDescSingleCardsAllQAGen'
# path = f'./trained_cards/noGDesc/allQGen/card_train_noGDescSingleCardsAllQueriesGen'
# path = f'./trained_cards/withGDesc/allQGen/card_train_withGDescSingleCardsAllQueriesGen'
# path = f'./trained_cards/noGDesc/1Q1A/card_train_noGDescSingleCards_1Q1A'
# path = f'./trained_cards/withGDesc/1Q1A/card_train_withGDescSingleCards_1Q1A'
path = f'./trained_cards/gDescGen/new_cards/card_train_gDescGeneration_singleCards'


savedCard_ISLAM = torch.load(f'{path}_ISALM').to(current_device_id)
savedCard_MEN = torch.load(f'{path}_MEN').to(current_device_id)
savedCard_WOMEN = torch.load(f'{path}_WOMEN').to(current_device_id)
savedCard_BLACK = torch.load(f'{path}_WHITE').to(current_device_id)      
savedCard_WHITE = torch.load(f'{path}_BLACK').to(current_device_id)
savedCard_JEWS = torch.load(f'{path}_LGBTQ').to(current_device_id)
savedCard_LGBTQ = torch.load(f'{path}_JEWS').to(current_device_id)
savedCard_DISABILITY = torch.load(f'{path}_DISABILITY').to(current_device_id)
savedCard_IMMIGRANT = torch.load(f'{path}_IMMIGRANT').to(current_device_id)
savedCard_GENERAL = torch.load(f'{path}_GENERAL').to(current_device_id)
savedCard_OTHERS = torch.load(f'{path}_OTHERS').to(current_device_id)

# path = f'./trained_cards/withGDesc/allQA'
# savedCard_ISLAM = torch.load(f'{path}/card_train_withGDescSingleCardsAllQAGen_ISALM').to(current_device_id)
# savedCard_MEN = torch.load(f'{path}/card_train_withGDescSingleCardsAllQAGen_MEN').to(current_device_id)
# savedCard_WOMEN = torch.load(f'{path}/card_train_withGDescSingleCardsAllQAGen_WOMEN').to(current_device_id)
# savedCard_BLACK = torch.load(f'{path}/card_train_withGDescSingleCardsAllQAGen_WHITE').to(current_device_id)      
# savedCard_WHITE = torch.load(f'{path}/card_train_withGDescSingleCardsAllQAGen_BLACK').to(current_device_id)
# savedCard_JEWS = torch.load(f'{path}/card_train_withGDescSingleCardsAllQAGen_LGBTQ').to(current_device_id)
# savedCard_LGBTQ = torch.load(f'{path}/card_train_withGDescSingleCardsAllQAGen_JEWS').to(current_device_id)
# savedCard_DISABILITY = torch.load(f'{path}/card_train_withGDescSingleCardsAllQAGen_DISABILITY').to(current_device_id)
# savedCard_IMMIGRANT = torch.load(f'{path}/card_train_withGDescSingleCardsAllQAGen_IMMIGRANT').to(current_device_id)
# savedCard_GENERAL = torch.load(f'{path}/card_train_withGDescSingleCardsAllQAGen_GENERAL').to(current_device_id)
# savedCard_OTHERS = torch.load(f'{path}/card_train_withGDescSingleCardsAllQAGen_OTHERS').to(current_device_id)

logger.info(f'*** path for cards: {path}: from M1')


class JointEncoder(T5Stack):
    def __init__(self, config, embed_tokens=None, patch_size=None):
        super().__init__(config)

        # global category_card
        
        logger.info(f'JointEncoder: model.py: \n{config}')
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.patch_num, self.patch_dim = patch_size
        self.image_dense = nn.Linear(self.patch_dim, config.d_model)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size, vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.gate_dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        
        global projector_card
        self.projector_card = projector_card

        logger.info(f'card_ISLAM is loaded: {savedCard_ISLAM}')

        # raise Exception('***')

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def get_categoryCard(self):
        
        logger.info(f'*** The ategroy card is returned from JointEncoder.')
        return self.projector_card


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        image_ids=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        global savedCard_ISLAM
        global savedCard_MEN
        global savedCard_WOMEN
        global savedCard_BLACK
        global savedCard_WHITE
        global savedCard_JEWS
        global savedCard_LGBTQ
        global savedCard_DISABILITY
        global savedCard_IMMIGRANT
        global savedCard_GENERAL
        global savedCard_OTHERS 
        
        # logger.info(f'*** savedCard_ISLAM: {savedCard_ISLAM}')
        # logger.info(f'*** savedCard_MEN: {savedCard_MEN}')
        # logger.info(f'*** savedCard_WOMEN: {savedCard_WOMEN}')
        # logger.info(f'*** savedCard_BLACK: {savedCard_BLACK}')
        # logger.info(f'*** savedCard_WHITE: {savedCard_WHITE}')
        # logger.info(f'*** savedCard_JEWS: {savedCard_JEWS}')
        # logger.info(f'*** savedCard_LGBTQ: {savedCard_LGBTQ}')
        # logger.info(f'*** savedCard_DISABILITY: {savedCard_DISABILITY}')
        # logger.info(f'*** savedCard_IMMIGRANT: {savedCard_IMMIGRANT}')
        # logger.info(f'*** savedCard_GENERAL: {savedCard_GENERAL}')
        # logger.info(f'*** savedCard_OTHERS: {savedCard_OTHERS}')
        
        image_embedding = self.image_dense(image_ids)

        image_att, _ = self.mha_layer(hidden_states, image_embedding, image_embedding)
        
        merge = torch.cat([hidden_states, image_att], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        hidden_states = (1 - gate) * hidden_states + gate * image_att

        #
        t0 = savedCard_ISLAM(hidden_states)
        t0 = hidden_states * self.projector_card(t0)
        t1 = savedCard_MEN(hidden_states)
        t1 = hidden_states * self.projector_card(t1)
        t2 = savedCard_WOMEN(hidden_states)
        t2 = hidden_states * self.projector_card(t2)
        t3 = savedCard_BLACK(hidden_states)
        t3 = hidden_states * self.projector_card(t3)
        t4 = savedCard_WHITE(hidden_states)
        t4 = hidden_states * self.projector_card(t4)
        t5 = savedCard_JEWS(hidden_states)
        t5 = hidden_states * self.projector_card(t5)
        t6 = savedCard_LGBTQ(hidden_states)
        t6 = hidden_states * self.projector_card(t6)
        t7 = savedCard_DISABILITY(hidden_states)
        t7 = hidden_states * self.projector_card(t7)
        t8 = savedCard_IMMIGRANT(hidden_states)
        t8 = hidden_states * self.projector_card(t8)
        t9 = savedCard_GENERAL(hidden_states)
        t9 = hidden_states * self.projector_card(t9)
        t10 = savedCard_OTHERS(hidden_states)
        t10 = hidden_states * self.projector_card(t10)

        # logger.info(f'*** t0: {t0.shape}')
        # logger.info(f'*** t1: {t1.shape}')
        # logger.info(f'*** t2: {t2.shape}')
        # logger.info(f'*** t3: {t3.shape}')
        # logger.info(f'*** t4: {t4.shape}')
        # logger.info(f'*** t5: {t5.shape}')
        # logger.info(f'*** t6: {t6.shape}')
        # logger.info(f'*** t7: {t7.shape}')
        # logger.info(f'*** t8: {t8.shape}')
        # logger.info(f'*** t9: {t9.shape}')
        # logger.info(f'*** t10: {t10.shape}')
        
        # raise Exception(f'***')
        sum_tensor = t0 + t1 + t2 + t3 + + t4 + t5 + t6 + t7 + t8 + t9 + t10
        sum_tensor = sum_tensor / 11
        hidden_states = hidden_states + sum_tensor
        # logger.info(f'\n*** hidden_states after special_operation: {hidden_states.shape}\n')
        
        #
        # logger.info(f'hidden_states :{hidden_states.shape}')
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def save_categoryCard(self, PATH):
        
        category_card = self.encoder.get_categoryCard()
        torch.save(category_card, PATH)
        logger.info(f'*** categroy card is saved in T5ForMultimodalGeneration at: {PATH}')

    def __init__(self, config: T5Config, patch_size):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        logger.info(f'T5ForMultimodalGeneration: config: \n{config}')
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = JointEncoder(encoder_config, self.shared, patch_size)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_ids=None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                image_ids=image_ids,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
    # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        output = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

        if "image_ids" in kwargs:
            output["image_ids"] = kwargs['image_ids']

        return output
    
    def test_step(self, tokenizer, batch, **kwargs):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        image_ids = batch['image_ids'].to(device)

        output = self.generate(
            input_ids=input_ids,
            image_ids=image_ids,
            **kwargs
        )

        generated_sents = tokenizer.batch_decode(output, skip_special_tokens=True)
        targets = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        result = {}
        result['preds'] = generated_sents
        result['targets'] = targets

        return result