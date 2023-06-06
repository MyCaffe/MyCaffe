import copy
import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple
from utility import load_batch, DebugFunction
from tft_torch.tft import InputChannelEmbedding, VariableSelectionNetwork, GatedResidualNetwork
from tft_torch.base_blocks import TimeDistributed, NullTransform

#
# @see [Parameters in Tensorflow Keras RNN and CUDNN RNN](https://kaixih.github.io/keras-cudnn-rnn/) by Kaixi Hou, 2021
# @see [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) by PyTorch
#
def get_cudnn_lstm_weights(lstm):
    num_layers = lstm.num_layers
    hidden_size = lstm.hidden_size
    wts = []
    for param in lstm.state_dict():
        param_val = lstm.state_dict()[param]
        wts.append(param_val)
    
    all_wts = []
    idx = 0
    for i in range(0, num_layers):
        wtii = wts[idx][:hidden_size,:]
        wtif = wts[idx][hidden_size:hidden_size*2,:]
        wtig = wts[idx][hidden_size*2:hidden_size*3,:]
        wtio = wts[idx][hidden_size*3:hidden_size*4,:]
        idx = idx + 1

        wthi = wts[idx][:hidden_size,:]
        wthf = wts[idx][hidden_size:hidden_size*2,:]
        wthg = wts[idx][hidden_size*2:hidden_size*3,:]
        wtho = wts[idx][hidden_size*3:hidden_size*4,:]
        idx = idx + 1

        bii = wts[idx][:hidden_size]
        bif = wts[idx][hidden_size:hidden_size*2]
        big = wts[idx][hidden_size*2:hidden_size*3]
        bio = wts[idx][hidden_size*3:hidden_size*4]
        idx = idx + 1
    
        bhi = wts[idx][:hidden_size]
        bhf = wts[idx][hidden_size:hidden_size*2]
        bhg = wts[idx][hidden_size*2:hidden_size*3]
        bho = wts[idx][hidden_size*3:hidden_size*4]
        idx = idx + 1

        wts1 = [wtii, wtif, wtio, wtig, wthi, wthf, wtho, wthg]
        b1 = [bii, bif, bio, big, bhi, bhf, bho, bhg]
    
        shape = [-1]
        weights = [torch.reshape(torch.transpose(x, 0, 1), shape) for x in wts1]
        biases = [torch.reshape(x, shape) for x in b1]
        cudnnwts = torch.concat(weights + biases, axis=0)
        all_wts.append(cudnnwts)
    
    full_cudnnwts = torch.concat(all_wts, axis=0) if num_layers > 1 else all_wts[0]
    return full_cudnnwts
       
def apply_temporal_selection(temporal_representation: torch.tensor,
                             static_selection_signal: torch.tensor,
                             temporal_selection_module: VariableSelectionNetwork,
                             tag="",
                             path="") -> Tuple[torch.tensor, torch.tensor]:
    num_samples, num_temporal_steps, _ = temporal_representation.shape

    debug = DebugFunction.apply

    DebugFunction.trace(temporal_representation, tag + ".temporal_representation")
    temporal_representation = debug(temporal_representation)
    DebugFunction.trace(static_selection_signal, tag + ".static_selection_signal")
    static_selection_signal = debug(static_selection_signal)

    # replicate the selection signal along time
    time_distributed_context = replicate_along_time(static_signal=static_selection_signal,
                                                         time_steps=num_temporal_steps)
    # Dimensions:
    # time_distributed_context: [num_samples x num_temporal_steps x state_size]
    # temporal_representation: [num_samples x num_temporal_steps x (total_num_temporal_inputs * state_size)]

    # for applying the same selection module on all time-steps, we stack the time dimension with the batch dimension
    temporal_flattened_embedding = stack_time_steps_along_batch(temporal_representation)
    time_distributed_context = stack_time_steps_along_batch(time_distributed_context)
    # Dimensions:
    # temporal_flattened_embedding: [(num_samples * num_temporal_steps) x (total_num_temporal_inputs * state_size)]
    # time_distributed_context: [(num_samples * num_temporal_steps) x state_size]

    # applying the selection module across time
    temporal_selection_output, temporal_selection_weights = temporal_selection_module(
        flattened_embedding=temporal_flattened_embedding, context=time_distributed_context)
    # Dimensions:
    # temporal_selection_output: [(num_samples * num_temporal_steps) x state_size]
    # temporal_selection_weights: [(num_samples * num_temporal_steps) x (num_temporal_inputs) x 1]

    # Reshape the selection outputs and selection weights - to represent the temporal dimension separately
    temporal_selection_output = temporal_selection_output.view(num_samples, num_temporal_steps, -1)
    temporal_selection_weights = temporal_selection_weights.squeeze(-1).view(num_samples, num_temporal_steps, -1)
    # Dimensions:
    # temporal_selection_output: [num_samples x num_temporal_steps x state_size)]
    # temporal_selection_weights: [num_samples x num_temporal_steps x num_temporal_inputs)]

    DebugFunction.trace(temporal_selection_output, tag + ".temporal_selection_output")
    temporal_selection_output = debug(temporal_selection_output)
    DebugFunction.trace(temporal_selection_weights, tag + ".temporal_selection_weights")
    temporal_selection_weights = debug(temporal_selection_weights)

    return temporal_selection_output, temporal_selection_weights

def replicate_along_time(static_signal: torch.tensor, time_steps: int) -> torch.tensor:
    """
    This method gets as an input a static_signal (non-temporal tensor) [num_samples x num_features],
    and replicates it along time for 'time_steps' times,
    creating a tensor of [num_samples x time_steps x num_features]

    Args:
        static_signal: the non-temporal tensor for which the replication is required.
        time_steps: the number of time steps according to which the replication is required.

    Returns:
        torch.tensor: the time-wise replicated tensor
    """
    time_distributed_signal = static_signal.unsqueeze(1).repeat(1, time_steps, 1)
    return time_distributed_signal

def stack_time_steps_along_batch(temporal_signal: torch.tensor) -> torch.tensor:
    """
    This method gets as an input a temporal signal [num_samples x time_steps x num_features]
    and stacks the batch dimension and the temporal dimension on the same axis (dim=0).

    The last dimension (features dimension) is kept as is, but the rest is stacked along dim=0.
    """
    return temporal_signal.view(-1, temporal_signal.size(-1))

def apply_sequential_processing(selected_historical: torch.tensor, selected_future: torch.tensor,
                                c_seq_hidden: torch.tensor, c_seq_cell: torch.tensor,
                                past_lstm, future_lstm, post_lstm_gating, lstm_layers, path) -> torch.tensor:
    """
    This part of the model is designated to mimic a sequence-to-sequence layer which will be used for local
    processing.
    On that part the historical ("observed") information will be fed into a recurrent layer called "Encoder" and
    the future information ("known") will be fed into a recurrent layer called "Decoder".
    This will generate a set of uniform temporal features which will serve as inputs into the temporal fusion
    decoder itself.
    To allow static metadata to influence local processing, we use "c_seq_hidden" and "c_seq_cell" context vectors
    from the static covariate encoders to initialize the hidden state and the cell state respectively.
    The output of the recurrent layers is gated and fused with a residual connection to the input of this block.
    """
    DebugFunction.set_output_path(path,0)
    debug = DebugFunction.apply
    DebugFunction.trace(selected_historical, "tft.selected_historical1")
    selected_historical = debug(selected_historical)
    DebugFunction.trace(selected_future, "tft.selected_future1")
    selected_future = debug(selected_future)
    DebugFunction.trace(c_seq_hidden, "tft.c_seq_hidden1")
    c_seq_hidden = debug(c_seq_hidden)
    DebugFunction.trace(c_seq_cell, "tft.c_seq_cell1")
    c_seq_cell = debug(c_seq_cell)

    # concatenate the historical (observed) temporal signal with the futuristic (known) temporal singal, along the
    # time dimension
    lstm_input = torch.cat([selected_historical, selected_future], dim=1)

    DebugFunction.trace(lstm_input, "tft.lstm_input1")
    lstm_input = debug(lstm_input)

    # the historical temporal signal is fed into the first recurrent module
    # using the static metadata as initial hidden and cell state
    # (initial cell and hidden states are replicated for feeding to each layer in the stack)

    c_seq_hidden1 = c_seq_hidden.unsqueeze(0)

    DebugFunction.trace(c_seq_hidden1, "tft.c_seq_hidden1")
    c_seq_hidden1 = debug(c_seq_hidden1)

    c_seq_hidden2 = c_seq_hidden1.repeat(lstm_layers, 1, 1)

    DebugFunction.trace(c_seq_hidden2, "tft.c_seq_hidden2")
    c_seq_hidden2 = debug(c_seq_hidden2)

    c_seq_cell1 = c_seq_cell.unsqueeze(0)

    DebugFunction.trace(c_seq_cell1, "tft.c_seq_cell1")
    c_seq_cell1 = debug(c_seq_cell1)

    c_seq_cell2 = c_seq_cell1.repeat(lstm_layers, 1, 1)

    DebugFunction.trace(c_seq_cell2, "tft.c_seq_cell2")
    c_seq_cell2 = debug(c_seq_cell2)

    past_lstm_output, hidden = past_lstm(selected_historical, (c_seq_hidden2, c_seq_cell2))
    past_lstm.save_wts()

    DebugFunction.trace(past_lstm_output, "tft.past_lstm_output1")
    past_lstm_output = debug(past_lstm_output)
    DebugFunction.trace(hidden[0], "tft.hidden0_1")
    debug(hidden[0])
    DebugFunction.trace(hidden[1], "tft.hidden1_1")
    debug(hidden[1])

    # the future (known) temporal signal is fed into the second recurrent module
    # using the latest (hidden,cell) state of the first recurrent module
    # for setting the initial (hidden,cell) state.
    future_lstm_output, _ = future_lstm(selected_future, hidden)
    future_lstm.save_wts()

    DebugFunction.trace(future_lstm_output, "tft.future_lstm_output1")
    future_lstm_output = debug(future_lstm_output)

    # concatenate the historical recurrent output with the futuristic recurrent output, along the time dimension
    lstm_output = torch.cat([past_lstm_output, future_lstm_output], dim=1)

    DebugFunction.trace(lstm_output, "tft.lstm_output1")
    lstm_output = debug(lstm_output)

    # perform gating to the recurrent output signal, using a residual connection to input of this block
    gated_lstm_output = post_lstm_gating(lstm_output, residual=lstm_input)

    DebugFunction.trace(gated_lstm_output, "tft.gated_lstm_output1")
    gated_lstm_output = debug(gated_lstm_output)

    return gated_lstm_output

def apply_static_enrichment(gated_lstm_output: torch.tensor,
                            static_enrichment_signal: torch.tensor,
                            static_enrichment_grn, state_size, path="") -> torch.tensor:
    """
    This static enrichment stage enhances temporal features with static metadata using a GRN.
    The static enrichment signal is an output of a static covariate encoder, and the GRN is shared across time.
    """
    if path != "":
        DebugFunction.set_output_path(path, 0)
    debug = DebugFunction.apply

    num_samples, num_temporal_steps, _ = gated_lstm_output.shape

    DebugFunction.trace(gated_lstm_output, "tft.statenr.gated_lstm_output.ase")
    gated_lstm_output = debug(gated_lstm_output)

    DebugFunction.trace(static_enrichment_signal, "tft.statenr.static_enrichment_signal.ase")
    static_enrichment_signal = debug(static_enrichment_signal)

    # replicate the selection signal along time
    time_distributed_context1 = replicate_along_time(static_signal=static_enrichment_signal,
                                                         time_steps=num_temporal_steps)
    # Dimensions:
    # time_distributed_context: [num_samples x num_temporal_steps x state_size]

    DebugFunction.trace(time_distributed_context1, "tft.statenr.time_distributed_context1.ase")
    time_distributed_context1 = debug(time_distributed_context1)


    # for applying the same GRN module on all time-steps, we stack the time dimension with the batch dimension
    flattened_gated_lstm_output = stack_time_steps_along_batch(gated_lstm_output)

    DebugFunction.trace(flattened_gated_lstm_output, "tft.statenr.flattened_gated_lstm_output.ase")
    flattened_gated_lstm_output = debug(flattened_gated_lstm_output)

    time_distributed_context2 = stack_time_steps_along_batch(time_distributed_context1)

    DebugFunction.trace(time_distributed_context2, "tft.statenr.time_distributed_context2.ase")
    time_distributed_context2 = debug(time_distributed_context2)

    # Dimensions:
    # flattened_gated_lstm_output: [(num_samples * num_temporal_steps) x state_size]
    # time_distributed_context: [(num_samples * num_temporal_steps) x state_size]

    # applying the GRN using the static enrichment signal as context data
    enriched_sequence1 = static_enrichment_grn(flattened_gated_lstm_output,
                                                   context=time_distributed_context2)
    # Dimensions:
    # enriched_sequence: [(num_samples * num_temporal_steps) x state_size]

    DebugFunction.trace(enriched_sequence1, "tft.statenr.enriched_sequence1.ase")
    enriched_sequence1 = debug(enriched_sequence1)

    # reshape back to represent temporal dimension separately
    enriched_sequence = enriched_sequence1.view(num_samples, -1, state_size)
    # Dimensions:
    # enriched_sequence: [num_samples x num_temporal_steps x state_size]

    DebugFunction.trace(enriched_sequence, "tft.statenr.enriched_sequence.ase")
    enriched_sequence = debug(enriched_sequence)

    return enriched_sequence

def apply_self_attention(enriched_sequence: torch.tensor,
                         num_historical_steps: int,
                         num_future_steps: int,
                         multihead_attn, post_attention_gating, target_window_start_idx, path):
    DebugFunction.set_output_path(path, 0)
    debug = DebugFunction.apply

    # create a mask - so that future steps will be exposed (able to attend) only to preceding steps
    output_sequence_length = num_future_steps - target_window_start_idx

    mask1 = torch.zeros(output_sequence_length,
                                  num_historical_steps + target_window_start_idx,
                                  device=enriched_sequence.device)
    mask2 = torch.triu(torch.ones(output_sequence_length, output_sequence_length,
                                            device=enriched_sequence.device),
                                 diagonal=1)

    mask = torch.cat([mask1, mask2], dim=1)
    DebugFunction.trace(mask, "tft.asa.mask")

    # Dimensions:
    # mask: [output_sequence_length x (num_historical_steps + num_future_steps)]

    DebugFunction.trace(enriched_sequence, "tft.asa.enriched_sequence")
    enriched_sequence = debug(enriched_sequence)

    q_in = enriched_sequence[:, (num_historical_steps + target_window_start_idx):, :]
    DebugFunction.trace(q_in, "tft.asa.q_in")
    q_in = debug(q_in)

    k_in = enriched_sequence
    DebugFunction.trace(k_in, "tft.asa.k_in")
    k_in = debug(k_in)

    v_in = enriched_sequence
    DebugFunction.trace(v_in, "tft.asa.v_in")
    v_in = debug(v_in)

    # apply the InterpretableMultiHeadAttention mechanism
    post_attention, attention_outputs, attention_scores = multihead_attn(
        q=q_in,  # query
        k=k_in,  # keys
        v=v_in,  # values
        mask=mask.bool())

    enrseq = enriched_sequence
    DebugFunction.trace(enrseq, "tft.asa.enrseq")
    enrseq = debug(enrseq)
    enrseq_residual = enriched_sequence[:, (num_historical_steps + target_window_start_idx):, :]

    DebugFunction.trace(enrseq_residual, "tft.asa.enrseq_residual")
    enrseq_residual = debug(enrseq_residual)
    DebugFunction.trace(post_attention, "tft.asa.post_attention")
    post_attention = debug(post_attention)
    DebugFunction.trace(attention_outputs, "tft.asa.attention_outputs")
    attention_outputs = debug(attention_outputs)
    DebugFunction.trace(attention_scores, "tft.asa.attention_scores")
    attention_scores = debug(attention_scores)
    # Dimensions:
    # post_attention: [num_samples x num_future_steps x state_size]
    # attention_outputs: [num_samples x num_future_steps x state_size]
    # attention_scores: [num_samples x num_future_steps x num_total_steps]

    # Apply gating with a residual connection to the input of this stage.
    # Because the output of the attention layer is only for the future time-steps,
    # the residual connection is only to the future time-steps of the temporal input signal
    gated_post_attention = post_attention_gating(
        x=post_attention,
        residual=enrseq_residual)
    # Dimensions:
    # gated_post_attention: [num_samples x num_future_steps x state_size]
    DebugFunction.trace(gated_post_attention, "tft.asa.gated_post_attention")
    gated_post_attention = debug(gated_post_attention)

    return gated_post_attention, attention_scores
