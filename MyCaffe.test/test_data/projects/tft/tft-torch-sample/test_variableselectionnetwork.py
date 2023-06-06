import copy
import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple
from utility import load_batch, DebugFunction
from tft_torch.tft import InputChannelEmbedding, VariableSelectionNetwork, GatedResidualNetwork
from tft_torch.base_blocks import TimeDistributed, NullTransform

debug = DebugFunction.apply
DebugFunction.set_output_path(0)

num_categorical = 7
categorical_cardinalities = [2,3,8,13,72,6,28]
dropout = 0.0
num_numeric = 4
state_size = 64

is_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if is_cuda else "cpu")

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

def apply_temporal_selection(self, temporal_representation: torch.tensor,
                                 static_selection_signal: torch.tensor,
                                 temporal_selection_module: VariableSelectionNetwork
                                 ) -> Tuple[torch.tensor, torch.tensor]:
   num_samples, num_temporal_steps, _ = temporal_representation.shape

   # replicate the selection signal along time
   time_distributed_context = self.replicate_along_time(static_signal=static_selection_signal,
                                                        time_steps=num_temporal_steps)
   # Dimensions:
   # time_distributed_context: [num_samples x num_temporal_steps x state_size]
   # temporal_representation: [num_samples x num_temporal_steps x (total_num_temporal_inputs * state_size)]

   # for applying the same selection module on all time-steps, we stack the time dimension with the batch dimension
   temporal_flattened_embedding = self.stack_time_steps_along_batch(temporal_representation)
   time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
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

   return temporal_selection_output, temporal_selection_weights


static_covariate_encoder = GatedResidualNetwork(input_dim=state_size,
                                                        hidden_dim=state_size,
                                                        output_dim=state_size,
                                                        dropout=dropout)
static_encoder_selection = copy.deepcopy(static_covariate_encoder)


historical_ts_selection = VariableSelectionNetwork(
            input_dim=state_size,
            num_inputs=num_numeric + num_categorical,
            hidden_dim=state_size,
            dropout=dropout,
            context_dim=state_size)

strPath = "data/favorita/weights"

idx = 33
for param in static_covariate_encoder.state_dict():
        strFile = strPath + "/" + str(idx) + "_" + param + ".npy"
        data = np.load(strFile)
        static_covariate_encoder.state_dict()[param] = torch.from_numpy(data).to(device)        
        idx = idx + 1

idx = 135
for param in historical_ts_selection.state_dict():
        strFile = strPath + "/" + str(idx) + "_" + param + ".npy"
        data = np.load(strFile)
        historical_ts_selection.state_dict()[param] = torch.from_numpy(data).to(device)        
        idx = idx + 1

historical_ts_selection.to(device)

historical_ts_rep = np.load("test/iter_0/hist_processed_input.npy")
historical_ts_rep = torch.from_numpy(historical_ts_rep).to(device)

static_ts_rep = np.load("test/iter_0/stat_processed_input.npy")
staticl_ts_rep = torch.from_numpy(static_ts_rep).to(device)

c_selection = static_encoder_selection(static_ts_rep)

selected_historical, historical_selection_weights = apply_temporal_selection(
            temporal_representation=historical_ts_rep,
            static_selection_signal=c_selection,
            temporal_selection_module=historical_ts_selection)

DebugFunction.trace(selected_historical, "selected_historical")

p0 = selected_historical.clone()

p0 = p0 * 0 + 1

loss = (selected_historical - p0).sum()

DebugFunction.trace(loss, "varselnet_loss")
loss = debug(loss)

loss.backward()

print("done!");


