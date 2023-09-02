#
# Augmened version of the CFC cell from the open-source GitHub project: https://github.com/raminmh/CfC by Ramin Hasani, 2021
# distributed under the Apache License 2.0: https://github.com/raminmh/CfC/blob/main/LICENSE
#
from this import s
from tkinter import W
import torch
import torch.nn as nn
import numpy as np
from utility import DebugFunction

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)

    def forward(self, inputs, states):
        output_state, cell_state = states

        z = self.input_map(inputs) + self.recurrent_map(output_state)
        i, ig, fg, og = z.chunk(4, 1)

        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        forget_gate = self.sigmoid(fg + 1.0)
        output_gate = self.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = self.tanh(new_cell) * output_gate

        return output_state, new_cell


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class CfcCell(nn.Module):
    def __init__(self, input_size, hidden_size, hparams):
        super(CfcCell, self).__init__()
        
        self._debug = False
        if "debug" in hparams:
            self._debug = hparams["debug"]
            self.debugfn = DebugFunction.apply
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hparams = hparams
        self._no_gate = False
        if "no_gate" in self.hparams:
            self._no_gate = self.hparams["no_gate"]
        self._minimal = False
        if "minimal" in self.hparams:
            self._minimal = self.hparams["minimal"]

        if self.hparams["backbone_activation"] == "silu":
            backbone_activation = nn.SiLU
        elif self.hparams["backbone_activation"] == "relu":
            backbone_activation = nn.ReLU
        elif self.hparams["backbone_activation"] == "tanh":
            backbone_activation = nn.Tanh
        elif self.hparams["backbone_activation"] == "gelu":
            backbone_activation = nn.GELU
        elif self.hparams["backbone_activation"] == "lecun":
            backbone_activation = LeCun
        else:
            raise ValueError("Unknown activation")
        self.layer_list = [
            nn.Linear(input_size + hidden_size, self.hparams["backbone_units"]),
            backbone_activation(),
        ]
        for i in range(1, self.hparams["backbone_layers"]):
            self.layer_list.append(
                nn.Linear(
                    self.hparams["backbone_units"], self.hparams["backbone_units"]
                )
            )
            self.layer_list.append(backbone_activation())
            if "backbone_dr" in self.hparams.keys():
                self.layer_list.append(torch.nn.Dropout(self.hparams["backbone_dr"]))
        self.backbone = nn.Sequential(*self.layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.ff1 = nn.Linear(self.hparams["backbone_units"], hidden_size)
        if self._minimal:
            self.w_tau = torch.nn.Parameter(
                data=torch.zeros(1, self.hidden_size), requires_grad=True
            )
            self.A = torch.nn.Parameter(
                data=torch.ones(1, self.hidden_size), requires_grad=True
            )
        else:
            self.ff2 = nn.Linear(self.hparams["backbone_units"], hidden_size)
            self.time_a = nn.Linear(self.hparams["backbone_units"], hidden_size)
            self.time_b = nn.Linear(self.hparams["backbone_units"], hidden_size)
        self.init_weights()

    def init_weights(self):
        init_gain = self.hparams.get("init")
        if init_gain is not None:
            for w in self.parameters():
                if w.dim() == 2:
                    torch.nn.init.xavier_uniform_(w, gain=init_gain)

    def save_weights(self, strPath, strTag):
        idx = 0
        DebugFunction.set_output_path(strPath, 0)
        for i in range(0, len(self.layer_list)):
            if isinstance(self.layer_list[i], nn.Linear):
                DebugFunction.trace(self.layer_list[i].weight, strTag + "bb_" + str(idx) + ".weight", "weights")
                DebugFunction.trace(self.layer_list[i].bias, strTag + "bb_" + str(idx) + ".bias", "weights")
                idx += 1
        DebugFunction.trace(self.ff1.weight, strTag + "ff1.weight", "weights")
        DebugFunction.trace(self.ff1.bias, strTag + "ff1.bias", "weights")
        DebugFunction.trace(self.ff2.weight, strTag + "ff2.weight", "weights")
        DebugFunction.trace(self.ff2.bias, strTag + "ff2.bias", "weights")
        DebugFunction.trace(self.time_a.weight, strTag + "time_a.weight", "weights")
        DebugFunction.trace(self.time_a.bias, strTag + "time_a.bias", "weights")
        DebugFunction.trace(self.time_b.weight, strTag + "time_b.weight", "weights")
        DebugFunction.trace(self.time_b.bias, strTag + "time_b.bias", "weights")

    def debug_log(self, x, name):
        if self._debug:
            tag = "" if self._tag is None else str(self._tag) + "."
            DebugFunction.trace(x, tag + name)
            return self.debugfn(x)
        else:
            return x

    def forward(self, input, hx, ts, tag = None):
        self._tag = tag
        input = self.debug_log(input, "cell_input")
        hx = self.debug_log(hx, "cell_hx")
        ts = self.debug_log(ts, "cell_ts")

        batch_size = input.size(0)
        ts = ts.view(batch_size, 1)
        x = torch.cat([input, hx], 1)
        
        x1 = self.debug_log(x, "cell_x1")

        #for i in range(0, len(self.layer_list)):
        #    x1 = self.layer_list[i](x1)
        #    x1 = self.debug_log(x1, "cell_x1_{}".format(i))
        #x2 = x1

        x2 = self.backbone(x1)
        x2 = self.debug_log(x2, "cell_x2")

        if self._minimal:
            # Solution
            ff1 = self.ff1(x2)
            new_hidden = (
                -self.A
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Cfc

            x2a = x2.clone()
            x2b = x2.clone()
            x2c = x2.clone()
            x2d = x2.clone()

            x2a = self.debug_log(x2a, "cell_x2a")

            x3 = self.ff1(x2a)
            x3 = self.debug_log(x3, "cell_x3")
            
            ff1 = self.tanh(x3)
            ff1 = self.debug_log(ff1, "cell_ff1")

            x2b = self.debug_log(x2b, "cell_x2b")

            x4 = self.ff2(x2b)
            x4 = self.debug_log(x4, "cell_x4")

            ff2 = self.tanh(x4)
            ff2 = self.debug_log(ff2, "cell_ff2")

            x2c = self.debug_log(x2c, "cell_x2c")

            t_a = self.time_a(x2c)
            t_a = self.debug_log(t_a, "cell_t_a")

            x2d = self.debug_log(x2d, "cell_x2d")

            t_b = self.time_b(x2d)
            t_b = self.debug_log(t_b, "cell_t_b")

            t_interp1_a = t_a * ts
            self.debug_log(t_interp1_a, "cell_t_interp1_a")

            t_interp1 = t_interp1_a + t_b
            self.debug_log(t_interp1, "cell_t_interp1")

            t_interp = self.sigmoid(t_interp1)
            t_interp = self.debug_log(t_interp, "cell_t_interp")

            if self._no_gate:
                new_hidden = ff1 + t_interp * ff2
                new_hidden = self.debug_log(new_hidden, "cell_new_hidden")
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
                new_hidden = self.debug_log(new_hidden, "cell_new_hidden")
        return new_hidden



class Cfc(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        out_feature,
        hparams,
        return_sequences=False,
        use_mixed=False,
        use_ltc=False,
    ):
        super(Cfc, self).__init__()

        self._debug = False
        self._debug_init = False
        if "debug" in hparams and hparams["debug"]:
            self._debug = True
            if self._debug:
                self.debugfn = DebugFunction.apply
            if "debug_init" in hparams and hparams["debug_init"]:
                self._debug_init = True

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.return_sequences = return_sequences

        if use_ltc:
            self.rnn_cell = LTCCell(in_features, hidden_size, hparams)
        else:
            self.rnn_cell = CfcCell(in_features, hidden_size, hparams)
        self.use_mixed = use_mixed
        if self.use_mixed:
            self.lstm = LSTMCell(in_features, hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.out_feature)

    def debug_log(self, x, name):
        if self._debug:
            DebugFunction.trace(x, name)
            return self.debugfn(x)
        else:
            return x

    def save_weights(self, strPath, strTag):
        DebugFunction.set_output_path(strPath, 0)
        self.rnn_cell.save_weights(strPath, strTag + "rnn_cell.")
        DebugFunction.trace(self.fc.weight, strTag + "fc.weight", "weights")
        DebugFunction.trace(self.fc.bias, strTag + "fc.bias", "weights")

    def forward(self, x, timespans=None, mask=None):
        if self._debug_init:
            x = self.debug_log(x, "x")
            if timespans != None:
                timespans = self.debug_log(timespans, "timespans")
            if mask != None:
                mask = self.debug_log(mask, "mask")

        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        true_in_features = x.size(2)
        h_state = torch.zeros((batch_size, self.hidden_size), device=device)
        if self.use_mixed:
            c_state = torch.zeros((batch_size, self.hidden_size), device=device)
        output_sequence = []
        if mask is not None:
            forwarded_output = torch.zeros((batch_size, self.out_feature), device=device)
            forwarded_input = torch.zeros((batch_size, true_in_features), device=device)
            time_since_update = torch.zeros((batch_size, true_in_features), device=device)

            forwarded_input = self.debug_log(forwarded_input, "forwarded_input")
            forwarded_output = self.debug_log(forwarded_output, "forwarded_output")
            time_since_update = self.debug_log(time_since_update, "time_since_update")

        for t in range(seq_len):
            inputs = x[:, t]
            inputs = self.debug_log(inputs, "inputs_a_" + str(t))

            ts = timespans[:, t].squeeze()
            ts = self.debug_log(ts, "ts_a_" + str(t))

            inputs1 = inputs
            if mask is not None:
                if mask.size(-1) == true_in_features:
                    mask1 = mask[:, t]
                    mask1_inv = 1 - mask1

                    mask1 = self.debug_log(mask1, "mask1_" + str(t))
                    mask1_inv = self.debug_log(mask1_inv, "mask1_inv_a_" + str(t))

                    forwarded_input = (mask1 * inputs + mask1_inv * forwarded_input)

                    ts1 = ts.view(batch_size, 1)
                    ts1 = self.debug_log(ts1, "ts1_" + str(t))
                    time_since_update1 = ts1 + time_since_update
                    time_since_update1 = self.debug_log(time_since_update1, "time_since_udpate1_" + str(t))

                    time_since_update = time_since_update1 * mask1_inv
                else:
                    forwarded_input = inputs

                forwarded_input = self.debug_log(forwarded_input, "forwarded_input_" + str(t))
                time_since_update = self.debug_log(time_since_update, "time_since_update_" + str(t))

                if (
                    true_in_features * 2 < self.in_features
                    and mask.size(-1) == true_in_features
                ):
                    # we have 3x in-features
                    inputs1 = torch.cat(
                        (forwarded_input, time_since_update, mask[:, t]), dim=1
                    )
                else:
                    # we have 2x in-feature
                    inputs1 = torch.cat((forwarded_input, mask[:, t]), dim=1)

            if self.use_mixed:
                h_state, c_state = self.lstm(inputs1, (h_state, c_state))

            h_state = self.debug_log(h_state, "h_state_" + str(t))
            inputs1 = self.debug_log(inputs1, "inputs1_" + str(t))
            ts = self.debug_log(ts, "ts_" + str(t))

            h_state1 = self.rnn_cell.forward(inputs1, h_state, ts)
            h_state1 = self.debug_log(h_state1, "h_state1_" + str(t))

            if mask is not None:
                mask1 = mask[:, t]
                cur_mask, _ = torch.max(mask1, dim=1)
                cur_mask = cur_mask.view(batch_size, 1)
                current_output = self.fc(h_state1)

                cur_mask_inv = (1.0 - cur_mask)

                cur_mask = self.debug_log(cur_mask, "cur_mask_" + str(t))
                cur_mask_inv = self.debug_log(cur_mask_inv, "cur_mask_inv_" + str(t))
                current_output = self.debug_log(current_output, "current_output_" + str(t))

                forwarded_output = (
                    cur_mask * current_output + cur_mask_inv * forwarded_output
                )
                forward_output = self.debug_log(forwarded_output, "forwarded_output_" + str(t))
            if self.return_sequences:
                output_sequence.append(self.fc(h_state1))

            h_state = h_state1

        if self.return_sequences:
            readout = torch.stack(output_sequence, dim=1)
        elif mask is not None:
            readout = forwarded_output
        else:
            readout = self.fc(h_state1)

        readout = self.debug_log(readout, "readout")

        return readout


class LTCCell(nn.Module):
    def __init__(
        self,
        in_features,
        units,
        hparams,
        ode_unfolds=6,
        epsilon=1e-8,
    ):
        super(LTCCell, self).__init__()
        self._tag = None
        self._debug = False
        if "debug" in hparams:
            self._debug = hparams["debug"]
            self.debugfn = DebugFunction.apply
        self.in_features = in_features
        self.units = units
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        # self.softplus = nn.Softplus()
        self.softplus = nn.Identity()
        self._allocate_parameters()

    @property
    def state_size(self):
        return self.units

    @property
    def sensory_size(self):
        return self.in_features

    def add_weight(self, name, init_value):
        param = torch.nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _erev_initializer(self, shape=None):
        return np.random.default_rng().choice([-1, 1], size=shape)

    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(
                self._erev_initializer((self.state_size, self.state_size))
            ),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(
                self._erev_initializer((self.sensory_size, self.state_size))
            ),
        )

        self._params["input_w"] = self.add_weight(
            name="input_w",
            init_value=torch.ones((self.sensory_size,)),
        )
        self._params["input_b"] = self.add_weight(
            name="input_b",
            init_value=torch.zeros((self.sensory_size,)),
        )


    def _sigmoid(self, v_pre, mu, sigma, tag):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        v_pre = self.debug_log(v_pre, tag + ".v_pre")

        mu = self.debug_log(mu, tag + ".mu")
        sigma = self.debug_log(sigma, tag + ".sigma")

        mues = v_pre - mu
        mues = self.debug_log(mues, tag + ".mues")

        x = sigma * mues
        x = self.debug_log(x, tag + ".x")

        x1 = torch.sigmoid(x)
        x1 = self.debug_log(x1, tag + ".x1")
        return x1

    def save_weights(self):
        self.debug_log(self._params["gleak"], "gleak", "weights")
        self.debug_log(self._params["vleak"], "vleak", "weights")
        self.debug_log(self._params["cm"], "cm", "weights")
        self.debug_log(self._params["sigma"], "sigma", "weights")
        self.debug_log(self._params["mu"], "mu", "weights")
        self.debug_log(self._params["w"], "w", "weights")
        self.debug_log(self._params["erev"], "erev", "weights")
        self.debug_log(self._params["sensory_sigma"], "sensory_sigma", "weights")
        self.debug_log(self._params["sensory_mu"], "sensory_mu", "weights")
        self.debug_log(self._params["sensory_w"], "sensory_w", "weights")
        self.debug_log(self._params["sensory_erev"], "sensory_erev", "weights")
        self.debug_log(self._params["input_w"], "input_w", "weights")
        self.debug_log(self._params["input_b"], "input_b", "weights")

    def debug_log(self, x, name, subpath = None):
        if self._debug:
            tag = "" if self._tag is None else str(self._tag) + "."
            DebugFunction.trace(x, tag + name, subpath)
            x = self.debugfn(x)
        return x

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        inputs = self.debug_log(inputs, "inputs.a");
        v_pre = self.debug_log(v_pre, "v_pre");
        elapsed_time = self.debug_log(elapsed_time, "elapsed_time.a");
        self._params["sensory_w"] = self.debug_log(self._params["sensory_w"], "sensory_w")
        self._params["sensory_mu"] = self.debug_log(self._params["sensory_mu"], "sensory_mu")
        self._params["sensory_sigma"] = self.debug_log(self._params["sensory_sigma"], "sensory_sigma")
        self._params["sensory_erev"] = self.debug_log(self._params["sensory_erev"], "sensory_erev")
        sensory_erev = self._params["sensory_erev"].clone()

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_sigmoid = self._sigmoid(inputs, self._params["sensory_mu"], self._params["sensory_sigma"], "sig")
        sensory_w_sigmoid = self.debug_log(sensory_w_sigmoid, "sensory_w_sigmoid")
        sensory_w_activation = self.softplus(self._params["sensory_w"]) * sensory_w_sigmoid

        sensory_w_activation = self.debug_log(sensory_w_activation, "sensory_w_activation")

        sensory_rev_activation = sensory_w_activation * sensory_erev

        sensory_rev_activation = self.debug_log(sensory_rev_activation, "sensory_rev_activation")

        sensory_w_activation = self.debug_log(sensory_w_activation, "sensory_w_activation.b")

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        w_numerator_sensory = self.debug_log(w_numerator_sensory, "w_numerator_sensory")
        w_denominator_sensory = self.debug_log(w_denominator_sensory, "w_denominator_sensory")

        # cm/t is loop invariant
        cm_t = self.softplus(self._params["cm"]).view(1, -1) / (
            (elapsed_time + 1) / self._ode_unfolds
        )

        self._params["cm"] = self.debug_log(self._params["cm"], "cm")
        cm_t = self.debug_log(cm_t, "cm_t")

        self._params["gleak"] = self.debug_log(self._params["gleak"], "gleak")
        self._params["vleak"] = self.debug_log(self._params["vleak"], "vleak")
        self._params["erev"] = self.debug_log(self._params["erev"], "erev")
        self._params["w"] = self.debug_log(self._params["w"], "w")
        self._params["mu"] = self.debug_log(self._params["mu"], "mu")
        self._params["sigma"] = self.debug_log(self._params["sigma"], "sigma")

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds):
            v_pre = self.debug_log(v_pre, str(t) + ".v_pre.a")
            wt_w = self._params["w"].clone()
            wt_w = self.debug_log(wt_w, str(t) + ".w")
            
            wt_mu = self._params["mu"].clone()
            wt_mu = self.debug_log(wt_mu, str(t) + ".mu")
            
            wt_sigma = self._params["sigma"].clone()
            wt_sigma = self.debug_log(wt_sigma, str(t) + ".sigma")

            w_sigmoid = self._sigmoid(v_pre, wt_mu, wt_sigma, "sig" + str(t))
            w_sigmoid = self.debug_log(w_sigmoid, str(t) + ".w_sigmoid")

            w_activation = self.softplus(wt_w) * w_sigmoid
            w_activation = self.debug_log(w_activation, str(t) + ".w_activation")

            wt_erev = self._params["erev"].clone()
            wt_erev = self.debug_log(wt_erev, str(t) + ".erev")
            
            rev_activation = w_activation * wt_erev
            rev_activation = self.debug_log(rev_activation, str(t) + ".rev_activation")

            w_numerator_sensory = self.debug_log(w_numerator_sensory, str(t) + ".w_numerator_sensory")
            w_denominator_sensory = self.debug_log(w_denominator_sensory, str(t) + ".w_denominator_sensory")

            # Reduce over dimension 1 (=source neurons)
            w_numerator1 = torch.sum(rev_activation, dim=1)
            w_numerator1 = self.debug_log(w_numerator1, str(t) + ".w_numerator1")
           
            w_numerator = w_numerator1 + w_numerator_sensory
            w_numerator = self.debug_log(w_numerator, str(t) + ".w_numerator.a")

            w_activation = self.debug_log(w_activation, str(t) + ".w_activation.a")

            w_denominator1 = torch.sum(w_activation, dim=1)
            w_denominator1 = self.debug_log(w_denominator1, str(t) + ".w_denominator1")

            w_denominator = w_denominator1 + w_denominator_sensory
            w_denominator = self.debug_log(w_denominator, str(t) + ".w_denominator.a")

            wt_gleak = self._params["gleak"].clone()
            wt_gleak = self.debug_log(wt_gleak, str(t) + ".gleak.a")
            wt_vleak = self._params["vleak"].clone()
            wt_vleak = self.debug_log(wt_vleak, str(t) + ".vleak")

            cm_t = self.debug_log(cm_t, str(t) + ".cm_t.b")
            v_pre = self.debug_log(v_pre, str(t) + ".v_pre.b")

            numerator1 = cm_t * v_pre
            numerator1 = self.debug_log(numerator1, str(t) + ".numerator1")

            numerator2 = self.softplus(wt_gleak) * wt_vleak
            numerator2 = self.debug_log(numerator2, str(t) + ".numerator2")
            numerator3 = numerator1 + numerator2
            numerator3 = self.debug_log(numerator3, str(t) + ".numerator3")

            numerator = (numerator3 + w_numerator)

            wt_gleak_b = wt_gleak.clone();
            wt_gleak_b = self.debug_log(wt_gleak_b, str(t) + ".gleak.b")
            
            numerator = self.debug_log(numerator, str(t) + ".numerator")

            denominator1 = cm_t + self.softplus(wt_gleak_b)
            denominator1 = self.debug_log(denominator1, str(t) + ".denominator1")

            denominator = denominator1 + w_denominator
            denominator = self.debug_log(denominator, str(t) + ".denominator")

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)
            if torch.any(torch.isnan(v_pre)):
                breakpoint()

            v_pre = self.debug_log(v_pre, str(t) + ".v_pre1.c")
            
        return v_pre

    def _map_inputs(self, inputs):

        inputs = self.debug_log(inputs, "mi.inputs.a")
        self._params["input_w"] = self.debug_log(self._params["input_w"], "mi.input_w")

        inputs1 = inputs * self._params["input_w"]

        inputs1 = self.debug_log(inputs1, "mi.inputs.b")
        self._params["input_b"] = self.debug_log(self._params["input_b"], "mi.input_b")

        inputs2 = inputs1 + self._params["input_b"]
        inputs2 = self.debug_log(inputs2, "mi.inputs.c")

        return inputs2

    def _map_outputs(self, state):
        output = state
        output = output * self._params["output_w"]
        output = output + self._params["output_b"]
        return output

    def _clip(self, w):
        return torch.nn.ReLU()(w)

    def apply_weight_constraints(self):
        self._params["w"].data = self._clip(self._params["w"].data)
        self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
        self._params["cm"].data = self._clip(self._params["cm"].data)
        self._params["gleak"].data = self._clip(self._params["gleak"].data)

    def forward(self, input, hx, ts, tag = None):
        # Regularly sampled mode (elapsed time = 1 second)
        ts = ts.view((-1, 1))
        inputs = self._map_inputs(input)

        self._tag = tag
        input = self.debug_log(input, "cell_input")
        hx = self.debug_log(hx, "cell_hx")
        ts = self.debug_log(ts, "cell_ts")

        next_state = self._ode_solver(inputs, hx, ts)
        next_state = self.debug_log(next_state, "next_state")

        # outputs = self._map_outputs(next_state)

        return next_state

