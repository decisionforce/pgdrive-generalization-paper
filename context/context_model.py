import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import get_activation_fn, try_import_tf

tf1, tf, tfv = try_import_tf()


@DeveloperAPI
class RecurrentNetwork(TFModelV2):
    """Helper class to simplify implementing RNN models with TFModelV2.

    Instead of implementing forward(), you can implement forward_rnn() which
    takes batches with the time dimension added already.

    Here is an example implementation for a subclass
    ``MyRNNClass(RecurrentNetwork)``::

        def __init__(self, *args, **kwargs):
            super(MyModelClass, self).__init__(*args, **kwargs)
            cell_size = 256

            # Define input layers
            input_layer = tf.keras.layers.Input(
                shape=(None, obs_space.shape[0]))
            state_in_h = tf.keras.layers.Input(shape=(256, ))
            state_in_c = tf.keras.layers.Input(shape=(256, ))
            seq_in = tf.keras.layers.Input(shape=(), dtype=tf.int32)

            # Send to LSTM cell
            lstm_out, state_h, state_c = tf.keras.layers.LSTM(
                cell_size, return_sequences=True, return_state=True,
                name="lstm")(
                    inputs=input_layer,
                    mask=tf.sequence_mask(seq_in),
                    initial_state=[state_in_h, state_in_c])
            output_layer = tf.keras.layers.Dense(...)(lstm_out)

            # Create the RNN model
            self.rnn_model = tf.keras.Model(
                inputs=[input_layer, seq_in, state_in_h, state_in_c],
                outputs=[output_layer, state_h, state_c])
            self.register_variables(self.rnn_model.variables)
            self.rnn_model.summary()
    """

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        assert seq_lens is not None
        padded_inputs = input_dict["obs_flat"]
        max_seq_len = tf.shape(padded_inputs)[0] // tf.shape(seq_lens)[0]
        output, new_state = self.forward_rnn(
            add_time_dimension(
                padded_inputs, max_seq_len=max_seq_len, framework="tf"), state,
            seq_lens)
        return tf.reshape(output, [-1, self.num_outputs]), new_state

    def forward_rnn(self, inputs, state, seq_lens):
        """Call the model with the given input tensors and state.

        Arguments:
            inputs (dict): observation tensor with shape [B, T, obs_size].
            state (list): list of state tensors, each with shape [B, T, size].
            seq_lens (Tensor): 1d tensor holding input sequence lengths.

        Returns:
            (outputs, new_state): The model output tensor of shape
                [B, T, num_outputs] and the list of new state tensors each with
                shape [B, size].

        Sample implementation for the ``MyRNNClass`` example::

            def forward_rnn(self, inputs, state, seq_lens):
                model_out, h, c = self.rnn_model([inputs, seq_lens] + state)
                return model_out, [h, c]
        """
        raise NotImplementedError("You must implement this for a RNN model")

    def get_initial_state(self):
        """Get the initial recurrent state values for the model.

        Returns:
            list of np.array objects, if any

        Sample implementation for the ``MyRNNClass`` example::

            def get_initial_state(self):
                return [
                    np.zeros(self.cell_size, np.float32),
                    np.zeros(self.cell_size, np.float32),
                ]
        """
        raise NotImplementedError("You must implement this for a RNN model")


def create_context_embedding(context_input, seq_in, cell_size, context_embedding_dim, name):
    state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h_{}".format(name))
    state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c_{}".format(name))
    # seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

    # Preprocess observation with a hidden layer and send to LSTM cell
    lstm_out, state_h, state_c = tf.keras.layers.LSTM(
        cell_size,
        return_sequences=True,
        return_state=True,
        name="lstm_{}".format(name))(
        # inputs=input_layer,
        inputs=context_input,
        mask=tf.sequence_mask(seq_in),
        initial_state=[state_in_h, state_in_c])

    # Postprocess LSTM output with another hidden layer and compute values
    context_embedding = tf.keras.layers.Dense(
        context_embedding_dim, activation=tf.keras.activations.linear, name="logits_{}".format(name)
    )(lstm_out)

    context_embedding = tf.reshape(context_embedding, [-1, context_embedding_dim])
    return context_embedding, state_in_h, state_in_c, state_h, state_c


class FullyConnectedNetworkWithContext(RecurrentNetwork):
    """Generic fully connected network implemented in ModelV2 API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(FullyConnectedNetworkWithContext, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens", [])
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")
        free_log_std = model_config.get("free_log_std")

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2
            self.log_std_var = tf.Variable(
                [0.0] * num_outputs, dtype=tf.float32, name="log_std")
            self.register_variables([self.log_std_var])

        # We are using obs_flat, so take the flattened shape as input.
        # raw_inputs = tf.keras.layers.Input(shape=(int(np.product(obs_space.shape)),), name="observations")

        assert hasattr(obs_space, "original_space")
        self.context_input_dim = obs_space.original_space[0].shape[0]
        self.obs_dim = obs_space.original_space[1].shape[0]
        assert self.context_input_dim + self.obs_dim == int(np.product(obs_space.shape))

        # === Define common variables ===
        obs_input = tf.keras.layers.Input(shape=(None, self.obs_dim), name="observations")

        # TODO FIXME change context embedding dim!!
        # self.cell_size = model_config["lstm_cell_size"]
        self.cell_size = 64

        # TODO FIXME change context embedding dim!!!
        self.context_embedding_dim = 64

        context_input = tf.keras.layers.Input(shape=(None, self.context_input_dim), name="context_input")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # === Build input for policy network ===
        context_embedding, state_in_h, state_in_c, state_out_h, state_out_c = \
            create_context_embedding(context_input, seq_in, self.cell_size, self.context_embedding_dim, "policy")
        inputs = tf.concat([context_embedding, tf.reshape(obs_input, [-1, self.obs_dim])], axis=1)

        # === Build policy network ===
        # Last hidden layer output (before logits outputs).
        last_layer = inputs
        # The action distribution outputs.
        logits_out = None
        i = 1

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
            i += 1

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            logits_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                last_layer = tf.keras.layers.Dense(
                    hiddens[-1],
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_layer)
            if num_outputs:
                logits_out = tf.keras.layers.Dense(
                    num_outputs,
                    name="fc_out",
                    activation=None,
                    kernel_initializer=normc_initializer(0.01))(last_layer)
            # Adjust num_outputs to be the number of nodes in the last layer.
            else:
                self.num_outputs = (
                        [int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        # Concat the log std vars to the end of the state-dependent means.
        if free_log_std and logits_out is not None:
            def tiled_log_std(x):
                return tf.tile(
                    tf.expand_dims(self.log_std_var, 0), [tf.shape(x)[0], 1])

            log_std_out = tf.keras.layers.Lambda(tiled_log_std)(inputs)
            logits_out = tf.keras.layers.Concatenate(axis=1)([logits_out, log_std_out])

        # === Build value network ===
        last_vf_layer = None
        assert not vf_share_layers
        # Build context model for value network
        context_embedding_vf, state_in_h_vf, state_in_c_vf, state_out_h_vf, state_out_c_vf = \
            create_context_embedding(context_input, seq_in, self.cell_size, self.context_embedding_dim, name="vf")
        inputs_vf = tf.concat([context_embedding_vf, tf.reshape(obs_input, [-1, self.obs_dim])], axis=1)
        if not vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            last_vf_layer = inputs_vf
            i = 1
            for size in hiddens:
                last_vf_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_value_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_vf_layer)
                i += 1

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(
            last_vf_layer if last_vf_layer is not None else last_layer
        )

        # Build the model
        self.base_model = tf.keras.Model(
            [context_input, obs_input, seq_in, state_in_h, state_in_c, state_in_h_vf, state_in_c_vf],
            [(logits_out if logits_out is not None else last_layer), value_out,
             state_out_h, state_out_c, state_out_h_vf, state_out_c_vf]
        )
        self.register_variables(self.base_model.variables)

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def forward_rnn(self, inputs, state, seq_lens):
        # RNN input is 3D, (batch, time, feature).
        # We shrink the time here
        # inputs = tf.reshape(inputs, [-1, self.context_input_dim + self.obs_dim])
        inputs = tf.split(inputs, [self.context_input_dim, self.obs_dim], axis=2)
        model_out, self._value_out, h, c, h_vf, c_vf = self.base_model(inputs + [seq_lens] + state)
        return model_out, [h, c, h_vf, c_vf]

    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    def forward(self, input_dict, state, seq_lens):
        return super().forward(input_dict, state, seq_lens)
