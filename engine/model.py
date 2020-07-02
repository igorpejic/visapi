import tensorflow as tf

# Embed input sequence [batch_size, seq_length, from_] -> [batch_size, seq_length, to_]
def embed_seq(input_seq, _from, _to, is_training, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope("embedding"): # embed + BN input set
        W_embed = tf.get_variable("weights", [1, _from, _to], initializer=initializer)
        embedded_input = tf.nn.conv1d(input_seq, W_embed, 1, "VALID", name="embedded_input")
        embedded_input = tf.layers.batch_normalization(embedded_input, axis=2, training=is_training, name='layer_norm', reuse=None)
        return embedded_input


# Apply multihead attention to a 3d tensor with shape [batch_size, seq_length, n_hidden].
# Attention size = n_hidden should be a multiple of num_head
# Returns a 3d tensor with shape of [batch_size, seq_length, n_hidden]
def multihead_attention(inputs, num_units=None, num_heads=16, dropout_rate=0.1, is_training=True):
    with tf.variable_scope("multihead_attention", reuse=None):
        # Linear projections
        Q = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        K = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        V = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # num_heads*[batch_size, seq_length, seq_length]
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs) # num_heads*[batch_size, seq_length, seq_length]
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # num_heads*[batch_size, seq_length, n_hidden/num_heads]
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # [batch_size, seq_length, n_hidden]   
        # Residual connection
        outputs += inputs # [batch_size, seq_length, n_hidden]
        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
 
    return outputs


# Apply point-wise feed forward net to a 3d tensor with shape [batch_size, seq_length, n_hidden]
# Returns: a 3d tensor with the same shape and dtype as inputs
def feedforward(inputs, num_units=[2048, 512], is_training=True):
    with tf.variable_scope("ffn", reuse=None):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]   
    return outputs
