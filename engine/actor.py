import tensorflow as tf
import numpy as np

from solution_checker import SolutionChecker

distr = tf.contrib.distributions

class DynamicMultiRNN(object):
    """
        Implementation of a dynamic multi-cell RNN

        Attributes:
            action_size(int) -- Number of actions available
            batch_size(int) -- Batch size.
            num_activations(int) --  Number of activations in the LSTM cell
            num_layers(int) -- Number of stacked LSTM layers
            state_maxServiceLength(int) -- Max input sequence length

            positions[Batch, seq_length] -- outputs the position
    """

    def __init__(self, action_size, batch_size, input_, input_len_, num_activations, num_layers):

        self.action_size = action_size
        self.batch_size = batch_size
        self.num_activations = num_activations
        self.num_layers = num_layers

        self.positions = []
        self.outputs = []

        self.input_ = input_
        self.input_len_ = input_len_

        # Variables initializer
        initializer = tf.contrib.layers.xavier_initializer()

        # Generate multiple LSTM cell
        cells = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(self.num_activations, state_is_tuple=True) for _ in range(self.num_layers)], state_is_tuple=True)

        # LSTMs internal state
        c_initial_states = []
        h_initial_states = []

        # Initial state (tuple) is trainable but same for all batch
        for i in range(self.num_layers):
            first_state = tf.get_variable("var{}".format(i), [1, self.num_activations], initializer=initializer)
            #first_state = tf.Print(first_state, ["first_state", first_state], summarize=10)

            c_initial_state = tf.tile(first_state, [self.batch_size, 1])
            h_initial_state = tf.tile(first_state, [self.batch_size, 1])

            c_initial_states.append(c_initial_state)
            h_initial_states.append(h_initial_state)

        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(c_initial_states[idx], h_initial_states[idx])
             for idx in range(self.num_layers)]
        )

        states_series, current_state = tf.nn.dynamic_rnn(cells, input_, initial_state=rnn_tuple_state, sequence_length=input_len_)
        #states_series = tf.Print(states_series, ["states_series", states_series, tf.shape(states_series)], summarize=10)

        self.outputs = tf.layers.dense(states_series, self.action_size, activation=tf.nn.softmax)       # [Batch, seq_length, action_size]
        #self.outputs = tf.Print(self.outputs, ["outputs", self.outputs, tf.shape(self.outputs)],summarize=10)

        # Multinomial distribution
        prob = tf.contrib.distributions.Categorical(probs=self.outputs)

        # Sample from distribution
        self.positions = prob.sample()        # [Batch, seq_length]
        self.positions = tf.cast(self.positions, tf.int32)
        #self.positions = tf.Print(self.positions, ["position", self.positions, tf.shape(self.positions)], summarize=10)



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


# Encode input sequence [batch_size, seq_length, n_hidden] -> [batch_size, seq_length, n_hidden]
def encode_seq(input_seq, input_dim, num_stacks, num_heads, num_neurons, is_training, dropout_rate=0.):
    with tf.variable_scope("stack"):
        for i in range(num_stacks): # block i
            with tf.variable_scope("block_{}".format(i)): # Multihead Attention + Feed Forward
                input_seq = multihead_attention(input_seq, num_units=input_dim, num_heads=num_heads, dropout_rate=dropout_rate, is_training=is_training)
                input_seq = feedforward(input_seq, num_units=[num_neurons, input_dim], is_training=is_training)
        return input_seq # encoder_output is the ref for actions [Batch size, Sequence Length, Num_neurons]
            

# From a query (decoder output) [Batch size, n_hidden] and a set of reference (encoder_output) [Batch size, seq_length, n_hidden]
# predict a distribution over next decoder input
def pointer(encoded_ref, query, mask, W_ref, W_q, v, C=10., temperature=1.0):
    encoded_query = tf.expand_dims(tf.matmul(query, W_q), 1) # [Batch size, 1, n_hidden]
    scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query), [-1]) # [Batch size, seq_length]
    scores = C*tf.tanh(scores/temperature) # control entropy
    masked_scores =  tf.clip_by_value(scores -100000000.*mask, -100000000., 100000000.) # [Batch size, seq_length]
    return masked_scores


# From a query [Batch size, n_hidden], glimpse at a set of reference vectors (ref) [Batch size, seq_length, n_hidden]
def full_glimpse(ref, from_, to_, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope("glimpse"):
        W_ref_g =tf.get_variable("W_ref_g",[1,from_, to_],initializer=initializer)
        W_q_g =tf.get_variable("W_q_g",[from_, to_],initializer=initializer)
        v_g =tf.get_variable("v_g",[to_],initializer=initializer)
        # Attending mechanism
        encoded_ref_g = tf.nn.conv1d(ref, W_ref_g, 1, "VALID", name="encoded_ref_g") # [Batch size, seq_length, n_hidden]
        scores_g = tf.reduce_sum(v_g * tf.tanh(encoded_ref_g), [-1], name="scores_g") # [Batch size, seq_length]
        attention_g = tf.nn.softmax(scores_g, name="attention_g")
        # 1 glimpse = Linear combination of reference vectors (defines new query vector)
        glimpse = tf.multiply(ref, tf.expand_dims(attention_g,2))
        glimpse = tf.reduce_sum(glimpse,1)
        return glimpse


# Embed input sequence [batch_size, seq_length, from_] -> [batch_size, seq_length, to_]
def embed_seq(input_seq, from_, to_, is_training, BN=True, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope("embedding"): # embed + BN input set
        W_embed = tf.get_variable("weights",[1,from_, to_], initializer=initializer)
        embedded_input = tf.nn.conv1d(input_seq, W_embed, 1, "VALID", name="embedded_input")
        if BN == True: embedded_input = tf.layers.batch_normalization(embedded_input, axis=2, training=is_training, name='layer_norm', reuse=None)
        return embedded_input


class Actor(object):
    
    def __init__(self, config):

        self.config = config
        # Data config
        self.batch_size = config.batch_size # batch size
        self.n = config.n # number of bins
        self.w = config.w # width of box to fit in
        self.h = config.h # height of box to fit in
        self.dimension = config.dimension # dimensions of a box
        
        # Network config
        self.input_embed = config.input_embed # dimension of embedding space
        self.num_neurons = config.num_neurons # dimension of hidden states (encoder)
        self.num_stacks = config.num_stacks # encoder num stacks
        self.num_heads = config.num_heads # encoder num heads
        self.query_dim = config.query_dim # decoder query space dimension
        self.num_units = config.num_units # dimension of attention product space (decoder and critic)
        self.num_neurons_critic = config.num_neurons_critic # critic n-1 layer num neurons
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
        
        # Training config (actor and critic)
        self.global_step = tf.Variable(0, trainable=False, name="global_step") # actor global step
        self.global_step2 = tf.Variable(0, trainable=False, name="global_step2") # critic global step
        self.init_B = config.init_B # critic initial baseline
        self.lr_start = config.lr_start # initial learning rate
        self.lr_decay_step = config.lr_decay_step # learning rate decay step
        self.lr_decay_rate = config.lr_decay_rate # learning rate decay rate
        self.is_training = config.is_training # swith to False if test mode

        self.count_non_placed_tiles = config.count_non_placed_tiles
        self.combinatorial_reward = config.combinatorial_reward

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = tf.placeholder(tf.float32, [None, self.n, self.dimension], name="input_coordinates")
        
        with tf.variable_scope("actor"): self.encode_decode()
        with tf.variable_scope("critic"): self.build_critic()
        with tf.variable_scope("environment"): self.build_reward()
        with tf.variable_scope("optimizer"): self.build_optim()
        self.merged = tf.summary.merge_all()    
        
        
    def encode_decode(self):
        actor_embedding = embed_seq(
            input_seq=self.input_, from_=self.dimension, to_= self.input_embed,
            is_training=self.is_training, BN=True, initializer=self.initializer)
        actor_encoding = encode_seq(
            input_seq=actor_embedding, input_dim=self.input_embed,
            num_stacks=self.num_stacks, num_heads=self.num_heads,
            num_neurons=self.num_neurons, is_training=self.is_training)
        if self.is_training == False:
            actor_encoding = tf.tile(actor_encoding,[self.batch_size,1,1])
        
        idx_list, log_probs, entropies = [], [], [] # tours index, log_probs, entropies
        mask = tf.zeros((self.batch_size, self.n)) # mask for actions
        
        n_hidden = actor_encoding.get_shape().as_list()[2] # input_embed
        W_ref = tf.get_variable("W_ref",[1, n_hidden, self.num_units],initializer=self.initializer)
        W_q = tf.get_variable("W_q",[self.query_dim, self.num_units],initializer=self.initializer)
        v = tf.get_variable("v",[self.num_units],initializer=self.initializer)
        
        encoded_ref = tf.nn.conv1d(actor_encoding, W_ref, 1, "VALID") # actor_encoding is the ref for actions [Batch size, seq_length, n_hidden]
        query1 = tf.zeros((self.batch_size, n_hidden)) # initial state
        query2 = tf.zeros((self.batch_size, n_hidden)) # previous state
        query3 = tf.zeros((self.batch_size, n_hidden)) # previous previous state
            
        W_1 =tf.get_variable("W_1",[n_hidden, self.query_dim],initializer=self.initializer) # update trajectory (state)
        W_2 =tf.get_variable("W_2",[n_hidden, self.query_dim],initializer=self.initializer)
        W_3 =tf.get_variable("W_3",[n_hidden, self.query_dim],initializer=self.initializer)
    
        for step in range(self.n): # sample from POINTER      
            query = tf.nn.relu(tf.matmul(query1, W_1) + tf.matmul(query2, W_2) + tf.matmul(query3, W_3))
            logits = pointer(encoded_ref=encoded_ref, query=query, mask=mask, W_ref=W_ref, W_q=W_q, v=v, C=self.config.C, temperature=self.config.temperature)
            prob = distr.Categorical(logits) # logits = masked_scores
            idx = prob.sample()
            
            idx_list.append(idx) # tour index
            log_probs.append(prob.log_prob(idx)) # log prob
            entropies.append(prob.entropy()) # entropies
            mask = mask + tf.one_hot(idx, self.n) # mask
            
            idx_ = tf.stack([tf.range(self.batch_size,dtype=tf.int32), idx],1) # idx with batch   
            query3 = query2
            query2 = query1
            query1 = tf.gather_nd(actor_encoding, idx_) # update trajectory (state)
            
        idx_list.append(idx_list[0]) # return to start
        self.tour = tf.stack(idx_list, axis=1) # permutations
        self.log_prob = tf.add_n(log_probs) # corresponding log-probability for backprop
        self.entropies = tf.add_n(entropies)
        tf.summary.scalar('log_prob_mean', tf.reduce_mean(self.log_prob))
        tf.summary.scalar('entropies_mean', tf.reduce_mean(self.entropies))
        
        
    def build_reward(self): # reorder input % tour and return tour length (euclidean distance)
        self.permutations = tf.stack(  # this just creates a vectors with repeating idxs so we can gather later
            [
                tf.tile(tf.expand_dims(tf.range(self.batch_size,dtype=tf.int32), 1), [1, self.n+1]),
                self.tour
            ], 2
        )
        if self.is_training==True:
            self.ordered_input_ = tf.gather_nd(self.input_,self.permutations)
        else:
            self.ordered_input_ = tf.gather_nd(tf.tile(self.input_,[self.batch_size,1,1]),self.permutations)

        solution_checker = SolutionChecker(self.n, self.w, self.h)
        sess = tf.Session()
        rewards = tf.py_func(
            solution_checker.get_rewards, [self.ordered_input_, self.count_non_placed_tiles, self.combinatorial_reward], tf.float32)
        
        self.reward = rewards
        tf.summary.scalar('reward_mean', tf.reduce_mean(rewards))

            
    def build_critic(self):
        critic_embedding = embed_seq(input_seq=self.input_, from_=self.dimension, to_= self.input_embed, is_training=self.is_training, BN=True, initializer=self.initializer)
        critic_encoding = encode_seq(input_seq=critic_embedding, input_dim=self.input_embed, num_stacks=self.num_stacks, num_heads=self.num_heads, num_neurons=self.num_neurons, is_training=self.is_training)
        frame = full_glimpse(ref=critic_encoding, from_=self.input_embed, to_=self.num_units, initializer=tf.contrib.layers.xavier_initializer()) # Glimpse on critic_encoding [Batch_size, input_embed]
        
        with tf.variable_scope("ffn"): #  2 dense layers for predictions
            h0 = tf.layers.dense(frame, self.num_neurons_critic, activation=tf.nn.relu, kernel_initializer=self.initializer)
            w1 = tf.get_variable("w1", [self.num_neurons_critic, 1], initializer=self.initializer)
            b1 = tf.Variable(self.init_B, name="b1")
            self.predictions = tf.squeeze(tf.matmul(h0, w1)+b1)
            tf.summary.scalar('predictions_mean', tf.reduce_mean(self.predictions))
            
    def build_optim(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops): # Update moving_mean and moving_variance for BN
            
            with tf.name_scope('reinforce'):
                lr1 = tf.train.natural_exp_decay(learning_rate=self.lr_start, global_step=self.global_step, decay_steps=self.lr_decay_step, decay_rate=self.lr_decay_rate, staircase=False, name="learning_rate1") # learning rate actor
                tf.summary.scalar('lr', lr1)
                opt1 = tf.train.AdamOptimizer(learning_rate=lr1) # Optimizer
                self.loss = tf.reduce_mean(tf.stop_gradient(self.reward-self.predictions)*self.log_prob, axis=0) # loss actor
                gvs1 = opt1.compute_gradients(self.loss) # gradients
                capped_gvs1 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs1 if grad is not None] # L2 clip
                self.trn_op1 = opt1.apply_gradients(grads_and_vars=capped_gvs1, global_step=self.global_step) # minimize op actor
                
            with tf.name_scope('state_value'):
                lr2 = tf.train.natural_exp_decay(learning_rate=self.lr_start, global_step=self.global_step2, decay_steps=self.lr_decay_step, decay_rate=self.lr_decay_rate, staircase=False, name="learning_rate2") # learning rate critic
                opt2 = tf.train.AdamOptimizer(learning_rate=lr2) # Optimizer
                loss2 = tf.losses.mean_squared_error(self.reward, self.predictions) # loss critic
                gvs2 = opt2.compute_gradients(loss2) # gradients
                capped_gvs2 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs2 if grad is not None] # L2 clip
                self.trn_op2 = opt2.apply_gradients(grads_and_vars=capped_gvs2, global_step=self.global_step2) # minimize op critic
