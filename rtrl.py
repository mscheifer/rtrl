import numpy as np
import theano
import theano.tensor as T
import shutil
import collections
import re
import lasagne
import load_books as bk
import time
import argparse

cmd_arg_def = argparse.ArgumentParser(description="Learn from text")

cmd_arg_def.add_argument("-d", "--debug", help=("Make it easier to catch errors""in the theano implementation"), action="store_true")

cmd_arg_def.add_argument("-a", "--addr_dims", metavar='address_dimensions',
type=int, required=True, help="Number of dimensons in the address")

# hunch that addr dims * 2 is good
cmd_arg_def.add_argument("-l", "--depth", metavar='memory_depth', type=int,
required=True, help="Size of the vector stored at each memory location")

cmd_arg_def.add_argument("-e", "--epochs", metavar='number_of_epochs', type=int,
default=100, help="Number of training epochs")

cmd_args = cmd_arg_def.parse_args()

if cmd_args.debug:
    theano.config.optimization='None'
    theano.config.exception_verbosity='high'
    theano.config.compute_test_value='warn'

np.random.seed(5678)

np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

# I think for the memory mode we should have a depth that is twice the address
# size. That way one memory location has info to point to 2, in practice it's
# enough info to compute any of the other memory locations based on the
# weights. Also, just having a depth greater than the address size means we can
# store where to go next as well as some bits about what to output

# If the write vector was highwayed from the read vector then we would have a
# constant error for the same memory location. But if we initialize the bias to
# -1 like in the paper to encourage the constant error, how does the initial
# information get injected into the value to overrwite the 0s? How did the
# original paper do it?
# ACTUALLY, do we even need highway? seems like the memory write alread is a highway

# So if we initialize everything to 0 then we end up with the same values for
# all of the hidden weights because they all have the same gradients at each
# step. We can fix this with random initialization but there is still the
# possibility that during training, two weight rows could converge to then
# basically just do the same thing and be redundant. If that ever becomes a
# problem we can get around it with dropout. The original dropout paper talks
# about this where features learn to be closely dependent on each other.
#
# Dropout with the memory will need to be different. If we drop out address
# weights we need to make sure that we don't drop out both bits in the same
# column, otherwise the read/write values would go straight to 0. We could also
# do some interesting stuff with droping areas of memory.
#
# Drop out 0.2 means multiply vector by 1 / (1 - 0.2) to compensate
#
# If we implement multiple ticks per time step, we can drop out some ticks.

# Highway style activations should help with vanishing gradients at the start of
# training, but what about adapting to changes later on, like catastrophic
# interference (solving that). Once the model has learned to forget, how can it
# learn to remember again?
# Or does the gradient information always carry though anyway, even when it is
# unused?

learning_rate = 0.1 # Only used if updater requires it

l2_param = 0.00

epochs = cmd_args.epochs

address_dims = cmd_args.addr_dims
address_size = address_dims * 2
memory_depth = cmd_args.depth

def expand_memory(address): # address is flattened vec of pairs of 2 values
    assert address.ndim == 1
    # this should work no matter what the depth dimension of the address space is
    # (any more dimensions beyond the length of address)
    n_dims = address_dims
    expanded_address = T.ones([2] * n_dims)

    for dim in range(address_dims):
        i = dim * 2
        dim_vals = address[i:i+2]
        to_broadcast = T.reshape(dim_vals, [1] * dim + [2] + [1] * (n_dims - dim - 1))
        expanded_address *= to_broadcast
    
    # Then add 1 broadcastable dimension for the depth
    return T.shape_padright(expanded_address, 1)

def read_memory(memory, address):
    return T.sum(expand_memory(address) * memory, axis=range(0, memory.ndim - 1))

def write_memory(memory, address, write_value):
    assert write_value.ndim == 1
    expanded_address = expand_memory(address)
    assert expanded_address.ndim == memory.ndim

    return (memory * (1 - expanded_address)) + (write_value * expanded_address)

def inject_grad(onto, grads):
    ret = onto
    for wrt, grad in grads:
        assert grad.ndim == wrt.ndim + onto.ndim

        # we want to select the grads in 'grad' that match each element of 'onto'
        # and then multiply those by wrt to get the proper differentiation
        has_grad = T.sum(grad * wrt, axis=range(onto.ndim, grad.ndim))
        ret += has_grad - theano.gradient.disconnected_grad(has_grad)
    return ret

# write address + write value + read address
mem_system_feats = address_size + memory_depth + address_size

# input + read value
input_size = len(bk.character_set) + memory_depth

def make_weights(n_in, n_out, name):
    initial_value = np.random.normal(scale = 1 / np.sqrt((n_in + n_out) / 2),
        size = (n_in, n_out))
    return theano.shared(value=initial_value, name=name)

# Using a forget weight based on the simpler version of highway networks

w = make_weights(input_size + 1, mem_system_feats, 'w') # +1 for bias
output_w = make_weights(input_size + 1, len(bk.character_set), 'output_w') # +1 for bias

prev_read_address = theano.shared(np.zeros(address_size), 'adrr')

prev_memory = theano.shared(np.zeros(((2,) * address_dims) + (memory_depth,)), 'mem')

saved_a_grad = theano.shared(np.zeros(prev_read_address.get_value().shape + w.get_value().shape))
saved_m_grad = theano.shared(np.zeros(prev_memory.get_value().shape + w.get_value().shape))

read_address_with_grads = inject_grad(prev_read_address, [(w, saved_a_grad)])
memory_with_grads = inject_grad(prev_memory, [(w, saved_m_grad)])

# Here trying an alternate way of doing biases by appending '1' to the x, y_prev
# vector and then adding an additional column to the weights. Doing this because
# it is simpler. Should test if this is more performant
def effective_x(input, hidden_state):
    return T.concatenate([input, hidden_state, [1]])

# lengths should sum up to the vector length
def split_vector_by_lengths(vector, lengths):
    assert vector.ndim == 1

    acc_pos = 0
    ret = ()
    for length in lengths:
        ret += (vector[acc_pos:acc_pos + length],)
        acc_pos += length
 
    return ret

def memory_activation(x, w, memory):

    full_feature_vector = T.nnet.sigmoid(T.dot(x, w))

    write_address, write_value, read_address = split_vector_by_lengths(full_feature_vector, 
        [address_size, memory_depth, address_size])

    return write_address, write_value, read_address

def softmax_log_likelihood(unscaled_output, target_logits):
    softmax = T.nnet.softmax(unscaled_output)
    return T.nnet.categorical_crossentropy(softmax, target_logits).sum()

#split input to our desired size first
#previous state should have the injected RTRL gradients
def partial_bptt(input, previous_read_address, memory, weights, output_weights):

    def step(input_i, read_address, mem, ws, output_ws):

        combined_x = effective_x(input_i, read_memory(mem, read_address))

        write_address, write_value, next_read_address = (
            memory_activation(combined_x, ws, mem))

        external_output = T.dot(combined_x, output_weights)

        next_memory = write_memory(mem, write_address, write_value)
        
        return [external_output, next_read_address, next_memory]

    outputs, updates = theano.scan(step, sequences=input, outputs_info=
        (None, dict(initial = previous_read_address), dict(initial = memory)),
         non_sequences=[weights, output_weights])

    assert len(updates) == 0

    external_outputs = outputs[0]
    read_addresses = outputs[1]
    memories = outputs[2]
    return external_outputs, read_addresses[-1], memories[-1]

input_logits = T.ivector('inputs')
input_logits.tag.test_value = bk.logits("abbabaabba", input_logits.dtype)

xs = T.extra_ops.to_one_hot(input_logits, len(bk.character_set))

target_logits = T.ivector('targets')
target_logits.tag.test_value = bk.logits("bbabaabbab", target_logits.dtype)

outputs, read_address, memory = (partial_bptt
    (xs, read_address_with_grads, memory_with_grads, w, output_w))

bptt_cost = softmax_log_likelihood(outputs, target_logits)
if l2_param > 0: bptt_cost += l2_param * T.sum((w ** 2) / 2) # L2 regularization

j_read_address_w = T.jacobian(read_address, w)
j_memory_w = T.reshape(T.jacobian(T.flatten(memory), w),
     prev_memory.get_value().shape + w.get_value().shape)
# Reshape will make things broadcastable by default, but then updating fails
# because shared variables are not broadcastable by default and broadcastable
# has to match for a shared var update. We don't want to boradcast anyway. 
# This is only a problem if the memory depth is 1, so make the depth not BCable
j_memory_w = T.unbroadcast(j_memory_w, prev_memory.ndim - 1)

update_a_grad = saved_a_grad, j_read_address_w
update_m_grad = saved_m_grad, j_memory_w
update_address = prev_read_address, read_address
update_memory = prev_memory, memory

weight_updates = list(lasagne.updates.adadelta(bptt_cost, [w, output_w]).items())

# online updates. Should try epochwise as well
bptt_func = theano.function([input_logits, target_logits], [outputs, bptt_cost],
    updates=[update_a_grad, update_m_grad, update_address, update_memory] + weight_updates)

#TODO: does this for sure execute on the GPU? What about if I did .set_value() instaed.
#that would be easier to read.
reset_state = theano.function([], [], updates=[
    (prev_read_address, T.zeros_like(prev_read_address)),
    (prev_memory, T.zeros_like(prev_memory)),
    (saved_a_grad, T.zeros_like(saved_a_grad)),
    (saved_m_grad, T.zeros_like(saved_m_grad))])

bptt_len = 30 # for testing RTRL
#bptt_len = mem_system_feats # some number on the order of feats, to get a^3 not a^4 runtime

for i in range(epochs):

    start_time = time.time()

    total_cost = 0

    for seq in bk.train_books: #TODO: random mini batch?

        # Reset the state between sequences. Not trying to be real disingenous
        # to the model, it's not useful for what we're trying to do. The input
        # we give it doesn't match it's prediciton even when it predicted the
        # correct thing for the first character of a new sequence. I think it's
        # not worth my time to figure out why that's difficult for it to learn
        reset_state()

        prev_last_input = seq[0]
        #Range starts at 1 because we don't try to predict the first character
        for segment in [seq[start : start + bptt_len] for start in range(1, len(seq), bptt_len)]:

            predictions, cost = bptt_func(bk.logits(prev_last_input + segment[:-1],
                 input_logits.dtype), bk.logits(segment, target_logits.dtype))
            prev_last_input = segment[-1]

            total_cost += cost

#            print("Raw     :", predictions)
            print("Predict :", bk.output_as_str(predictions))
            print("Expected:", segment)

        print("")

    avg_cost = total_cost / sum(len(seq) for seq in bk.train_books)
    print("End epoch:", i, "Cost:", avg_cost, "Time:", str(time.time() - start_time) + "s", "\n")

print("Hidden weigths:", str(w.get_value().shape))
print(w.get_value())
