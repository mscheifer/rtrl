import numpy as np
import theano
import theano.tensor as T
import load_books as bk
import shutil
import collections
import re

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

np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

learning_rate = 0.1

l2_param = 0.00

epochs = 200

max_line_size = 80

def char_vec(char):
    vec = np.zeros(len(bk.character_set))
    if char != 0: vec[bk.char_indices[char]] = 1
    return vec

def inject_grad(onto, grads):
    # onto should be a vector
    ret = onto
    for wrt, grad in grads:
        # grad should have one more axis than wrt, which is the size of onto

        # we want to select the grads in 'grad' that match each element of 'onto'
        # and then multiply those by wrt to get the proper differentiation
        has_grad = T.sum(grad * wrt, axis=range(1, len(grad.get_value().shape)))
        ret += has_grad - theano.gradient.disconnected_grad(has_grad)
    return ret

hidden_features = 6 #(784 - char size) or something?

feats = hidden_features + len(bk.character_set)

input_size = len(bk.character_set)

def make_weights(n_in, n_out):
    rng = np.random.RandomState(5678)
    initial_value = rng.normal(scale = 1 / np.sqrt((n_in + n_out) / 2),
        size = (n_in, n_out))
    return theano.shared(value=initial_value, name='w')

# Using a forget weight based on the simpler version of highway networks

w = make_weights(input_size + feats + 1, feats) # +1 for bias
f = make_weights(input_size + feats + 1, feats) # +1 for bias
x = T.dvector('x')

saved_w_grad = theano.shared(np.zeros((feats,) + w.get_value().shape))
saved_f_grad = theano.shared(np.zeros((feats,) + w.get_value().shape))

y_prev = theano.shared(np.zeros(feats))

rtrl_grads = [(w, saved_w_grad), (f, saved_f_grad)]

# Maybe During runtime, not training. If the user is giving info, that can be the x.

hidden_state = inject_grad(y_prev, rtrl_grads)

# Here trying an alternate way of doing biases by appending '1' to the x, y_prev
# vector and then adding an additional column to the weights. Doing this because
# it is simpler. Should test if this is more performant
effective_x = T.concatenate([x, hidden_state, [1]])

forget = T.nnet.sigmoid(T.dot(effective_x, f))

# Simple highway activation
y = T.nnet.sigmoid(T.dot(effective_x, w)) * forget + (1 - forget) * hidden_state

jy_w = T.jacobian(y, w)
jy_f = T.jacobian(y, f)

expected = T.dvector('expected')

#TODO: gotta deal with vanishing gradients. 1. highway net function (lookup if this is similar
#to GRUs). 2. softplus (better relu)

#TODO; what optimizers are good for online learning. Rmsprop, adagrad, adadelta, adam, etc

#TODO: softmax ?

#TODO: categorical cross entropy because can only be in one category?

#Thinking about categorical cross entropy. Binary cross entropy is like combining two predictions
#into 1. The value is true and the value is false as the inverse of the value. Then you use both of
#those in the cost to say how accurate it was at predicting each of those things, even though they
#are mutually exclusive and thus tied together. Categorical just has the 0 side so it is more like
#treating just the probability of something happening, and not the propbability of it not happening.
#But what real effect does this have on the gradients? And why do people think it makes so much sense
#with softmax?

# So if we initialize everything to 0 then we end up with the same values for
# all of the hidden weights because they all have the same gradients at each
# step. We can fix this with random initialization but there is still the
# possibility that during training, two weight rows could converge to then
# basically just do the same thing and be redundant. If that ever becomes a
# problem we can get around it with dropout. The original dropout paper talks
# about this where features learn to be closely dependent on each other.

#cost = T.sum((expected - output) ** 2) / 2 # mean squared error

def bin_cross_entropy(output):

    # Clip the outputs to avoid INF cost as log(0) = INF (and log(1 - 1) = INF)
    min_out = np.finfo(theano.config.floatX).tiny
    max_out = np.dtype(theano.config.floatX).type(1.0) - np.finfo(theano.config.floatX).epsneg

    output = T.clip(output, min_out, max_out)

    cost = T.nnet.binary_crossentropy(output, expected).sum()
    if l2_param > 0: cost += l2_param * T.sum((w ** 2) / 2) # L2 regularization
    return cost

output = y[:len(bk.character_set)]

cost = bin_cross_entropy(output)

w_grad = T.grad(cost, wrt=w)
f_grad = T.grad(cost, wrt=f)

update_w_grad = saved_w_grad, jy_w
update_f_grad = saved_f_grad, jy_f
update_state = y_prev, y
# online updates. Should try epochwise as well
update_weight = w, w - (learning_rate * w_grad)
update_forget = f, f - (learning_rate * f_grad)

time_step_func = theano.function([x, expected], [output, cost], \
    updates=[update_w_grad, update_f_grad, update_weight, update_forget, update_state])

for i in range(epochs):

    total_cost = 0

    for seq in bk.train_books: #TODO: random mini batch?

        outputs = collections.deque(seq[0], maxlen=max_line_size)

        print("\n")# make 2 line space for the overrite printing

        for (index, elem), next_elem in zip(enumerate(seq[:-1]), seq[1:]):

            out, cost = time_step_func(char_vec(elem), char_vec(next_elem))

            outputs.append(bk.index_chars[np.argmax(out)])

            total_cost += cost
            
            print_start = (index + 1) - (len(outputs) - 1)

            def remove_whitespace(s):
                return re.sub(r"\s", " ", s)

            print("\033[F\033[F" + remove_whitespace("".join(list(outputs))), "Cost: ", cost)
            print(remove_whitespace(seq[print_start:print_start + len(outputs)]))

        print("")

    print("End epoch:", i, "Cost:", total_cost / sum(len(seq) for seq in bk.train_books), "\n")

print("Hidden weigths:", str(w.get_value()[:,len(bk.character_set):].shape))
print(w.get_value()[:,len(bk.character_set):])
