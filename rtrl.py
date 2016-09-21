import numpy as np
import theano
import theano.tensor as T
import load_books as bk
import sys

#theano.printing.pydotprint(jyFunc, outfile = 'rtrl.png', scan_graphs=True)

learning_rate = 0.1

epochs = 2

max_line_size = 80

def char_vec(char):
    vec = np.zeros(len(bk.character_set))
    if char != 0: vec[bk.char_indices[char]] = 1
    return vec

def inject_grad(onto, wrt, grad):
    # we want to select the grads in 'grad' that match each element of 'onto'
    # and then multiply those by wrt to get the proper differentiation
    has_grad = T.sum(wrt * grad, axis=[1,2])
    return onto + has_grad - theano.gradient.disconnected_grad(has_grad)

hidden_features = 2 #(784 - char size) or something?

feats = hidden_features + len(bk.character_set)

input_size = len(bk.character_set)

saved_grad_shape = [feats, feats, input_size + feats]

w = theano.shared(np.zeros([feats, input_size + feats]), name='w')
x = T.dvector('x')

saved_grad = theano.shared(np.zeros(saved_grad_shape))

y_prev = theano.shared(np.zeros(feats))

y = T.nnet.sigmoid(T.dot(w, T.concatenate([x, inject_grad(y_prev, w, saved_grad)])))

jy = T.jacobian(y, w)

expected = T.dvector('expected')

output = y[:len(bk.character_set)]

#TODO: biases

#TODO: softmax ?

#TODO: categorical cross entropy because can only be in one category?

#TODO: if we somehow end up with exactly 0 for an output (possibly even values close to 0 then
# some rounding happens, we will get INF and then NAN from the logs in cross entropy. How can we
# prevent this?

#cost = T.sum((expected - y[:len(character_set)]) ** 2) / 2 # mean squared error
cost = T.nnet.binary_crossentropy(output, expected).sum()

grad = T.grad(cost, wrt=w)

update_grad = saved_grad, jy
# online updates. Should try epochwise as well
update_weight = w, w - (learning_rate * grad)
update_state = y_prev, y

time_step_func = theano.function([x, expected], [output], \
    updates=[update_grad, update_weight, update_state])

#TODO: cap to 80 characters for the 2 lines

for i in xrange(epochs):

    print "Begin epoch: " + str(i)

    for seq in bk.train_books: #TODO: random mini batch?

        outputs = seq[0]

        print "\n" # make space for the overrite printing

        for elem, next_elem in zip(seq[:-1], seq[1:]):
            out = time_step_func(char_vec(elem), char_vec(next_elem))
            outputs += bk.index_chars[np.argmax(out)]
            
            print_start = max(0, len(outputs) - max_line_size)

            print("\033[F\033[F" + outputs[print_start:].replace('\n', ' '))
            print(seq[print_start:len(outputs)].replace('\n', ' '))

        print ""
