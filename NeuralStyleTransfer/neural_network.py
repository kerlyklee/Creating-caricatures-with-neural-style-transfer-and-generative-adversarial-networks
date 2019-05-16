# code below was part of DeepLearning.ai CNNs course assignment

import tensorflow as tf
from nst_utils import *


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.transpose(tf.reshape(a_C, shape=[n_H*n_W, -1]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, shape=[n_H*n_W, -1]))

    J_content = tf.reduce_sum((a_C_unrolled - a_G_unrolled)**2)/(4*n_H*n_W*n_C)

    return J_content


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A,tf.transpose(A))

    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, -1]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, -1]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum((GS - GG)**2)/(4*(n_H*n_W*n_C)**2)

    return J_style_layer


def compute_style_cost(model, sess):
    """
    Computes the overall style cost from several chosen layers
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    J_style = 0

    for layer_name, coeff in CONFIG.STYLE_LAYERS:

        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha*J_content + beta*J_style

    return J


def prepare_network(content_image, style_image, learning_rate=1.0):
    '''
    Runs a tensor flow session in which it prepares the graph of cost function
    for optimisation. Returns the session and a tensor flow optimizer object
    needed for 'train_network' method.
    Arguments:
    model_path -- string, path to VGG-19 model
    content_image -- numpy array, content image preprocessed for VGG-19
    style-image -- numpy array, style image preprocessed for VGG-19
    Returns:
    sess -- tensorflow session object
    train_step -- tensor flow optimizer to be minimized
    model -- dict, a loaded VGG model
    '''

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    model = load_vgg_model("imagenet-vgg-verydeep-19.mat")

    # define graph for computing content cost
    sess.run(model['input'].assign(content_image))

    # this layer is used to define content cost
    out = model['conv4_2']
    a_C = sess.run(out)
    a_G = out
    J_content = compute_content_cost(a_C, a_G)

    # define graph for computing style cost
    sess.run(model['input'].assign(style_image))
    J_style = compute_style_cost(model, sess)

    # total cost
    J = total_cost(J_content, J_style)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(J)

    return sess, train_step, model

def train_network(sess, train_step, model, content_image, output_path,
                  num_iterations, save_intermediate):
    '''
    Optimises the cost function defined by 'prepare_network' method.
    Arguments:
    sess -- tensorflow session
    train_step -- optimizer.minimze(cost)
    model -- dictionary containing pretrained VGG model
    content_image -- array, content_image
    output_path -- string, generated image path
    num_iterations -- int, number of iterations of optimisation
    save_intermediate -- bool, whether to save intermediate generated images or not
    Returns:
    generated_image -- array, generated image
    '''
    # initalized image to be generated
    generated_image = generate_noise_image(content_image)

    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(generated_image))

    printProgressBar(0, num_iterations, prefix='Progress:', suffix='Complete', length=50)
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])

        if save_intermediate:
            if i % (int(num_iterations/10)) == 0:
                # save current generated image
                save_image('/output/iter' + str(i) + ".jpg", generated_image)

        printProgressBar(i + 1, num_iterations, prefix='Progress:',
                         suffix='Complete', length=50)

    # save last generated image
    name = 'tulemus'
    result_ext = '.jpg'
    result_name = name + result_ext
    save_image(result_name, generated_image)

    return generated_image