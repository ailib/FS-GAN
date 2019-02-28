from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import xlwt

'''os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.98)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
'''
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--batch_size", type=int, default=3, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--lamda", type=float, default=0.001)
parser.add_argument("--N_GPU",type=int, default=2)
parser.add_argument("--N_layer", type=int, default=0)
parser.add_argument("--layerl1_weight", type=float, default=0)
parser.add_argument("--img_h",type=int, default=320)
parser.add_argument("--img_w",type=int, default=480)
parser.add_argument("--h_ratio",type=float, default=1)
parser.add_argument("--w_ratio",type=float, default=1)


a = parser.parse_args()

EPS = 1e-12

Examples = collections.namedtuple("Examples", "paths, inputs, targets, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, layers, de_layers, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def conv(batch_input, filter_size, out_channels, stride, name="conv"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE ):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]

        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")

        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def Pool(x,name):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

def batchnorm(input,name="bn"):
    with tf.variable_scope("batchnorm", reuse=tf.AUTO_REUSE ):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable(name + "offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable(name + "scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def deconv(batch_input, out_channels):
    batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
    filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
    # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
    #     => [batch, out_height, out_width, out_channels]
    #padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
    conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
    return conv


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        width = tf.shape(raw_input)[1] # [height, width, channels]
        a_images = preprocess(raw_input[:,:width//2,:])
        b_images = preprocess(raw_input[:,width//2:,:])

        inputs, targets = [b_images, a_images]

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)
        #r = tf.image.resize_images(r, [a.img_h, a.img_w], method=tf.image.ResizeMethod.AREA)


        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        #r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)
        h, w, _ = r.get_shape()
        if h < a.img_h or w < a.img_w:
            r = tf.image.resize_images(r, [a.img_h, a.img_w], method=tf.image.ResizeMethod.AREA)
        if a.mode =="train":
            offsize1 = h - a.img_h
            offsize2 = w - a.img_w
            if offsize1 > 0 and offsize2 > 0:
                offsize = offsize1 if ( offsize1 < offsize2 ) else offsize2
                offset = tf.cast(tf.floor(tf.random_uniform([2], 0, offsize, seed=seed)), dtype=tf.int32)
                r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], a.img_h, a.img_w)
            else:
                r = tf.image.resize_images(r, [a.img_h, a.img_w], method=tf.image.ResizeMethod.AREA)

        if a.mode =="test":
            #a.h_ratio = h/a.img_h
            #a.w_ratio = w/a.img_w
            #a.h_ratio=tf.divide(h, a.img_h, name=None)
            #a.w_ratio=tf.divide(w, a.img_w, name=None)
            r = tf.image.resize_images(r, [a.img_h, a.img_w], method=tf.image.ResizeMethod.AREA)
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, groundtruth_inputs, generator_outputs_channels):

    layers = []
    with tf.variable_scope("encoder_1"):
        x = conv(generator_inputs, filter_size=3, out_channels=a.ngf, stride=1, name="con_1")
        layers.append(x)
        y = conv(groundtruth_inputs, filter_size=3, out_channels=a.ngf, stride=1, name="con_1")
        l1 =  tf.reduce_mean(tf.abs(x - y))
        tf.add_to_collection("L1_encoder_1",l1)   # 480*320*ngf  layers[0]


    layer_specs = [
        a.ngf,      # encoder_2: [batch, 480, 320, ngf] => [batch, 240, 160, ngf]
        a.ngf * 2,  # encoder_3: [batch, 240, 160, ngf] => [batch, 120, 80, ngf * 2]
        a.ngf * 4,  # encoder_4: [batch, 120,  80, ngf * 2] => [batch, 60, 40, ngf * 4]
        a.ngf * 8,  # encoder_5: [batch, 60, 40, ngf * 4] => [batch, 30, 20, ngf * 8]
        a.ngf * 8,  # encoder_6: [batch, 30, 20, ngf * 8] => [batch, 15, 10, ngf * 8]
        ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            x = lrelu(layers[-1], 0.2)
            x = conv(x, 3, out_channels, 1, name="con_1")
            x = batchnorm(x, name="con1_bn")
            x = lrelu(x, 0.2)
            x = conv(x, 4, out_channels, 2, name="con_2")
            x = batchnorm(x, name="con2_bn")
            layers.append(x)

            y = lrelu(y, 0.2)
            y = conv(y, 3, out_channels, 1, name="con_1")
            y = batchnorm(y, name="con1_bn")
            y = lrelu(y, 0.2)
            y = conv(y, 4, out_channels, 2, name="con_2")
            y = batchnorm(y, name="con2_bn")
            l1 = tf.reduce_mean(tf.abs(x - y))
            tf.add_to_collection("L1_encoder_%d" % (len(layers)), l1)

    de_layers = []
    layer_specs_de = [
        (a.ndf * 8, 0.0),  # decoder_1: [batch, 15,10, ndf * 8] => [batch, 30, 20, ndf * 8]
        (a.ndf * 4, 0.0),  # decoder_2: [batch, 30, 20, ndf * 8] => [batch, 60, 40, ndf * 4]
        (a.ndf * 2, 0.0),  # decoder_3: [batch, 60, 40, ndf * 4] => [batch, 120, 80, ndf* 2 ]
        (a.ndf , 0.0),  # decoder_4:[batch, 120, 80, ndf * 2 ] => [batch, 240, 160, ndf]
        (a.ndf , 0.0),  # decoder_5:[batch, 240, 160, ndf ] => [batch, 480, 320, ndf]
    ]
    for (out_channels, dropout) in layer_specs_de:
        with tf.variable_scope("decoder_%d" % (len(de_layers) + 1)):
            if len(de_layers) == 0:
                x = layers[-1]
            else:
                x = tf.concat([de_layers[-1], layers[len(layers) - len(de_layers)-1]], 3)
            x = tf.nn.relu(x)
            x = deconv(x, out_channels)
            x = batchnorm(x)
            de_layers.append(x)

    # decoder_6:[batch, 480, 320, ndf ] => [batch, 480, 320, 3]
    with tf.variable_scope("decoder_%d" % (len(de_layers)+1)):
        x = tf.concat([de_layers[-1], layers[len(layers) - len(de_layers)-1]], 3)
        x = tf.nn.relu(x)
        x = conv(x, 3, generator_outputs_channels, 1, name="con_1")
        x = tf.tanh(x)
        de_layers.append(x)

    return de_layers[-1], layers, de_layers



def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        grad = tf.concat(grads,0)
        grad = tf.reduce_mean(grad,0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    return average_grads

def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        layers_dis = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)
        layers_dis.append(input)

        layer_specs_dis = [
            a.ngf,  # encoder_3: [batch, 256, 256, ngf] => [batch, 128, 128, ngf]
            a.ngf * 2,  # encoder_4: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            a.ngf * 4,  # encoder_5: [batch, 64,  64, ngf * 2] => [batch, 32, 32, ngf * 4]
            a.ngf * 8,  # encoder_6: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ]

        for n_layer, out_channels in enumerate(layer_specs_dis):
            with tf.variable_scope("layer_%d" % (n_layer + 1)):
                x = lrelu(layers_dis[-1], 0.2)
                x = conv(x, 3, out_channels, 1, name="con_1")
                x = batchnorm(x, name="con1_bn")
                x = lrelu(x, 0.2)
                x = conv(x, 3, out_channels, 2, name="con_2")
                x = batchnorm(x, name="con2_bn")
                layers_dis.append(x)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_5"):
            x = lrelu(layers_dis[-1], 0.2)
            x = conv(x, 3, 1, 1, name="con_1")
            x = tf.sigmoid(x)
            layers_dis.append(x)

        return layers_dis[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs, layers, de_layers = create_generator(inputs, targets, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        all_layer_l1 = 0
        for i in range(a.N_layer):
            all_layer_l1 += tf.multiply(a.layerl1_weight * math.log(i+1), tf.get_collection("L1_encoder_%d" % (i+1)))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight + all_layer_l1

    with tf.name_scope("discriminator_train"):
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        tower_disgrads = []
        for i in range(a.N_GPU):
            with tf.device('/gpu:%d' % i):
                discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
                discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
                tower_disgrads.append(discrim_grads_and_vars)
        discrim_grads_and_vars = average_gradients(tower_disgrads)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)

            tower_gengrads = []
            for i in range(a.N_GPU):
                with tf.device('/gpu:%d' % i):
                    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                    gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                    tower_gengrads.append(gen_grads_and_vars)
            gen_grads_and_vars = average_gradients(tower_gengrads)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        layers = layers,
        de_layers = de_layers,
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        #a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)


    def convert(image):
        #if a.mode =="test":
            #new_h = a.img_h * a.h_ratio
            #new_w = a.img_w * a.w_ratio
            #image = tf.image.resize_images(image, [new_h, new_w], method=tf.image.ResizeMethod.AREA)
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", outputs)


    with tf.name_scope("encode_summary"):
        for i in range(len(model.layers)):
            en_layer = tf.image.convert_image_dtype(model.layers[i], dtype=tf.int32)
            b, h, w, c = en_layer.get_shape()
            tf.summary.image("encode_layers_%d" % i, tf.image.convert_image_dtype(tf.split(en_layer, num_or_size_splits=c, axis=3)[0], dtype=tf.uint8))

    with tf.name_scope("decode_summary"):
        for i in range(len(model.de_layers)):
            de_layer = tf.image.convert_image_dtype(model.de_layers[i], dtype=tf.int32)
            b, h, w, c = de_layer.get_shape()
            tf.summary.image("decode_layers_%d" % i, tf.image.convert_image_dtype(tf.split(de_layer, num_or_size_splits=c, axis=3)[0], dtype=tf.uint8))

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    saver = tf.train.Saver(max_to_keep=1,save_relative_paths=True)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            max_steps = min(examples.steps_per_epoch, max_steps)
            all_time = 0
            for step in range(max_steps):
                start = time.time()
                results = sess.run(display_fetches)
                time_spend = time.time()
                all_time = all_time + time_spend - start
                #table.write(step, 0, time_spend)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("test_time:", all_time/max_steps)
        else:
            # training
            start = time.time()

            file = xlwt.Workbook(encoding='utf-8')
            table = file.add_sheet('GGAN_W0', cell_overwrite_ok=True)

            j = 1
            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("gen_loss_L1", results["gen_loss_L1"])

                    #if (train_epoch == 1) and (train_step == 50):
                        #j = 1

                    epoch_value = np.float64(train_epoch)
                    table.write(j, 0, epoch_value)
                    l1_value = np.float64(results["gen_loss_L1"])
                    table.write(j, 1, l1_value)
                    file.save('GGAN_W0.xls')
                    j = j + 1

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
