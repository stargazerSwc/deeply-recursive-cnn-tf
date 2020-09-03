# coding=utf8

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import super_resolution as sr
import super_resolution_utilty as util


# 0903
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


del_all_flags(tf.flags.FLAGS)

import argparse
import os

parser = argparse.ArgumentParser(description='test Super Resolution Models')

# Model
parser.add_argument('--initial_lr', default=0.001,type=float, help="Initial learning rate")
parser.add_argument('--lr_decay', default=0.5,type=float, help="Learning rate decay rate when it does not reduced during specific epoch")
parser.add_argument('--lr_decay_epoch', default=4, type=int,help="Decay learning rate when loss does not decrease")
parser.add_argument('--beta1', default=0.1, type=float,help="Beta1 form adam optimizer")
parser.add_argument('--beta2', default=0.1,type=float, help="Beta2 form adam optimizer")
parser.add_argument('--momentum', default=0.9, type=float,help="Momentum for momentum optimizer and rmsprop optimizer")
parser.add_argument('--feature_num', default=96, type=int,help="Number of CNN Filters")
parser.add_argument('--cnn_size', default=3, type=int,help="Size of CNN filters")
parser.add_argument('--inference_depth', default=9, type=int,help="Number of recurrent CNN filters")
parser.add_argument('--batch_num', default=64, type=int,help="Number of mini-batch images for training")
parser.add_argument('--batch_size', default=41, type=int,help="Image size for mini-batch")
parser.add_argument('--stride_size', default=21, type=int,help="Stride size for mini-batch")
parser.add_argument('--optimizer', default="adam", help="Optimizer: can be [gd, momentum, adadelta, adagrad, adam, rmsprop]")
parser.add_argument('--loss_alpha', default=1,type=int, help="Initial loss-alpha value (0-1). Don't use intermediate outputs when 0.")
parser.add_argument('--loss_alpha_zero_epoch', default=25, type=int,help="Decrease loss-alpha to zero by this epoch")
parser.add_argument('--loss_beta', default=0.0001, type=float,help="Loss-beta for weight decay")
parser.add_argument('--weight_dev', default=0.001, type=float,help="Initial weight stddev")
parser.add_argument('--initializer',default="he", help="Initializer: can be [uniform, stddev, diagonal, xavier, he]")

# Image Processing
parser.add_argument('--scale', default=2, type=int,help="Scale for Super Resolution (can be 2 or 4)")
parser.add_argument('--max_value', default=255.0, type=float,help="For normalize image pixel value")
parser.add_argument('--channels', default=1, type=int,help="Using num of image channels. Use YCbCr when channels=1.")
parser.add_argument('--jpeg_mode',default=False,help="Using Jpeg mode for converting from rgb to ycbcr")
parser.add_argument('--residual',default=False, help="Using residual net")

# Training or Others
parser.add_argument('--is_training', default=True, type=float,help="Train model with 91 standard images")
parser.add_argument('--dataset', default="set5",help="Test dataset. [set5, set14, bsd100, urban100, all, test] are available")
parser.add_argument('--training_set', default="ScSR",  help="Training dataset. [ScSR, Set5, Set14, Bsd100, Urban100] are available")
parser.add_argument('--evaluate_step',default=20, type=int,help="steps for evaluation")
parser.add_argument('--save_step', default=2000, type=int,help="steps for saving learned model")
parser.add_argument('--end_lr', default=1e-5, type=float,help="Training end learning rate")
parser.add_argument('--checkpoint_dir', default="model",  help="Directory for checkpoints")
parser.add_argument('--cache_dir', default="cache",  help="Directory for caching image data. If specified, build image cache")
parser.add_argument('--data_dir', default="data",  help="Directory for test/train images")
parser.add_argument('--load_model', default=False, type=float,help="Load saved model before start")
parser.add_argument('--model_name', default="",  help="model name for save files and tensorboard log")

# Debugging or Logging
parser.add_argument('--output_dir', default="output",  help="Directory for output test images")
parser.add_argument('--log_dir', default="tf_log", help="Directory for tensorboard log")
parser.add_argument('--debug', default=False, type=float,help="Display each calculated MSE and weight variables")
parser.add_argument('--initialise_log', default=True, type=float,help="Clear all tensorboard log before start")
parser.add_argument('--visualize', default=True, type=float,help="Save loss and graph data")
parser.add_argument('--summary', default=False, type=float,help="Save weight and bias")

parser.add_argument('--file', default="",  help="Test filename")

if __name__ == '__main__':
	opt = parser.parse_args()
	print("Super Resolution (tensorflow version:%s)" % tf.__version__)
	print("%s\n" % util.get_now_date())

	if opt.model_name is "":
		model_name = "model_F%d_D%d_LR%f" % (opt.feature_num, opt.inference_depth, opt.initial_lr)
	else:
		model_name = "model_%s" % opt.model_name
	model = sr.SuperResolution(opt, model_name=model_name)

	test_filenames = [opt.file]
	opt.load_model = True

	model.build_embedding_graph()
	model.build_inference_graph()
	model.build_reconstruction_graph()
	model.build_optimizer()
	model.init_all_variables(load_initial_data=opt.load_model)

	model.do_super_resolution(opt.file, opt.output_dir)
