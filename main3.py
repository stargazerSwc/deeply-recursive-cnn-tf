# coding=utf8

"""
Deeply-Recursive Convolutional Network for Image Super-Resolution
by Jiwon Kim, Jung Kwon Lee and Kyoung Mu Lee Department of ECE, ASRI, Seoul National University, Korea

Paper: http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Kim_Deeply-Recursive_Convolutional_Network_CVPR_2016_paper.html

Test implementation using TensorFlow library.

Author: Jin Yamanaka
Many thanks for: Masayuki Tanaka and Shigesumi Kuwashima
Project: https://github.com/jiny2001/deeply-recursive-cnn-tf
"""

# import tensorflow as tf
# 0903
import tensorflow.compat.v1 as tf
import super_resolution as sr
import super_resolution_utilty as util



#0903
def del_all_flags(FLAGS):
    flags_dict = opt._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        opt.__delattr__(keys)

del_all_flags(tf.opt.FLAGS)

import argparse
import os
parser = argparse.ArgumentParser(description='Train Super Resolution Models')

# Model
paser.add_argument('--initial_lr', default=0.001, type=float, help='Initial learning rate')
paser.add_argument('--lr_decay', default=0.5, help="Learning rate decay rate when it does not reduced during specific epoch")
paser.add_argument('--lr_decay_epoch', default=4,type=int,help= "Decay learning rate when loss does not decrease")
paser.add_argument('--beta1', default=0.1,type=float, help="Beta1 form adam optimizer")
paser.add_argument('--beta2', default=0.1, type=float,help="Beta2 form adam optimizer")
paser.add_argument('--momentum',default= 0.9, type=float,help="Momentum for momentum optimizer and rmsprop optimizer")
paser.add_argument('--feature_num', default=96, type=int,help="Number of CNN Filters")
paser.add_argument('--cnn_size', default=3, type=int,help="Size of CNN filters")
paser.add_argument('--inference_depth', default=9, type=int,help="Number of recurrent CNN filters")
paser.add_argument('--batch_num', default=64, type=int,help="Number of mini-batch images for training")
paser.add_argument('--batch_size', default=41, type=int,help="Image size for mini-batch")
paser.add_argument('--stride_size', default=21, type=int,help="Stride size for mini-batch")
paser.add_argument('--optimizer', default="adam", help="Optimizer: can be [gd, momentum, adadelta, adagrad, adam, rmsprop]")
paser.add_argument('--loss_alpha', default=1, type=int,help="Initial loss-alpha value (0-1). Don't use intermediate outputs when 0.")
paser.add_argument('--loss_alpha_zero_epoch', default=25, help="Decrease loss-alpha to zero by this epoch")
paser.add_argument('--loss_beta', default=0.0001, type=float,help="Loss-beta for weight decay")
paser.add_argument('--weight_dev', default=0.001, type=float,help="Initial weight stddev")
paser.add_argument('--initializer', default="he", help="Initializer: can be [uniform, stddev, diagonal, xavier, he]")

# Image Processing
paser.add_argument('--scale', default=2, type=int,help="Scale for Super Resolution (can be 2 or 4)")
paser.add_argument('--max_value', default=255.0, type=float,help="For normalize image pixel value")
paser.add_argument('--channels', default=1, type=int,help="Using num of image channels. Use YCbCr when channels=1.")
paser.add_argument('--jpeg_mode', default=False, help="Using Jpeg mode for converting from rgb to ycbcr")
paser.add_argument('--residual', default=False, help="Using residual net")

# Training or Others
paser.add_argument('--is_training', default=True, help="Train model with 91 standard images")
paser.add_argument('--dataset', default="set5", help="Test dataset. [set5, set14, bsd100, urban100, all, test] are available")
paser.add_argument('--training_set', default="ScSR", help="Training dataset. [ScSR, Set5, Set14, Bsd100, Urban100] are available")
paser.add_argument('--evaluate_step', default=20, type=int,help="steps for evaluation")
paser.add_argument('--save_step', default=2000, type=int,help="steps for saving learned model")
paser.add_argument('--end_lr', default=1e-5, type=float,help="Training end learning rate")
paser.add_argument('--checkpoint_dir', default="model", help="Directory for checkpoints")
paser.add_argument('--cache_dir', default="cache", help="Directory for caching image data. If specified, build image cache")
paser.add_argument('--data_dir', default="data", help="Directory for test/train images")
paser.add_argument('--load_model', default=False, help="Load saved model before start")
paser.add_argument('--model_name', default="", help="model name for save files and tensorboard log")

# Debugging or Logging
paser.add_argument('--output_dir', default="output", help="Directory for output test images")
paser.add_argument('--log_dir', default="tf_log", help="Directory for tensorboard log")
paser.add_argument('--debug', default=False, help="Display each calculated MSE and weight variables")
paser.add_argument('--initialise_log', default=True, help="Clear all tensorboard log before start")
paser.add_argument('--visualize', default=True, help="Save loss and graph data")
paser.add_argument('--summary', default=False, help="Save weight and bias")


def main(_):
  opt = paser.parse_args()

  print("Super Resolution (tensorflow version:%s)" % tf.__version__)
  print("%s\n" % util.get_now_date())

  if opt.model_name is "":
    model_name = "model_F%d_D%d_LR%f" % (opt.feature_num, opt.inference_depth, opt.initial_lr)
  else:
    model_name = "model_%s" % opt.model_name
  model = sr.SuperResolution(FLAGS, model_name=model_name)

  test_filenames = util.build_test_filenames(opt.data_dir, opt.dataset, opt.scale)
  if opt.is_training:
    if opt.dataset == "test":
      training_filenames = util.build_test_filenames(opt.data_dir, opt.dataset, opt.scale)
    else:
      training_filenames =  util.get_files_in_directory(opt.data_dir + "/" + opt.training_set + "/")

    print("Loading and building cache images...")
    model.load_datasets(opt.cache_dir, training_filenames, test_filenames, opt.batch_size, opt.stride_size)
  else:
    opt.load_model = True

  model.build_embedding_graph()
  model.build_inference_graph()
  model.build_reconstruction_graph()
  model.build_optimizer()
  model.init_all_variables(load_initial_data=opt.load_model)

  if opt.is_training:
    train(training_filenames, test_filenames, model)
  
  psnr = 0
  total_mse = 0
  for filename in test_filenames:
    mse = model.do_super_resolution_for_test(filename, opt.output_dir)
    total_mse += mse
    psnr += util.get_psnr(mse)

  print ("\n--- summary --- %s" % util.get_now_date())
  model.print_steps_completed()
  util.print_num_of_total_parameters()
  print("Final MSE:%f, PSNR:%f" % (total_mse / len(test_filenames), psnr / len(test_filenames)))

  
def train(training_filenames, test_filenames, model):

  mse = model.evaluate()
  model.print_status(mse)

  while model.lr > opt.end_lr:
  
    logging = model.step % opt.evaluate_step == 0
    model.build_training_batch()
    model.train_batch(log_mse=logging)

    if logging:
      mse = model.evaluate()
      model.print_status(mse)

    if model.step > 0 and model.step % opt.save_step == 0:
      model.save_model()

  model.end_train_step()
  model.save_all()

  if opt.debug:
    model.print_weight_variables()
    

if __name__ == '__main__':
  tf.app.run()
