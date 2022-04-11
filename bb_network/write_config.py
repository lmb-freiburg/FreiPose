from __future__ import print_function, unicode_literals


def write(cfg, out_file, train_record, test_record, type):
    if type == 'mobilenet':
        cfg_str = get_cfg_as_str(cfg, train_record, test_record)
    else:
        cfg_str = get_cfg_as_str_faster_rcnn(cfg, train_record, test_record)

    with open(out_file, 'w') as fo:
        fo.write(cfg_str)
    print('Wrote cfg file to %s' % out_file)


def get_cfg_as_str_faster_rcnn(cfg, train_record, test_record):
    str = '''
    model {
      faster_rcnn {
        num_classes: 1
        image_resizer {
          keep_aspect_ratio_resizer {
            min_dimension: 300
            max_dimension: 300
          }
        }
        feature_extractor {
          type: "faster_rcnn_resnet50"
          first_stage_features_stride: 16
        }
        first_stage_anchor_generator {
          grid_anchor_generator {
            height_stride: 16
            width_stride: 16
            scales: 0.25
            scales: 0.5
            scales: 1.0
            scales: 2.0
            aspect_ratios: 0.5
            aspect_ratios: 1.0
            aspect_ratios: 2.0
          }
        }
        first_stage_box_predictor_conv_hyperparams {
          op: CONV
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            truncated_normal_initializer {
              stddev: 0.00999999977648
            }
          }
        }
        first_stage_nms_score_threshold: 0.0
        first_stage_nms_iou_threshold: 0.699999988079
        first_stage_max_proposals: 100
        first_stage_localization_loss_weight: 2.0
        first_stage_objectness_loss_weight: 1.0
        initial_crop_size: 14
        maxpool_kernel_size: 2
        maxpool_stride: 2
        second_stage_box_predictor {
          mask_rcnn_box_predictor {
            fc_hyperparams {
              op: FC
              regularizer {
                l2_regularizer {
                  weight: 0.0
                }
              }
              initializer {
                variance_scaling_initializer {
                  factor: 1.0
                  uniform: true
                  mode: FAN_AVG
                }
              }
            }
            use_dropout: false
            dropout_keep_probability: 1.0
          }
        }
        second_stage_post_processing {
          batch_non_max_suppression {
            score_threshold: 0.300000011921
            iou_threshold: 0.600000023842
            max_detections_per_class: 100
            max_total_detections: 100
          }
          score_converter: SOFTMAX
        }
        second_stage_localization_loss_weight: 2.0
        second_stage_classification_loss_weight: 1.0
      }
    }
    train_config {
      batch_size: 1
      data_augmentation_options {
        random_horizontal_flip {
        }
      }
      optimizer {
        momentum_optimizer {
          learning_rate {
            manual_step_learning_rate {
              initial_learning_rate: 0.000300000014249
              schedule {
                step: 900000
                learning_rate: 2.99999992421e-05
              }
              schedule {
                step: 1200000
                learning_rate: 3.00000010611e-06
              }
            }
          }
          momentum_optimizer_value: 0.899999976158
        }
        use_moving_average: false
      }
      gradient_clipping_by_norm: 10.0
      fine_tune_checkpoint: "./data/faster_rcnn_resnet50_coco_2018_01_28/model.ckpt"
      from_detection_checkpoint: true
    '''
    str += '   num_steps: %d' % cfg['bb_train_steps']
    str += '''
    }    
      train_input_reader: {
        tf_record_input_reader { 
    '''
    str += '    input_path: "%s"' % train_record
    str += '''
        }
     
    '''
    str += '   label_map_path: "%s"' % cfg['bb_label_map_path']
    str += '''
      }
      
      eval_config: {
        num_examples: 50   # these are the number of batches to evaluate on, 1998 samples
        use_moving_averages: false
        max_evals: 1
      }
      
      eval_input_reader: {
        tf_record_input_reader {
    '''
    str += '    input_path: "%s"' % test_record
    str += '''
        }
     
    '''
    str += '   label_map_path: "%s"' % cfg['bb_label_map_path']
    str += '''
        shuffle: false
        num_readers: 1
      #  num_epochs: 1  #this makes it fail
      }

    '''
    return str


def get_cfg_as_str(cfg, train_record, test_record):
    str = ''' # BB train config created with write_config script.
    model {
      ssd {
        num_classes: 1
        box_coder {
          faster_rcnn_box_coder {
            y_scale: 10.0
            x_scale: 10.0
            height_scale: 5.0
            width_scale: 5.0
          }
        }
        matcher {
          argmax_matcher {
            matched_threshold: 0.5
            unmatched_threshold: 0.5
            ignore_thresholds: false
            negatives_lower_than_unmatched: true
            force_match_for_each_row: true
          }
        }
        similarity_calculator {
          iou_similarity {
          }
        }
        anchor_generator {
          ssd_anchor_generator {
            num_layers: 6
            min_scale: 0.1
            max_scale: 0.8
            aspect_ratios: 1.0
            aspect_ratios: 2.0
            aspect_ratios: 0.5
            aspect_ratios: 3.0
            aspect_ratios: 0.3333
          }
        }
        image_resizer {
          fixed_shape_resizer {
            height: 300
            width: 300
          }
        }
        box_predictor {
          convolutional_box_predictor {
            min_depth: 0
            max_depth: 0
            num_layers_before_predictor: 0
            use_dropout: false
            dropout_keep_probability: 0.8
            kernel_size: 1
            box_code_size: 4
            apply_sigmoid_to_scores: false
            conv_hyperparams {
              activation: RELU_6,
              regularizer {
                l2_regularizer {
                  weight: 0.00004
                }
              }
              initializer {
                truncated_normal_initializer {
                  stddev: 0.03
                  mean: 0.0
                }
              }
              batch_norm {
                train: true,
                scale: true,
                center: true,
                decay: 0.9997,
                epsilon: 0.001,
              }
            }
          }
        }
        feature_extractor {
          type: 'ssd_mobilenet_v2'
          min_depth: 16
          depth_multiplier: 1.0
          conv_hyperparams {
            activation: RELU_6,
            regularizer {
              l2_regularizer {
                weight: 0.00004
              }
            }
            initializer {
              truncated_normal_initializer {
                stddev: 0.03
                mean: 0.0
              }
            }
            batch_norm {
              train: true,
              scale: true,
              center: true,
              decay: 0.9997,
              epsilon: 0.001,
            }
          }
        }
        loss {
          classification_loss {
            weighted_sigmoid {
              anchorwise_output: true
            }
          }
          localization_loss {
            weighted_smooth_l1 {
              anchorwise_output: true
            }
          }
          hard_example_miner {
            num_hard_examples: 3000
            iou_threshold: 0.99
            loss_type: CLASSIFICATION
            max_negatives_per_positive: 3
            min_negatives_per_image: 0
          }
          classification_weight: 1.0
          localization_weight: 1.0
        }
        normalize_loss_by_num_matches: true
        post_processing {
          batch_non_max_suppression {
            score_threshold: 1e-8
            iou_threshold: 0.6
            max_detections_per_class: 100
            max_total_detections: 100
          }
          score_converter: SIGMOID
        }
      }
    }
    
    train_config: {
      batch_size: 6
      optimizer {
        rms_prop_optimizer: {
          learning_rate: {
            exponential_decay_learning_rate {
              initial_learning_rate: 0.004
              decay_steps: 350000
              decay_factor: 0.75
            }
          }
          momentum_optimizer_value: 0.9
          decay: 0.9
          epsilon: 1.0
        }
      } 
    '''
    str += 'fine_tune_checkpoint: "%s" ' % cfg['bb_fine_tune_checkpoint']
    str += '''
        from_detection_checkpoint: true
        # Note: The below line limits the training process to 200K steps, which we
        # empirically found to be sufficient enough to train the pets dataset. This
        # effectively bypasses the learning rate schedule (the learning rate will
        # never decay). Remove the below line to train indefinitely.
    '''
    str += '   num_steps: %d' % cfg['bb_train_steps']
    str += '''
     
        data_augmentation_options {
          random_horizontal_flip {
          }
        }
        data_augmentation_options {
          ssd_random_crop {
          }
        }
        data_augmentation_options {
          random_pixel_value_scale {  # per pixel scaling factor
            minval: 0.8
            maxval: 1.2
          }
        }
        data_augmentation_options {  # mixture of brightness, hue, sat, and contrast
          random_distort_color {
          }
        }
        data_augmentation_options {  # scaling of the image
          random_image_scale {
              min_scale_ratio: 0.8
              max_scale_ratio: 1.0
          }
        }
        # DATA AUG OPTIONS:
        # https://stackoverflow.com/questions/44906317/what-are-possible-values-for-data-augmentation-options-in-the-tensorflow-object
        # https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py
      }
      
      
      train_input_reader: {
        tf_record_input_reader { 
    '''
    str += '    input_path: "%s"' % train_record
    str += '''
        }
     
    '''
    str += '   label_map_path: "%s"' % cfg['bb_label_map_path']
    str += '''
      }
      
      eval_config: {
        num_examples: 50   # these are the number of batches to evaluate on, 1998 samples
        use_moving_averages: false
        max_evals: 1
      }
      
      eval_input_reader: {
        tf_record_input_reader {
    '''
    str += '    input_path: "%s"' % test_record
    str += '''
        }
     
    '''
    str += '   label_map_path: "%s"' % cfg['bb_label_map_path']
    str += '''
        shuffle: false
        num_readers: 1
      #  num_epochs: 1  #this makes it fail
      }
      '''
    return str

