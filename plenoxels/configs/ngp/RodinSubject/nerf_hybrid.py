config = {
 'expname': 'rodin_subject_hybrid',
 'logdir': './logs/rodin',
 'device': 'cuda:0',

 'data_downsample': 1.0,
 'data_dirs': ['/mnt/blob2/render_output_hd/subject'],
 'num_subjects': 8,
 'contract': False,
 'ndc': False,
 'rootdir': '/mnt/blob2/render_output_hd',
 'max_tr_frames': 295,
 'max_ts_frames': 5,

 # Optimization settings
 'num_steps': 5001,
 'num_epochs': 30,
 'batch_size': 8196,
 'finetune_mlp': True,
 'optim_type': 'adam',
 'scheduler_type': 'warmup_cosine',
 'lr': 0.0002,
 'optim_type_triplane': 'adam',
 'scheduler_type_triplane': 'warmup_cosine',
 'lr_triplane': 0.003,

 # Regularization
 'plane_tv_weight': 0.0001,
#  'plane_tv_weight_proposal_net': 0.0001,
#  'histogram_loss_weight': 1.0,
#  'distortion_loss_weight': 0.001,

 # Training settings
 'save_every': 5000,
 'valid_every': 5000,
 'save_outputs': True,
 'train_fp16': True,

 # Raymarching settings
#  'single_jitter': False,
#  'num_samples': 48,
#  # proposal sampling
#  'num_proposal_samples': [256, 128],
#  'num_proposal_iterations': 2,
#  'use_same_proposal_network': False,
#  'use_proposal_weight_anneal': True,
#  'proposal_net_args_list': [
#    {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [64, 64, 64]},
#    {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [128, 128, 128]}
#  ],

 # Model settings
 'ngp_subject': True,
 'multiscale_res': [1],
 'density_activation': 'trunc_exp',
 'concat_features_across_scales': True,
 'linear_decoder': False,
 'grid_config': [{
   'grid_dimensions': 2,
   'input_coordinate_dim': 3,
   'output_coordinate_dim': 32,
   'resolution': [512, 512, 512]
 }],
}
