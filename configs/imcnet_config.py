import types

def get_args_parser():
    args = types.SimpleNamespace()
    # ---------------------------- Path ---------------------------- #
    args.data_name = 'rotate_rocket'#rotate_rocket
    args.save_path = 'artifacts/avgpool_cross'

    # ----------------------- Data Paramters ----------------------- #
    args.batch_size = 4
    args.scale = 3
    args.height = 480
    args.width = 640
    args.train_scene_limit = 2000
    args.test_scene_limit = 500

    # ---------------------- Train Parameters ---------------------- #
    args.n_epoch = 5000
    args.train_epoch = 20000
    args.seed = 700
    #args.lr = 0.000159
    args.lr = 1e-4
    args.weight_decay = 1e-3
    args.clip_max_norm = 0.0
    args.dist_thresh = 5
    args.loss_weights = [1., 1.]
    args.log_interval = 1000
    args.save_interval = 20
    args.distributed = False

    # ----------------------- Model Patameters ---------------------- #
    args.d_coarse_model = 256
    args.d_fine_model = 128
    args.n_coarse_layers = 4
    args.n_fine_layers = 1
    args.n_heads = 8
    args.backbone_name = 'resnet101'
    args.matching_name = 'dual_softmax'
    args.match_threshold = 0.2
    args.window_size = 5
    args.border = 1
    args.load = "/four_disk/wxn_work/Train_New_Model/train_window_topk/artifacts/norm/resnet101-dual_softmax_dim256-128_depth256-128/100_model_RGBD_normalize_best_5033.9_92.3.pth"
    args.load = None
    args.sinkhorn_iterations = 100


    # --------------- Distributed Training Parameters --------------- #
    args.world_size = 2
    args.dist_url = 'tcp://192.168.1.218:12345'

    return args
