from datasets.depth_dataset import build_depth
from datasets.sar_opt_dataset import build_so
from datasets.nirscene1_dataset import build_nir
from datasets.rgbd_dataset import build_rgbd
from datasets.SE_sar_opt_dataset import build_SE
from datasets.space_sar_opt_dataset import build_SP
from datasets.rotate_space_sar_opt_dataset import build_Rotate_SpaceNet
from datasets.rotate_rocket_dataset import build_Rotate_Rocket
from datasets.dataset2 import build_gl
def build_dataset(args):
    if args.data_name == 'gl':
        train_data_file="/home/ly/Documents/zkj/stage1/train/train.txt"
        test_data_file="/home/ly/Documents/zkj/stage1/train/test.txt"
        return build_gl(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'so':
        train_data_file="/home/ly/data/rockets/rockets_subject3_data/rockets/stage/train/train.txt"
        test_data_file="/home/ly/data/rockets/rockets_subject3_data/rockets/stage/train/test.txt"
        return build_so(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'rotate_rocket':
        train_data_file="/home/ly/Documents/zkj/stage1/train/train.txt"
        test_data_file="/home/ly/Documents/zkj/stage1/train/test.txt"
        return build_Rotate_Rocket(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'nir':
        train_data_file="/home/ly/Documents/zkj/dataset/nirscene1/train.txt"
        test_data_file="/home/ly/Documents/zkj/dataset/nirscene1/test.txt"
        return build_nir(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'rgbd':
        train_data_file="/home/ly/Documents/zkj/img/train.txt"
        test_data_file="/home/ly/Documents/zkj/img/val.txt"
        return build_rgbd(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'SE':
        train_data_file="/home/ly/Documents/zkj/ROIs/se_train.txt"
        test_data_file="/home/ly/Documents/zkj/ROIs/se_train.txt"
        return build_SE(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'SP':
        train_data_file="/home/ly/Documents/zkj/spacenet/se_train.txt"
        test_data_file="/home/ly/Documents/zkj/spacenet/se_test.txt"
        return build_SP(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'RSP':
        train_data_file="/home/ly/Documents/zkj/spacenet/se_train.txt"
        test_data_file="/home/ly/Documents/zkj/spacenet/se_test.txt"
        return build_Rotate_SpaceNet(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
