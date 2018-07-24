# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--lambda1",
                        type=int,
                        default=100)
    parser.add_argument("--batch_size",
                        type=int,
                        default=24)
    parser.add_argument("--noise_size",
                        type=int,
                        default=0)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=500)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/mnt/disk1/dat/lchen63/grid/data/pickle/")
    parser.add_argument("--model_dir",
                        type=str,
                        default="/mnt/disk1/dat/lchen63/grid/model/")
    parser.add_argument("--sample_dir",
                        type=str,
                        default="/mnt/disk1/dat/lchen63/data/grid/sample/model1/")
    parser.add_argument("--log_dir",
                        type=str,
                        default="/mnt/disk1/dat/lchen63/grid/data/log/model4/")
    parser.add_argument("--model_number",
                        type=str,
                        default=1)
    parser.add_argument('--device_ids', type=str, default='0,1,2,3')
    parser.add_argument('--dataset', type=str, default='grid')
    parser.add_argument('--num_thread', type=int, default=12)
    parser.add_argument('--flownet_pth', type=str, help='path of flownets model')
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--lr_corr', type=float, default=0.0001)
    parser.add_argument('--lr_flownet', type=float, default=1e-4)
    parser.add_argument('--fake_corr', type=bool, default=True)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()


if __name__ == "__main__":
    config = parse_args()
    config.is_train = True
    if config.model_number=='1':
        import trainer_model1 as trainer
    elif config.model_number=='2':
        import trainer_model2 as trainer
    elif config.model_number=='3':
        import trainer_model3 as trainer
    elif config.model_number=='4':
        import trainer_model4 as trainer
    elif config.model_number=='5':
        import trainer_model5 as trainer
    elif config.model_number=='vgan' :
        import trainer_model_vgan as trainer
    elif config.model_number=='vgg' :
        import trainer_model_vgg as trainer
    elif config.model_number=='embedding' :
        import trainer_embeddings as trainer
    elif config.model_number=='r_perceptual':
        import trainer_r_perceptual as trainer
    elif config.model_number == 'base':
        import trainer_base as trainer
    elif config.model_number == 'base_r':
        import trainer_base_r as trainer
    elif config.model_number == 'difference':
        import trainer_difference as trainer
    elif config.model_number == 'perceptual':
        import trainer_perceptual as trainer
    elif config.model_number == 'flownets':
        import flownet.ft_grid_flow as trainer
    elif config.model_number == 'farneback':
        import trainer_farneback as trainer
    elif config.model_number == 'flownet_real_fake':
        import trainer_flownet as trainer
    elif config.model_number == 'warp':
        import trainer_warp as trainer
    elif config.model_number == 'embedding_corr':
        import trainer_embedding_corr as trainer
    elif config.model_number == 'flownet_diff':
        import trainer_flow_diff as trainer
    elif config.model_number == 'flownet_pool':
        import trainer_flownet_pool as trainer
    elif config.model_number == 'base_c_p_d':
        import trainer_corr_perceptual_difference as trainer
    else:
        print 'wrong model number!!!!!!!!!!!!!!!!!!!!'
    main(config)
