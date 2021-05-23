from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from Datasets.Helen import HelenDataset
from torchvision import transforms
from TrainingBackBone.transforms import Normalize
from preprocess import ToPILImage, Resize, Stage1_ToTensor
from TrainingBackBone.augmentation import Stage1Augmentation
from TrainingBackBone.args import get_args

args = get_args()


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


root_dirs = "../datas/data"
parts_root_dir = "../datas/parts"

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt",
    'test': "testing.txt"
}

transforms_list = {
    'train':
        transforms.Compose([
            ToPILImage(),
            Resize((args.size, args.size)),
            Stage1_ToTensor(),
            Normalize(mean=[0.369, 0.314, 0.282], std=[0.282, 0.251, 0.238])
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((args.size, args.size)),
            Stage1_ToTensor(),
            Normalize(mean=[0.369, 0.314, 0.282], std=[0.282, 0.251, 0.238])
        ]),
    'test':
        transforms.Compose([
            ToPILImage(),
            Resize((args.size, args.size)),
            Stage1_ToTensor(),
            Normalize(mean=[0.369, 0.314, 0.282], std=[0.282, 0.251, 0.238])
        ])
}


def get_loader(mode='train', batch_size=1, shuffle=True, num_workers=4, datamore=0, dataset=HelenDataset,
               root_dir=None, return_dataset=False):
    # DataLoader
    if root_dir is None:
        root_dir = root_dirs
    # DataLoader
    Dataset = {x: HelenDataset(txt_file=txt_file_names[x],
                               root_dir=root_dir,
                               parts_root_dir=parts_root_dir,
                               transform=transforms_list[x],
                               stage='stage1'
                               )
               for x in ['train', 'val', 'test']
               }

    if datamore == 0:
        datas_loader = DataLoaderX(Dataset[mode], batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers)
    else:
        augmentation = Stage1Augmentation(dataset=HelenDataset,
                                          txt_file=txt_file_names,
                                          root_dir=root_dir,
                                          parts_root_dir=parts_root_dir,
                                          resize=(args.size, args.size),
                                          stage='stage1'
                                          )
        enhaced_datasets = augmentation.get_dataset()
        enhaced_datasets.update({'test':
                                     Dataset['test']
                                 })
        datas_loader = DataLoaderX(enhaced_datasets[mode], batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers)

    if return_dataset:
        return Dataset[mode]
    return datas_loader
