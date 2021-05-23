from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from torchvision import transforms
from TrainRefine.args import get_args
from TrainRefine.refindata import RefineDataSet
from TrainRefine.transforms import RandomHorizontalFlip, RandomAffine, GaussianNoise, Normalize, Resize, ToTensor

args = get_args()


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


root_dir = "/bigdata/yinzi/patchset/parts"

transforms_list = {
    'train':
        transforms.Compose([
            Resize((64, 64)),
            RandomHorizontalFlip(),
            RandomAffine(degrees=[-90, 90], translate=[0.1, 0.1],
                         scale=[0.8, 1.5]),
            ToTensor(),
            Normalize(mean=[0.369, 0.314, 0.282], std=[0.282, 0.251, 0.238])
        ]),
    'val':
        transforms.Compose([
            Resize((64, 64)),
            ToTensor(),
            Normalize(mean=[0.369, 0.314, 0.282], std=[0.282, 0.251, 0.238])
        ]),
    'test':
        transforms.Compose([
            Resize((64, 64)),
            ToTensor(),
            Normalize(mean=[0.369, 0.314, 0.282], std=[0.282, 0.251, 0.238])
        ])
}

if args.datamore != 1:
    transforms_list['train'] = transforms.Compose([
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=[0.369, 0.314, 0.282], std=[0.282, 0.251, 0.238])
    ])

# DataLoader

Dataset = {x: RefineDataSet(root_dir=root_dir,
                         mode=x,
                         transform=transforms_list[x]
                         )
           for x in ['train', 'val', 'test']
           }

dataloader = {}
dataloader['train'] = DataLoaderX(Dataset['train'], batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)

dataloader['val'] = DataLoaderX(Dataset['val'], batch_size=1,
                                shuffle=False, num_workers=4)
dataloader['test'] = DataLoaderX(Dataset['test'], batch_size=1,
                                 shuffle=False, num_workers=4)


def get_loader(mode='train'):
    return dataloader[mode]
