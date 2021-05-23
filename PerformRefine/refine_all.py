import sys

sys.path.append("../")
from TrainingBackBone.gen_data import get_loader
import os
import timeit
import torch.nn.functional as F
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from utils.calc_funcs import get_dets, affine_crop, affine_mapback
from TrainRefine.model import RefineModel
import matplotlib.pyplot as plt
from tqdm import tqdm

test_dataloader = get_loader(mode='test')

rough_pred_path = "/data/yinzi/bd_refine/rough_preds"
refined_out_path = "/data/yinzi/bd_refine/refined_outs"

os.makedirs(rough_pred_path, exist_ok=True)
os.makedirs(refined_out_path, exist_ok=True)

step = 0
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
size_set = [8, 8, 8, 8, 16, 16]
refine_net = RefineModel().to(device)

state_dict = "checkpoints/refine/asadasd.pth.tar"
refine_net.load_state_dict(torch.load(state_dict, map_location=device))
refine_net.eval()

for batch in tqdm(test_dataloader):
    x, y = batch['image'].to(device), batch['labels'].to(device)
    name = batch['name']
    N, C, H, W = x.shape
    rough_mask = TF.to_tensor(Image.open(os.path.join(rough_pred_path, f'{name[0]}.png')))

    rough_mask_one_hot = torch.zeros((N, 9, H, W), device=device)

    for i in range(9):
        rough_mask_one_hot[:, i] = (rough_mask == i).long()
    # rough_all_mouth, _ = torch.max(rough_mask_one_hot[:, 6:9], dim=1, keepdim=True)
    # rough_mask_one_hot = torch.cat([rough_mask_one_hot[:, 1:6], rough_all_mouth], dim=1)
    final_refiend = []
    for i in range(9):
        boxes = get_dets(rough_mask_one_hot[:, i:i + 1], patch_size=size_set[i], out_fmt='cxcywh')
        # boxes_xyxy = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        image_patches, theta = affine_crop(x, boxes[0], is_label=False)
        rough_patches, _ = affine_crop(rough_mask_one_hot[:, i:i + 1], theta=theta, is_label=True)

        image_patches = F.interpolate(image_patches, size=(64, 64), align_corners=True)
        rough_patches = F.interpolate(rough_patches, size=(64, 64), mode='nearest')
        refined = refine_net(image_patches, pred=rough_patches)
        refined_mask = refined.argmax(dim=1, keepdim=True)
        refined_mask = F.interpolate(refined_mask, size=(size_set[i], size_set[i]), mode='nearest')
        final_refiend.append(affine_mapback(refined_mask, theta=theta))

    start = timeit.default_timer()
    step += 1

    end = timeit.default_timer()
    print(str(end - start))

    final_refiend = torch.cat(final_refiend, dim=1)
    final_refiend = final_refiend.argmax(final_refiend, dim=1, keepdim=False).cpu()
    plt.imshow(final_refiend[0])
    plt.pause(0.01)
    final_refiend = TF.to_pil_image(final_refiend[0])
    final_refiend.save(os.path.join(refined_out_path, f'{name[0]}.png'), format="PNG",
                       compress_level=0)
