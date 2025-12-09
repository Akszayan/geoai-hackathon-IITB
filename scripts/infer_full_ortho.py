# scripts/infer_full_ortho.py
"""
Sliding-window inference for a single ortho TIFF.

Usage:
python -u scripts/infer_full_ortho.py \
  --ortho raw_data/orthos/PB/28996_NADALA_ORTHO/28996_NADALA_ORTHO.tif \
  --checkpoint models/deeplabv3_pb_best.pth \
  --out outputs/predictions/
"""
import argparse, os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights

def make_model(num_classes, device):
    try:
        model = deeplabv3_resnet50(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V2, num_classes=num_classes)
    except Exception:
        model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
    model.to(device).eval()
    return model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ortho", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--device", default="cuda")
    return p.parse_args()

def normalize_tile(img, mean, std):
    img = img.astype('float32') / 255.0
    img = (img - mean.reshape(1,1,3)) / std.reshape(1,1,3)
    img = np.transpose(img, (2,0,1))  # CHW
    return img

def main():
    args = parse_args()
    ortho_path = Path(args.ortho)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"{ortho_path.stem}_pred_mask.tif"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = make_model(args.num_classes, device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck['model'])
    model.eval()

    # mean/std from training
    mean = np.load("data/meta/mean.npy")
    std  = np.load("data/meta/std.npy")

    with rasterio.open(ortho_path) as src:
        meta = src.meta.copy()
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs
        dtype = 'uint8'

        # output file
        out_meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': dtype,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw'
        }
        # create output writer
        with rasterio.open(out_tif, 'w', **out_meta) as dst:
            tile = args.tile_size
            stride = tile - args.overlap
            # initialize a big array to write gradually? We'll write per-window to file
            # iterate windows
            for y in range(0, height, stride):
                h = tile if y + tile <= height else height - y
                for x in range(0, width, stride):
                    w = tile if x + tile <= width else width - x
                    win = Window(x, y, w, h)
                    try:
                        arr = src.read([1,2,3], window=win)  # 3 x h x w
                    except Exception as e:
                        print("read err", e)
                        continue
                    arr = np.moveaxis(arr, 0, 2)  # h,w,3
                    if arr.shape[0] != tile or arr.shape[1] != tile:
                        pad_h = tile - arr.shape[0]
                        pad_w = tile - arr.shape[1]
                        arr = np.pad(arr, ((0,pad_h),(0,pad_w),(0,0)), constant_values=0)

                    img = normalize_tile(arr, mean, std)
                    img_t = torch.from_numpy(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        out = model(img_t)['out']  # 1,C,H,W
                        pred = out.argmax(1).squeeze(0).cpu().numpy().astype('uint8')  # H,W

                    # crop back to original window
                    pred = pred[:h, :w]
                    # write to destination window
                    dst.write(pred, 1, window=win)
            print("Saved prediction:", out_tif)

if __name__ == "__main__":
    main()
