import torch
from PIL import Image
import torchvision.transforms as T
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import argparse
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='images')
    parser.add_argument('--output_dir', type=str, default='vision_features')
    parser.add_argument('--img_type', type=str, default="vit", choices=['detr', 'vit'], help='type of image features')
    args = parser.parse_args()
    return args

def extract_features(img_type, input_image):
    if img_type == "vit":
        config = resolve_data_config({}, model=vit_model)
        transform = create_transform(**config)
        with torch.no_grad():
            img = Image.open(input_image).convert("RGB")
            input = transform(img).unsqueeze(0)
            feature = vit_model.forward_features(input)
        return feature
    
    if img_type == "detr":
        transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            img = Image.open(input_image).convert("RGB")
            input = transform(img).unsqueeze(0)
            feature = detr_model(input)[-1]
        return feature
    
    if img_type == 'clip':

        image_transform = T.Compose( [
            T.Resize( (256, 256) ),
            T.ToTensor()    ])

        image = Image.open(input_image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        itransformed_mage = image_transform(image)
        # print(itransformed_mage.shape)
        inputs = clip_processor(text="", images=itransformed_mage, padding=True, return_tensors="pt")
        outputs = clip_model(**inputs)
        img_emb = outputs.image_embeds
        # print(f'{img_emb.shape} # {outputs.text_embeds.shape}')

        # print(f'\n***')
        

        return img_emb



if __name__ == '__main__':
    args = parse_args()
    print("args",args)
    all_images = os.listdir(args.data_root)
    tmp = []
    name_map = {}
    all_images.sort(key=lambda x:int(x))
    print(len(all_images))
    
    if args.img_type == "vit":
        vit_model = timm.create_model("vit_large_patch32_384", pretrained=True, num_classes=0)
        vit_model.eval()
    
    if args.img_type == "detr":
        detr_model = torch.hub.load('cooelf/detr', 'detr_resnet101_dc5', pretrained=True)
        detr_model.eval()

    if args.img_type == 'clip':
        
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        from transformers import CLIPProcessor, CLIPModel

        clip_version = f'laion/CLIP-ViT-H-14-laion2B-s32B-b79K' # 1024'
        clip_model = CLIPModel.from_pretrained(clip_version)
        clip_processor = CLIPProcessor.from_pretrained(clip_version)
        print(f'CLIP: {clip_version} is Loaded.')

    
    for idx, image in enumerate(tqdm(all_images)):
        if idx % 100 == 0: print(idx)
        if os.path.exists(os.path.join(args.data_root, image, "image.png")):
            curr_dir = os.path.join(args.data_root, image, "image.png")
        else:
            curr_dir = os.path.join(args.data_root, image, "choice_0.png")
        feature = extract_features(args.img_type, curr_dir)
        tmp.append(feature.detach().cpu())
        name_map[str(image)] = idx
    
    res = torch.cat(tmp).cpu()
    print(res.shape)
    torch.save(res, os.path.join(args.output_dir, args.img_type +'.pth'))
    with open(os.path.join(args.output_dir, 'name_map.json'), 'w') as outfile:
        json.dump(name_map, outfile)