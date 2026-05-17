import os
import cv2
import torch
import numpy as np
from model import UNet
from config import Config

def load_model(model_path, use_channels_last=False):
    model = UNet(in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=Config.DEVICE)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(Config.DEVICE)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model

def preprocess_image(image_path):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if Config.USE_PREPROC:
        meta = torch.load(os.path.join(Config.PREPROC_DIR, "meta.pt"), map_location="cpu")
        mean = meta["mean"]
        std = meta["std"]

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    img = img / 255.0
    img = (img - mean) / std
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img.unsqueeze(0)
    return img

def predict_image(model, image_path, non_blocking=False, use_channels_last=False):
    img = preprocess_image(image_path)
    if use_channels_last:
        img = img.to(Config.DEVICE, memory_format=torch.channels_last, non_blocking=non_blocking)
    else:
        img = img.to(Config.DEVICE, non_blocking=non_blocking)
    
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).squeeze(0)
    
    return pred.cpu().numpy()

def overlay_mask_on_image(image_path, pred_mask, alpha=0.5):
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    
    pred_mask = (pred_mask * 255).astype(np.uint8)
    
    mask_color = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
    mask_color[pred_mask > 127] = [0, 0, 255]
    
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original_img, contours, -1, (0, 255, 0), 2)
    
    overlay = cv2.addWeighted(mask_color, alpha, original_img, 1 - alpha, 0)
    
    return overlay

def save_prediction(pred_mask, save_path):
    pred_mask = (pred_mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, pred_mask)

def main():
    model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    device = torch.device(Config.DEVICE)
    use_channels_last = Config.USE_CHANNELS_LAST and device.type == 'cuda'
    non_blocking = Config.USE_NON_BLOCKING and device.type == 'cuda'
    print(f"Channels Last: {'Enabled' if use_channels_last else 'Disabled'}")
    print(f"Non-Blocking Transfer: {'Enabled' if non_blocking else 'Disabled'}")

    model = load_model(model_path, use_channels_last)
    print(f"模型加载成功: {model_path}")
    
    test_dir = os.path.join(Config.BASE_DIR, "images")
    if not os.path.exists(test_dir):
        print(f"测试目录不存在: {test_dir}")
        return
    
    save_dir = os.path.join(Config.RESULT_DIR, "predictions")
    os.makedirs(save_dir, exist_ok=True)
    
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))])
    
    print(f"开始推理，共 {len(test_files)} 张图像...")
    
    for i, filename in enumerate(test_files):
        image_path = os.path.join(test_dir, filename)
        pred_mask = predict_image(model, image_path, non_blocking, use_channels_last)
        
        overlay_img = overlay_mask_on_image(image_path, pred_mask, alpha=0.5)
        
        save_path = os.path.join(save_dir, filename.replace('.jpg', '_overlay.png').replace('.JPG', '_overlay.png'))
        cv2.imwrite(save_path, overlay_img)
        
        mask_save_path = os.path.join(save_dir, filename.replace('.jpg', '_mask.png').replace('.JPG', '_mask.png'))
        save_prediction(pred_mask, mask_save_path)
        
        if (i + 1) % 10 == 0:
            print(f"已处理: {i + 1}/{len(test_files)}")
    
    print(f"推理完成，结果保存到: {save_dir}")

if __name__ == "__main__":
    main()