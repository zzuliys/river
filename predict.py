import os
import cv2
import torch
import numpy as np
from model import UNet
from config import Config

def load_model(model_path):
    model = UNet(in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    img = img / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img.unsqueeze(0)
    return img

def predict_image(model, image_path):
    img = preprocess_image(image_path)
    img = img.to(Config.DEVICE)
    
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
    
    model = load_model(model_path)
    print(f"模型加载成功: {model_path}")
    
    test_dir = os.path.join(Config.BASE_DIR, "test", "images")
    if not os.path.exists(test_dir):
        print(f"测试目录不存在: {test_dir}")
        return
    
    save_dir = os.path.join(Config.RESULT_DIR, "predictions")
    os.makedirs(save_dir, exist_ok=True)
    
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))])
    
    print(f"开始推理，共 {len(test_files)} 张图像...")
    
    for i, filename in enumerate(test_files):
        image_path = os.path.join(test_dir, filename)
        pred_mask = predict_image(model, image_path)
        
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