import os
import json
import cv2
import numpy as np

def json_to_mask(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for shape in data['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)
        
        if label in ['water', 'river', 'lake', 'water_body']:
            cv2.fillPoly(mask, [points], 255)
        elif label == 'background':
            cv2.fillPoly(mask, [points], 0)
    
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    mask_path = os.path.join(output_dir, f'{base_name}.png')
    cv2.imwrite(mask_path, mask)
    
    return mask_path

def batch_convert(json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        mask_path = json_to_mask(json_path, output_dir)
        print(f"转换完成: {json_file} → {os.path.basename(mask_path)}")
    
    print(f"\n共转换 {len(json_files)} 个文件")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='将LabelMe JSON标注转换为PNG掩码')
    parser.add_argument('--input', '-i', required=True, help='JSON文件目录')
    parser.add_argument('--output', '-o', required=True, help='输出PNG目录')
    args = parser.parse_args()
    
    batch_convert(args.input, args.output)