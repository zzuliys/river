# 默认：随机抽 10 张验证集图片
python3 predict.py

# 或代码调用单张
python3 -c "
from predict import load_model, predict_single
predict_single('path/to/image.jpg')
"