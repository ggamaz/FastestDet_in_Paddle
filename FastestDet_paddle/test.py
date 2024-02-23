import os
import cv2
import time
import argparse
import paddle
from paddle.vision.transforms import Resize
from paddle.vision.transforms import Transpose
from paddle.vision.transforms import Normalize

from utils.tool import *
from module.detector import Detector
#python3 test.py --yaml configs/coco.yaml --weight checkpoint/weight_AP05\:0.071551_170-epoch.pdparams --img /home/galaxy/MyWorks/python_expriments/experiment_5/val/img_06_4404612700_01306.jpg
if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="", help='.yaml config')
    parser.add_argument('--weight', type=str, default=None, help='.weight config')
    parser.add_argument('--img', type=str, default='', help='The path of test image')
    parser.add_argument('--thresh', type=float, default=0.65, help='The path of test image')
    parser.add_argument('--onnx', action="store_true", default=False, help='Export onnx file')
    parser.add_argument('--torchscript', action="store_true", default=False, help='Export torchscript file')
    parser.add_argument('--cpu', action="store_true", default=False, help='Run on cpu')

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"
    assert os.path.exists(opt.weight), "请指定正确的模型路径"
    assert os.path.exists(opt.img), "请指定正确的测试图像路径"

    # 选择推理后端
    if opt.cpu:
        print("run on cpu...")
        paddle.set_device("cpu")
    else:
        if paddle.is_compiled_with_cuda():
            print("run on gpu...")
            paddle.set_device("gpu")
        else:
            print("run on cpu...")
            paddle.set_device("cpu")     

    # 解析yaml配置文件
    cfg = LoadYaml(opt.yaml)    
    print(cfg) 

    # 模型加载
    print("load weight from:%s"%opt.weight)
    model = Detector(cfg.category_num, True)
    model.set_state_dict(paddle.load(opt.weight))
    # sets the module in eval mode
    model.eval()
    
    # 数据预处理
    # transform = paddle.vision.transforms.Compose([
    #     Resize((cfg.input_height, cfg.input_width)),
    #     Transpose((2, 0, 1)),
    #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    ori_img = cv2.imread(opt.img)
    res_img = cv2.resize(ori_img, (cfg.input_width, cfg.input_height), interpolation = cv2.INTER_LINEAR)
    img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)
    img = paddle.to_tensor(img.transpose(0, 3, 1, 2), )
    img = img.astype('float32') / 255.0
    
    
    # 导出onnx模型
    if opt.onnx:
        paddle.onnx.export(model,                     # model being run
                          img,                       # model input (or a tuple for multiple inputs)
                          "./FastestDet.onnx",       # where to save the model (can be a file or file-like object)
                          export_params=True,        # store the trained parameter weights inside the model file
                          opset_version=11,          # the ONNX version to export the model to
                          do_constant_folding=True)  # whether to execute constant folding for optimization
        print("onnx export success...")                 

    # 导出torchscript模型
    if opt.torchscript:
        paddle.jit.save(model, "./FastestDet.pt")
        print("to convert torchscript to pnnx/ncnn: ./pnnx FastestDet.pt inputshape=[1,3,%d,%d]" % (cfg.input_height, cfg.input_height))

    # 模型推理
    start = time.perf_counter()
    preds = model(img)
    end = time.perf_counter()
    inference_time = (end - start) * 1000.
    print("forward time:%fms"%inference_time)
    print(preds.shape)
    # 特征图后处理
    output = handle_preds(preds, paddle.get_device(), opt.thresh)
    print(np.array(output).shape)
    # 加载label names
    LABEL_NAMES = []
    with open(cfg.names, 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())
    
    H, W, _ = ori_img.shape
    scale_h, scale_w = H / cfg.input_height, W / cfg.input_width
    # 绘制预测框
    for box in output[0]:
        print(box)
        box = box.tolist()
       
        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * W), int(box[1] * H)
        x2, y2 = int(box[2] * W), int(box[3] * H)

        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
        cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
    cv2.imwrite("result.png", ori_img)
