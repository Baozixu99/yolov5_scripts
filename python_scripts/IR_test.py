import torch
import torchvision
from openvino import Core, Type, Tensor
from openvino.runtime import PartialShape, Layout
from openvino.preprocess import PrePostProcessor, ColorFormat
import cv2
import numpy as np
import time
import os
import statistics  # 统计模块
import socket
# 和 issros 交互模块
import threading
import queue
import json
import hashlib
COLORS = [
    [0, 255, 0],    # 绿 (Green)
    [0, 255, 255],   # 黄 (Yellow)
    [0, 0, 255],    # 红 (Red)
    [255, 0, 0],    # 蓝 (Blue)
    [255, 0, 255],  # 品红 (Magenta)
]

# =================================================================
# 威胁权重配置：ID与权重的映射
THREAT_CONFIG = {
    1: 50.0,  # 火炮 (Artillery)：极高威胁
    2: 20.0,  # 坦克 (Tank)：高威胁
    3: 5.0,   # 军用汽车 (Military Car)：中等威胁
    0: 1.0,   # 士兵 (Soldier)：低威胁
}

# =================================================================
# 新增：单体目标威胁等级标签 (用于显示在检测框上)
# =================================================================
OBJECT_LEVEL_LABELS = {
    0: "LOW",       # 士兵 -> 低威胁
    1: "EXTREME",   # 火炮 -> 极高威胁
    2: "HIGH",      # 坦克 -> 高威胁
    3: "MEDIUM"     # 军车 -> 中威胁
}


def log_task_result(img_name, threat_level, threat_score, detections):
    """
    生成符合测试用例三要求的精简数据包，并实时写入本地文件
    """
    # 1. 构建精简的目标列表 (只保留核心字段)
    # 0:士兵, 1:火炮, 2:坦克, 3:军车
    NAME_MAP = {0: "Soldier", 1: "Artillery", 2: "Tank", 3: "Military Car"}
    
    clean_targets = []
    for det in detections:
        # det: [x1, y1, x2, y2, conf, cls_id]
        cls_id = int(det[5])
        clean_targets.append({
            "id": cls_id,
            "obj": NAME_MAP.get(cls_id, "Unknown"),
            "box": [int(det[0]), int(det[1]), int(det[2]), int(det[3])], # 坐标取整，减少字节
            # "conf": round(float(det[4]), 2) # 如果seL4不需要置信度，这行可以注释掉以进一步精简
        })

    # 2. 构建核心数据包 (Payload)
    # 这是需要传输给 seL4 的原始内容
    payload = {
        "ts": f"{time.time():.3f}",       # 时间戳 (Timestamp)
        "img": img_name,                  # 帧ID (Frame ID)
        "lvl": threat_level,              # 威胁等级 (Threat Level)
        "scr": round(threat_score, 1),    # 威胁分数 (Score)
        "cnt": len(clean_targets),        # 目标数量 (Count)
        "tgt": clean_targets              # 目标详情 (Targets)
    }

    # 3. 序列化与完整性校验
    # separators=(',', ':') 去除空格，使 JSON 最紧凑
    json_str = json.dumps(payload, separators=(',', ':'))
    
    # 计算校验和 (Simulate integrity check for Test Case 3)
    checksum = hashlib.md5(json_str.encode()).hexdigest()[:8]
    
    # 4. 构造最终日志行 (带校验头)
    # 格式: [CHECKSUM] JSON_STRING
    log_line = f"[{checksum}] {json_str}"

    # 5. 实时写入文件 (Append mode + Flush)
    log_file = "/home/mission_output.log"
    try:
        with open(log_file, "a") as f:
            f.write(log_line + "\n")
            # with 语句退出时会自动 flush，确保断电数据不丢
    except Exception as e:
        print(f"日志写入失败: {e}")

    return log_line, len(json_str)

def get_threat_level(detections, img_w, img_h):
    """
    计算威胁等级
    :param detections: list of [x1, y1, x2, y2, conf, cls_id]
    """
    total_score = 0.0
    if not detections:
        return 0.0, "SECURE", (0, 255, 0) # 绿色安全

    img_area = img_w * img_h

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        
        # 1. 获取基础分
        base_score = THREAT_CONFIG.get(cls_id, 0.5) 
        
        # 2. 计算距离/大小系数 (框越大，威胁越大)
        box_area = (x2 - x1) * (y2 - y1)
        size_ratio = box_area / img_area
        size_factor = 1.0 + (size_ratio * 2.0)
        
        total_score += base_score * size_factor

    # 3. 等级评定
    if total_score == 0:
        return 0.0, "SECURE", (0, 255, 0)      # 绿色
    elif total_score < 5.0:
        return total_score, "LOW", (0, 255, 255)       # 黄色
    elif total_score < 20.0:
        return total_score, "CAUTION", (0, 165, 255)   # 橙色
    elif total_score < 50.0:
        return total_score, "DANGER", (0, 0, 255)      # 红色
    else:
        return total_score, "CRITICAL", (0, 0, 128)    # 深红/紫色

def load_labels(label_path):
    """读取标签文件并返回类别ID集合"""
    true_classes = set()
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) > 0:  # 处理空行
                    class_id = int(parts[0])
                    true_classes.add(class_id)
    return true_classes

def calculate_accuracy(pred_classes, true_classes):
    """计算单张图片的预测准确率"""
    if not true_classes:  # 无标签情况处理
        return 0.0 if pred_classes else 1.0
    
    if not pred_classes:  # 无预测情况处理
        return 0.0
    
    # 计算匹配的类别数
    correct = len(pred_classes & true_classes)
    # 计算准确率（预测类别与真实类别交集的比例）
    return correct / len(true_classes)

def Inference(compiled_model, image_file):
    start_time = time.perf_counter()

    # 1. 安全检查与读取
    if not os.path.exists(image_file):
        print(f"[Error] File does not exist: {image_file}")
        return None, 0, None

    img0 = cv2.imread(image_file)
    if img0 is None:
        print(f"[Error] Failed to read image with cv2: {image_file}")
        return None, 0, None

    # print("Original Image Size:", img0.shape)
    # img_name = os.path.basename(image_file)

    # 2. 获取模型期望的输入尺寸 (N, C, H, W)
    # 因为我们在 main 里已经 reshape 过了，或者是静态模型，这里直接读 shape
    input_layer = compiled_model.input(0)
    input_shape = input_layer.shape
    
    # 解析 NCHW。通常 shape 为 [1, 3, 640, 640]
    # 如果 shape 获取失败或维度不对，默认使用 640
    if len(input_shape) == 4:
        target_h, target_w = input_shape[2], input_shape[3]
    else:
        target_h, target_w = 640, 640

    # 3. 手动预处理 (替代导致崩溃的 PPP)
    # [Resize] 调整大小
    img = cv2.resize(img0, (target_w, target_h))
    
    # [Color] BGR -> RGB (OpenCV 默认 BGR，模型通常要 RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # [Normalize] 归一化 0-255 -> 0.0-1.0 & 转 float32
    img = img.astype(np.float32) / 255.0

    # [Layout] HWC -> CHW (关键步骤：OpenCV 是 HWC，OpenVINO 输入需要 CHW)
    # 原始: [640, 640, 3] -> 目标: [3, 640, 640]
    img = img.transpose((2, 0, 1))

    # [Batch] 增加 Batch 维度 -> [1, 3, 640, 640]
    img = np.expand_dims(img, axis=0)

    # [Memory] 确保内存连续 (非常重要！防止底层 C++ 访问越界导致段错误)
    img = np.ascontiguousarray(img)

    # 4. 封装 Tensor 并推理
    input_tensor = Tensor(img)

    # 创建推理请求 (注意：频繁创建可能会微损耗性能，若追求极致可移到函数外复用)
    infer_request = compiled_model.create_infer_request()
    infer_request.infer(inputs={0: input_tensor})

    # 5. 获取输出
    output_tensor = infer_request.get_output_tensor()

    # 计算耗时（毫秒）
    inference_time = (time.perf_counter() - start_time) * 1000

    # 返回 PyTorch Tensor 以兼容后续的 non_max_suppression 代码
    return torch.from_numpy(output_tensor.data), inference_time, img0

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        print("---x shape---:", x.shape)
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

#Aqua
def send_image_to_server(image_path, server_ip, server_port):
    """
    将标记好的图片发送到指定的服务器

    Args:
        image_path (str): 标记好的图片的保存路径
        server_ip (str): 服务器的IP地址
        server_port (int): 服务器的端口号
    """
    print("in send_image_to_server###########")
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"图片文件不存在: {image_path}")
        return

    # 读取图片文件为字节流
    with open(image_path, 'rb') as f:
        image_data = f.read()
        print("###succeed read image###")

    # 创建一个 TCP Socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # 连接到服务器
            s.connect((server_ip, server_port))
            print(f"已连接到服务器: {server_ip}:{server_port}")

            # 发送图片数据长度
            image_len = len(image_data).to_bytes(4, byteorder='big')
            s.sendall(image_len)

            # 发送图片数据
            s.sendall(image_data)
            print(f"图片已发送: {image_path}")

        except Exception as e:
            print(f"发送图片时出错: {e}")

# ==========================================
# 修改部分 2：D2L 风格绘图函数 (细线 + 小字 + 实心背景)
# ==========================================
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    优化的绘图函数，模仿 D2L 风格
    """
    # 1. 强制使用细线和小字体
    tl = 2  # 线宽固定为 2 (细线)
    font_scale = 0.5  # 字体大小固定为 0.5 (小字)
    font_thickness = 1  # 字体粗细固定为 1
    
    # 2. 获取坐标整数值
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    
    # 3. 绘制矩形框
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    # 4. 绘制标签
    if label:
        # 计算文本尺寸
        t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=font_thickness)[0]
        padding = 2 # 文字背景的留白
        
        # 5. 确定文字位置 (优先画在左上角外部，如果溢出则画在内部)
        if c1[1] - t_size[1] - (padding*2) < 0: 
             # 画在框内
             text_origin = (c1[0] + padding, c1[1] + t_size[1] + padding)
             # 背景矩形坐标
             rect_c1 = (c1[0], c1[1])
             rect_c2 = (c1[0] + t_size[0] + padding*2, c1[1] + t_size[1] + padding*2)
        else:
             # 画在框外
             text_origin = (c1[0] + padding, c1[1] - padding)
             # 背景矩形坐标
             rect_c1 = (c1[0], c1[1] - t_size[1] - padding*2)
             rect_c2 = (c1[0] + t_size[0] + padding*2, c1[1])
             
        # 绘制与边框同色的实心背景
        cv2.rectangle(img, rect_c1, rect_c2, color, -1, cv2.LINE_AA)
        
        # 6. 绘制白色文字
        cv2.putText(img, label, text_origin, 0, font_scale, [255, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

# 调用推理模型的函数
def start_inference(compiled_model, img_dir):
    IMAGE_DIR = img_dir
    # 初始化统计量
    total_images = 0
    total_correct = 0
    #total_predictions = 0

    # 时间统计量
    time_records = []
    total_start = time.perf_counter()

    # print(f"Processing: {IMAGE_DIR}")

    # 1. 执行推理
    prediction, elapsed_time, img0 = Inference(compiled_model, IMAGE_DIR)
    
    if prediction is None:
        print("推理失败，跳过")
        return

    time_records.append(elapsed_time)

    # 2. NMS 后处理
    ans = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300)
    
    # 3. 准备尺寸参数
    input_layer = compiled_model.input(0)
    input_shape = input_layer.shape
    if len(input_shape) == 4:
        input_h = input_shape[2] 
        input_w = input_shape[3] 
    else:
        input_h, input_w = 640, 640

    height, width = img0.shape[:2]
    scale_x = width / input_w
    scale_y = height / input_h

    # 4. 解析检测框并存储
    valid_detections = [] 
    pred_classes = set() 

    # 定义名称映射 (用于显示在框上和日志中)
    NAME_MAP = {
        0: "Soldier",       # 士兵
        1: "Artillery",     # 火炮
        2: "Tank",          # 坦克
        3: "Military Car"   # 军用汽车
    }

    if len(ans[0]) > 0:
        for det in ans[0]:
            xyxy = det[:4]
            conf = float(det[4])
            cls_id = int(det[5])
            
            # 还原坐标
            x1 = int(xyxy[0] * scale_x)
            y1 = int(xyxy[1] * scale_y)
            x2 = int(xyxy[2] * scale_x)
            y2 = int(xyxy[3] * scale_y)
            
            # 边界裁剪
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            if x2 <= x1 or y2 <= y1: continue

            # 存入列表用于威胁计算
            valid_detections.append([x1, y1, x2, y2, conf, cls_id])
            pred_classes.add(cls_id)

            # -----------------------------------------------------------
            # 核心修改：绘制目标框 (格式：LEVEL: Name)
            # -----------------------------------------------------------
            # 1. 获取颜色
            color_idx = cls_id % len(COLORS)
            color = COLORS[color_idx] 
            
            # 2. 获取威胁等级文本 (从全局配置 OBJECT_LEVEL_LABELS 获取)
            # 如果 ID 不在配置中，默认显示 ID 数字
            level_text = OBJECT_LEVEL_LABELS.get(cls_id, str(cls_id))
            
            # 3. 获取目标名称
            class_name = NAME_MAP.get(cls_id, "Unknown")

            # 4. 组合标签: "HIGH: Tank"
            label = f"{level_text}: {class_name}"
            
            plot_one_box([x1, y1, x2, y2], img0, color=color, label=label, line_thickness=2)
            # -----------------------------------------------------------

    # 5. 计算整图威胁等级
    threat_score, threat_level, threat_color = get_threat_level(valid_detections, width, height)

    # 6. 绘制 HUD (左上角仪表盘)
    overlay = img0.copy()
    cv2.rectangle(overlay, (0, 0), (380, 60), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img0, 1 - alpha, 0, img0)
    
    info_text = f"LEVEL: {threat_level}  SCORE: {int(threat_score)}"
    cv2.putText(img0, info_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, threat_color, 2, cv2.LINE_AA)

    # 高威胁警示框 (红框)
    if threat_score >= 20.0:
        cv2.rectangle(img0, (0, 0), (width-1, height-1), threat_color, 10)

    # 7. 保存与回传
    output_dir = os.path.join(os.path.dirname(IMAGE_DIR), "detections")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(IMAGE_DIR))
    cv2.imwrite(output_path, img0)
    # print(f"Saved detection image to: {output_path}")

    server_ip = "10.70.123.17"
    server_port = 9996
    send_image_to_server(output_path, server_ip, server_port)

    # =================================================================
    # 8. [新增] 生成测试用例数据包 (写入 mission_output.log)
    # =================================================================
    log_line = log_task_result(
        os.path.basename(IMAGE_DIR), 
        threat_level, 
        threat_score, 
        valid_detections
    )

    # 9. 打印终端日志
    counts = {}
    for det in valid_detections:
        cid = int(det[5])
        c_name = NAME_MAP.get(cid, str(cid)) 
        counts[c_name] = counts.get(c_name, 0) + 1
    
    if counts:
        target_str = ", ".join([f"{name} x{num}" for name, num in counts.items()])
    else:
        target_str = "None"

    print("=" * 60)
    print(f" [Threat Report] {os.path.basename(IMAGE_DIR)}")
    print(f" > Level   : {threat_level} (Score: {threat_score:.1f})")
    print(f" > Targets : {target_str}")
    print(f" > Latency : {elapsed_time:.2f} ms")
    print(f" > Data Log: {log_line[:50]}... (Saved)") # 只打印前50字符提示
    print("=" * 60)
        # 读取标签
        #base_name = os.path.splitext(pic)[0]
        #label_path = os.path.join(LABELS_DIR, f"{base_name}.txt")
        #true_classes = load_labels(label_path)

        # 统计准确率
        #accuracy = calculate_accuracy(pred_classes, true_classes)
        #total_images += 1
        #total_predictions += len(pred_classes)
        #total_correct += len(pred_classes) * accuracy

        # 打印单张图片推理准确结果和推理耗时
    # print(f"{pic}")
    print(f"  Inference Time: {elapsed_time:.2f}ms")
    print(f"  Predicted: {pred_classes}")
        #print(f"  Predicted: {pred_classes} | True: {true_classes}")
        #print(f"  Image Accuracy: {accuracy:.2%}")

    # 总耗时统计
    total_time = (time.perf_counter() - total_start) * 1000  # 转换为毫秒

    # 输出时间统计报告
    print("\nTime Statistics:")
    print(f"Total Images Processed: {len(time_records)}")
    print(f"Total Inference Time: {total_time:.2f}ms")
    print(f"Average Time per Image: {statistics.mean(time_records):.2f}ms")
    print(f"Fastest Inference: {min(time_records):.2f}ms")
    print(f"Slowest Inference: {max(time_records):.2f}ms")
    # print(f"Time Standard Deviation: {statistics.stdev(time_records):.2f}ms")

    # 输出总体统计
    #print("\nFinal Statistics:")
    #print(f"Total Images Processed: {total_images}")
    #print(f"Total Predictions Made: {total_predictions}")

    #if total_predictions > 0:
    #    overall_accuracy = total_correct / total_predictions
    #    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    #else:
    #    print("No valid predictions made for accuracy calculation")

    #if total_images > 0:
    #    detection_rate = total_predictions / total_images
    #    print(f"Detection Rate: {detection_rate:.2f} predictions per image")

# 多线程的推理函数
def inference_worker(q, shutdown_event, compiled_model):
    while not shutdown_event.is_set() or not q.empty():
        try:
            img_name=q.get(False)
            # 路径格式为：绝对路径前缀 + 设备名（UAV1/3）+ 图像类型（0/1）+ 调度阶段（stage_0~stage_4）
            # IMAGE_DIR = 'D:\Desktop\jupyterWorkspace_BAC\is2ros_yolov5_inference\labels_and_dataset\dataset\my_data\\' + img_name
            IMAGE_DIR = '/home/jupyterWorkspace/is2ros_yolov5_inference_zc/labels_and_dataset/dataset/my_data/'+ img_name

            print(IMAGE_DIR)
            #print("Yolov5 收到 issros 能力匹配结果，开始推理检测对应传感器采集的图片......")
            # print(IMAGE_DIR)
            # TODO 调度推理程序
            start_inference(compiled_model, IMAGE_DIR)
            # 当前消息处理完毕
            q.task_done()
        except queue.Empty:
            continue
def image_server_thread(port, save_dir, uav_name, q):
    os.makedirs(save_dir, exist_ok=True)
    counter = 1  # 每个 UAV 独立计数

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # ==================== 接收端性能优化 ====================
        # 允许端口复用
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 禁用延迟发送（Nagle算法）
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # 增大接收缓冲区到 2MB 
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        except Exception:
            pass
        # ======================================================

        s.bind(('0.0.0.0', port))
        s.listen(100)
        print(f"{uav_name} 接收线程已启动，监听端口 {port}")

        while counter < 1999: # 稍微改大一点循环上限
            try:
                conn, addr = s.accept()
                # 增大超时时间，大图传输容错
                conn.settimeout(15) 
                
                # 接收图像大小（4 字节）
                img_len_bytes = b''
                while len(img_len_bytes) < 4:
                    chunk = conn.recv(4 - len(img_len_bytes))
                    if not chunk: break
                    img_len_bytes += chunk
                
                if len(img_len_bytes) < 4:
                    conn.close()
                    continue
                    
                img_len = int.from_bytes(img_len_bytes, byteorder='big')
                # 简化日志，只打印一次
                # print(f"[调试]: {uav_name} 期望接收: {img_len} 字节")

                # 接收图像内容
                received = b''
                total_received = 0
                
                # 【关键优化】使用 memoryview 预分配内存，比 += 字符串拼接快得多
                # 这里为了兼容性保持 bytes +=，但增大了 chunk_size
                while total_received < img_len:
                    remaining = img_len - total_received
                    # 【关键优化】一次读 64KB 而不是 8KB，减少系统调用次数，提速 8 倍
                    chunk_size = min(65536, remaining) 
                    
                    try:
                        data = conn.recv(chunk_size)
                        if not data: break
                        received += data
                        total_received += len(data)
                    except socket.timeout:
                        print(f"[错误] {uav_name} 接收超时")
                        break
                    except Exception as e:
                        print(f"[错误] {uav_name} 接收异常: {e}")
                        break

                # 验证接收完整性
                if total_received != img_len:
                    print(f"[丢包警告] {uav_name} 仅接收 {total_received}/{img_len} (丢失 {img_len-total_received})")
                    conn.close()
                    continue
                
                print(f"[成功] {uav_name} 接收图片_{counter} (大小: {img_len/1024:.1f}KB)")

                # 保存文件
                filename = f"{save_dir}/{uav_name}_{counter}.png"
                # 直接写入，不重复打开
                with open(filename, 'wb') as f:
                    f.write(received)
                    
                # 加入推理队列
                img_name = f"{uav_name}_{counter}.png"
                q.put(img_name)
                
                counter += 1
                conn.close()
                
            except Exception as e:
                print(f"[异常] {uav_name} 连接错误: {e}")
                continue
# ================== 新增：握手服务线程 ==================
def handshake_server():
    HANDSHAKE_MSG = "011111"
    srt = "3040"
    server_ip = "10.70.123.17"
    server_port = 65432
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
        conn.connect((server_ip, server_port))
        print(f"已连接到服务器: {server_ip}:{server_port}")

        while True:
        
            try:
                # 接收客户端发送的握手消息
                received = conn.recv(1024).decode("utf-8")
                print(f"接受到server消息为{received}")
                conn.sendall(srt.encode("utf-8"))
            except Exception as e:
                print(" 出现异常:", e)
            #finally:
                #conn.close()

# ------------------------- 主程序重构 -------------------------

if __name__ == "__main__":

    # uav_configs = [
    #     (8891, "D:\Desktop\jupyterWorkspace_BAC\is2ros_yolov5_inference\labels_and_dataset\dataset\my_data", "uav1"),
    #     (8892, "D:\Desktop\jupyterWorkspace_BAC\is2ros_yolov5_inference\labels_and_dataset\dataset\my_data", "uav2"),
    #     (8893, "D:\Desktop\jupyterWorkspace_BAC\is2ros_yolov5_inference\labels_and_dataset\dataset\my_data", "uav3"),
    # ]
    # # uav_configs = [
    # #     (8891, "/home/bousew/jupyterWorkspace/is2ros_yolov5_inference/labels_and_dataset/dataset/my_data/", "uav1"),
    # #     (8892, "/home/bousew/jupyterWorkspace/is2ros_yolov5_inference/labels_and_dataset/dataset/my_data/", "uav2"),
    # #     (8893, "/home/bousew/jupyterWorkspace/is2ros_yolov5_inference/labels_and_dataset/dataset/my_data/", "uav3"),
    # # ]
    # q = queue.Queue()
    # threads = []
    # for port, save_dir, uav_name in uav_configs:
    #     t = threading.Thread(target=image_server_thread, args=(port, save_dir, uav_name, q))
    #     t.daemon = False
    #     t.start()
    #     threads.append(t)
    # print("服务器已启动，等待 UAV 图像传输...")
    #
    # for t in threads:
    #     t.join()

    # 初始化模型配置参数
    # 2025推荐使用统一设备API，支持"AUTO", "GPU", "MYRIAD"等
    #DEVICE = "GPU"
    DEVICE = "CPU"

    # 模型路径
    # MODEL_XML = 'D:\Desktop\jupyterWorkspace_BAC\is2ros_yolov5_inference\yolov5_scripts\IR_models\IR_model_with_ovc\\best.xml'
    # MODEL_BIN = 'D:\Desktop\jupyterWorkspace_BAC\is2ros_yolov5_inference\yolov5_scripts\IR_models\IR_model_with_ovc\\best.bin'
    MODEL_XML = '/home/jupyterWorkspace/is2ros_yolov5_inference_zc/yolov5_scripts/IR_models/IR_model_with_ovc/best.xml'
    MODEL_BIN = '/home/jupyterWorkspace/is2ros_yolov5_inference_zc/yolov5_scripts/IR_models/IR_model_with_ovc/best.bin'


    # 初始化OpenVINO Core
    core = Core()
    # 禁用缓存，避免文件锁问题
    core.set_property({'CACHE_DIR': ''})
    # 加载模型（自动处理xml+bin）
    print("Loading model...")
    model = core.read_model(MODEL_XML)
    print("***debug1")
    # 动态输入配置（示例）
    if model.inputs[0].partial_shape.is_dynamic:
        model.reshape({0: PartialShape([1,3,640,640])})
    print("***debug2")
    # 预处理配置（可选）
    # ppp = PrePostProcessor(model)
    # print("***debug3")
    # ppp.input().tensor() \
    #     .set_layout(Layout("NHWC")) \
    #     .set_color_format(ColorFormat.RGB) \
    #     .set_element_type(Type.f32)
    #  # 模型内部布局（如需要可设为 NCHW）
    # print("***debug4")
    # ppp.input().preprocess() \
    #     .convert_layout("NCHW")
    # print("***debug5")
    # model = ppp.build()
    # print("***debug6")
    #
    # # 编译模型（自动设备发现）
    compiled_model = core.compile_model(model, DEVICE)
    print("***debug7")
    #
    # 打印输入输出信息
    print(f"Input shape: {compiled_model.input(0).shape}")
    print(f"Output shape: {compiled_model.output(0).shape}")

    # uav_configs = [
    #     (8891, "D:\Desktop\jupyterWorkspace_BAC\is2ros_yolov5_inference\labels_and_dataset\dataset\my_data", "uav1"),
    #     (8892, "D:\Desktop\jupyterWorkspace_BAC\is2ros_yolov5_inference\labels_and_dataset\dataset\my_data", "uav2"),
    #     (8893, "D:\Desktop\jupyterWorkspace_BAC\is2ros_yolov5_inference\labels_and_dataset\dataset\my_data", "uav3"),
    # ]

    uav_configs = [
        # (8891, "/home/jupyterWorkspace/is2ros_yolov5_inference_zc/labels_and_dataset/dataset/my_data/", "uav1"),
        (8892, "/home/jupyterWorkspace/is2ros_yolov5_inference_zc/labels_and_dataset/dataset/my_data/", "uav2"),
        # (8893, "/home/jupyterWorkspace/is2ros_yolov5_inference_zc/labels_and_dataset/dataset/my_data/", "uav3"),
    ]

    q = queue.Queue()
    threads = []
    print("***debug8")
    # ====启动握手服务线程 ====
    #hs_thread = threading.Thread(target=handshake_server)
    #hs_thread.daemon = True
    #hs_thread.start()
    
    for port, save_dir, uav_name in uav_configs:
        t = threading.Thread(target=image_server_thread, args=(port, save_dir, uav_name, q))
        t.daemon = False
        t.start()
        threads.append(t)
        print("***debug9")
    print("服务器已启动，等待 UAV 图像传输...")

    shutdown_event = threading.Event()

    # 启动工作线程
    inference_worker_thread = threading.Thread(target = inference_worker,
                                                args = (q, shutdown_event, compiled_model))
    inference_worker_thread.start()
    print("执行推理线程")

    for t1 in threads:
        t1.join()
    # 向工作线程发送结束信号
    shutdown_event.set()
    # 等待工作线程结束
    inference_worker_thread.join()
    # 等待队列处理完毕
    q.join()
    print("Inference has done")



