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

    # 读取图像并预处理
    img0 = cv2.imread(image_file)
    print("Original Image Size:", img0.shape)
    img_name = os.path.basename(image_file)
    print(f"Image name: {img_name}")

    # 获取模型输入要求
    input_tensor = compiled_model.input(0)
    input_shape = input_tensor.shape
    h, w = input_shape[1], input_shape[2]

    # 预处理流程
    img = cv2.resize(img0, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0

    # 创建 OpenVINO Tensor
    input_tensor = Tensor(img)

    # 执行推理
    infer_request = compiled_model.create_infer_request()
    infer_request.infer(inputs={0: input_tensor})

    # 获取输出
    output_tensor = infer_request.get_output_tensor()

    # 计算耗时（毫秒）
    inference_time = (time.perf_counter() - start_time) * 1000

    return torch.from_numpy(output_tensor.data), inference_time, img0  # 返回原始图片

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

    # 执行推理
    print("Starting inference and accuracy calculation...")

    # pic_list = os.listdir(IMAGE_DIR)
    # for pic in pic_list:
    #     if not pic.lower().endswith(('.png', '.jpg', '.jpeg')):
    #         continue
    #
    #     # 执行推理并获取耗时
    #     image_path = os.path.join(IMAGE_DIR, pic)
    prediction, elapsed_time,img0 = Inference(compiled_model, IMAGE_DIR)
    time_records.append(elapsed_time)

        # 后处理
    print(f"Prediction shape: {prediction.shape}")
    ans = non_max_suppression(prediction,
                                conf_thres=0.25,
                                iou_thres=0.45,
                                max_det=300)
        #cls_ids = ans[0][:,-1].unique()
        #print(f"{pic} detected classes: {cls_ids}")
    pred_classes = set(ans[0][:,-1].int().tolist())


    # 获取模型输入尺寸和原始图片尺寸
    input_shape = compiled_model.input(0).shape
    input_h, input_w = input_shape[1], input_shape[2]
    height, width = img0.shape[:2]
    
    # 比例缩放坐标
    scale_x = width / input_w
    scale_y = height / input_h
    
    # 绘制检测框
    if len(ans[0]) > 0:
        for det in ans[0]:
            xyxy, conf, cls_id = det[:4], det[4], det[5]
            # 缩放坐标到原始图片尺寸
            x1 = int(xyxy[0] * scale_x)
            y1 = int(xyxy[1] * scale_y)
            x2 = int(xyxy[2] * scale_x)
            y2 = int(xyxy[3] * scale_y)
            # 裁剪坐标到图片边界
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, width), min(y2, height)
            # 绘制矩形框
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制文本
            label = f"{int(cls_id)}: {conf:.2f}"
            cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
        # 保存带有检测框的图片
        output_dir = os.path.join(os.path.dirname(IMAGE_DIR), "detections")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(IMAGE_DIR))
        cv2.imwrite(output_path, img0)
        print(f"Saved detection image to: {output_path}")

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
            IMAGE_DIR = '/home/jupyterWorkspace/is2ros_yolov5_inference/labels_and_dataset/dataset/my_data/'+ img_name

            print(IMAGE_DIR)
            #print("Yolov5 收到 issros 能力匹配结果，开始推理检测对应传感器采集的图片......")
            # print(IMAGE_DIR)
            # TODO 调度推理程序
            start_inference(compiled_model, IMAGE_DIR)
            # 当前消息处理完毕
            q.task_done()
        except queue.Empty:
            continue
def image_server_thread(port, save_dir, uav_name,q):
    os.makedirs(save_dir, exist_ok=True)
    counter = 1  # 每个 UAV 独立计数

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', port))
        s.listen(100)
        print(f"{uav_name} 接收线程已启动，监听端口 {port}")

        while counter < 20:
            conn, addr = s.accept()
            with conn:
                print(f"{uav_name} 收到连接来自: {addr}")

                # 接收图像大小（4 字节）
                img_len_bytes = conn.recv(4)
                if not img_len_bytes:
                    continue
                img_len = int.from_bytes(img_len_bytes, byteorder='big')

                # 接收图像内容
                received = b''
                while len(received) < img_len:
                    data = conn.recv(4096)
                    if not data:
                        break
                    received += data

                # 保存文件

               # filename = f"{save_dir}\\{uav_name}_{counter}.png"
                filename = f"{save_dir}/{uav_name}_{counter}.png"
                with open(filename, 'wb') as f:
                    f.write(received)
                print(f"已保存: {filename}")

                img_name = f"{uav_name}_{counter}.png"
                q.put(img_name)
                t_id = threading.current_thread()
                print(f"now queue size:  {q.qsize()} and thread_id: {t_id}")
                counter += 1
                print(f"{uav_name} 当前计数器: {counter}")

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
    DEVICE = "AUTO"

    # 模型路径
    # MODEL_XML = 'D:\Desktop\jupyterWorkspace_BAC\is2ros_yolov5_inference\yolov5_scripts\IR_models\IR_model_with_ovc\\best.xml'
    # MODEL_BIN = 'D:\Desktop\jupyterWorkspace_BAC\is2ros_yolov5_inference\yolov5_scripts\IR_models\IR_model_with_ovc\\best.bin'
    MODEL_XML = '/home/jupyterWorkspace/is2ros_yolov5_inference/yolov5_scripts/IR_models/IR_model_with_ovc/best.xml'
    MODEL_BIN = '/home/jupyterWorkspace/is2ros_yolov5_inference/yolov5_scripts/IR_models/IR_model_with_ovc/best.bin'


    # 初始化OpenVINO Core
    core = Core()

    # 加载模型（自动处理xml+bin）
    print("Loading model...")
    model = core.read_model(MODEL_XML)

    # 动态输入配置（示例）
    if model.inputs[0].partial_shape.is_dynamic:
        model.reshape({0: PartialShape([1,3,640,640])})

    # 预处理配置（可选）
    ppp = PrePostProcessor(model)
    ppp.input().tensor() \
        .set_layout(Layout("NHWC")) \
        .set_color_format(ColorFormat.RGB) \
        .set_element_type(Type.f32)
     # 模型内部布局（如需要可设为 NCHW）
    ppp.input().preprocess() \
        .convert_layout("NCHW") \
        .mean([0.485, 0.456, 0.406]) \
        .scale([0.229, 0.224, 0.225])
    model = ppp.build()
    #
    # # 编译模型（自动设备发现）
    compiled_model = core.compile_model(model, DEVICE)
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
        (8091, "/home/jupyterWorkspace/is2ros_yolov5_inference/labels_and_dataset/dataset/my_data/", "uav1"),
        (8092, "/home/jupyterWorkspace/is2ros_yolov5_inference/labels_and_dataset/dataset/my_data/", "uav2"),
        (8093, "/home/jupyterWorkspace/is2ros_yolov5_inference/labels_and_dataset/dataset/my_data/", "uav3"),
    ]

    q = queue.Queue()
    threads = []
    for port, save_dir, uav_name in uav_configs:
        t = threading.Thread(target=image_server_thread, args=(port, save_dir, uav_name, q))
        t.daemon = False
        t.start()
        threads.append(t)
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



