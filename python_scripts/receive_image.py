import os
import socket
import argparse
import json


def build_filename(prefix: str, counter: int, suffix: str = ".png") -> str:
    """Build a per-UAV sequential image filename."""
    return f"{prefix}_{counter}{suffix}"


def recv_exact(conn, size):
    """Receive exactly `size` bytes, or return None if connection closes early."""
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def receive_image(save_dir, listen_host="0.0.0.0", listen_port=9996, prefix="uav"):
    """
    接收客户端上传的图片并保存到指定目录。

    命名规则：
    {prefix}_{序号}.png
    例如：uav1_1.png, uav1_2.png

    Args:
        save_dir (str): 保存图片的目录
        listen_host (str): 监听地址
        listen_port (int): 监听端口
        prefix (str): 文件名前缀，建议按来源区分，例如 uav1 / uav2
    """
    os.makedirs(save_dir, exist_ok=True)
    counter = 1
    uav_counters = {}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((listen_host, listen_port))
        s.listen(1)
        print(f"服务器已启动，监听 {listen_host}:{listen_port}，等待连接...")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"收到连接来自: {addr}")

                first4 = recv_exact(conn, 4)
                if first4 is None:
                    print("未收到完整的协议头，跳过此次连接")
                    continue

                current_uav = prefix

                # 新协议: UVP1 + meta_len + meta_json + image_len + image_bytes
                if first4 == b"UVP1":
                    meta_len_bytes = recv_exact(conn, 4)
                    if meta_len_bytes is None:
                        print("未收到完整元信息长度，跳过此次连接")
                        continue

                    meta_len = int.from_bytes(meta_len_bytes, byteorder="big")
                    meta_bytes = recv_exact(conn, meta_len)
                    if meta_bytes is None:
                        print("未收到完整元信息，跳过此次连接")
                        continue

                    try:
                        meta = json.loads(meta_bytes.decode("utf-8"))
                        current_uav = meta.get("uav", prefix)
                    except Exception:
                        current_uav = prefix

                    image_len_bytes = recv_exact(conn, 4)
                    if image_len_bytes is None:
                        print("未收到完整的图片长度头，跳过此次连接")
                        continue
                    image_len = int.from_bytes(image_len_bytes, byteorder="big")
                else:
                    # 旧协议兼容: first4 直接是 image_len
                    image_len = int.from_bytes(first4, byteorder="big")

                received = bytearray()
                while len(received) < image_len:
                    data = conn.recv(min(65536, image_len - len(received)))
                    if not data:
                        break
                    received.extend(data)

                if len(received) != image_len:
                    print(f"接收不完整: {len(received)}/{image_len} 字节，跳过保存")
                    continue

                if current_uav not in uav_counters:
                    uav_counters[current_uav] = 1

                filename = build_filename(current_uav, uav_counters[current_uav])
                save_path = os.path.join(save_dir, filename)

                with open(save_path, "wb") as f:
                    f.write(received)

                print(f"已保存图片: {save_path}")
                uav_counters[current_uav] += 1
                counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive images and save them as uavX_X.png")
    parser.add_argument("--save-dir", default="images_received", help="Directory to store received images")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=9996, help="Listen port")
    parser.add_argument("--uav-name", default="uav1", help="UAV prefix used in filenames, e.g. uav1")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    receive_image(args.save_dir, listen_host=args.host, listen_port=args.port, prefix=args.uav_name)