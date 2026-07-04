import struct
import numpy as np


# 這裡是解析 COLMAP 的 binary 格式，轉換成我們需要的相機參數格式
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        #print(f"Found {num_cameras} cameras in {path_to_model_file}")
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1] 
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = 4 if model_id == 1 else 3 # SIMPLE_PINHOLE 有 3 個

            params = read_next_bytes(fid, num_params * 8, "d" * num_params)
            cameras[camera_id] = {
                "model": model_id, "width": width, "height": height, "params": np.array(params)
            }
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        #print(f"Found {num_images} images in {path_to_model_file}")
        for _ in range(num_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5]) # 四元數 [w, x, y, z]
            tvec = np.array(binary_image_properties[5:8]) # 平移向量 [tx, ty, tz]
            camera_id = binary_image_properties[8]

            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            read_next_bytes(fid, num_points2D * 24, "ddq" * num_points2D)
            
            images[image_id] = {
                "qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": image_name
            }
    return images

    