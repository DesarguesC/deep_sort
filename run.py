import cv2, os, shutil
from deep_sort import DeepSORT  # 导入DeepSORT模块，具体路径根据你的设置而定
intput_dir = './inputs/exe-11/'
files_list = os.listdir(input_dir)
files_list = [x for x in files_list if 'ipynb' not in x]

output_dir = './test_out/cells'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    
os.mkdir(output_dir)

# 创建DeepSORT跟踪器
tracker = DeepSORT()

# 读取之前保存的帧图像，并进行目标跟踪
for frame_count in range(len(files_list)):  # num_frames表示帧数
    # 读取图像帧
    frame = cv2.imread(f'frame_{frame_count:06}.jpg')

    # 在帧上进行目标检测和跟踪
    tracked_objects = tracker.update(frame)

    # 在图像上绘制目标框等信息
    for obj in tracked_objects:
        # 绘制目标框
        cv2.rectangle(frame, (obj.left, obj.top), (obj.right, obj.bottom), (0, 255, 0), 2)
        # 绘制目标ID
        cv2.putText(frame, str(obj.track_id), (obj.left, obj.top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示图像帧（可选）
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)  # 等待1毫秒

# 释放资源
cv2.destroyAllWindows()

