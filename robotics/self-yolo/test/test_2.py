import cv2
from ultralytics import YOLO

# モデルの読み込み（最新のyolov5su.ptを推奨）
model = YOLO("yolov5su.pt")

def detect_video(model, video_source): # 引数からデフォルトの0を外すと安全
    cap = cv2.VideoCapture(video_source)
    
    # 動画が正常に開けたか確認
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("動画の終わりに到達したか、読み込みに失敗しました。")
            break
        
        # stream=Trueにすると、長い動画でもメモリを節約できます
        results = model(frame, stream=True)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 座標の取得
                x1, y1, x2, y2 = box.xyxy[0]
                # クラス名や信頼度も取得するとより便利です
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'
                
                # 枠の描画
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # ラベルの描画
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Video Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 使用例：動画ファイルのパスを指定
# Dockerの場合は、マウントしたディレクトリ内のパスを指定してください
video_file_path = "/home/jovyan/work/data/791732962.833074.mp4"
detect_video(model, video_file_path)