from twilio.rest import Client

account_sid = input("account_sid : ")
auth_token = input("auth_token : ")
phone_numbers = [
]
client = Client(account_sid, auth_token)

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
#model = YOLO('yolov8s.pt')
model = YOLO('yolov8n.pt')

# 동영상 파일 사용시
# video_path = "path/to/your/video/file.mp4"
# cap = cv2.VideoCapture(video_path)

CONFIDENCE_THRESHOLD = 0.3

# webcam 사용시 
cap = cv2.VideoCapture(0)

x_offset = 480
y_offset = 0
cnt = 0
s_img = cv2.imread("wait2.png", -1)
y1, y2 = y_offset, y_offset + s_img.shape[0]
x1, x2 = x_offset, x_offset + s_img.shape[1]
alpha_s = s_img[:, :, 3] / 255.0
alpha_l = 1.0 - alpha_s
check_yn = False

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        for data in results[0].boxes.data.tolist():
            confidence = float(data[4])
            if confidence < CONFIDENCE_THRESHOLD :
                continue
            label = int(data[5])
            if label == 67 :
                cnt+=1
                print("cell-phone detected! cnt : ", cnt)

            if not check_yn and cnt  >= 50 :
                for c in range(0, 3):
                    annotated_frame[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                      alpha_l * annotated_frame[y1:y2, x1:x2, c])
                print("잠깐! 모르는 사람과 통화중이신가요?")

        
        if not check_yn and cv2.waitKey(1) & 0xFF == ord("1"):
            check_yn = True
            
            for phone_number in phone_numbers :
                message = client.messages.create(
                to=phone_number,
                from_="+12568125366",
                body="신한은행 성수동지점 ATM기 [6001]\n 보이스피싱 의심사례 발생!\n 즉시 확인 부탁드립니다.")
            
                print(message.sid)
        elif not check_yn and cv2.waitKey(1) & 0xFF == ord("2"):
            check_yn = True
        
        elif check_yn and cv2.waitKey(1) & 0xFF == ord("3"):
            check_yn = False
            cnt = 0
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q") or cv2.waitKey(1) & 0xFF == ord("Q") or cv2.waitKey(1) & 0xFF == ord("ㅂ"):
            break
            
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()