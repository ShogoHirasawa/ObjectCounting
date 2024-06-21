import cv2
import csv
from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("./testdata/test.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

with open('count_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Object_ID", "Class_Name", "Count"])

# Define region points as a polygon with 5 points
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

frame_number = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)
    print(dir(tracks))

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)



    with open('count_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for track in tracks:
            writer.writerow([frame_number, track['bbox'], track['class'], track['score']])

    frame_number += 1


cap.release()
video_writer.release()
cv2.destroyAllWindows()