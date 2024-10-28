import cv2
import numpy as np
import math
import os
import sys
from sklearn.cluster import KMeans
from ultralytics import YOLO
import time

def calculate_angle(line):
    angle = math.atan2(line[3] - line[1], line[2] - line[0])
    angle = math.degrees(angle) - 90
    angle = (360 - angle) % 360
    if angle == 0:
        angle = 360
    return angle

def calculate_lenght(line):
    if len(line) < 4:
        return 0
    if not isinstance(line, np.ndarray):
        line = np.array(line)
    return abs(np.linalg.norm(line[0:2] - line[2:4]))

def line_equation(line):
    A = line[3] - line[1]
    B = line[0] - line[2]
    C = A * line[0] + B * line[2]
    return A, B, -C

def line_distance(line1, line2):

    vector1 = np.array([line1[2] - line1[0], line1[3] - line1[1]])
    vector2 = np.array([line2[2] - line2[0], line2[3] - line2[1]])

    vector_between_lines = np.array([line2[0] - line1[0], line2[1] - line1[1]])

    distance = np.abs(np.cross(vector1, vector_between_lines)) / np.linalg.norm(vector1)

    return distance

#============================================================================================

def rescale(image, max_lenght):
    return cv2.resize(image, (max_lenght, max_lenght))

def find_lines(clock):

    gray = cv2.cvtColor(clock, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=20)

    return lines


def group_hands(hands, center_x, center_y):

    hand_group = {0: [], 1: [], 2: []}
    angles = []
    t = 0.2
    width_point = np.array([center_x - center_x*t, center_x + center_x*t])
    height_point = np.array([center_y - center_y*t, center_y + center_y*t])
    for i in range(len(hands)):
        x1, y1, x2, y2 = hands[i]
        if (width_point[0]  < x1 < width_point[1] and height_point[0] < y1 < height_point[1]):
            angles.append(calculate_angle([x1, y1, x2, y2]))
        elif (width_point[0] < x2 < width_point[1] and height_point[0] < y2 < height_point[1]):
            hands[i] = [x2, y2, x1, y1]
            angles.append(calculate_angle([x2, y2, x1, y1]))

    angles = np.array(angles).reshape(-1, 1)
    n = 3
    while (n > 0):
        try:
            kmeans = KMeans(n_clusters=n, n_init=10)
            kmeans.fit(angles)
        except:
            n -= 1
        else:
            break
    if n == 0:
        return hand_group
    labels = kmeans.labels_

    for label, hand in zip(labels, hands):
        hand_group[label].append(hand)

    return hand_group


def find_hands(lines, center_x, center_y, t=0.5):

    hands = []
    width_point = np.array([center_x - center_x*t, center_x + center_x*t])
    height_point = np.array([center_y - center_y*t, center_y + center_y*t])
    if lines is not None:
        for line in list(lines):
            x1, y1, x2, y2 = list(line[0])
            if (width_point[0]  < x1 < width_point[1] and height_point[0] < y1 < height_point[1]):
                hands.append((x1, y1, x2, y2))
            elif (width_point[0] < x2 < width_point[1] and height_point[0] < y2 < height_point[1]):
                hands.append((x1, y1, x2, y2))

    return hands

def get_hands(hand_group):

    hour_label = min_label = sec_label = -1

    min_thickness = 99999
    for label, hands in list(hand_group.items()):
        hands.sort(key=lambda hand: calculate_lenght(hand), reverse=True)
        if len(hands) >= 2:
            thickness = line_distance(hands[0], hands[1])
            if thickness < min_thickness:
                min_thickness = thickness
                sec_label = label
        else:
            sec_label = label
            break

    labels = [label for label in list(hand_group.keys()) if label != sec_label]
    hands1 = list(hand_group[labels[0]])
    hands2 = list(hand_group[labels[1]])
    if not hands1:
        hour_label = min_label = labels[1]
    elif not hands2:
        hour_label = min_label = labels[0]
    else:
        if calculate_lenght(hands1[0]) < calculate_lenght(hands2[0]):
            hour_label = labels[0]
            min_label = labels[1]
        else:
            hour_label = labels[1]
            min_label = labels[0]

    return hour_label, min_label, sec_label

def get_time(hand_group, hour_label, min_label, sec_label):

    hour = min = sec = 0

    if hand_group[hour_label]:
        x1, y1, x2, y2 = hand_group[hour_label][0]
        hour_angle = calculate_angle([x1, -y1, x2, -y2])
        hour = int(hour_angle / 30)
    else:
        hour = 0

    if hand_group[min_label]:
        x1, y1, x2, y2 = hand_group[min_label][0]
        min_angle = calculate_angle([x1, -y1, x2, -y2])
        min = int(min_angle / 6)
    else:
        min = hour

    if hand_group[sec_label]:
        x1, y1, x2, y2 = hand_group[sec_label][0]
        sec_angle = calculate_angle([x1, -y1, x2, -y2])
        sec = int(sec_angle / 6)
    else:
        sec = min
    
    return hour, min, sec

def draw_hand(clock, hour_hands, min_hands, sec_hands,):

    for hand in hour_hands:
        cv2.line(clock, hand[0:2], hand[2:4], (0, 0, 255), 2)
    for hand in min_hands:
        cv2.line(clock, hand[0:2], hand[2:4], (255, 0, 0), 2)
    for hand in sec_hands:
        cv2.line(clock, hand[0:2], hand[2:4], (0, 255, 0), 2)


def read_clock(image, x1, y1, x2, y2):

    clock = image[y1:y2, x1:x2]

    clock = rescale(clock, max_lenght=500)
    
    width, height, _ = clock.shape
    center_x, center_y = width//2, height//2

    lines = find_lines(clock)

    hands = find_hands(lines, center_x, center_y, t=0.5)

    hand_group = group_hands(hands, center_x, center_y)

    hour_label, min_label, sec_label = get_hands(hand_group)

    hour, min, sec = get_time(hand_group, hour_label, min_label, sec_label)

    draw_hand(clock, hand_group[hour_label], hand_group[min_label], hand_group[sec_label])

    text = f"{str(hour).zfill(2)}:{str(min).zfill(2)}:{str(sec).zfill(2)}"
    cv2.putText(clock, text, (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    print(text)

    return clock


def video(video):

    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video)

    while cap.isOpened():

        success, frame = cap.read()

        if success:

            results = model(frame, verbose=False, classes=[74])

            annotated_frame = results[0].plot()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result[0].boxes.numpy()
                    for box in boxes:
                        if box.cls == 74:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            annotated_frame = read_clock(annotated_frame, x1, y1, x2, y2)
                else:
                    print("No clocks detected.")

            cv2.imshow("YOLOv8 Inference", annotated_frame)

        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def image(path):

    model = YOLO('yolov8n.pt')

    image = cv2.imread(path)

    results = model(image, classes=[74])

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result[0].boxes.numpy()
            for box in boxes:
                if box.cls == 74:
                    frame = box.xyxy[0]
                    x1, y1, x2, y2 = map(int, frame)
                    image = read_clock(image, x1, y1, x2, y2)
        else:
            print("No clocks detected.")

    cv2.imshow('clock', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    if len(sys.argv) == 1:
        video(0)
    if len(sys.argv) == 2:

        path = sys.argv[1]
        if not os.path.isfile(path):
            exit(1)

        
        _, ext = os.path.splitext(path)
        if ext.lower() == '.mp4':
            video(path)
        else:
            image(path)

