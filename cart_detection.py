import cv2
import numpy as np
import os
import math
import time
import argparse

from shapely.geometry import Polygon
from pathlib import Path

import helpers
from yolov8 import YOLOv8
from yolov8 import utils

class CartDetector():
    def __init__(self, model_path, polygon= None) -> None:
        self.detector =  YOLOv8(model_path)
        self.H = np.float32([[580, 0], [1620, 0],
                    [383, 1080], [1800, 1080]])
        if polygon is not None: 
            self.polygon = np.array(polygon)
        else:
            self.polygon = np.array([[370, 3], [300, 207], [200, 410], [140, 850], [660, 970], [1535, 980], [1900, 900], [1850, 290], [1660, 0]])
        print(self.polygon)
        self.poly = Polygon(self.polygon)
        self.area_within_thresh = 0.7
        self.score_thresh = 0.1
        self.area_thresh = 8000

        self.frame_width = None
        self.frame_height = None

    def process_frame(self, frame, frame_order, base_name_video= None, out_vis= False):
        result = {}
        self.frame_height, self.frame_width = frame.shape[:2]
        boxes, scores, class_ids = self.detector(frame)
        #cv2.imwrite(f'output_vision/{frame_order}.jpg', frame)
        selected_indices = np.where(scores >= self.score_thresh)[0]
        selected_boxes = []
        selected_scores = []
        selected_class_ids = []
        for indice in selected_indices:
            box = boxes[indice]
            score = scores[indice]
            class_id = class_ids[indice]
            if self.check_box(box, boxes[selected_indices]) and class_id == 0:
                selected_boxes.append(box)
                selected_scores.append(score)
                selected_class_ids.append(class_id)

        result['boxes'] = selected_boxes
        result['count'] = len(selected_boxes)
        result['scores'] = selected_scores
        
        if out_vis:
            combined_img = utils.draw_detections(frame, selected_boxes, selected_scores, selected_class_ids)
            polygon_graph = self.polygon.reshape((-1, 1, 2))
            cv2.polylines(combined_img, [polygon_graph], isClosed=True, color=(0, 255, 0), thickness=3)
            cv2.putText(combined_img, str(result['count']), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            if base_name_video is not None:
                if not os.path.exists(f'output_vision/{base_name_video}'):
                    os.makedirs(f'output_vision/{base_name_video}')
                cv2.imwrite(f'output_vision/{base_name_video}/{frame_order}.jpg', combined_img)
                all_boxes_image = self.detector.draw_detections(frame)
                cv2.imwrite(f'output_vision/{base_name_video}/{frame_order}_all.jpg', all_boxes_image)
        return result, combined_img
    
    def check_box(self, box, selected_boxes_via_score):
        area_within = self.is_box_area_within_polygon(box)
        if (area_within > self.area_within_thresh):
            if (self.box_area(box) < self.area_thresh) and box[3] < self.frame_height//5:
                return False
            elif ((box[2] - box[0]) > 1.5*(box[3] - box[1]) or (self.box_area(box) < 15000)) and box[3] > self.frame_height//5:
                 return False
            #elif self.box_area(box) < 20000:
            #    near_boxes = self.find_nearby_boxes(box, selected_boxes_via_score)
            #    area_inside = 0
            #    '''
            #    for key in near_boxes.keys():
            #        near_box = near_boxes[key]
            #        area_inside_curr = self.compute_percentage_inside(box, near_box)
            #        if area_inside_curr > area_inside:
            #            area_inside = area_inside_curr 
            #    '''
            #    return area_inside < 0.95
            else:
                 return True
        else:
            return False
        
    def box_area(self, box):
        
        return (box[2] - box[0]) * (box[3] - box[1])

    def compute_intersection_area(self, box1, box2):
        
        x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
        y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
        
        intersection_area = x_overlap * y_overlap

        return intersection_area

    def compute_percentage_inside(self, box1, box2):
        
        if self.box_area(box1) > self.box_area(box2):
             box2, box1 = box1, box2

        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        intersection_area = self.compute_intersection_area(box1, box2)
        percentage_inside = intersection_area / area_box1

        return percentage_inside


    def is_box_area_within_polygon(self, box):

        box_area_value = self.box_area(box)

        intersection_area = self.poly.intersection(Polygon([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])])).area

        overlap_ratio = intersection_area / box_area_value

        return overlap_ratio

    def find_nearby_boxes(self, current_box, all_boxes, threshold=10):

        near_boxes = {'left': None, 'right': None, 'top': None, 'bottom': None}
    
        for i in range(all_boxes.shape[0]):
            box = all_boxes[i, :]
            if list(box) == list(current_box):
                continue
           
            x_distance = abs(current_box[0] - box[0])
            y_distance = abs(current_box[1] - box[1])

            if box[0] < current_box[0] and x_distance < threshold:
                if near_boxes['left'] is None or box[0] > near_boxes['left'][0]:
                    near_boxes['left'] = box

            if box[2] > current_box[2] and x_distance < threshold:
                if near_boxes['right'] is None or box[2] < near_boxes['right'][2]:
                    near_boxes['right'] = box

            if box[1] < current_box[1] and y_distance < threshold:
                if near_boxes['top'] is None or box[1] > near_boxes['top'][1]:
                    near_boxes['top'] = box

            if box[3] > current_box[3] and y_distance < threshold:
                if near_boxes['bottom'] is None or box[3] < near_boxes['bottom'][3]:
                    near_boxes['bottom'] = box
            
        return near_boxes

    def process_video(self, video_path, path_out, min_order = 5, out_vis=False):
        if os.path.exists(path_out):
            os.remove(path_out)
        
        helpers.write_title(path_out)
        
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(os.path.basename(video_path), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (1920, 1080))
        frame_number = 0
        fps =  5#math.ceil(cap.get(cv2.CAP_PROP_FPS))
        count = 0
        print(f'{os.path.basename(video_path)}: fps {fps}')
        base_name = os.path.basename(video_path).split('.')[0]
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_number % (min_order * fps) == 0:
                    t1 = time.time()
                    result, combined_img = self.process_frame(frame=frame, frame_order=frame_number, base_name_video= base_name, out_vis=out_vis)
                    out.write(combined_img)
                    str_data = helpers.convert_data_raw(result)
                    time_detect= helpers.convert_time(min_order*count)
                    print(f'{time_detect}: Frame: {frame_number}, Time: {time.time() - t1}, value: {str_data}, fps: {cap.get(cv2.CAP_PROP_FPS)}')

                    str_data = f'{count}, {time_detect} {str_data}'
                    count += 1
                    with open(path_out, 'a') as file:
                        file.write(str_data + "\n")
            except Exception as e:
                print(f"Error processing frame: {e}")

            frame_number += 1
        cap.release()
        out.release()

def read_path(base_folder ):
	paths = []
	for p, d, f in os.walk(base_folder):
		for file in f:
			if file.endswith('.avi') or file.endswith('.AVI'):
				# print("file" , file)
				paths.append(file)

	return paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, 

        default="./videos",
        help="Path to input video file"
    )
    parser.add_argument(
        "--model", type=str,

        default= './models/last.onnx',
        help= "Path to model Yolov8"
    )
    parser.add_argument(
        "--output", type=str, 

        default="./results",
        help="Path to input video folder"
    )

    args = parser.parse_args()
    model_path = args.model

    detection = CartDetector(model_path=model_path)
    print('detection: ', detection.detector)
    base_folder_path = args.input
    base_save_folder=  args.output
    Path(base_save_folder).mkdir(parents=True, exist_ok=True)

    id_folder = base_folder_path.split("/")[-1]
    sub_path_save = os.path.join(base_save_folder, id_folder)
    Path(sub_path_save).mkdir(parents=True, exist_ok=True)
    print(sub_path_save, id_folder)
    paths = read_path(base_folder_path)
    print("list video " ,paths)
    for file in paths:
        base = os.path.basename(file)
        sub_name = base.split('.')[0]
        path_out =  os.path.join(sub_path_save, f'{sub_name}.csv')
        in_video_path = os.path.join(base_folder_path, file)

        detection.process_video(video_path=in_video_path, path_out=path_out, min_order= 1, out_vis=True)
    