import cv2
from Detector import Detector
import norfair
import numpy as np
import random
from Straight import Straight
import torch
from typing import List


class Tracker:
    def __init__(self, straights: list[Straight], masks: list):
        self.__straights: list[Straight] = straights
        self.__masks = masks
        self.__tracking: dict = {}

        for i in range(500):
            self.__tracking[i] = []

        self.__detector = Detector("yolov5s")
        self.__tracker = norfair.Tracker(distance_function="euclidean", distance_threshold=100)

    def get_tracking(self):
        return self.__tracking

    def track(self, video_path: str, show: bool = False, save: bool = True) -> dict:
        video = norfair.Video(input_path=video_path, output_path="outputs/"+video_path.split("/")[-1])

        colors: list = []
        for i in range(len(self.__straights)):
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        last_frame = None
        for frame in video:
            last_frame = frame
            for i in range(len(self.__straights)):
                self.__straights[i].paint(frame, color=colors[i], thickness=2)

            yolo_detections = self.__detector(
                frame,
                conf_threshold=0.25,
                iou_threshold=0.45,
                image_size=800,
                classes=[2, 3, 5, 7]
                # Filtrar por clases, solo queremos detectar vehiculos
            )

            for index, row in yolo_detections.pandas().xyxy[0].iterrows():
                label: str = str(row["class"])
                x1: int = int(row["xmin"])
                y1: int = int(row["ymin"])

                if label == "2":
                    label = "Car"
                elif label == "3":
                    label = "Moto"
                elif label == "5":
                    label = "Bus"
                elif label == "7":
                    label = "Truck"

                cv2.putText(frame, label, (x1 + 25, y1 + 25), cv2.QT_FONT_NORMAL, 0.7, (170, 220, 12), 1)

            detections = self.__yolo_detections_to_norfair_detections(
                yolo_detections, track_points="centroid"
            )

            tracked_objects = self.__tracker.update(detections=detections)
            norfair.draw_points(frame, detections)
            norfair.draw_tracked_objects(frame, tracked_objects)

            if len(tracked_objects) > 0:
                for tracked_object in tracked_objects:
                    centroid: np.ndarray = tracked_object.estimate
                    id: int = tracked_object.global_id

                    self.__tracking[id].append(centroid)
            if save:
                video.write(frame)
            if show:
                cv2.imshow("", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        if last_frame is not None:
            for i in range(len(self.__straights)):
                self.__straights[i].paint(last_frame, color=colors[i], thickness=2)
            cv2.imwrite("inputs/trayectorias_sh.png", last_frame)

        track: dict[int, list] = self.get_tracking()
        return track

    def get_tracking_info(self) -> np.ndarray:
        zone1 = self.__masks[0]
        zone2 = self.__masks[1]
        zone3 = self.__masks[2]
        zone4 = self.__masks[3]

        res_matrix: np.ndarray = np.zeros((4, 4), np.uint8)

        for k, v in self.__tracking.items():
            id_ = k
            lista_prueba = self.__tracking[id_]

            if len(lista_prueba) > 2:
                x_ini = int(lista_prueba[0][0][0])
                y_ini = int(lista_prueba[0][0][1])

                x_f = int(lista_prueba[-1][0][0])
                y_f = int(lista_prueba[-1][0][1])

                distance = (x_f - x_ini, y_f - y_ini)

                if x_f >= 800:
                    x_f = 799

                if y_f >= 600:
                    y_f = 599

                if np.linalg.norm(distance) > 100:
                    if zone1[y_ini, x_ini] == 255:
                        if zone2[y_f, x_f] == 255:
                            res_matrix[0][1] += 1
                        elif zone3[y_f, x_f] == 255:
                            res_matrix[0][2] += 1
                        elif zone4[y_f, x_f] == 255:
                            res_matrix[0][3] += 1

                    elif zone2[y_ini, x_ini] == 255:
                        if zone1[y_f, x_f] == 255:
                            res_matrix[1][0] += 1
                        elif zone3[y_f, x_f] == 255:
                            res_matrix[1][2] += 1
                        elif zone4[y_f, x_f] == 255:
                            res_matrix[1][3] += 1

                    if zone3[y_ini, x_ini] == 255:
                        if zone1[y_f, x_f] == 255:
                            res_matrix[2][0] += 1
                        elif zone2[y_f, x_f] == 255:
                            res_matrix[2][1] += 1
                        elif zone4[y_f, x_f] == 255:
                            res_matrix[2][3] += 1

                    if zone4[y_ini, x_ini] == 255:
                        if zone1[y_f, x_f] == 255:
                            res_matrix[3][0] += 1
                        elif zone2[y_f, x_f] == 255:
                            res_matrix[3][1] += 1
                        elif zone3[y_f, x_f] == 255:
                            res_matrix[3][2] += 1
        return res_matrix

    def print_matrix(self, matrix: np.ndarray) -> None:
        print('----')
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                print(f"tr_{i+1}{j+1} = {matrix[i][j]},", end=' ')
            print()

    def __yolo_detections_to_norfair_detections(self, yolo_detections: torch.tensor, track_points: str = "centroid") \
            -> List[norfair.Detection]:
        norfair_detections: List[norfair.Detection] = []

        if track_points == "centroid":
            detections_as_xywh = yolo_detections.xywh[0]
            for detection_as_xywh in detections_as_xywh:
                centroid = np.array(
                    [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
                )
                scores = np.array([detection_as_xywh[4].item()])
                norfair_detections.append(
                    norfair.Detection(
                        points=centroid,
                        scores=scores,
                        label=int(detection_as_xywh[-1].item()),
                    )
                )
        elif track_points == "bbox":
            detections_as_xyxy = yolo_detections.xyxy[0]
            for detection_as_xyxy in detections_as_xyxy:
                bbox = np.array(
                    [
                        [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                        [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                    ]
                )
                scores = np.array(
                    [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
                )
                norfair_detections.append(
                    norfair.Detection(
                        points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                    )
                )

        return norfair_detections
