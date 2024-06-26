import cv2
from Detector import Detector
from EfficientNetEmbedding import EfficientNetEmbedding
from norfair.filter import OptimizedKalmanFilterFactory
from norfair import Detection
import norfair
import numpy as np
from PIL import Image
from Straight import Straight
import torch
from torchvision import transforms
from typing import List


class Tracker:
    def __init__(self, straights: list[Straight], masks: list):
        self.__straights: list[Straight] = straights
        self.__masks = masks
        self.__tracking: dict = {}

        for i in range(500):
            self.__tracking[i] = [0] * 20

        self.__detector = Detector("yolov7-d6.pt")
        self.__embedding = EfficientNetEmbedding()
        self.__embedding.eval()
        self.__transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.__tracker = norfair.Tracker(initialization_delay=1,
                                         distance_function="euclidean",
                                         distance_threshold=50,
                                         past_detections_length=5,
                                         #reid_distance_function=self.__euclidean_distance,
                                         #reid_distance_threshold=30,
                                         )

    def get_tracking(self):
        return self.__tracking

    def center(self, points):
        return [np.mean(np.array(points), axis=0)]

    def track(self, video_path: str, show: bool = False, save: bool = True) -> dict:
        video = norfair.Video(input_path=video_path, output_path="outputs/" + video_path.split("/")[-1])

        i: int = 0
        for frame in video:
            yolo_detections = self.__detector(
                frame,
                conf_threshold=0.5,
                iou_threshold=0.5,
                image_size=frame.shape[1],
                classes=[2, 3, 5, 7]
            )

            bbox_detections = self.__yolo_detections_to_norfair_detections(
                yolo_detections, track_points="bbox"
            )

            # for detection in bbox_detections:
            #     cut = norfair.get_cutout(detection.points, frame)
#
            #     if cut.shape[0] > 0 and cut.shape[1] > 0:
            #         cut = Image.fromarray(cut)
            #         cut = self.__transform(cut).unsqueeze(0)
            #         with torch.no_grad():
            #             detection.embedding = self.__embedding(cut)

            tracked_objects = self.__tracker.update(detections=bbox_detections)
            # norfair.draw_points(frame, detections)
            # norfair.draw_tracked_objects(frame, tracked_objects)

            if len(tracked_objects) > 0:
                to_draw: List["TrackedObject"] = []
                for tracked_object in tracked_objects:
                    bbox: np.ndarray = tracked_object.estimate

                    centroidx: float = (bbox[0, 0] + bbox[1, 0]) / 2
                    centroidy: float = (bbox[0, 1] + bbox[1, 1]) / 2

                    centroid: np.ndarray = np.array([centroidx, centroidy])
                    id: int = tracked_object.global_id

                    cv2.putText(frame, str(id), (int(centroid[0] - 18), int(centroid[1] - 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 2, (0, 255, 0), -1
                               )

                    if np.linalg.norm(centroid - self.__tracking[id][(i + 1) % 20]) > 20:
                        self.__tracking[id][i % 20] = centroid
                        to_draw.append(tracked_object)

                # norfair.draw_tracked_objects(frame, to_draw)
            i += 1
            if save:
                video.write(frame)
            if show:
                cv2.imshow("", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

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
                print(f"tr_{i + 1}{j + 1} = {matrix[i][j]},", end=' ')
            print()

    def __euclidean_distance(self, matched_not_init_object, unmatched_object):
        second_embedding = unmatched_object.last_detection.embedding

        if second_embedding is None:
            for detection in reversed(unmatched_object.past_detections):
                if detection.embedding is not None:
                    second_embedding = detection.embedding
                    break
            else:
                return 1000

        for detection_fst in matched_not_init_object.past_detections:
            if detection_fst.embedding is None:
                continue

            distance = torch.dist(detection_fst.embedding, second_embedding, p=2).item()

            if distance < 30:
                return distance

        return 1000

    def __yolo_detections_to_norfair_detections(self, yolo_detections: torch.tensor, track_points: str = "centroid") \
            -> List[norfair.Detection]:
        norfair_detections: List[Detection] = []

        if track_points == "centroid":
            detections_as_xywh = yolo_detections.xywh[0]
            for detection_as_xywh in detections_as_xywh:
                centroid = np.array(
                    [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
                )
                scores = np.array([detection_as_xywh[4].item()])
                norfair_detections.append(
                    Detection(
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
                    Detection(
                        points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                    )
                )

        return norfair_detections
