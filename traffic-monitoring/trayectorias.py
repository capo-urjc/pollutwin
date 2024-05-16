import cv2
from matplotlib import pyplot as plt
import numpy as np
from Straight import Straight
from Tracker import Tracker


def main():
    points: list[list[int]] = []
    with open("config/sherbrooke/straights.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            values: list[int] = line.split(',')
            x: int = int(values[0])
            y: int = int(values[1])
            points.append([x, y])

    straights: list[Straight] = [Straight(points[0], points[1]),
                                 Straight(points[2], points[3]),
                                 Straight(points[4], points[5]),
                                 Straight(points[6], points[7]),
                                 ]

    masks: list[np.ndarray] = []
    with open("config/sherbrooke/masks.txt", 'r') as f:
        lines = f.readlines()
        splitted_line: list[int] = lines[0].split(',')
        shape: tuple[int, int] = (int(splitted_line[0]), int(splitted_line[1]))
        for line in lines[1:]:
            values: list[int] = line.split(',')

            mask: np.ndarray = np.zeros(shape)
            mask[int(values[0]):int(values[1]), int(values[2]):int(values[3])] = 255
            masks.append(mask)

    tracker = Tracker(straights, masks)

    track: dict = tracker.track("sherbrooke_video.avi", True, False)

    for k, v in track.items():
        id_ = k
        lista_prueba = track[id_]
        if len(lista_prueba) > 2:
            x_ini = lista_prueba[0][0][0]
            y_ini = lista_prueba[0][0][1]

            x_f = lista_prueba[-1][0][0]
            y_f = lista_prueba[-1][0][1]

            distance = (x_f - x_ini, y_f - y_ini)
            if np.linalg.norm(distance) > 100:
                for j in range(len(lista_prueba)):
                    x = lista_prueba[j][0][0]
                    y = lista_prueba[j][0][1]
                    plt.plot(x, y, marker='.', color="yellow", alpha=0.1)

    mat_res = tracker.get_tracking_info()
    tracker.print_matrix(mat_res)


if __name__ == '__main__':
    main()
