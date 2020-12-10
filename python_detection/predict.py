from darkflow.net.build import TFNet
import cv2
import os
import numpy as np
import traceback
class Predict:

    def __init__(self):
        self.axis = list()
        self.cnt = 0
        self.video_path = './data/test.mp4'
        self.options = {"model": "./cfg/tiny-yolo-4c.cfg", "load": -1, "threshold": 0.7}

    def init_seg(self, event, x, y, flags, param):
        if len(self.axis) == 4:
            return;
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.axis.append([x, y])

    def draw_line(self, frame):

        if len(self.axis) == 4:
            pts = np.array(self.axis, np.int32)  # 각 꼭지점은 2차원 행렬로 선언
            #pts = pts.reshape((-1, 1, 2))
            #print(pts)
            cv2.line(frame, (self.axis[0][0], self.axis[0][1]), (self.axis[1][0], self.axis[1][1]), (255, 255, 255), 3)
            cv2.line(frame, (self.axis[1][0], self.axis[1][1]), (self.axis[2][0], self.axis[2][1]), (255, 255, 255), 3)
            cv2.line(frame, (self.axis[2][0], self.axis[2][1]), (self.axis[3][0], self.axis[3][1]), (255, 255, 255), 3 )
            #frame = cv2.polylines(frame, [pts], True, (0, 255, 0))
        return frame

    def run(self):
        tfnet = TFNet(self.options)

        cap = cv2.VideoCapture(self.video_path)

        save_point = []
        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret : break
            frame = cv2.resize(frame, (960, 540))



            # 물체 인식
            result = tfnet.return_predict(frame)
            for data in result:
                print(data)
                frame = cv2.rectangle(frame, (data['topleft']['x'], data['topleft']['y']), (data['bottomright']['x'], data['bottomright']['y']), (0, 0, 255), 2)

                save_point.append([data['bottomright']['x'], data['bottomright']['y']])

            cv2.namedWindow("img")
            cv2.imshow('img', self.draw_line(frame))
            cv2.setMouseCallback('img', self.init_seg)
            if cnt == 0:
                cv2.waitKey(0)
                cnt += 1

            if cv2.waitKey(1) == ord('q'):
                break

        img = np.zeros((540, 960, 3), np.uint8)

        check_point = []
        # 5점의 평균이 10 이하 일 시 멈춤.
        check_index = 0


        back_out_line = []
        for idx, point in enumerate(save_point):
            try:
                img = cv2.line(img, (point[0], point[1]), (point[0], point[1]), (255, 0, 0), 5)
                try:
                    # 각 라인 위치
                    dist1 = self.dist(point, self.axis[0], self.axis[1])
                    dist2 = self.dist(point, self.axis[1], self.axis[2])
                    dist3 = self.dist(point, self.axis[2], self.axis[3])
                except:
                    print(traceback.format_exc())
                    continue

                # +- 30 픽셀 위치 찾기 실제 무릎도 안되는 위치.
                if dist2 < 80:
                    img = cv2.line(img, (save_point[idx][0], save_point[idx][1]), (save_point[idx][0], save_point[idx][1]),
                                   (0, 0, 255), 5)
                    back_out_line.append([idx, point])


            except:
                print(traceback.format_exc())
        cv2.imshow("point", self.draw_line(img))
        cv2.waitKey(0)


        # 판단
        first_idx = -1
        for idx, point in enumerate(back_out_line):
            img = self.draw_line(img)
            if (idx > 0 ) and (idx < len(back_out_line)-1):
                try:
                    # 가까운 인덱스가 아니면 무시
                    if back_out_line[idx][0] - back_out_line[idx-1][0] > 3: continue
                    prev = back_out_line[idx][1][1] - back_out_line[idx-1][1][1]
                    next = back_out_line[idx+1][1][1] - back_out_line[idx][1][1]
                    if prev >= 0 and next <= 0 :
                        img = cv2.line(img, (back_out_line[idx][1][0], back_out_line[idx][1][1]),
                                       (back_out_line[idx][1][0], back_out_line[idx][1][1]), (0, 255, 0), 5)
                        print ("증가 감소 ")
                        if first_idx == -1 :
                            first_idx = idx


                    """
                    prev     = (back_out_line[idx][1][1] - back_out_line[idx-1][1][1]) / (back_out_line[idx][1][0] - back_out_line[idx-1][1][0])
                    next = (back_out_line[idx+1][1][1] - back_out_line[idx][1][1]) / (back_out_line[idx+1][1][0] - back_out_line[idx][1][0])
                    print (back_out_line[idx][1][0], back_out_line[idx][1][1])
                    print(back_out_line[idx-1][1][0], back_out_line[idx-1][1][1])
                    print(back_out_line[idx+1][1][0], back_out_line[idx+1][1][1])
                    # 떨어짐
                    if prev >= 0 and next >= 0 :
                        img = cv2.line(img, (back_out_line[idx][1][0], back_out_line[idx][1][1]),
                                       (back_out_line[idx][1][0], back_out_line[idx][1][1]), (0, 255, 0), 5)

                    # 상승 변곡 + 반대로 나아감
                    elif prev >= 0 and next <= 0 :
                        img = cv2.line(img, (back_out_line[idx][1][0], back_out_line[idx][1][1]),
                                       (back_out_line[idx][1][0], back_out_line[idx][1][1]), (0, 255, 255), 5)
                        print("변곡 ")


                    # 상승
                    elif prev <= 0 and next <= 0 :
                        img = cv2.line(img, (back_out_line[idx][1][0], back_out_line[idx][1][1]),
                                       (back_out_line[idx][1][0], back_out_line[idx][1][1]), (255, 255, 0), 5)

                    # 하락 변곡 + 반대로 나아감
                    elif prev <=0 and next >= 0:
                        img = cv2.line(img, (back_out_line[idx][1][0], back_out_line[idx][1][1]),
                                       (back_out_line[idx][1][0], back_out_line[idx][1][1]), (255, 255, 255), 5)

                    else:
                        print("???")
                    """
                    # 기울기가 + 이면 떨어짐
                    # 기울기가 - 면 올라감
                    print(prev, next)
                except:
                    print(traceback.format_exc())
                    pass
        cv2.imshow("point2", img)
        cv2.waitKey(0)
        # 미세 조정
        result = ""
        if (not first_idx == -1) and (self.axis[1][1] + 2 > back_out_line[first_idx][1][1] or self.axis[2][1] + 2 > back_out_line[first_idx][1][1]) :
            result = "IN!!"
        else:
            result = "OUT!!"

        cv2.putText(img, result, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("result", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    # P - 점 , A-B 두 직선의 거리
    def dist(self, P, A, B):
        area = abs((A[0] - P[0]) * (B[1] - P[1]) - (A[1] - P[1]) * (B[0] - P[0]))
        AB = ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5
        return (area / AB)

    def y_dist(self, P, fuc_place):
        min = 99999
        for fuc in fuc_place:
            y = fuc[0] * P[0] + fuc[1]
            if abs(y-P[1]) < min:
                min = abs(y-P[1])
        return min

    def func_check(self, point1, point2):
        # 기울기
        a = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point1[1] - a*point1[0]
        return [a, b]

if __name__ == "__main__":
    start = Predict()
    start.run()
