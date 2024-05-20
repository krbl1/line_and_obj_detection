import cv2
import numpy as np
import matplotlib.pyplot as plt
import obj_det.object_detection as od

def masking(screen, triangle):
    # height = screen.shape[0]
    # width = screen.shape[1]

    mask = np.zeros_like(screen)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(screen, mask)
    return masked_image

def canny(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaus = cv2.GaussianBlur(im, (5, 5), 0)
    can = cv2.Canny(gaus, 50, 100)
    return can

def display_lines(image, lines):
    
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            print(x1, y1, x2, y2, 'lines')
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

def get_coord(image, line_params):
    if True:
        slope, intercept = line_params
        print(slope, intercept)
        # y1 = image.shape[0]
        y1 = 910
        y2 = int(y1*(4/5))
        x1 = int((y1-intercept)/slope)
        x2 = int((y2-intercept)/slope)
        print(np.array([x1, y1, x2, y2]), 'mass')
        return np.array([x1, y1, x2, y2])
    else:
        return np.array([0, 0, 1, 1])



def avarege_lines(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            params = np.polyfit((x1, x2), (y1, y2), 1)
            slope = params[0]
            intercept = params[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            elif slope >= 0:
                right_fit.append((slope, intercept))
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = get_coord(image, left_fit_average)
        else:
            left_line = np.array([0, 0, 1, 1])
        
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = get_coord(image, right_fit_average)
        else:
            right_line = np.array([0, 0, 1, 1])
   
       
        # left_line = get_coord(image, left_fit_average)
        # right_line = get_coord(image, right_fit_average)

        return np.array([left_line, right_line])
    else:
        left_line = np.array([0, 0, 1, 1])
        right_line = np.array([0, 0, 1, 1])
        return np.array([left_line, right_line])



def video_cap(vid):
    cap = cv2.VideoCapture(vid)
   
    while True:
            success, img = cap.read()
            img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
            can = canny(img)
            trian = np.array([
        [(400, 910),(1420, 910),(1020, 504)]
         ])
            mask_for_detection = np.array([
        [(0, 910),(img.shape[1], 910),(img.shape[1], 0), (0, 0)]
         ])
        #     trian = np.array([
        # [(200, img.shape[0]),(1100, img.shape[0]),(550, 250)]
        #  ])

            triangle_img = masking(can, trian)
            first_img_with_mask = masking(img, mask_for_detection)

            lines = cv2.HoughLinesP(triangle_img, 2, np.pi/180, 200, np.array([]), minLineLength = 40, maxLineGap=5)
            averaged_lines = avarege_lines(img, lines)
            image_with_lines = display_lines(img, averaged_lines)
            

            combo_img = cv2.addWeighted(img, 0.8, image_with_lines, 1, 1)
            
            detections = od.detect_objects(first_img_with_mask)
            for (box, class_id) in detections:
                x, y, w, h = box
                label = str(od.classes[class_id])
                cv2.rectangle(combo_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(combo_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            


            cv2.waitKey(1)
            cv2.imshow("Result", combo_img)

            # plt.imshow(combo_img)
            # plt.show()


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break





video_cap('line_dettection_with_obj_det/drive.mov')
