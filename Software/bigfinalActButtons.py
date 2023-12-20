import cv2
import numpy as np
import queue
import threading

import time


import Jetson.GPIO as GPIO











def rectify_stereo_images(imgL, imgR, map1L, map2L,map1R, map2R):

    d_imgL = cv2.cuda_GpuMat()
    d_imgL.upload(imgL)


    d_imgR = cv2.cuda_GpuMat()
    d_imgR.upload(imgR)

    # Rectify the images
    d_rectified_imgL = cv2.cuda.remap(d_imgL, map1L, map2L, interpolation=cv2.INTER_LINEAR)
    d_rectified_imgR = cv2.cuda.remap(d_imgR, map1R, map2R, interpolation=cv2.INTER_LINEAR)
    
    rectified_imgL = d_rectified_imgL.download()
    rectified_imgR = d_rectified_imgR.download()
    return rectified_imgL, rectified_imgR


def disparity_to_distance(disparity, focal_length, baseline):
    if disparity == 0:
        return float('inf')  # Return infinity if disparity is 0
    else:
        return focal_length * baseline / disparity




def calculate_average_distance(disparity_map, center_x, center_y, window_size=5):
    # Define the half window size
    half_size = window_size // 2

        # Ensure center is within the bounds of the disparity_map
    if (center_x - half_size) < 0 or (center_y - half_size) < 0 or \
       (center_x + half_size) >= disparity_map.shape[1] or (center_y + half_size) >= disparity_map.shape[0]:
        return np.nan  # Return NaN if the window is out of bounds

    region = disparity_map[center_y-half_size:center_y+half_size+1,
                           center_x-half_size:center_x+half_size+1]

    # Ensure that the region is not empty
    if region.size == 0:
        return np.nan  # Return NaN if the region is empty

    Q1 = np.percentile(region, 25)
    Q3 = np.percentile(region, 75)
    IQR = Q3 - Q1

    # Define the outlier margin
    outlier_margin = 1.5 * IQR

    # Filter out the outliers
    filtered_region = region[(region >= Q1 - outlier_margin) & (region <= Q3 + outlier_margin)]

    # Calculate the average distance
    if filtered_region.size == 0:
        # Handle the case where all values are outliers
        return np.nan
    else:
        return np.mean(filtered_region)



def calculate_distance_at_coordinates(rectified_imgL, rectified_imgR, x, y, wls_filter, left_matcher, right_matcher, focal_length, baseline):
    # Compute disparity maps
    disparity_left = left_matcher.compute(rectified_imgL, rectified_imgR).astype(np.float32) / 16
    disparity_right = right_matcher.compute(rectified_imgR, rectified_imgL).astype(np.float32) / 16
    
    # Apply the WLS filter
    filtered_disp = wls_filter.filter(disparity_left, rectified_imgL, None, disparity_right)
    
    # Calculate the average disparity for the specified coordinates
    avg_disparity = calculate_average_distance(filtered_disp, x, y, window_size=10)
    
    # Check if avg_disparity is not NaN before converting to distance
    if not np.isnan(avg_disparity):
        avg_distance = disparity_to_distance(avg_disparity, focal_length, baseline)
        return avg_distance
    else:
        return int(0)  # Return None if the average disparity is NaN



def undistort_fisheye_single(fisheye_frame0, camera_matrix, dist_coeffs, d_map1, d_map2):
    # Crop the frame to the left half (960x1280)
    h, w = fisheye_frame0.shape[:2]
    cropped_frame0 = fisheye_frame0[:, :w // 2]
    cropped_frame1 = fisheye_frame0[:, w// 2:]
    #print(cropped_frame.shape)
    d_frame0 = cv2.cuda_GpuMat()
    d_frame0.upload(cropped_frame0)

    d_frame1 = cv2.cuda_GpuMat()
    d_frame1.upload(cropped_frame1)

    d_undistorted_frame0 = cv2.cuda.remap(d_frame0, d_map1, d_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    d_undistorted_frame1 = cv2.cuda.remap(d_frame1, d_map1, d_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    undistorted_frame0 = d_undistorted_frame0.download()
    undistorted_frame1 = d_undistorted_frame1.download()

    return undistorted_frame0,undistorted_frame1

def undistort_fisheye(fisheye_frame0,fisheye_frame1, camera_matrix, dist_coeffs, d_map1, d_map2):
    # Crop the frame to the left half (960x1280)
    h, w = fisheye_frame0.shape[:2]
    cropped_frame0 = fisheye_frame0[:, :w // 2]
    h, w = fisheye_frame1.shape[:2]
    cropped_frame1 = fisheye_frame1[:, :w // 2]
    #print(cropped_frame.shape)
    d_frame0 = cv2.cuda_GpuMat()
    d_frame0.upload(cropped_frame0)
    d_frame1 = cv2.cuda_GpuMat()
    d_frame1.upload(cropped_frame1)

        #d_undistorted_frame = cv2.cuda_GpuMat()
    d_undistorted_frame0 = cv2.cuda.remap(d_frame0, d_map1, d_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    d_undistorted_frame1 = cv2.cuda.remap(d_frame1, d_map1, d_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


    undistorted_frame0 = d_undistorted_frame0.download()
    undistorted_frame1 = d_undistorted_frame1.download()

 


    return undistorted_frame0,undistorted_frame1

def create_birds_eye_view(img1, img2, img3, img4):
    # Define the dimensions of the images
    IMAGE_H = 1280
    IMAGE_W = 960

    j = 30
    x = 320 - j
    y = 320 + j
    i = 0

    A = [x, x]
    B = [x, y]
    C = [y, x]
    D = [y, y]

    # Define the source points (unchanged for all images)
    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])

    # Create a list of destination points for each image
    dst_points = [
        np.float32([A, C, [0 - i, 0], [640 + i, 0]]),
        np.float32([B, A, [0, 640 + i], [0, 0 - i]]),
        np.float32([C, D, [640, 0- i], [640, 640 + i]]),
        np.float32([D, B, [640 + i, 640], [0 - i, 640]])
    ]

    # List of images
    images = [img1, img2, img3, img4]

    # Create an empty canvas to overlay the images
    overlay = np.zeros((640, 640, 3), dtype=np.uint8)
    d_overlay = cv2.cuda_GpuMat(overlay)
    d_overlay.upload(overlay)
    # Loop through the images, apply perspective transformation, and overlay them on the canvas
    for i, img in enumerate(images):
        # Apply np slicing for ROI crop
        img = img[400:(400 + IMAGE_H), 0:IMAGE_W]
        d_img = cv2.cuda_GpuMat()
        d_img.upload(img)
        # Apply perspective transformation with the corresponding dst points
        d_warped_img = cv2.cuda.warpPerspective(d_img, cv2.getPerspectiveTransform(src, dst_points[i]), (640, 640))

        # Overlay the warped image on the canvas with transparency
        d_overlay = cv2.cuda.addWeighted(d_overlay, 1, d_warped_img, 1, 0)
    overlay=d_overlay.download()
    return overlay
    


def object_detection_on_image(input_image, input_imageR, net, class_names_file, output_image,  map1L, map2L,map1R, map2R, wls_filter, left_matcher, right_matcher, focal_length, baseline):


    # Load COCO class names (common object classes)
    with open(class_names_file, "r") as f:
        classes = f.read().strip().split('\n')

    # Read the input image
    frame = input_image

    height, width, _ = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Run the forward pass
    outputs = net.forward(output_layer_names)

    # Initialize lists to store detected objects and their confidence scores
    boxes = []
    confidences = []
    class_ids = []  # List for storing class_ids
    distances = []

    # Minimum confidence threshold for object detection
    confidence_threshold = 0.5

    # Process the output
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([int(x), int(y), int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)  # Store the class_id
                #rectify left and right
                rectL,rectR = rectify_stereo_images(frame, input_imageR,map1L, map2L,map1R, map2R)
                #calculate distance using the coordinate of the center of the boudning box
                average_distance= calculate_distance_at_coordinates(rectL, rectR, center_x, center_y, wls_filter, left_matcher, right_matcher, focal_length, baseline)
                distances.append(average_distance)

    # Apply non-maximum suppression to remove duplicate and low-confidence boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]  # Retrieve the correct label
            confidence = confidences[i]
            average_distance=distances[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {average_distance:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the output image
    return frame


def object_detection(input_image, input_imageR, net, class_names_file, output_image):


    # Load COCO class names (common object classes)
    with open(class_names_file, "r") as f:
        classes = f.read().strip().split('\n')

    # Read the input image
    frame = input_image

    height, width, _ = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Run the forward pass
    outputs = net.forward(output_layer_names)

    # Initialize lists to store detected objects and their confidence scores
    boxes = []
    confidences = []
    class_ids = []  # List for storing class_ids
    distances = []

    # Minimum confidence threshold for object detection
    confidence_threshold = 0.5

    # Process the output
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([int(x), int(y), int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)  # Store the class_id


    # Apply non-maximum suppression to remove duplicate and low-confidence boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]  # Retrieve the correct label
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Save the output image
    return frame




def resize_and_pad(img, target_width=1024, target_height=600, bg_color=(0, 0, 0)):
    # Calculate target aspect ratio
    target_ratio = target_width / target_height
    img_ratio = img.shape[1] / img.shape[0]

    if img_ratio > target_ratio:
        new_width = target_width
        new_height = int(target_width / img_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * img_ratio)

    resized_img = cv2.resize(img, (new_width, new_height))

    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=bg_color)
    return padded_img


def capture_frame(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            #print("Error: Couldn't read frame from the camera.")
            #stop_event.set()
            continue
        else:
            if not frame_queue.full():  # Only keep a limited number of frames in the queue
                frame_queue.put(frame)
            else:
                continue

def main():
    # Fisheye camera intrinsic parameters
    cap0 = cv2.VideoCapture(0)
    cap0.release()
    camera_matrix = np.array([[448.8026, 0, 687.2021],  # fx, 0, cx
                              [0, 448.8026, 496.6160],  # 0, fy, cy
                              [0, 0, 1]], dtype="double")

    dist_coeffs = np.array([-6.4734e-04, 1.6494e-06, 2.8839e-09, 0.0])  # Distortion coefficients



    frame_queue2 = queue.Queue(maxsize=2)
    frame_queue3 = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    stop_event2 = threading.Event()

    cap2 = cv2.VideoCapture(2)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    cap3 = cv2.VideoCapture(3)
    cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    capture_thread2 = threading.Thread(target=capture_frame, args=(cap2, frame_queue2, stop_event))
    capture_thread3 = threading.Thread(target=capture_frame, args=(cap3, frame_queue3, stop_event))
    cap2.release()
    cap3.release()
    cap0 = cv2.VideoCapture(0)

    cap1 = cv2.VideoCapture(1)

#    cap3 = cv2.VideoCapture(3)


    # Use a dictionary to save the last frames from each camera
    last_frames = {}

    # Initialize the variable to track which camera pair is active (0,1) or (2,3)
    active_cameras = (0, 1)
    # Set the camera resolution.
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    #cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    #cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    print("Streaming... Press 'q' to quit.")
    







    K1 = np.array([[450.30224988, 0., 695.08230627],
               [0., 451.58675105, 472.29942655	],
               [0., 0., 1.]])
    D1 = np.array([-0.13305652, 0.43907021, -0.81787499, 0.49425498])

    K2 = np.array([[454.95119115, 0., 722.04808981],
               [0., 452.73858773, 469.37623925],
               [0., 0., 1.]])
    D2 = np.array([0.01583902, -0.12057456, 0.00435042, 0.13741291])


#K1=K2
#D1=D2


    R = np.array([[ 0.99983059, 0.01113066, -0.01465931],
              [-0.01098986, 0.99989304, 0.00965096],
              [0.01476516, -0.00948822, 0.99984597]])
    T = np.array([[-1.37833047], [-0.09558202], [0.10675889]])

    # Since K1 and K2 are very similar, we can take the focal length from one of them
    focal_length = K1[0, 0]  # Assuming fx and fy are approximately equal

    # The translation vector T gives the position of the second camera relative to the first
    # Assuming the baseline is the x-component of T, given that this is the usual setup
    baseline = np.linalg.norm(T)  # The baseline magnitude

    window_size = 5
    min_disp = 0
    num_disp = 16*6 - min_disp
    # Prepare the WLS filter
    left_matcher = cv2.StereoSGBM_create(
      minDisparity=min_disp,
      numDisparities=num_disp,
      blockSize=window_size,
      uniquenessRatio=10,
      speckleWindowSize=100,
      speckleRange=32,
      disp12MaxDiff=1,
      P1=8*3*window_size**2,
      P2=32*3*window_size**2,
      preFilterCap=63,
      mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
     )
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Set WLS filter parameters
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.2)
 
    image_size=(1280, 960)
    # Compute rectification transforms
    R1, R2, P1, P2, Q , _ , _= cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY
    )
    # Compute the undistortion and rectification transformation map
    map1L, map2L = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    d_map1L = cv2.cuda_GpuMat(map1L)
    d_map2L = cv2.cuda_GpuMat(map2L)
    d_map1R = cv2.cuda_GpuMat(map1R)
    d_map2R = cv2.cuda_GpuMat(map2R)





    

    ret0, frame = cap0.read()
    h, w = frame.shape[:2]
    cropped_frame = frame[:, :w // 2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), camera_matrix, cropped_frame.shape[:2], cv2.CV_32F)
    d_map1 = cv2.cuda_GpuMat(map1)
    d_map2 = cv2.cuda_GpuMat(map2)
    # Create a single named window
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    show_undistorted1 = False
    show_undistorted2 = False
    show_object_detection1 = False
    show_object_detection2 = False
    show_distance1 = False
    show_distance2 = False
    show_birds_eye = False
    current_display = None


    model_weights = "yolov3-tiny.weights"
    model_config = "yolov3-tiny.cfg"
    class_names_file = "coco.names"
    output_image = "output_image.jpg"
    # Load the YOLO model
    net = cv2.dnn.readNet(model_weights, model_config)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


                

    # Pin Definitions
    input_pins = [7, 11, 13, 15]  # These are the GPIO pins you will use for input

    # Set up the GPIO channel
    GPIO.setmode(GPIO.BOARD)  # Use the BOARD pin numbering scheme
    GPIO.setup(input_pins, GPIO.IN)  # Set all the pins you are using as input

    # Initialize a variable for frame rate
    frame_rate = 0
    start_time = cv2.getTickCount()

    #capture_thread0.start()
    #capture_thread1.start()
    v=0
    ta=False #toggle to make new cam0,1 threads
    ta2=False #toggle to make new cam2,3 threads
    ta3=False #toggle to close cam0,1 threads
    ta4=False #toggle to close cam2,3 threads
    visit2=False
    visit3=False

    #while not stop_event.is_set():
    while True:
      if show_birds_eye:
        if not ta3 and visit2:
                stop_event.set()
                capture_thread0.join()
                capture_thread1.join()
                ta=False
                ta3=True
                visit2=False
        if not ta4 and visit3:
                stop_event2.set()
                capture_thread2.join()
                capture_thread3.join()
                ta2=False
                ta4=True
                visit3=False
        # Switch between camera pairs if bird's eye view is enabled
        if active_cameras == (0, 1):
                # Capture frames from cameras 0 and 1
                ret0, frame0 = cap0.read()
                ret1, frame1 = cap1.read()
                if ret0:
                    last_frames['cap0'] = frame0
                if ret1:
                    last_frames['cap1'] = frame1
                # Release cameras 0 and 1
                time.sleep(0.2)
                cap0.release()
                cap1.release()
                # Open cameras 2 and 3
                cap2 = cv2.VideoCapture(2)
                cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

                cap3 = cv2.VideoCapture(3)
                cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                time.sleep(0.2)
                active_cameras = (2, 3)
        else:
                # Capture frames from cameras 2 and 3
                ret2, frame2 = cap2.read()
                ret3, frame3 = cap3.read()
                if ret2:
                    last_frames['cap2'] = frame2
                if ret3:
                    last_frames['cap3'] = frame3
                # Release cameras 2 and 3
                time.sleep(0.2)
                cap2.release()
                cap3.release()
                # Open cameras 0 and 1
                cap0 = cv2.VideoCapture(0)
                cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

                cap1 = cv2.VideoCapture(1)
                cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                time.sleep(0.2)
                active_cameras = (0, 1)
                
                v=v+1
       # Use the last saved frames for bird's eye view
        frames = [last_frames.get('cap0'), last_frames.get('cap1'),last_frames.get('cap2'), last_frames.get('cap3')]
            # Check if we have all frames
        if (v>1):

             undistorted_frame0,undistorted_frame1 = undistort_fisheye(frames[0],frames[1], camera_matrix, dist_coeffs,d_map1,d_map2)
             undistorted_frame2,undistorted_frame3 = undistort_fisheye(frames[2],frames[3], camera_matrix, dist_coeffs,d_map1,d_map2)
             images = [undistorted_frame0, undistorted_frame1, undistorted_frame2, undistorted_frame3]
             current_display = create_birds_eye_view(*images)        
        else:
          continue

        # ret2, frame2 = cap2.read()
      #  # ret3, frame3 = cap3.read()
      else:
        if show_undistorted2:
          if active_cameras == (0, 1):
              if not ta3:
                stop_event.set()
                capture_thread0.join()
                capture_thread1.join()
                ta=False
                ta3=True
              cap0.release()
              cap1.release()
              # Open cameras 2 and 3
              cap2 = cv2.VideoCapture(2)
              cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
              cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

              cap3 = cv2.VideoCapture(3)
              cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
              cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

              active_cameras = (2, 3)
          if not ta2:
              stop_event2.clear()
              frame_queue2 = queue.Queue(maxsize=2)
              frame_queue3 = queue.Queue(maxsize=2)   
              capture_thread2 = threading.Thread(target=capture_frame, args=(cap2, frame_queue2, stop_event2))
              capture_thread3 = threading.Thread(target=capture_frame, args=(cap3, frame_queue3, stop_event2))
              capture_thread2.start()
              capture_thread3.start()
              ta2=True
              ta4=False
              visit3=True
          try:
            frame2 = frame_queue2.get(timeout=0.1)
            frame3 = frame_queue3.get(timeout=0.1)
            undistorted_frame2,undistorted_frame3 = undistort_fisheye(frame2,frame3, camera_matrix, dist_coeffs,d_map1,d_map2)
          except queue.Empty:
             continue
            # Display frame rate on the image

          current_display = np.hstack((undistorted_frame2, undistorted_frame3))

        else:
         if active_cameras == (2, 3):
                if not ta4:
                 stop_event2.set()
                 capture_thread2.join()
                 capture_thread3.join()
                 ta2=False
                 ta4=True
                # Release cameras 2 and 3
                cap2.release()
                cap3.release()
                # Open cameras 0 and 1
                cap0 = cv2.VideoCapture(0)
                cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                cap1 = cv2.VideoCapture(1)
                cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                active_cameras = (0, 1)

         try:
             if not ta:
              stop_event.clear()
              frame_queue0 = queue.Queue(maxsize=2)
              frame_queue1 = queue.Queue(maxsize=2)   
              capture_thread0 = threading.Thread(target=capture_frame, args=(cap0, frame_queue0, stop_event))
              capture_thread1 = threading.Thread(target=capture_frame, args=(cap1, frame_queue1, stop_event))
              capture_thread0.start()
              capture_thread1.start()
              ta=True
              ta3=False
              visit2=True
             frame0 = frame_queue0.get(timeout=0.1)
             frame1 = frame_queue1.get(timeout=0.1)
         except queue.Empty:
             continue  # Skip this iteration if no frames are available




        # this part of the code shows the front and back feed
        if show_undistorted1:
            # Measure frame rate
            
           # Get the undistorted frame
            undistorted_frame0,undistorted_frame1 = undistort_fisheye(frame0,frame1, camera_matrix, dist_coeffs,d_map1,d_map2)

           # undistorted_frame2 = undistort_fisheye(frame2, camera_matrix, dist_coeffs)
           # undistorted_frame3 = undistort_fisheye(frame3, camera_matrix, dist_coeffs)
            
     
            current_display = np.hstack((undistorted_frame0, undistorted_frame1))
        

        # this part of the code shows the sides feed
        #elif show_undistorted2:
                # Release cameras 0 and 1
            
        # this part of the code shows the object detection of the front feed
        elif show_object_detection1:
             undistorted_frame0,undistorted_frame1  = undistort_fisheye_single(frame0, camera_matrix, dist_coeffs,d_map1,d_map2)


             current_display = object_detection(undistorted_frame0,undistorted_frame1, net, class_names_file, output_image)




        # this part of the code shows the object detection of the back feed
        elif show_object_detection2:
             undistorted_frame0,undistorted_frame1  = undistort_fisheye_single(frame1, camera_matrix, dist_coeffs,d_map1,d_map2)

             current_display = object_detection(undistorted_frame0,undistorted_frame1, net, class_names_file, output_image)




        # this part of the code shows the birds eye view
        #elif show_birds_eye:


        # this part of the code shows the distance of the front feed
        elif show_distance1:
             undistorted_frame0,undistorted_frame1  = undistort_fisheye_single(frame0, camera_matrix, dist_coeffs,d_map1,d_map2)



             current_display = object_detection_on_image(undistorted_frame0,undistorted_frame1, net, class_names_file, output_image, d_map1L, d_map2L,d_map1R, d_map2R, wls_filter, left_matcher, right_matcher, focal_length,baseline)

        # this part of the code shows the distance of the back feed
        elif show_distance2:
             undistorted_frame0,undistorted_frame1  = undistort_fisheye_single(frame1, camera_matrix, dist_coeffs,d_map1,d_map2)


             current_display = object_detection_on_image(undistorted_frame0,undistorted_frame1, net, class_names_file, output_image, d_map1L, d_map2L,d_map1R, d_map2R, wls_filter, left_matcher, right_matcher, focal_length,baseline)

      #frame_rate = cv2.getTickFrequency() / (cv2.getTickCount() - start_time)
      #start_time = cv2.getTickCount()
      #font = cv2.FONT_HERSHEY_SIMPLEX
      #cv2.putText(current_display, f'Frame Rate: {frame_rate:.2f} FPS', (10, 30), font, 1, (0, 255, 0), 2)
      #cv2.putText(current_display, f'Frame Rate: {frame_rate:.2f} FPS', (10, 30), font, 1, (0, 255, 0), 2)
      current_display = resize_and_pad(current_display)
      cv2.imshow("Camera Feed", current_display)
      cv2.waitKey(1)

        # Check for user input to change the feed to display
      #key = cv2.waitKey(1) & 0xFF
      # Read the value of the pins
      input_values = [GPIO.input(pin) for pin in input_pins]
      combined_value = (input_values[0] << 3) | (input_values[1] << 2) | (input_values[2] << 1) | input_values[3]


      if combined_value == 2:
          #stop_event.set()
           break
      elif combined_value == 15:
           print("birdeye")
           show_undistorted1 = False
           show_undistorted2 = False
           show_object_detection1 = False
           show_object_detection2 = False
           show_distance1 = False
           show_distance2 = False
           show_birds_eye = True
      elif combined_value == 14:
           print("distort1")
           show_undistorted1 = True
           show_undistorted2 = False
           show_object_detection1 = False
           show_object_detection2 = False
           show_distance1 = False
           show_distance2 = False
           show_birds_eye = False
      elif combined_value == 13:
           print("distort2")
           show_undistorted1 = False
           show_undistorted2 = True
           show_object_detection1 = False
           show_object_detection2 = False
           show_distance1 = False
           show_distance2 = False
           show_birds_eye = False
      elif combined_value == 11:
           print("obj1")
           show_undistorted1 = False
           show_undistorted2 = False
           show_object_detection1 = True
           show_object_detection2 = False
           show_distance1 = False
           show_distance2 = False
           show_birds_eye = False
      elif combined_value == 10:
           print("obj2")
           show_undistorted1 = False
           show_undistorted2 = False
           show_object_detection1 = False
           show_object_detection2 = True
           show_distance1 = False
           show_distance2 = False
           show_birds_eye = False
      elif combined_value == 9:
           print("dist1")
           show_undistorted1 = False
           show_undistorted2 = False
           show_object_detection1 = False
           show_object_detection2 = False
           show_distance1 = True
           show_distance2 = False
           show_birds_eye = False
      elif combined_value == 7:
           print("dist2")
           show_undistorted1 = False
           show_undistorted2 = False
           show_object_detection1 = False
           show_object_detection2 = False
           show_distance1 = False
           show_distance2 = True
           show_birds_eye = False



    if active_cameras == (0, 1):
      stop_event.set()
      capture_thread0.join()
      capture_thread1.join()
      cap0.release()
      cap1.release()
    else:
      cap2.release()
      cap3.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

