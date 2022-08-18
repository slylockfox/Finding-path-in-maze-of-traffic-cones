# See https://github.com/slylockfox/Finding-path-in-maze-of-traffic-cones

## Load the trained model

import torch
import torchvision

model = torchvision.models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 5)

# Next, load the trained weights from the ``best_model.pth`` file that you uploaded

model.load_state_dict(torch.load('best_model_cones_8.pth'))

# Currently, the model weights are located on the CPU memory execute the code below to transfer to the GPU device.

device = torch.device('cuda')
model = model.to(device)

## Create the preprocessing function

# We have now loaded our model, but there's a slight issue.  The format that we trained our model doesnt *exactly* match the format of the camera.  To do that, 
# we need to do some *preprocessing*.  This involves the following steps

# 1. Convert from BGR to RGB
# 2. Convert from HWC layout to CHW layout
# 3. Normalize using same parameters as we did during training (our camera provides values in [0, 255] range and training loaded images in [0, 1] range so we need to scale by 255.0
# 4. Transfer the data from CPU memory to GPU memory
# 5. Add a batch dimension

import cv2
import numpy as np

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x

# Great! We've now defined our pre-processing function which can convert images from the camera format to the neural network input format.

# Now, let's start and display our camera.  You should be pretty familiar with this by now.

# import traitlets
# from IPython.display import display
# import ipywidgets.widgets as widgets
from jetbot import Camera, bgr8_to_jpeg

camera = Camera.instance(width=224, height=224)
# image = widgets.Image(format='jpeg', width=224, height=224)
# camera_link = traitlets.dlink((camera, 'value'), (image, 'value'), transform=bgr8_to_jpeg)
# display(widgets.HBox([image]))

# We'll also create our robot instance which we'll need to drive the motors.

from jetbot import Robot

robot = Robot()

# robot.left(0.3)
# robot.stop()
# robot.right(0.3)

# Next, we will write function which will collect frame-by-frame images with telementry (output stats and actions) for further analysis, debugging and controller tuning
# > To avoid possible confusion - the folder *images* stores the camera snapshots collected during the robot run. This is NOT the folder with training images. 
# > 
# > Also, the location of On-Screen_Display (OSD) messages with telemetry data is optimized for default (224 x 224 pixels) image size

import glob
import os
from PIL import Image, ImageFont, ImageDraw

font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 20)
font_probs = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
path_to_img_folder = 'images'

# we have this "try/except" statement because these next functions can throw an error if the directories exist already
try:
    os.makedirs(path_to_img_folder)
except FileExistsError:
    print('Directory not created because they already exist')

def save_frames_with_telemetry(raw_image,current_probs,exploration,action,n_frames_stuck):
    prob_free,prob_left,prob_right,prob_blocked,prob_stop = current_probs
    # save image with telemetry for performance analysis (On-screen display data is for for 224x224 frame size) 
    img_file_nm = path_to_img_folder + '/img' + str(int(100.*time.time())) + '.jpeg'
    img = Image.fromarray(raw_image)
    
    draw = ImageDraw.Draw(img)
    draw.text((5,0),  "F", (255,255,0), font=font)
    draw.text((5,20), "L", (255,255,0), font=font)
    draw.text((5,40), "R", (255,255,0), font=font)
    draw.text((5,60), "B", (255,255,0), font=font)
    draw.text((5,80), "S", (255,255,0), font=font)
    
    def fill_color(prob,threshold,below_color=(0,255,0,255),above_color=(255,0,0,255)):
        color = below_color
        if prob > threshold:
            color = above_color
        return color
    
    draw.rectangle([20,5,  int(20+80*prob_free) ,  18], fill=fill_color(prob_free, 0.5), outline=None)
    draw.rectangle([20,25, int(20+80*prob_left) ,  38], fill=fill_color(prob_left, 0.5), outline=None)
    draw.rectangle([20,45, int(20+80*prob_right),  58], fill=fill_color(prob_right, 0.5), outline=None)
    draw.rectangle([20,65, int(20+80*prob_blocked),78], fill=fill_color(prob_blocked, 0.5), outline=None)
    draw.rectangle([20,85, int(20+80*prob_stop),   78], fill=fill_color(prob_stop, 0.5), outline=None)

    draw.text((22,4),  str(round(prob_free,2)), (0,0,0), font=font_probs)
    draw.text((22,24), str(round(prob_left,2)), (0,0,0), font=font_probs)
    draw.text((22,44), str(round(prob_right,2)), (0,0,0), font=font_probs)
    draw.text((22,64), str(round(prob_blocked,2)), (0,0,0), font=font_probs)
    draw.text((22,84), str(round(prob_stop,2)), (0,0,0), font=font_probs)
    
    # display frame number
    draw.text((5,104), 'FRAME #' + str(frame_counter), (0,0,0), font=font_probs)

    # display state of n_frames_stuck counter
    if exploration:
        draw.rectangle([5,104,170,119], fill=(255,0,0,255), outline=None)
    draw.text((5,124), 'FRAMES STUCK: #' + str(n_frames_stuck), (0,0,0), font=font_probs)

    action_displayed = action[0] + " " + str(action[1])
    if (action[0] == "FWRD"):
        action_displayed = action[0] + " " + str(action[1])
        draw.text((105,4), action_displayed, (255,0,0), font=font_probs)
    if (action[0] == "LEFT"):
        draw.text((105,24), action_displayed, (255,0,0), font=font_probs)
    if (action[0] == "RGHT"):
        draw.text((105,44), action_displayed, (255,0,0), font=font_probs)
    if (action[0] == "STOP"):
        draw.text((105,64), action_displayed, (255,0,0), font=font_probs)
    
    draw = ImageDraw.Draw(img)
    img.save(img_file_nm)    


# Next, we'll create a function that will get called whenever the camera's value changes.  This function will do the following steps

# 1. Pre-process the camera image
# 2. Execute the neural network
# 3. While the neural network output indicates we're blocked, we'll turn left, otherwise we go forward.
# 4. (Optional) - save each frame on the disk along with telementry for further analysis, PD controller tuning and debugging 

import torch.nn.functional as F
import time

# Simple PD controller (Kp - proportional term, Kd - derivative term)
Kp = 0.18 # 0.23 on Jetbot-43
Kd = 0.05

frwd_value = 0.3                      # Default value to drive forward (0 = no action, 1 = full motor capacity)
rot_value_when_exploring = 0.3        # Default value to spin to the right when robot is in exploration mode (0 = no action, 1 = full motor capacity)
min_prob_free_to_drive_frwd = 0.25    # Min probability prob_free for robot to drive forward 
min_prob_blocked = 0.25 # MJS: added
min_prob_to_stop = 0.5  # MJS: added
max_n_frames_stuck = 20               # Limit on the number of frames the robot is stuck for. Once this limit is reached, robot goes into exploration mode (makes large right turn)
frame_counter = 0                     # Frame counter 
n_frames_stuck = 0                    # Initialize counter of the number of successive frames the robot is stuck for
exploration = False                   # Initialize binary variable which determines if robot is in exploration mode (when True.) Used to mark the related frames with red background  
data_log = []                         # Initialize the array whcih will store a history of telemetry readings and robot actions (for analysis and tuning)
recent_detections = []                 # Initialize the array to store the last frame data

def update(change):
    global robot, frame_counter, n_frames_stuck, exploration
    x = change['new'] 
    x = preprocess(x)
    y = model(x)
    
    # apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
    y = F.softmax(y, dim=1)
    
    y = y.flatten()
   
    # extract probabilities of blocked, free, left and right
    prob_blocked = float(y[0])
    prob_free = float(y[1])
    prob_left = float(y[2])
    prob_right = float(y[3])
    prob_stop = float(y[4])
 
    # update list of recent detections
    while (len(recent_detections) >= 2):
        recent_detections.pop(0)
    recent_detections.append([prob_free,prob_left,prob_right,prob_blocked,prob_stop])
    
    # check if robot got stuck and update n_frames_stuck counter
    if prob_blocked > min_prob_blocked: # was if prob_free < min_prob_free_to_drive_frwd:
        n_frames_stuck = n_frames_stuck + 1 
    else:
        n_frames_stuck = 0
        
    # calculate errors at times t (current) and t-1 (prev)    
    # error(t) and error(t-1): prob_left-prob_right   
    if len(recent_detections) == 2:
        current_probs = recent_detections[1]
        prev_probs = recent_detections[0]
    else:
        current_probs = [prob_free,prob_left,prob_right,prob_blocked,prob_stop]
        prev_probs = current_probs
                
    # error = prob_left-prob_right        
    current_error = current_probs[1] - current_probs[2]
    prev_error    = prev_probs[1] - prev_probs[2]

    # increment frame counter 
    frame_counter = frame_counter + 1
    
    # define functions which deterine (and return) robot actions
    def forward(value):
        robot.forward(value)
        return ("FWRD",round(value,2))

    def left(value):
        robot.left(value)
        return ("LEFT",round(value,2))

    def right(value):
        robot.right(value)
        return ("RGHT",round(value,2))
    
    def stop():
        robot.stop()
        return ("STOP",0)
    
    def backup(value):
        robot.backward(value)
        return ("STOP",round(value,2))
    
    action = ""
  
    # estimate rotational value to turn left (if negative) or right (if positive)
    # 0 = no action, 1 = full motor capacity)
    rot_value = - Kp * current_error - Kd * (current_error - prev_error)
    
    # store propotional and differential controller components for frame-by-frame analysis
    p_component = - Kp * current_error
    d_component = - Kd * (current_error - prev_error)
    
    # initalize binary flag showinf if robot rotates 
    robot_rotates = False
    
    # action logic
    # moving forward if there is no obstacles
    if prob_free > min_prob_free_to_drive_frwd:
        action = forward(frwd_value)
        
    # MJS: add stop action
    elif prob_stop > min_prob_to_stop:
        action = stop()

    # turn left or right if robot is not blocked for a long time
    elif n_frames_stuck < max_n_frames_stuck:
        robot_rotates = True
        if rot_value < 0.0:
            action = left(-rot_value)
        else:
            action = right(rot_value)

    # activate exploration mode - robot turns right by a large (45-90 degree) angle if it failed to move forward for [max_n_frames_stuck] recent frames
    else:
        exploration = True
        robot_rotates = True
        # action = right(rot_value_when_exploring)
        action = backup(frwd_value)
        time.sleep(0.5)
        n_frames_stuck = 0
    
    time.sleep(0.001)
    
    # MJS: save every Nth frame only
    if False: # frame_counter % 100 == 0:

        # save frames - images with telemetry and robot actions data 
        save_frames_with_telemetry(change['new'],current_probs,exploration,action,n_frames_stuck)

        # append frame's telemetry and robot action to the stored data 
        if not robot_rotates:
            rot_value = 0.
            p_component = 0.
            d_component = 0.
        if robot_rotates and exploration:
            rot_value = rot_value_when_exploring
            p_component = 0.
            d_component = 0.

        last_frame_data = [frame_counter, round(prob_free,3), round(prob_left,3), round(prob_right,3), round(prob_blocked,3), round(prob_stop,3),
                               action[0], action[1], round(rot_value,3), round(p_component,3), round(d_component,3), n_frames_stuck, exploration]
        data_log.append(last_frame_data)

    # reset variables
    exploration = False
    robot_rotates = False

update({'new': camera.value})  # we call the function once to initialize
robot.stop()

# We've created our neural network execution function, but now we need to attach it to the camera for processing. 

# We accomplish that with the ``observe`` function.

# > WARNING: This code will move the robot!! Please make sure your robot has clearance.  The collision avoidance should work, but the neural
# > network is only as good as the data it's trained on!

camera.observe(update, names='value')  # this attaches the 'update' function to the 'value' traitlet of our camera

# Great! If your robot is plugged in it should now be generating new commands with each new camera frame.  Perhaps start by placing your robot on the ground and seeing what it does when it reaches the cones.

# To stop this behavior, unattach this callback by executing the code below.

# camera.unobserve(update, names='value')
# robot.stop()

import time
time.sleep(20)
camera.unobserve(update, names='value')
robot.stop()
