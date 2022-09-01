#!/usr/bin/python3

import numpy as np
import torch, cv2, os, rospy
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.ros import create_humans_detection_msg
from sensor_msgs.msg import Image
from yolov7_ros.msg import HumansStamped, Human
from cv_bridge import CvBridge

def process_img_msg(image, args):
    model = args[0]
    device = args[1]
    visualize = args[2]
    queue_size = args[3]
    out_topic = args[4]
    skeleton_keypoints_out_topic = args[5]
    """ callback function for publisher """
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(image, "bgr8")    
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)   
    with torch.no_grad():
        output, _ = model(image)
        
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    #Visualization
    vis_topic = out_topic + "visualization" if out_topic.endswith("/") else out_topic + "/visualization"
    visualization_publisher = rospy.Publisher(vis_topic, Image, queue_size=queue_size  ) if visualize else None
    vis_msg = bridge.cv2_to_imgmsg(nimg)
    visualization_publisher.publish(vis_msg)

    #Keypoints Publisher
    skeleton_detection_publisher = rospy.Publisher(skeleton_keypoints_out_topic, HumansStamped, queue_size=queue_size)

    keypoints_array_msg =create_humans_detection_msg(output)

    skeleton_detection_publisher.publish(keypoints_array_msg)



def main():
    rospy.init_node("yolov7_human_pose")

    ns = rospy.get_name() + "/"

    weights_path = rospy.get_param(ns + "weights_path")
    img_topic = rospy.get_param(ns + "img_topic")
    out_topic = rospy.get_param(ns + "out_topic")
    skeleton_keypoints_out_topic = rospy.get_param(ns + "skeleton_keypoints_out_topic")
    queue_size = rospy.get_param(ns + "queue_size")
    visualize = rospy.get_param(ns + "visualize")
    
    #PARAMS TBD
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    weigths = torch.load(weights_path, map_location=device)
    model = weigths['model']
    _ = model.float().eval()
    if torch.cuda.is_available():
        model.half().to(device)
    
    callback_args = (model, device, visualize, queue_size, out_topic, skeleton_keypoints_out_topic)
    
    #Subscribe to Image
    rospy.Subscriber(img_topic, Image, process_img_msg, callback_args, queue_size)
    
    
    rospy.spin()


if __name__ == "__main__":
    main()









