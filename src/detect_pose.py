#!/usr/bin/python3

import string
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
from torchvision.transforms import ToTensor

class YoloV7_HPE:
    def __init__(self, weights: string ="../weights/yolov7-w6-pose.pt", device: str = "cuda"):
        #PARAMS TBD
        self.device = device
        self.weigths = torch.load(weights, map_location=device)
        self.model = self.weigths['model']
        _ = self.model.float().eval()
        if torch.cuda.is_available():
            self.model.half().to(device)
    

class Yolov7_HPEPublisher:
    def __init__(self, visualize: bool, device: string, queue_size: int, img_topic: str = "/my_camera_topic", weights: str = "../weights/yolov7-w6-pose.pt",
                out_img_topic: str ="yolov7_hpe", skeleton_keypoints_out_topic: str = "yolov7_hpe_skeletons"):
        """
        :param weights: path/to/yolo_weights.pt
        :param img_topic: name of the image topic to listen to
        :param out_img_topic: topic for visualization will be published under the
            namespace '/yolov7')
        :param skeleton_keypoints_out_topic: intersection over union threshold will be published under the
            namespace '/yolov7')
        :param device: device to do inference on (e.g., 'cuda' or 'cpu')
        :param queue_size: queue size for publishers
        :visualize: flag to enable publishing the detections visualized in the image
        """
        self.tensorize = ToTensor()
        self.model = YoloV7_HPE( weights = weights, device = device)
        self.bridge = CvBridge()
        
        #Subscribe to Image
        self.img_subscriber = rospy.Subscriber(img_topic, Image, self.process_img_msg)
        
        #Visualization Publisher
        self.out_img_topic = out_img_topic + "visualization" if out_img_topic.endswith("/") else out_img_topic + "/visualization"
        self.visualization_publisher = rospy.Publisher(self.out_img_topic, Image, queue_size=queue_size) if visualize else None
        
        #Keypoints Publisher
        self.skeleton_keypoints_out_topic = skeleton_keypoints_out_topic + "visualization" if out_img_topic.endswith("/") else skeleton_keypoints_out_topic + "/visualization"
        self.skeleton_detection_publisher = rospy.Publisher(skeleton_keypoints_out_topic, HumansStamped, queue_size=queue_size)

        rospy.loginfo("YOLOv7 initialization complete. Ready to start inference")

    def process_img_msg(self, image: Image):
        """ callback function for publisher """
        image = self.bridge.imgmsg_to_cv2(image, "bgr8")    
        image = letterbox(image, 960, stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        if torch.cuda.is_available():
            image = image.half().to(device)   
        with torch.no_grad():
            output, _ = self.model.model(image)
            
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.model.yaml['nc'], nkpt=self.model.model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
       
        #Publishing Keypoints
        if (len(output) > 0):
            keypoints_array_msg = create_humans_detection_msg(output)
            self.skeleton_detection_publisher.publish(keypoints_array_msg)
        

        #Publishing Visualization if Required
        if self.visualization_publisher:
            vis_msg = self.bridge.cv2_to_imgmsg(nimg)
            self.visualization_publisher.publish(vis_msg)


if __name__ == "__main__":
    rospy.init_node("yolov7_human_pose")

    ns = rospy.get_name() + "/"

    weights = rospy.get_param(ns + "weights")
    img_topic = rospy.get_param(ns + "img_topic")
    out_img_topic = rospy.get_param(ns + "out_img_topic")
    skeleton_keypoints_out_topic = rospy.get_param(ns + "skeleton_keypoints_out_topic")
    queue_size = rospy.get_param(ns + "queue_size")
    visualize = rospy.get_param(ns + "visualize")
    device = rospy.get_param(ns + "device")

    # some sanity checks
    if not os.path.isfile(weights):
        raise FileExistsError("Weights not found.")

    if not ("cuda" in device or "cpu" in device):
        raise ValueError("Check your device.")

    publisher = Yolov7_HPEPublisher(
        img_topic=img_topic,
        out_img_topic=out_img_topic, 
        skeleton_keypoints_out_topic = skeleton_keypoints_out_topic,
        weights=weights,
        device=device,
        visualize=visualize,
        queue_size=queue_size
    )
    
    rospy.spin()









