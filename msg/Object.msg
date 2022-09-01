# Object label
string label

# Object label ID
int16 label_id

# Object sub
string sublabel

# Object confidence level (1-99)
float32 confidence

# Object centroid position
float32[3] position

# Position covariance
float32[6] position_covariance

# 2D Bounding box projected to Camera image
yolov7_ros/BoundingBox2Di bounding_box_2d

# 3D dimensions (width, height, lenght)
float32[3] dimensions_3d

# Is skeleton available?
bool skeleton_available

# 2D Bounding box projected to Camera image of the person head
yolov7_ros/BoundingBox2Df head_bounding_box_2d

# 2D Person skeleton projected to Camera image
yolov7_ros/Skeleton2D skeleton_2d

