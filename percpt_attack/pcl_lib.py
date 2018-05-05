import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2


class Msger:
    
    def __init__(self):
        rospy.init_node('pcl2_pub_example')
        self.pcl_pub = rospy.Publisher("/my_pcl_topic", PointCloud2,queue_size=1)
        rospy.loginfo("Initializing pcl2 publisher node...")
        #give time to roscore to make the connections
        rospy.sleep(1.)

    def publish_pcl_msg(self, pcl):
        cloud_points =[p[:3] for p in pcl]
        #header
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        #create pcl from points
        scaled_polygon_pcl = pcl2.create_cloud_xyz32(header, cloud_points)
        #publish    
        rospy.loginfo("publishing sample pointcloud.. !")
        self.pcl_pub.publish(scaled_polygon_pcl)
        rospy.sleep(0.1)