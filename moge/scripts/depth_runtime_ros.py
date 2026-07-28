import faulthandler
faulthandler.enable()
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from threading import Lock
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import Image

class MoGeRuntimeROS(Node):
    def __init__(self):
        super().__init__("MoGe_runtime_ros_node")
        self.declare_parameters(namespace='',
                                parameters=[
                                    ("img_topic_name", "/cer/realsense_repeater/color_image"),
                                    ("synth_depth_topic_name", "/MoGe/depth"),
                                    ("device", "cuda"),
                                    ("token_number", 3000)
                                ])
        self.mutex = Lock() # Could be ignored since we have only a callback
        self.img_topic_name = self.get_parameter("img_topic_name").value
        self.synth_depth_topic_name = self.get_parameter("synth_depth_topic_name").value
        device = self.get_parameter("device").value
        # The model range is [1200, 3600]. The default resolution_level=9 uses the max (3600) number of tokens. 
        # More the tokens, more accurate and slower is the model
        self.token_number = self.get_parameter("token_number").value
        
        self.img_sub = self.create_subscription(Image, self.img_topic_name, self.img_callback, 10)
        self.depth_pub = self.create_publisher(Image, self.synth_depth_topic_name, 10)
        self.rgb_pub = self.create_publisher(Image, "/MoGe/rgb", 10)

        self.device = torch.device(device)
        self.moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(self.device)
        
    def img_callback(self, msg : Image):
        with self.mutex:
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            rgb = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0  # (3, H, W) float32 in [0, 1]
            output = self.moge.infer(rgb, num_tokens=self.token_number)
            depth = output["depth"].cpu().numpy().astype(np.float32)  # (H, W) metric scale, invalid pixels are inf
            depth_msg = Image()
            depth_msg.header = msg.header
            depth_msg.height = msg.height
            depth_msg.width = msg.width
            depth_msg.encoding = "32FC1"
            depth_msg.is_bigendian = 0
            depth_msg.step = msg.width * 4
            depth_msg.data = depth.tobytes()
            self.depth_pub.publish(depth_msg)
            self.rgb_pub.publish(msg)

def main():
    rclpy.init()
    node = MoGeRuntimeROS()
    print(f"Created {node.get_name()}")
    rclpy.spin(node)
    print(f"Shutting down...")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
