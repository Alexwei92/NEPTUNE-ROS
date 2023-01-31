import os
import sys
import rosbag
import tqdm

frame_to_keep = [
    't265_odom_frame',
    't265_pose_frame',
    't265_link',
    'd435i_link',
    'd435i_color_frame',
    'd435i_depth_frame',
    'd435i_depth_optical_frame',
    'd435i_aligned_depth_to_color_frame',
    'd435i_color_optical_frame',
]

def filter_tf(in_bag, out_bag, frame_to_keep):
    print("Reading from " + in_bag)
    bag = rosbag.Bag(in_bag, 'r')
    with rosbag.Bag(out_bag, 'w') as outbag:
        print("Writing to " + out_bag)
        for topic, msg, t in bag.read_messages():
            if topic == '/tf' and msg.transforms:
                transforms_to_keep = []
                for i in range(len(msg.transforms)):
                    if msg.transforms[i].header.frame_id in frame_to_keep:
                        transforms_to_keep.append(msg.transforms[i])
                    
                msg.transforms = transforms_to_keep
                outbag.write('/tf', msg, t)

            elif topic == '/tf_static' and msg.transforms:
                transforms_to_keep = []
                for i in range(len(msg.transforms)):
                    if msg.transforms[i].header.frame_id in frame_to_keep:
                        transforms_to_keep.append(msg.transforms[i])
                    
                msg.transforms = transforms_to_keep
                outbag.write('/tf_static', msg, t)

            else:
                outbag.write(topic, msg, t)


if __name__ == "__main__":
    in_bag = sys.argv[1]
    out_bag = sys.argv[2]

    try:
        filter_tf(in_bag, out_bag, frame_to_keep)
        print('Successful!')
    except Exception as e:
        print(e)

