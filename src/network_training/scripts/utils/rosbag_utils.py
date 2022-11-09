import numpy as np
import pandas

# from geometry_msgs.msg import Twist, Pose
from scipy.spatial import KDTree

def print_bag_topics(bag):
    """Print all topic names in the bag file"""
    topics = bag.get_type_and_topic_info()[1].keys()
    types = []
    freqs = []
    for topic_tuple in bag.get_type_and_topic_info()[1].values():
        types.append((topic_tuple[0]))
        if (topic_tuple[3]) is None:
            freqs.append(10000)
        else:
            freqs.append((topic_tuple[3]))
    print("Available topics: ")
    print('-' * 30)
    for topic, _type, freq in zip(topics, types, freqs):
        print("'%s'   (msg_type=%s, frequency=%.2fHz)" %(topic, _type, freq))
    print('-' * 30)

def print_topic_info(topic_info):
    """Print message information"""
    # print('-'*30)
    for key, value in topic_info.items():
        print("%s: %s" % (key, value))
    print('-'*30)

def get_topic_duration(start_msg, end_msg):
    """Get the duration of a topic"""
    if not hasattr(start_msg, 'header') or not hasattr(end_msg, 'header'):
        return None
    start_time = start_msg.header.stamp.secs + float(start_msg.header.stamp.nsecs) * 1e-9
    end_time = end_msg.header.stamp.secs + float(end_msg.header.stamp.nsecs) * 1e-9
    return round(end_time - start_time, 2)

def get_all_fields(msg):
    """Get all fields from a msg"""
    fields = []
    exclude_list = ['deserialize', 'deserialize_numpy', 'serialize', 'serialize_numpy']
    for field in dir(msg):
        if not field.startswith('__') and not field.startswith('_') and field[0].islower() and field not in exclude_list:
            fields.append(field)
    return fields

def get_topic_from_bag(bag, topic, printout=True):
    """Read topic data into pandas.Dataframe format"""
    has_found = False
    # read msg and time
    for _, msg, bag_time in bag.read_messages(topics=topic):
        if not has_found:
            fields = get_all_fields(msg)
            data = {key: [] for key in fields}
            data['bag_time'] = []
            data['ros_time'] = []
            has_found = True

        data['bag_time'].append(bag_time.to_sec())
        use_bag_time = True
        if hasattr(msg, "header") :
            use_bag_time = False
            ros_time = msg.header.stamp.secs + float(msg.header.stamp.nsecs) * 1e-9
            if ros_time == 0: # if timestamp not updated properly
                use_bag_time = True
        if use_bag_time:
            data['ros_time'].append(data['bag_time'][-1]) # use bag_time if msg.header.stamp not found
        else:
            data['ros_time'].append(ros_time) # use msg.header.stamp

        for field in fields:
            data[field].append(getattr(msg, field))
        
    if not has_found:
        print("Topic '%s' does not exist in the bag file!" % topic)
        return None

    # calculate frequency
    dt_total = 0
    time_type = 'bag_time'
    for i in range(1, len(data[time_type])):
        dt_total += data[time_type][i] - data[time_type][i-1]
    freq = (len(data[time_type]) - 1) / dt_total
    data['frequency'] = round(freq, 2)

    if printout:
        topic_info = {
            "topic": topic,
            "fields": fields,
            "messages": len(data['bag_time']),
            'frequency': str(data['frequency']) + 'Hz',
        }
        print_topic_info(topic_info)

    return pandas.DataFrame(data)

def parse_pose_msg(data):
    """Parse pose msg"""
    pose = {
        'position_x': [],
        'position_y': [],
        'position_z': [],
        'orientation_w': [],
        'orientation_x': [],
        'orientation_y': [],
        'orientation_z': [],
    }
    for msg in data:
        pose['position_x'].append(msg.pose.position.x)
        pose['position_y'].append(msg.pose.position.y)
        pose['position_z'].append(msg.pose.position.z)

        pose['orientation_w'].append(msg.pose.orientation.w)
        pose['orientation_x'].append(msg.pose.orientation.x)
        pose['orientation_y'].append(msg.pose.orientation.y)
        pose['orientation_z'].append(msg.pose.orientation.z)

    return pandas.DataFrame(pose)

def parse_twist_msg(data):
    """Parse twist msg"""
    twists = {
        'angular_x': [],
        'angular_y': [],
        'angular_z': [],
        'linear_x': [],
        'linear_y': [],
        'linear_z': [],
    }
    for msg in data:
        twists['angular_x'].append(msg.twist.angular.x)
        twists['angular_y'].append(msg.twist.angular.y)
        twists['angular_z'].append(msg.twist.angular.z)

        twists['linear_x'].append(msg.twist.linear.x)
        twists['linear_y'].append(msg.twist.linear.y)
        twists['linear_z'].append(msg.twist.linear.z)

    return pandas.DataFrame(twists)

def timesync_topics(topic_list, force_use_first=True, printout=True):
    """
    Time Synchronize different topics
    
    topic_list = [topic1, topic2, ...] and each topic is in pandas.Dataframe format
    """
    assert len(topic_list) > 1, "The number of synchronized topics should be larger than one!"

    base_index = 0
    min_freq = topic_list[0]['frequency'][0]
    max_freq = min_freq

    # find the lowest and highest frequencies from all topics
    for i in range(1, len(topic_list)):
        if topic_list[i]['frequency'][0] <  min_freq:
            base_index = i
            min_freq = topic_list[i]['frequency'][0]
        
        if topic_list[i]['frequency'][0] >  max_freq:
            max_freq = topic_list[i]['frequency'][0]
        
    freq_diff = max_freq - min_freq
    if freq_diff > 5 and printout:
        print('Warning: The maximum and minimum frequencies differ by %d Hz. Use with caution!' % (freq_diff))

    # if force to use the first topic as the base
    if force_use_first:
        base_index = 0
        min_freq = topic_list[0]['frequency'][0]

    if printout:
        print("Use the %d column as the base. The synchronized frequency is %.0fHz." % (base_index, min_freq))

    # query timestamp
    time_query = topic_list[base_index]['ros_time'].to_numpy().reshape(-1, 1)

    # KD Tree to find the closest timstamp
    topic_list_sync = []
    for i, topic in enumerate(topic_list):
        if i == base_index:
            topic_list_sync.append(topic)
            continue
        time_kd = KDTree(topic['ros_time'].to_numpy().reshape(-1, 1))
        _, query_idx = time_kd.query(time_query)
        topic_sync = topic.iloc[query_idx].copy() # copy() is needed to avoid warning msg
        topic_sync['new_index'] = np.arange(len(topic_sync)) # reset index in pandas dataframe
        topic_sync.set_index('new_index', inplace=True, drop=True)
        topic_sync['ros_time'] = time_query.copy()
        topic_list_sync.append(topic_sync)
        
    return time_query, topic_list_sync

def find_freq_sweep_start_end_time(rosout_msgs):
    """Find frequency sweep start and end time"""
    start_time = None
    end_time = None
    for i in range(len(rosout_msgs)):
        if rosout_msgs['msg'].iloc[i] == "Start frequency sweep":
            start_time = rosout_msgs['header'].iloc[i].stamp.secs + float(rosout_msgs['header'].iloc[i].stamp.nsecs) * 1e-9
        if rosout_msgs['msg'].iloc[i] == "End of frequency sweep":
            end_time = rosout_msgs['header'].iloc[i].stamp.secs + float(rosout_msgs['header'].iloc[i].stamp.nsecs) * 1e-9
    return start_time, end_time

def find_freq_sweep_start_end_time_from_command(command_msgs):
    """Find frequency sweep start and end time from command"""
    start_time = None
    end_time = None
    speed_cmd = command_msgs['linear_speed']
    last_cmd = speed_cmd.iloc[0]

    index = []
    for i in range(len(speed_cmd)):
        if speed_cmd.iloc[i] != last_cmd:
            index.append(i)
            last_cmd = speed_cmd.iloc[i]
    
    start_time = command_msgs['ros_time'].iloc[index[0]]
    end_time = command_msgs['ros_time'].iloc[index[1]]
    
    return start_time, end_time

def find_doublet_start_end_time(rosout_msgs):
    """Find doublet start and end time"""
    start_time = None
    end_time = None
    for i in range(len(rosout_msgs)):
        if rosout_msgs['msg'].iloc[i] == "Start doublet":
            start_time = rosout_msgs['header'].iloc[i].stamp.secs + float(rosout_msgs['header'].iloc[i].stamp.nsecs) * 1e-9
        if rosout_msgs['msg'].iloc[i] == "End of doublet":
            end_time = rosout_msgs['header'].iloc[i].stamp.secs + float(rosout_msgs['header'].iloc[i].stamp.nsecs) * 1e-9
    return start_time, end_time

def find_doublet_start_end_time_from_command(command_msgs):
    """Find doublet start and end time"""
    start_time = None
    end_time = None
    speed_cmd = command_msgs['linear_speed']
    last_cmd = speed_cmd.iloc[0]

    index = []
    for i in range(len(speed_cmd)):
        if speed_cmd.iloc[i] != last_cmd:
            index.append(i)
            last_cmd = speed_cmd.iloc[i]
    
    start_time = command_msgs['ros_time'].iloc[index[0]]
    end_time = command_msgs['ros_time'].iloc[index[1]]
    
    return start_time, end_time

def crop_data_with_start_end_time(topic_msgs, start_time, end_time):
    """Crop data using start and end time"""
    start_index = None
    end_index = None

    if start_time > topic_msgs['ros_time'].iloc[-1] or end_time < topic_msgs['ros_time'].iloc[0]:
        print('No data was found in the time range. Return None.')
        return None

    for i in range(len(topic_msgs)):
        if start_index is None and topic_msgs['ros_time'].iloc[i] >= start_time:
            start_index = i
        
        if end_index is None and topic_msgs['ros_time'].iloc[i] >= end_time:
            end_index = i - 1

    topic_msg_crop = topic_msgs.loc[start_index:end_index].copy()
    topic_msg_crop.reset_index(drop=True)
    topic_msg_crop[''] = np.arange(len(topic_msg_crop)) # reset index in pandas dataframe
    topic_msg_crop.set_index('', inplace=True, drop=True)

    return topic_msg_crop

def get_linear_acceleration_from_imu(imu_msgs):
    """Get linear acceleration from IMU msgs"""
    acc_x = []
    acc_y = []
    acc_z = []
    for accel in imu_msgs['linear_acceleration']:
        acc_x.append(accel.x)
        acc_y.append(accel.y)
        acc_z.append(accel.z)

    acc_x = np.array(acc_x)
    acc_y = np.array(acc_y)
    acc_z = np.array(acc_z)

    return acc_x, acc_y, acc_z

def get_angular_velocity_from_imu(imu_msgs):
    """Get angular velocity from IMU msgs"""
    gyro_x = []
    gyro_y = []
    gyro_z = []
    for gyro in imu_msgs['angular_velocity']:
        gyro_x.append(gyro.x)
        gyro_y.append(gyro.y)
        gyro_z.append(gyro.z)

    gyro_x = np.array(gyro_x)
    gyro_y = np.array(gyro_y)
    gyro_z = np.array(gyro_z)

    return gyro_x, gyro_y, gyro_z

def get_linear_x_from_odom(odom_msgs):
    """Get linear x from Odometry msg"""
    linear_x = []
    for msg in odom_msgs['twist']:
        linear_x.append(msg.twist.linear.x)

    linear_x = np.array(linear_x)
    
    return linear_x
