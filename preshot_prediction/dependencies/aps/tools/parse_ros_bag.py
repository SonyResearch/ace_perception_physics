#!/usr/bin/env python3
"""
Script to parse ROS2 bag files and extract message data.
Supports various message types and export formats.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from numpy import tri
import yaml

import multi_ball_triangulation.python_module as multi_ball_triangulation
import triangulation.python_module as triangulation
import triangulation.datalogger_pybind as triangulation_logger


try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    print(f"Error: Missing ROS2 dependencies. Please install rosbag2_py and rclpy.")
    print(f"  sudo apt install ros-<distro>-rosbag2-py python3-rclpy")
    sys.exit(1)


class ROS2BagParser:
    """Parser for ROS2 bag files."""
    
    def __init__(self, bag_path: str):
        """
        Initialize the parser with a bag file path.
        
        Args:
            bag_path: Path to the ROS2 bag directory
        """
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag file not found: {bag_path}")
        
        self.reader = SequentialReader()
        storage_options = StorageOptions(uri=str(self.bag_path), storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        self.reader.open(storage_options, converter_options)
        
        # Get topic metadata
        self.topic_types = self.reader.get_all_topics_and_types()
        self.topic_metadata = {t.name: t for t in self.topic_types}
        
    def get_topics(self) -> List[str]:
        """Get list of all topics in the bag."""
        return [t.name for t in self.topic_types]
    
    def get_topic_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all topics."""
        info = {}
        for topic in self.topic_types:
            info[topic.name] = {
                'type': topic.type,
                'serialization_format': topic.serialization_format,
            }
        return info
    
    def read_messages(self, topics: Optional[List[str]] = None, 
                     start_time: Optional[int] = None,
                     end_time: Optional[int] = None,
                     max_count: Optional[int] = None):
        """
        Read messages from the bag file.
        
        Args:
            topics: List of topic names to read. If None, read all topics.
            start_time: Start timestamp (nanoseconds). If None, start from beginning.
            end_time: End timestamp (nanoseconds). If None, read until end.
            max_count: Maximum number of messages to read. If None, read all.
            
        Yields:
            Tuple of (topic_name, message, timestamp)
        """
        # Set filter for topics if specified
        if topics:
            from rosbag2_py import StorageFilter
            storage_filter = StorageFilter(topics=topics)
            self.reader.set_filter(storage_filter)
        
        # Reset to beginning
        self.reader.reset_filter()
        if topics:
            from rosbag2_py import StorageFilter
            storage_filter = StorageFilter(topics=topics)
            self.reader.set_filter(storage_filter)
        
        count = 0
        while self.reader.has_next():
            if max_count and count >= max_count:
                break
                
            (topic, data, timestamp) = self.reader.read_next()
            
            # Apply time filters
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            
            # Deserialize message
            msg_type = self.topic_metadata[topic].type
            msg_class = get_message(msg_type)
            msg = deserialize_message(data, msg_class)
            
            yield topic, msg, timestamp
            count += 1
    
    def extract_to_dict(self, topics: Optional[List[str]] = None,
                       max_count: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Extract messages to a dictionary format.
        
        Args:
            topics: List of topic names to extract. If None, extract all.
            max_count: Maximum messages per topic.
            
        Returns:
            Dictionary mapping topic names to lists of message dictionaries
        """
        data = {}
        
        for topic, msg, timestamp in self.read_messages(topics=topics, max_count=max_count):
            if topic not in data:
                data[topic] = []
            
            msg_dict = {
                'timestamp': timestamp,
                'data': self._message_to_dict(msg)
            }
            data[topic].append(msg_dict)
        
        return data
    
    def _message_to_dict(self, msg) -> Dict[str, Any]:
        """Convert a ROS message to a dictionary."""
        result = {}
        
        # Get all slots (fields) of the message
        if hasattr(msg, 'get_fields_and_field_types'):
            fields = msg.get_fields_and_field_types().keys()
        elif hasattr(msg, '__slots__'):
            fields = msg.__slots__
        else:
            return str(msg)
        
        for field in fields:
            if not hasattr(msg, field):
                continue
                
            value = getattr(msg, field)
            
            # Handle nested messages
            if hasattr(value, '__slots__') or hasattr(value, 'get_fields_and_field_types'):
                result[field] = self._message_to_dict(value)
            # Handle lists
            elif isinstance(value, (list, tuple)):
                result[field] = [
                    self._message_to_dict(item) if hasattr(item, '__slots__') 
                    or hasattr(item, 'get_fields_and_field_types')
                    else item
                    for item in value
                ]
            # Handle bytes
            elif isinstance(value, bytes):
                result[field] = list(value)
            else:
                result[field] = value
        
        return result
    
    def export_to_json(self, output_path: str, topics: Optional[List[str]] = None,
                      max_count: Optional[int] = None):
        """Export bag data to JSON file."""
        data = self.extract_to_dict(topics=topics, max_count=max_count)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Exported to {output_path}")
    
    def export_to_yaml(self, output_path: str, topics: Optional[List[str]] = None,
                      max_count: Optional[int] = None):
        """Export bag data to YAML file."""
        data = self.extract_to_dict(topics=topics, max_count=max_count)
        
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print(f"Exported to {output_path}")

    def export_to_ace(self, output_path: str):
        """Export bag data to YAML file."""
        triangulation_topic="/sensors/ball_triangulation/points"
        topics=[triangulation_topic]
        data = self.extract_to_dict(topics=topics)
        writer=triangulation_logger.DataWriter(output_path,"","")
        import numpy as np


        for msg in data[triangulation_topic]:
            sequence_number=msg["data"]["header"]["sequence_number"]
            stamp_sec=msg["data"]["header"]["stamp"]["sec"]
            stamp_nanosec=msg["data"]["header"]["stamp"]["nanosec"]
            triangulated_points=[]
            for p in msg["data"]["points"]:
                tp=triangulation.TriangulatedPoint()
                tp.position=[p["position"]["x"], p["position"]["y"], p["position"]["z"]]
                tp.covariance=np.array(p["covariance"]).reshape(3,3)
                triangulated_points.append(tp)
            multi_ball_triangulation.write_to_logger(writer,sequence_number,stamp_sec,stamp_nanosec,triangulated_points)

    
    def print_summary(self):
        """Print a summary of the bag contents."""
        print(f"\nBag file: {self.bag_path}")
        print("\nTopics:")
        print("-" * 80)
        
        for topic in self.topic_types:
            print(f"  {topic.name}")
            print(f"    Type: {topic.type}")
            print(f"    Serialization: {topic.serialization_format}")
        
        print("\n" + "-" * 80)
        
        # Count messages per topic
        print("\nMessage counts:")
        topic_counts = {}
        for topic, _, _ in self.read_messages():
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        for topic, count in sorted(topic_counts.items()):
            print(f"  {topic}: {count}")
    
    def close(self):
        """Close the bag reader."""
        del self.reader

def convert_ros2_to_acelog(path: str, output_path: str):
    """
    Convert ROS2 bag file to AceLog format.
    """
    bag_parser = ROS2BagParser(path)
    data = bag_parser.extract_to_dict()
    
    
    print(f"Converted ROS2 bag to AceLog format at {output_path}")
    bag_parser.close()

def process_bag(path: str, output_path:str, args):

    try:
        # Initialize parser
        bag_parser = ROS2BagParser(path)
        
        # List topics
        if args.list_topics:
            print("\nAvailable topics:")
            for topic in bag_parser.get_topics():
                info = bag_parser.get_topic_info()[topic]
                print(f"  {topic} ({info['type']})")
            return
        
        # Print summary
        # if args.summary or not output_path:
        #     bag_parser.print_summary()
        
        # Export to file
        if output_path:
            if args.format == 'json' or output_path.endswith('.json'):
                bag_parser.export_to_json(output_path, topics=args.topics, 
                                         max_count=args.max_count)
            elif args.format == 'yaml' or output_path.endswith(('.yaml', '.yml')):
                bag_parser.export_to_yaml(output_path, topics=args.topics,
                                         max_count=args.max_count)
            elif args.format == 'ace' or output_path.endswith(('.ace', '.ace')):
                bag_parser.export_to_ace(output_path)
            else:
                print(f"Warning: Unknown format, defaulting to JSON")
                bag_parser.export_to_json(output_path, topics=args.topics,
                                         max_count=args.max_count)
        
        bag_parser.close()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Parse and extract data from ROS2 bag files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print bag summary
  %(prog)s /path/to/bag

  # Export all topics to JSON
  %(prog)s /path/to/bag -o output.json

  # Export specific topics to YAML
  %(prog)s /path/to/bag -t /camera/image /imu/data -o output.yaml

  # Export with message limit
  %(prog)s /path/to/bag -o output.json --max-count 100

  # List topics only
  %(prog)s /path/to/bag --list-topics
        """
    )
    
    parser.add_argument('bag_path', help='Path to ROS2 bag directory')
    parser.add_argument('-o', '--output', help='Output file path (JSON or YAML)')
    parser.add_argument('-t', '--topics', nargs='+', help='Specific topics to extract')
    parser.add_argument('--list-topics', action='store_true', 
                       help='List all topics and exit')
    parser.add_argument('--max-count', type=int, 
                       help='Maximum number of messages to process per topic')
    parser.add_argument('--format', choices=['json', 'yaml','ace'], default='ace',
                       help='Output format (default: ace)')
    parser.add_argument('--summary', action='store_true',
                       help='Print bag summary')
    
    parser.add_argument("--crawl",action='store_true', 
                       help='crawl folder for bags')
    
    args = parser.parse_args()
    
    if args.crawl:
        import os
        end="_ladybug/rosbag_augmented/rosbag_augmented_0.db3"
        for root, _, files in os.walk(args.bag_path):
            for file in files:
                bag_path=os.path.join(root,file)
                if bag_path.endswith(end):
                    print(f"Processing bag: {bag_path}")
                    stripped_filename=bag_path.replace(end,"")
                    game_name=os.path.basename(stripped_filename)
                    date=os.path.basename(os.path.dirname(stripped_filename))
                    output_path=f"{date}_{game_name}_multiball_triangulation.ace"

                    output_path=os.path.join(args.output,output_path)

                    process_bag(bag_path, output_path, args)
    else:
        process_bag(args.bag_path, args.output, args)

if __name__ == '__main__':
    main()
