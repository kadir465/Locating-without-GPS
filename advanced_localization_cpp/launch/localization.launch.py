from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_yaml", default_value=""),
            DeclareLaunchArgument("map_path", default_value=""),
            DeclareLaunchArgument("map_gsd_m_per_px", default_value="0.2"),
            DeclareLaunchArgument("camera_topic", default_value="/camera/image_mono"),
            DeclareLaunchArgument("px4_imu_topic", default_value="/fmu/out/sensor_combined"),
            DeclareLaunchArgument("flow_max_side_px", default_value="1280"),
            Node(
                package="advanced_localization_cpp",
                executable="localization_node",
                name="advanced_localization_cpp",
                output="screen",
                parameters=[
                    {
                        "config_yaml": LaunchConfiguration("config_yaml"),
                        "map_path": LaunchConfiguration("map_path"),
                        "map_gsd_m_per_px": ParameterValue(LaunchConfiguration("map_gsd_m_per_px"), value_type=float),
                        "camera_topic": LaunchConfiguration("camera_topic"),
                        "px4_imu_topic": LaunchConfiguration("px4_imu_topic"),
                        "flow_max_side_px": ParameterValue(LaunchConfiguration("flow_max_side_px"), value_type=int),
                    }
                ],
            ),
        ]
    )
