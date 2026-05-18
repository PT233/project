"""
Helper script to run pytest while stripping ROS paths from sys.path
to avoid the launch_testing_ros plugin INTERNALERROR.
"""
import sys
import os

# Remove ROS-related paths from sys.path before pytest loads plugins
ros_paths_to_remove = [p for p in sys.path if 'ros' in p.lower() or 'ros2' in p.lower()]
for p in ros_paths_to_remove:
    sys.path.remove(p)

# Also remove from PYTHONPATH env var if set
pythonpath = os.environ.get('PYTHONPATH', '')
if pythonpath:
    parts = [p for p in pythonpath.split(':') if 'ros' not in p.lower() and 'ros2' not in p.lower()]
    os.environ['PYTHONPATH'] = ':'.join(parts)

# Add project root to path
project_root = '/mnt/h/1.MasterDegree/17.RD/project'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
sys.exit(pytest.main(sys.argv[1:]))
