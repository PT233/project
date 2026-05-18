# conftest.py — disable ROS launch_testing plugin which causes INTERNALERROR
collect_ignore_glob = []


def pytest_configure(config):
    """Unregister the problematic ROS launch_testing_ros plugin."""
    try:
        config.pluginmanager.unregister(
            name="launch_testing_ros_pytest_entrypoint"
        )
    except Exception:
        pass
