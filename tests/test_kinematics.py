import numpy as np

from seima_mocap.kinematics import JointKinematicsEstimator, angle_3d


def test_angle_3d_right_angle():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert np.isclose(angle_3d(a, b, c), 90.0)


def test_angle_3d_straight_line():
    a = np.array([-1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    assert np.isclose(angle_3d(a, b, c), 180.0)


def test_joint_estimator_right_elbow():
    positions = {
        12: np.array([0.0, 1.0, 0.0]),  # right shoulder
        14: np.array([0.0, 0.0, 0.0]),  # right elbow
        16: np.array([1.0, 0.0, 0.0]),  # right wrist
    }
    estimator = JointKinematicsEstimator(alpha=1.0)
    result = estimator.update(positions, timestamp_s=0.0)
    assert "right_elbow" in result
    assert np.isclose(result["right_elbow"].angle_deg, 90.0)
    assert np.isclose(result["right_elbow"].angular_velocity_deg_s, 0.0)
