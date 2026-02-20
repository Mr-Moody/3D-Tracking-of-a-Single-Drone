"""Asynchronous Extended Kalman Filter for 3D drone tracking."""

from typing import Optional, Tuple, Union

import numpy as np

from src.camera import Camera


class DroneTrackerEKF():
    """9-state EKF: [x, y, z, vx, vy, vz, ax, ay, az].

    Fuses 2D detections from multiple cameras asynchronously.
    """

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 2.0,
    ):
        if initial_state is not None:
            np.asarray(initial_state, dtype=np.float64).ravel()
        else:
            self.state = np.zeros(9)
        
        if self.state.size != 9:
            raise ValueError("State must have 9 elements")
        
        self.state = self.state.reshape(9)

        if initial_covariance is not None:
            self.P = np.asarray(initial_covariance, dtype=np.float64)
        else:
            self.P = np.eye(9) * 100.0
        
        if self.P.shape != (9, 9):
            raise ValueError("Covariance must be 9x9")
        
        self.P = np.asarray(self.P, dtype=np.float64)

        self._q = process_noise_std**2
        self._r = measurement_noise_std**2
        self.R = np.eye(2) * self._r
        self.Q = self._build_process_noise(1.0)  # dt=1 for scaling; recomputed in predict

    def _build_F(self, dt: float) -> np.ndarray:
        """State transition matrix for constant-acceleration model."""
        dt2 = dt * dt
        
        F = np.eye(9)
        
        F[0, 3] = dt
        F[0, 6] = 0.5 * dt2
        F[1, 4] = dt
        F[1, 7] = 0.5 * dt2
        F[2, 5] = dt
        F[2, 8] = 0.5 * dt2
        F[3, 6] = dt
        F[4, 7] = dt
        F[5, 8] = dt
        
        return F

    def _build_process_noise(self, dt: float) -> np.ndarray:
        """Discrete process noise Q for constant-acceleration (discrete white noise)."""
        q = self._q
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        
        # Singer model with discrete white noise on acceleration
        Q = np.zeros((9, 9))
        
        Q[0, 0] = dt4 / 4
        Q[0, 3] = dt3 / 2
        Q[0, 6] = dt2 / 2
        Q[1, 1] = dt4 / 4
        Q[1, 4] = dt3 / 2
        Q[1, 7] = dt2 / 2
        Q[2, 2] = dt4 / 4
        Q[2, 5] = dt3 / 2
        Q[2, 8] = dt2 / 2
        Q[3, 0] = dt3 / 2
        Q[3, 3] = dt2
        Q[3, 6] = dt
        Q[4, 1] = dt3 / 2
        Q[4, 4] = dt2
        Q[4, 7] = dt
        Q[5, 2] = dt3 / 2
        Q[5, 5] = dt2
        Q[5, 8] = dt
        Q[6, 0] = dt2 / 2
        Q[6, 3] = dt
        Q[6, 6] = 1
        Q[7, 1] = dt2 / 2
        Q[7, 4] = dt
        Q[7, 7] = 1
        Q[8, 2] = dt2 / 2
        Q[8, 5] = dt
        Q[8, 8] = 1
        
        return Q * q

    def predict(self, dt: float) -> None:
        """Predict state and covariance forward by dt."""
        F = self._build_F(dt)
        Q = self._build_process_noise(dt)
        
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def update(self, camera: Camera, measurement_2d: Union[Tuple[float, float], np.ndarray], use_undistort: bool = False) -> None:
        """Update state with 2D measurement (u, v) from the given camera."""
        z = np.asarray(measurement_2d, dtype=np.float64).ravel()[:2]
        
        if use_undistort:
            z = np.array(camera.undistort_point(z[0], z[1]))
            
        pos = self.state[:3]
        
        z_pred = np.array(camera.project(pos))
        y = z - z_pred
        
        H_pos = camera.projection_jacobian(pos)
        
        H = np.zeros((2, 9))
        
        H[:, :3] = H_pos
        S = H @ self.P @ H.T + self.R
        
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        
        self.state = self.state + K @ y
        
        I_KH = np.eye(9) - K @ H
        
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def set_state(self, state: np.ndarray) -> None:
        """Set state vector [x,y,z,vx,vy,vz,ax,ay,az]."""
        self.state = np.asarray(state, dtype=np.float64).ravel()[:9].reshape(9)

    def set_covariance(self, P: np.ndarray) -> None:
        """Set covariance matrix (9x9)."""
        self.P = np.asarray(P, dtype=np.float64).reshape(9, 9)

    def get_position(self) -> np.ndarray:
        """Return current position (x, y, z)."""
        return self.state[:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Return current velocity (vx, vy, vz)."""
        return self.state[3:6].copy()
    
    def get_acceleration(self) -> np.ndarray:
        """Return current acceleration (ax, ay, az)."""
        return self.state[6:9].copy()
