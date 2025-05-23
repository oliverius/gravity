import math
import numpy as np
import matplotlib.pyplot as plt

class PlotG:
    def __init__(self, distance_cm: int, ball_diameter_cm:int, camera_tilt_deg: int, trajectory_px:tuple):
        positions_px, times = trajectory_px

        if len(positions_px) < 5:
            print("Not enough data points. Try checking your frame range or color mask.")
            exit()
        
        self.positions_px = positions_px
        self.times = times

        self.positions_m = self.pixel_to_meter_conversion(distance_cm, ball_diameter_cm, camera_tilt_deg)

        coeffs, g_estimate = self.fit()
        self.plot(coeffs, g_estimate)

    def pixel_to_meter_conversion(self, distance_cm: int, ball_diameter_cm:int, camera_tilt_deg: int) -> list:
        real_distance_m = (distance_cm - ball_diameter_cm) / 100

        start_px  = min(self.positions_px)
        end_px    = max(self.positions_px)
        total_px  = end_px - start_px

        # Correction for the camera position. 0 if not correction needed
        correction_factor = 1 / math.cos(math.radians(camera_tilt_deg))
        real_distance_m = real_distance_m * correction_factor

        pixel_to_meter = real_distance_m / total_px

        return [(p - start_px) * pixel_to_meter for p in self.positions_px]
    
    def fit(self):
        # === FIT y(t) = a*t² + b*t + c
        coeffs = np.polyfit(self.times, self.positions_m, 2)
        a = coeffs[0]
        g_estimate = 2 * a

        print(f"\nEstimated gravitational acceleration: {g_estimate:.4f} m/s²")
        return coeffs, g_estimate

    def plot(self, coeffs, g_estimate):
        t = np.array(self.times)
        y = np.array(self.positions_m)
        fit_y = np.polyval(coeffs, t)

        plt.figure()
        plt.plot(t, y, 'bo', label='Measured')
        plt.plot(t, fit_y, 'r-', label='Fitted parabola')
        
        plt.text(0.05, 0.80, f"g ≈ {g_estimate:.4f} m/s²",
            transform=plt.gca().transAxes,
            fontsize=12, color='black',
            verticalalignment='top')

        plt.xlabel('Time (s)')
        plt.ylabel('Vertical Position (m)')
        plt.title('Ball Drop Trajectory and Gravity Fit')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
