import numpy as np
from latline.wavelets import even_wavelet, odd_wavelet

class Sphere(object):
    """
    This class implements a simulated sphere. it includes the functions to compute the next position in the simulation
    and to compute the corresponding fluid velocities as measured by the lateral line sensors.
    """

    def __init__(self, cfg):
        """
        Initializes a sphere.
        :param cfg: Experiment config object
        """
        self.x = np.random.uniform(cfg.x_range[0], cfg.x_range[1])
        self.y = np.random.uniform(cfg.y_range[0], cfg.y_range[1])
        self.z = np.random.uniform(cfg.z_range[0], cfg.z_range[1] * 2 / 3)

        self.d_theta_range = cfg.d_theta_range
        self.v = cfg.v

        self.x_r = cfg.x_range
        self.y_r = cfg.y_range
        self.z_r = np.array(cfg.z_range) * 2 / 3
        self.phi = np.random.uniform(0, 2 * np.pi)
        self.theta = np.random.uniform(0, 2 * np.pi)
        self.s = np.linspace(cfg.sensor_range[0], cfg.sensor_range[1], cfg.N_sensors)
        self.a = cfg.a

        self.fluid_v = None

    def get_velocity(self):

        vx = self.v * np.cos(self.theta) * np.sin(self.phi)
        vy = self.v * np.sin(self.theta) * np.sin(self.phi)
        vz = self.v * np.cos(self.phi)

        return vx, vy, vz


    def compute_fluid_v(self):
        '''
        Computes current fluid velocity that is measured by the grid
        :return: Fluid velocity (array)
        '''

        vx = self.v * np.cos(self.theta) * np.sin(self.phi)
        vy = self.v * np.sin(self.theta) * np.sin(self.phi)
        vz = self.v * np.cos(self.phi)

        v_vec = np.asarray([vx, vy, vz])

        y1 = self.y - (-.5)
        y2 = self.y - .5

        y1z = np.asarray([0, y1, self.z])
        v_orth_x1 = np.dot(v_vec, y1z) / np.linalg.norm(y1z)

        y2z = np.asarray([0, y2, self.z])
        v_orth_x2 = np.dot(v_vec, y2z) / np.linalg.norm(y2z)

        phi_x1 = np.arctan2(v_orth_x1, vx)
        phi_x2 = np.arctan2(v_orth_x2, vx)

        even_psi_x1 = even_wavelet(self.s, self.x, np.sqrt(y1 ** 2 + self.z ** 2))
        odd_psi_x1 = odd_wavelet(self.s, self.x, np.sqrt(y1 ** 2 + self.z ** 2))

        even_psi_x2 = even_wavelet(self.s, self.x, np.sqrt(y2 ** 2 + self.z ** 2))
        odd_psi_x2 = odd_wavelet(self.s, self.x, np.sqrt(y2 ** 2 + self.z ** 2))

        Wx1 = np.sqrt(v_orth_x1 ** 2 + vx ** 2)
        Wx2 = np.sqrt(v_orth_x2 ** 2 + vx ** 2)

        return (Wx1 * self.a**3 * (odd_psi_x1 * np.sin(phi_x1) - even_psi_x1 * np.cos(phi_x1)) / np.linalg.norm(y1z) ** 3,
                Wx2 * self.a**3 * (odd_psi_x2 * np.sin(phi_x2) - even_psi_x2 * np.cos(phi_x2)) / np.linalg.norm(y2z) ** 3)

    def checkEdges(self, oldx, oldy, oldz):
        '''
        This function mirrors a sphere when it is going to beyond the borders
        :param oldx: old x location
        :param oldy: old y location
        :return: mirrored 'old' x- and y-locations
        '''
        def mirror_plus(p1, p2, b):
            p2_star = b - (p2 - b)
            p1_star = b + b - p1
            return p1_star, p2_star

        def mirror_min(p1, p2, a):
            p2_star = a + a - p2
            p1_star = a - (p1 - a)
            return p1_star, p2_star

        if self.x + self.a > self.x_r[1]:
            oldx, self.x = mirror_plus(oldx, self.x + self.a, self.x_r[1])
        elif self.x - self.a < self.x_r[0]:
            oldx, self.x = mirror_min(oldx, self.x - self.a, self.x_r[0])

        elif self.y + self.a > self.y_r[1]:
            oldy, self.y = mirror_plus(oldz, self.y + self.a, self.y_r[1])
        elif self.y - self.a < self.y_r[0]:
            oldy, self.y = mirror_min(oldz, self.y - self.a, self.y_r[0])

        elif self.z + self.a > self.z_r[1]:
            oldz, self.z = mirror_plus(oldz, self.z + self.a, self.z_r[1])
        elif self.z - self.a < self.z_r[0]:
            oldz, self.z = mirror_min(oldz, self.z - self.a, self.z_r[0])

        dx = self.x - oldx
        dy = self.y - oldy
        dz = self.z - oldz

        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        self.theta = np.arctan2(dy, dx)

        self.phi = np.arccos(dz / r)
        return oldx, oldy, oldz

    def step(self):
        """
        Update the angle (phi), x- and y-locations and obtain fluid velocity
        :return: new (x,y)-location and fluid velocity
        """
        self.phi += np.random.uniform(self.d_theta_range[0], self.d_theta_range[1])
        self.theta += np.random.uniform(self.d_theta_range[0], self.d_theta_range[1])

        dx = self.v * np.cos(self.theta) * np.sin(self.phi)
        dy = self.v * np.sin(self.theta) * np.sin(self.phi)
        dz = self.v * np.cos(self.phi)

        oldx = self.x
        oldy = self.y
        oldz = self.z

        self.x += dx
        self.y += dy
        self.z += dz

        oldx, oldy, oldz = self.checkEdges(oldx, oldy, oldz)
        _, _, _ = self.checkEdges(oldx, oldy, oldz)

        self.fluid_v = self.compute_fluid_v()