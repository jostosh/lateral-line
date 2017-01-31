import numpy as np
from latline.sphere import Sphere


class Latline(object):
    """
    This object implements a simulated lateral line experiment in which multiple spheres are moved through a basin and
    fluid velocities are captured by a horizontal sensor grid
    """

    def __init__(self, x_range, y_range, z_range, d_theta_range, sensor_range, n_sensors, min_spheres, max_spheres, v=0.05):

        self.spheres = []
        self.max = max_spheres
        self.min = min_spheres
        sphere_flags = np.random.binomial(1, 0.5, max_spheres)
        while not sphere_flags.any():
            sphere_flags = np.random.binomial(1, 0.5, max_spheres)

        for flag in sphere_flags:
            if flag:
                self.spheres.append(Sphere(x_range, y_range, z_range, d_theta_range, sensor_range, n_sensors, v))

    def step(self):
        if np.random.uniform(0, 1) > 0.95:
            if len(self.spheres) > self.min:
                del self.spheres[np.random.randint(0, len(self.spheres))]

        if np.random.uniform(0, 1) > 0.95:
            if len(self.spheres) < self.max:
                self.spheres.append(Sphere([-1, 1], [-1, 1], [0, 1.5], [-1, 1], [-1.5, 1.5], 32, 0.05))

        for sphere in self.spheres:
            sphere.step()

        fluid_v_0 = np.sum([sph.fluid_v[0] for sph in self.spheres], axis=0)
        fluid_v_1 = np.sum([sph.fluid_v[1] for sph in self.spheres], axis=0)

        xs = [sph.x for sph in self.spheres]
        ys = [sph.y for sph in self.spheres]
        zs = [sph.z for sph in self.spheres]

        return xs, ys, zs, fluid_v_0 * 1000, fluid_v_1 * 1000 # todo: why * 1000?