import numpy as np
from latline.sphere import Sphere


class Latline(object):
    """
    This object implements a simulated lateral line experiment in which multiple spheres are moved through a basin and
    fluid velocities are captured by a horizontal sensor grid
    """

    def __init__(self, cfg):

        self.max = cfg.max_spheres
        self.min = cfg.min_spheres
        self.spheres = [Sphere(cfg) for _ in range(np.random.randint(cfg.min_spheres, cfg.max_spheres))]
        self.cfg = cfg

    def step(self):
        if np.random.uniform(0, 1) > 0.95:
            if len(self.spheres) > self.min:
                del self.spheres[np.random.randint(0, len(self.spheres))]

        if np.random.uniform(0, 1) > 0.95:
            if len(self.spheres) < self.max:
                self.spheres.append(Sphere(self.cfg))

        for sphere in self.spheres:
            sphere.step()

        fluid_v_0 = np.sum([sph.fluid_v[0] for sph in self.spheres], axis=0)
        fluid_v_1 = np.sum([sph.fluid_v[1] for sph in self.spheres], axis=0)

        xs = [sph.x for sph in self.spheres]
        ys = [sph.y for sph in self.spheres]
        zs = [sph.z for sph in self.spheres]

        return xs, ys, zs, fluid_v_0, fluid_v_1