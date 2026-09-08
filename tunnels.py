import numpy as np
import random

STEP = 1.0


class TunnelGenerator:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.0])
        self.direction = np.array([0.0, 0.0, 1.0])
        self.points = [self.pos.copy()]

    def normalize(self, v):
        l = np.linalg.norm(v)
        if l == 0:
            return v
        return v / l

    def random_perpendicular(self):
        v = np.random.normal(size=3)
        v -= np.dot(v, self.direction) * self.direction
        return self.normalize(v)

    def add_point(self):
        self.pos += self.direction * STEP
        self.points.append(self.pos.copy())

    # -----------------------------
    # Wander
    # -----------------------------
    def wander(self, length):
        for _ in range(length):
            self.direction += np.random.normal(scale=0.08, size=3)
            self.direction = self.normalize(self.direction)
            self.add_point()

    # -----------------------------
    # Gentle curve
    # -----------------------------
    def curve(self, length):
        axis = self.random_perpendicular()

        for _ in range(length):
            self.direction += axis * 0.03
            self.direction = self.normalize(self.direction)
            self.add_point()

    # -----------------------------
    # Wave
    # -----------------------------
    def wave(self, length):
        axis = self.random_perpendicular()

        for i in range(length):
            angle = np.sin(i * 0.25) * 0.05
            self.direction += axis * angle
            self.direction = self.normalize(self.direction)
            self.add_point()

    # -----------------------------
    # Spiral
    # -----------------------------
    def spiral(self, length):
        axis = self.random_perpendicular()

        for _ in range(length):
            # slowly rotate around axis
            side = np.cross(self.direction, axis)
            self.direction += side * 0.08
            self.direction = self.normalize(self.direction)
            self.add_point()

    # -----------------------------
    # Sharp turn
    # -----------------------------
    def sharp_turn(self, length):
        axis = self.random_perpendicular()

        for _ in range(length):
            self.direction += axis * 0.12
            self.direction = self.normalize(self.direction)
            self.add_point()

    # -----------------------------
    # Hybrid generator
    # -----------------------------
    def generate(self, sections=20):

        methods = [
            (self.wander, 0.35),
            (self.curve, 0.20),
            (self.wave, 0.20),
            (self.spiral, 0.15),
            (self.sharp_turn, 0.10),
        ]

        funcs = [m[0] for m in methods]
        weights = [m[1] for m in methods]

        for _ in range(sections):
            func = random.choices(funcs, weights)[0]
            length = random.randint(20, 80)
            func(length)

        return np.array(self.points)


generator = TunnelGenerator()
points = generator.generate()

print(points[::40])
