import matplotlib.pyplot as plt
from random import uniform, gauss, random
from math import sqrt, sin, cos, pi, exp
from time import time

start = time()


class Space:
    def __init__(self, n, coordinates):
        self.n = n
        self.x = []
        self.y = []

        coordinates = coordinates.split()
        for i1 in range(0, 2 * n, 2):
            self.x.append(int(coordinates[i1]))
            self.y.append(int(coordinates[i1+1]))

    def distance(self, robot1, angle):
        """https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line"""

        dist = self.diagonal()
        x3, y3 = robot1.x, robot1.y
        x4, y4 = robot1.x + self.diagonal() * cos(angle), robot1.y + self.diagonal() * sin(angle)

        for i2 in range(self.n):
            x1, y1 = self.x[i2], self.y[i2]
            if i2 == self.n - 1:
                x2, y2 = self.x[0], self.y[0]
            else:
                x2, y2 = self.x[i2+1], self.y[i2+1]

            if (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) != 0:  # not parallel
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

                if 0 <= t <= 1 and 0 <= u <= 1:
                    point = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
                else:
                    point = None

                if point:
                    new_dist = sqrt((robot1.x - point[0]) ** 2 + (robot1.y - point[1]) ** 2)
                    if new_dist < dist:
                        dist = new_dist
        return dist

    def plot(self):
        for i3 in range(self.n):
            if i3 == self.n - 1:
                plt.plot([self.x[i3], self.x[0]], [self.y[i3], self.y[0]], 'k')
            else:
                plt.plot([self.x[i3], self.x[i3+1]], [self.y[i3], self.y[i3+1]], 'k')

    def bound(self):
        return min(self.x), min(self.y), max(self.x), max(self.y)

    def diagonal(self):
        min_x, min_y, max_x, max_y = self.bound()
        return sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)


class Robot:
    def __init__(self, space1, x=None, y=None):
        self.space = space1
        bound = self.space.bound()

        self.x = x if x else uniform(bound[0], bound[2])
        self.y = y if y else uniform(bound[1], bound[3])

        self.w = 1
        self.real = False
        self.distances = []

        self.l_noise = sl
        self.x_noise = sx
        self.y_noise = sy
        self.k = k

    def plot(self):
        if self.real:
            plt.plot(self.x, self.y, 'Dk')
        else:
            plt.plot(self.x, self.y, '.y')

    def move(self, x, y):
        self.x += x + gauss(0, self.x_noise)
        self.y += y + gauss(0, self.y_noise)

    def sense(self):
        self.distances = []
        for a in range(0, 360, 360 // self.k):
            angle = a * pi / 180
            self.distances.append(self.space.distance(self, angle))

    @staticmethod
    def gaussian(x, noise, rx):
        """https://en.wikipedia.org/wiki/Normal_distribution"""
        return exp(-((x - rx) ** 2) / (noise ** 2)) / sqrt(2 * pi * (noise ** 2)) / 2

    def probability(self, rx):
        res = 1
        for j in range(self.k):
            res *= self.gaussian(self.distances[j], 2 * self.l_noise, rx[j])
        self.w = res


space = Space(int(input()), input())
# space.plot()

m, k = input().split()
m, k = int(m), int(k)

sl, sx, sy = input().split()
sl, sx, sy = float(sl), float(sx), float(sy)

inp = input().split()
# if inp[0] == '1':
#     robot = Robot(space, int(inp[1]), int(inp[2]))
#     robot.real = True
#     robot.plot()

robots = []
N1, N2 = 1000, 100
for _ in range(N1):
    r = Robot(space)
    robots.append(r)
    # r.plot()

# plt.show()


def sense():
    max_w = robots[0].w
    real_x = [float(el) for el in input().split()]

    for r in robots:
        r.sense()
        r.probability(real_x)

        if r.w > max_w:
            max_w = r.w

    print(max_w)
    return max_w


def create_robots():
    global N1, N2
    new_robots = []
    index = int(random() * N1)
    beta = 0

    for _ in range(N2):
        beta += random() * 2 * max_w
        while beta > robots[index].w:
            beta -= robots[index].w
            index = (index + 1) % N1
        new_robot = Robot(space, robots[index].x, robots[index].y)
        new_robots.append(new_robot)

    N1 = N2

    return new_robots


def plot_all():
    space.plot()
    for r in robots:
        r.plot()
    # robot.plot()
    plt.show()


def move():
    move_x, move_y = input().split()
    move_x, move_y = int(move_x), int(move_y)

    for r in robots:
        r.move(move_x, move_y)

    # robot.move(move_x, move_y)


for i in range(m + 1):
    # sense
    max_w = sense()

    # create new robots
    robots = create_robots()

    # plot_all()

    # move new robots
    if i != m:
        move()
        # plot_all()


# print(robot.x, robot.y)
print(sum(r.x for r in robots) / N2)
print(sum(r.y for r in robots) / N2)

print(time() - start)
answers = [[2.99802, 3.03753], [1.9406, 8.00725], [1.59266, 8.63027], [3.62952, 15.1485]]
print(sum(r.x for r in robots) / N2 - answers[1][0])
print(sum(r.y for r in robots) / N2 - answers[1][1])
