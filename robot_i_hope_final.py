from random import uniform, gauss, random
from math import sqrt, sin, cos, pi, exp
from time import time
import matplotlib.pyplot as plt


class Space:
    def __init__(self, n, coordinates):
        self.n = n
        self.x = []
        self.y = []

        coordinates = coordinates.split()
        for i in range(0, 2 * n, 2):
            self.x.append(int(coordinates[i]))
            self.y.append(int(coordinates[i + 1]))

    def distance(self, robot, angle):
        """https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line"""

        dist = self.diagonal()
        x3, y3 = robot.x, robot.y
        x4, y4 = robot.x + self.diagonal() * cos(angle), robot.y + self.diagonal() * sin(angle)

        for i in range(self.n):
            x1, y1 = self.x[i], self.y[i]
            if i == self.n - 1:
                x2, y2 = self.x[0], self.y[0]
            else:
                x2, y2 = self.x[i + 1], self.y[i + 1]

            if (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) != 0:  # not parallel
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

                if 0 <= t <= 1 and 0 <= u <= 1:
                    point1 = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
                    point2 = x3 + u * (x4 - x3), y3 + u * (y4 - y3)
                else:
                    point1 = None

                if point1:
                    new_dist = sqrt((robot.x - point1[0]) ** 2 + (robot.y - point1[1]) ** 2)
                    if new_dist < dist:
                        dist = new_dist
        return dist

    def bound(self):
        return min(self.x), min(self.y), max(self.x), max(self.y)

    def diagonal(self):
        min_x, min_y, max_x, max_y = self.bound()
        return sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

    def plot(self):
        for i in range(self.n):
            if i == self.n - 1:
                plt.plot([self.x[i], self.x[0]], [self.y[i], self.y[0]], 'k')
            else:
                plt.plot([self.x[i], self.x[i + 1]], [self.y[i], self.y[i + 1]], 'k')


class Robot:
    def __init__(self, space, x=None, y=None, sl=0.1, sx=0.1, sy=0.1, k=36):
        self.space = space
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

    def __str__(self):
        return f'x: {self.x} y: {self.y} sl: {self.l_noise} sx: {self.x_noise} sy: {self.y_noise} k: {self.k} ' \
               f' dist: {self.distances}'

    def plot(self):
        if self.real:
            plt.plot(self.x, self.y, 'Dk')
        else:
            plt.plot(self.x, self.y, '.y')

    def move(self, x, y):
        self.x += x
        self.y += y
        if not self.real:
            self.x += gauss(0, self.x_noise)
            self.y += gauss(0, self.y_noise)

    def sense(self):
        self.distances = []
        for a in range(self.k):
            angle = a * 360 / self.k * pi / 180
            self.distances.append(self.space.distance(self, angle))

    @staticmethod
    def gaussian(x, noise, rx):
        """https://en.wikipedia.org/wiki/Normal_distribution"""
        return exp(-((x - rx) ** 2) / (noise ** 2) / 2) / sqrt(2 * pi * (noise ** 2))

    def probability(self, rx):
        res = 1
        for j in range(self.k):
            res *= self.gaussian(self.distances[j], self.l_noise, rx[j])
        self.w = res


def generate_robots(n, space, sl, sx, sy, k):
    robots = []
    for _ in range(n):
        robot = Robot(space, sl=sl, sx=sx, sy=sy, k=k)
        robots.append(robot)
    return robots


def sense(robots, real_x):
    max_w = 0

    for r in robots:
        r.sense()
        r.probability(real_x)

        if r.w > max_w:
            max_w = r.w
    return max_w


def create_robots(new_n, max_w, robots, space, sl):
    n = len(robots)
    new_robots = []
    index = int(random() * n)
    beta = 0

    for _ in range(new_n):
        beta += random() * 2 * max_w

        while beta > robots[index].w:
            beta -= robots[index].w
            index = (index + 1) % n

        new_robot = Robot(space, robots[index].x, robots[index].y, sl=sl, sx=robots[index].x_noise,
                          sy=robots[index].y_noise, k=robots[index].k)
        new_robots.append(new_robot)

    return new_robots


def move(move_x, move_y, robots):
    for r in robots:
        r.move(move_x, move_y)


def plot_robots(robots):
    for r in robots:
        r.plot()


def main():
    test_num = 2
    f = open('test' + str(test_num) + '.txt', 'r')
    start = time()

    # create space
    space = Space(int(f.readline()), f.readline())
    space.plot()

    # input data
    m, k = f.readline().split()
    m, k = int(m), int(k)
    sl, sx, sy = f.readline().split()
    sl, sx, sy = float(sl), float(sx), float(sy)
    inp = f.readline().split()

    # create real robot
    real_robot = Robot(space, x=6, y=11, sl=sl, sx=sx, sy=sy, k=k)
    real_robot.real = True
    real_robot.plot()

    # create first robots
    n1 = 1000
    n2 = 100
    robots = generate_robots(n1, space, sl * 5, sx, sy, k)
    plot_robots(robots)
    plt.show()

    for i in range(m + 1):
        if i < m - 2:
            f.readline()
            f.readline()
        else:
            # sense
            real_x = [float(el) for el in f.readline().split()]
            max_w = sense(robots, real_x)
            print(i, ':', max_w)

        # # create new robots
        # if max_w == 0:
        #     # something went wrong
        #     while max_w == 0:
        #         robots = generate_robots(n1, space, sl * 5, sx, sy, k)
        #         max_w = sense(robots, real_x)
        #
        #         space.plot()
        #         real_robot.plot()
        #         plot_robots(robots)
        #         plt.show()

            # everything is ok
            robots = create_robots(n2, max_w, robots, space, sl)

            space.plot()
            real_robot.plot()
            plot_robots(robots)
            plt.show()

            # move new robots
            if i != m:
                move_x, move_y = f.readline().split()
                move_x, move_y = float(move_x), float(move_y)
                move(move_x, move_y, robots)

                # move real robot
                real_robot.move(move_x, move_y)

            space.plot()
            real_robot.plot()
            plot_robots(robots)
            plt.show()

    # result
    print(sum(r.x for r in robots) / n2, end=' ')
    print(sum(r.y for r in robots) / n2)

    print('time', time() - start, 'sec')
    answers = [[2.99802, 3.03753], [1.9406, 8.00725], [1.59266, 8.63027], [3.62952, 15.1485]]
    print(sum(r.x for r in robots) / n2 - answers[test_num - 1][0])
    print(sum(r.y for r in robots) / n2 - answers[test_num - 1][1])


if __name__ == '__main__':
    main()

