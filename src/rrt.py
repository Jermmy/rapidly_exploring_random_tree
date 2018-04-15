import numpy as np
import cv2

class Node:

    def __init__(self, p, prev):
        self.p = p
        self.prev = prev

    def __str__(self):
        return str(self.p)


class Rob:

    def __init__(self, p, r):
        self.p = p
        self.r = r


class Param:

    def __init__(self, w, h, border_size, max_iter, smooth_iter, threshold):
        self.w = w
        self.h = h
        self.border_size = border_size
        self.max_iter = max_iter
        self.smooth_iter = smooth_iter
        self.threshold = threshold


def in_collision_node(obst, rob, p, rrt=None):
    '''
    :param obst: obstacles
    :param rob:  robot
    :return: true or false, whether the robot collide with obstacles
    '''
    for o in obst:
        dis = dist(o.p, p)
        if dis < rob.r + o.r:
            return True
    if rrt:
        for node in rrt:
            dis = dist(node.p, p)
            if dis < rob.r * 2:
                return True
    return False


def in_collision_path(obst, rob, p1, p2):
    '''
    :param obst: obstacles
    :param rob:  robot
    :param p1:
    :param p2:
    :return:
    '''
    d = dist(p1, p2)
    m = np.ceil(d / 0.3)
    t = np.linspace(0, 1, m)
    for i in range(1, len(t) - 1):
        p = p1 * (1 - t[i]) + p2 * t[i]
        if in_collision_node(obst, rob, p):
            return True
    return False



def sample(w_low, w_high, h_low, h_high):
    '''
    :param h: the height of map
    :param w: the width of map
    :return:
    '''
    # p = np.random.rand(2)
    p = np.array([np.random.randint(w_low, w_high), np.random.randint(h_low, h_high)])

    return p


def find_nearest(rrt, p):
    '''
    :param rrt:
    :param p:
    :return:
    '''
    min_p = rrt[0]
    min_dist = dist(min_p.p, p)
    for i in range(1, len(rrt)):
        d = dist(rrt[i].p, p)
        if min_dist > d:
            min_dist = d
            min_p = rrt[i]
    return min_p


def add_node(rrt, p, prev):
    node = Node(p, prev)
    rrt.append(node)
    return node

def dist(p1, p2):
    '''
    :param p:
    :param p_goal:
    :return:
    '''
    return np.linalg.norm(p1 - p2)


def get_path(rrt):
    '''
    :param rrt:
    :return:
    '''
    path = []
    node = rrt[-1]
    path.append(node.p)
    while node.prev != None:
        path.append(node.prev.p)
        node = node.prev
    path.reverse()
    return path


def smooth_path(path, rob, obst, smooth_iter, image):
    length = len(path)
    l = [0] * length
    for i in range(1, len(l)):
        l[i] = dist(path[i], path[i-1]) + l[i-1]

    iter = 0
    while iter <= smooth_iter:
        iter += 1
        s1 = np.random.rand(1) * l[-1]
        s2 = np.random.rand(1) * l[-1]
        if s2 < s1:
            s1, s2 = s2, s1
        for k in range(1, length):
            if s1 < l[k]:
                i = k - 1
                break
        for k in range(i + 1, length):
            if s2 < l[k]:
                j = k - 1
                break
        if j <= i:
            continue

        t1 = (s1 - l[i]) / (l[i+1] - l[i])
        gamma1 = (1 - t1) * path[i] + t1 * path[i+1]
        t2 = (s2 - l[j]) / (l[j+1] - l[j])
        gamma2 = (1 - t2) * path[j] + t2 * path[j+1]

        if in_collision_path(obst, rob, gamma1, gamma2):
            continue
        new_path = []
        new_path.extend(path[:i+1])
        new_path.append(gamma1.astype(np.int32))
        new_path.append(gamma2.astype(np.int32))
        new_path.extend(path[j+1:])
        path = new_path

        l = [0] * len(path)
        for i in range(2, len(l)):
            l[i] = dist(path[i], path[i - 1]) + l[i - 1]

    return path



def plan_path_rrt(rob, obst, p_start, p_goal, param):
    image = np.zeros(shape=[param.h, param.w, 3], dtype=np.int8)
    image.fill(255)
    cv2.rectangle(image, (0, 0), (param.w - 1, param.h - 1), (0, 0, 0), param.border_size)
    cv2.circle(image, (p_start[0], p_start[1]), rob.r, (255, 255, 0), -1)
    cv2.circle(image, (p_goal[0], p_goal[1]), rob.r, (0, 0, 255), -1)
    for o in obst:
        cv2.circle(image, (o.p[0], o.p[1]), o.r, (0, 0, 0), -1)
    cv2.imshow('image', image.astype(np.uint8))
    cv2.imwrite('../data/rrt_start.png', image.astype(np.uint8))
    cv2.waitKey()

    rrt = []
    rrt.append(Node(p_start, None))
    iter = 0
    while iter <= param.max_iter:
        iter += 1

        p = sample(0 + param.border_size, param.w - border_size, 0 + param.border_size, param.h - border_size)

        rob.p = p

        if in_collision_node(obst, rob, p, rrt):
            continue

        nearest = find_nearest(rrt, p)

        if in_collision_path(obst, rob, nearest.p, p):
            continue

        p_node = add_node(rrt, p, nearest)

        cv2.line(image, (nearest.p[0], nearest.p[1]), (p[0], p[1]), (255, 20, 147), 1)
        cv2.circle(image, (p[0], p[1]), rob.r, (255, 255, 0), -1)
        cv2.imshow('image', image.astype(np.uint8))

        if iter % 10 == 0:
            cv2.waitKey()
            cv2.imwrite('../data/rrt_' + str(iter) + ".png", image.astype(np.uint8))

        if dist(p, p_goal) < param.threshold:
            if in_collision_path(obst, rob, p, p_goal):
                continue
            add_node(rrt, p_goal, p_node)

            cv2.line(image, (p[0], p[1]), (p_goal[0], p_goal[1]), (255, 20, 147), 1)
            cv2.circle(image, (p_goal[0], p_goal[1]), rob.r, (255, 255, 0), -1)
            cv2.imshow('image', image.astype(np.uint8))
            cv2.imwrite('../data/rrt_' + str(iter+1) + ".png", image.astype(np.uint8))
            cv2.waitKey()

            path = get_path(rrt)

            cv2.circle(image, (path[0][0], path[0][1]), rob.r, (139, 105, 20), -1)
            for i in range(1, len(path)):
                cv2.line(image, (path[i-1][0], path[i-1][1]), (path[i][0], path[i][1]), (205, 205, 0), 1)
                cv2.circle(image, (path[i][0], path[i][1]), rob.r, (139, 105, 20), -1)
            cv2.imshow('image', image.astype(np.uint8))
            cv2.imwrite('../data/rrt_unsmooth.png', image.astype(np.uint8))
            cv2.waitKey()

            path = smooth_path(path, rob, obst, param.smooth_iter, image)

            cv2.circle(image, (path[0][0], path[0][1]), rob.r, (0, 0, 205), -1)
            for i in range(1, len(path)):
                cv2.line(image, (path[i - 1][0], path[i - 1][1]), (path[i][0], path[i][1]), (65, 105, 225), 1)
                cv2.circle(image, (path[i][0], path[i][1]), rob.r, (0, 0, 205), -1)
            cv2.imshow('image', image.astype(np.uint8))
            cv2.imwrite('../data/rrt_smooth.png', image.astype(np.uint8))
            cv2.waitKey()

            break


def obstacles_1(height, width):
    obsts = []
    for w in range(int(0.5 * width), int(0.75 * width)):
        obsts.append([w, int(1. / 8 * height)])
        obsts.append([w, int(3. / 8 * height)])
    for h in range(int(1. / 8 * height), int(3. / 8 * height)):
        obsts.append([int(1. / 2 * width), h])
        obsts.append([int(3. / 4 * width), h])

    for h in range(int(1. / 2 * height), int(3. / 4 * height)):
        obsts.append([int(1. / 8 * width), h])
        obsts.append([int(3. / 8 * width), h])
    for w in range(int(1. / 8 * width), int(3. / 8 * width)):
        obsts.append([w, int(1. / 2 * height)])
        obsts.append([w, int(3. / 4 * width)])

    obsts = np.array(obsts, dtype=np.int32)
    return obsts


def obstacles_2(height, width):
    obsts = []
    for w in range(int(1. / 3 * width), int(2. / 3 * width)):
        obsts.append([w, int(1. / 3 * height)])
        obsts.append([w, int(2. / 3 * height)])
    for h in range(int(1. / 3 * height), int(2. / 3 * height)):
        obsts.append([int(1. / 3 * width), h])
    for h in range(int(1. / 3 * height), int(6. / 12 * height)):
        obsts.append([int(2. / 3 * width), h])
    for h in range(int(6.5 / 12 * height), int(2. / 3 * height)):
        obsts.append([int(2. / 3 * width), h])

    obsts = np.array(obsts, dtype=np.int32)
    return obsts



if __name__ == '__main__':
    width, height, border_size = 200, 200, 2
    goal_threshold = 20
    max_iter, smooth_iter = 1000, 20
    param = Param(width, height, border_size, max_iter, goal_threshold, smooth_iter)
    p_start = np.array([20, 20], dtype=np.int32)
    p_goal = np.array([190, 190], dtype=np.int32)
    obj_coord = obstacles_1(height, width)
    rob = Rob(p_start, 2)
    obst = []
    for coord in obj_coord:
        obst.append(Rob(coord, 2))
    plan_path_rrt(rob, obst, p_start, p_goal, param)


