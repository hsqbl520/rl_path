import numpy as np
import matplotlib.pyplot as plt

# 定义地图大小和障碍物数量
MAP_SIZE = 50
NUM_OBSTACLES = 20

# 定义雷达探测距离和分辨率
RADAR_RANGE = 3
RADAR_RESOLUTION = 60

# 创建地图
map = np.zeros((MAP_SIZE, MAP_SIZE))

# 随机生成障碍物
for i in range(NUM_OBSTACLES):
    x = np.random.randint(MAP_SIZE)
    y = np.random.randint(MAP_SIZE)
    map[x, y] = 1

# 定义机器人初始位置和速度
robot_pos = np.array([np.random.randint(MAP_SIZE), np.random.randint(MAP_SIZE)])
robot_speed = 1

# 绘制地图和机器人
plt.imshow(map, cmap='gray')
plt.scatter(robot_pos[0], robot_pos[1], c='r', marker='o')
plt.xlim(0, MAP_SIZE)
plt.ylim(0, MAP_SIZE)

xMax = 500
# 开始探测
i = 0
while i <= xMax:
    # 获取机器人当前位置
    x, y = robot_pos[0], robot_pos[1]
    print(1)
    # 雷达探测
    for angle in np.arange(0, 360, RADAR_RESOLUTION):
        for r in np.arange(0, RADAR_RANGE, 0.5):
            px = int(np.round(x + r * np.cos(np.radians(angle))))
            py = int(np.round(y + r * np.sin(np.radians(angle))))
            if px < 0 or px >= MAP_SIZE or py < 0 or py >= MAP_SIZE:
                break
            if map[px, py] == 1:
                print("障碍物距离：", r)
                break

    # 移动机器人
    x += robot_speed * np.random.randn()
    y += robot_speed * np.random.randn()
    robot_pos = np.array([int(np.round(x)), int(np.round(y))])

    i = i+1
    # 绘制地图和机器人
    plt.pause(1)
    plt.imshow(map, cmap='gray')
    plt.scatter(robot_pos[0], robot_pos[1], c='r', marker='o')
    plt.xlim(0, MAP_SIZE)
    plt.ylim(0, MAP_SIZE)
    plt.draw()
plt.show()

