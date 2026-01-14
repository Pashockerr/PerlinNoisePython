import matplotlib.pyplot as plt 
import matplotlib
from random import randint
from math import sqrt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

def readPresentedOrDefaultInt(i_s, d):
    if len(i_s) == 0:
        return d
    return int(i_s)

def readPresentedOrDefaultFloat(i_s, d):
    if len(i_s) == 0:
        return d
    return float(i_s)

t_w = readPresentedOrDefaultInt(input("Input texture width(100): "), 100)
t_h = readPresentedOrDefaultInt(input("Input texture height(100): "), 100)
scale = readPresentedOrDefaultFloat(input("Input noise scale(1/20): "), 1/20)
v_w = readPresentedOrDefaultInt(input("Input vector grid width(100): "), 100)
v_h = readPresentedOrDefaultInt(input("Input vector grid height(100): "), 100)
v_d = readPresentedOrDefaultInt(input("Input vector grid deepness(100): "), 100)
anim_interval = readPresentedOrDefaultInt(input("Input animation interval in ms(50): "), 50)

gradient_vectors = [
    [1, 1, -1], 
    [1, -1 , -1],
    [-1, 1, -1],
    [-1, -1, -1],
    [1, 1, 1], 
    [1, -1 , 1],
    [-1, 1, 1],
    [-1, -1, 1]
]

def smoothstep(t):
    return 3*(t**2) - 2*(t**3)

def interp(x, p1, p2):  # p = [ x, v ]
    return p1[1] + (p2[1] - p1[1]) * smoothstep(((x - p1[0])/(p2[0] - p1[0])))

def triinterp(x, y, z, points):   # points = [ [[x,y,z], v] ]
    # Along x-axis
    vx1 = interp(x, [points[0][0][0], points[0][1]], [points[4][0][0], points[4][1]])
    vx2 = interp(x, [points[1][0][0], points[1][1]], [points[5][0][0], points[5][1]])
    vx3 = interp(x, [points[2][0][0], points[2][1]], [points[6][0][0], points[6][1]])
    vx4 = interp(x, [points[3][0][0], points[3][1]], [points[7][0][0], points[7][1]])

    # Along y-axis
    vy1 = interp(y, [points[0][0][1], vx1], [points[2][0][1], vx3])
    vy2 = interp(y, [points[1][0][1], vx2], [points[3][0][1], vx4])

    # Along z-axis
    return interp(z, [points[0][0][2], vy1], [points[1][0][2], vy2])

def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def getVector(start, end):
    return [end[0] - start[0], end[1] - start[1], end[2] - start[2]]

def distance(p1, p2):
    return sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)

def genVectorGrid3D(w, h, d, vg):
    # Populate vector grid
    for x in range(w):
        vg.append([])
        for y in range(h):
            vg[x].append([])
            for z in range(d):
                vector = gradient_vectors[randint(0, 7)]
                vg[x][y].append(vector)

def getValue(vg, x, y, z):
    if x > len(vg) or y > len(vg[0]) or z > len(vg[0][0]):
        print("Side mismatch!")
        return [[0]]
    
    # Determine corner vectors
    cornerVectors = [] # [ [[px, py, pz], [x, y, z]] ]
    sx = int(x // 1)
    sy = int(y // 1)
    sz = int(z // 1)
    for px in range(sx, sx + 2):
        for py in range(sy, sy + 2):
            for pz in range(sz, sz + 2):
                cornerVectors.append([[px, py, pz], vg[px][py][pz]])
    
    # Calculate dot products
    dp = []
    for corner in cornerVectors:
        offset = getVector(corner[0], [x, y, z])
        dp.append([corner[0], dot(corner[1], offset)])
    
    # Interpolation
    return triinterp(x, y, z, dp)

def renderTexture(img, grid, width, height, scale, z):
    for x in range(0, width):
        img.append([])
        for y in range(0, height):
            img[x].append(getValue(grid, x*scale, y*scale, z*scale))


grid = []
genVectorGrid3D(v_w, v_h, v_d, grid)

fig, ax = plt.subplots()
img = []
renderTexture(img, grid, t_w, t_h, scale, 0)
im = ax.imshow(img, animated=True, interpolation="nearest")

def update(i):
    global img, grid, im
    img=[]
    renderTexture(img, grid, t_w, t_h, scale, i)
    im.set_array(img)
    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=v_d,
    interval=anim_interval,
    blit=True,
    repeat=True
)

plt.show()
