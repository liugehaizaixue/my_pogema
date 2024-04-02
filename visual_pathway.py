import matplotlib.pyplot as plt

def plot_line(x0, y0, x1, y1):
    # Bresenham's line algorithm
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    points = []

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

def visualize_line(grid, x0, y0, x1, y1):
    points = plot_line(x0, y0, x1, y1)
    print(points)
    for point in points:
        x, y = point
        if 0 <= x < len(grid[0]) and 0 <= y < len(grid):
            grid[y][x] = 1
    
    plt.imshow(grid, cmap='gray', origin='lower')
    plt.plot([x0, x1], [y0, y1], marker='o', color='red')  # plot the line
    plt.title('Line Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Define grid size
grid_width = 20
grid_height = 20

# Initialize grid
grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]

# Define two points
start_point = (6, 6)
# end_point = (15, 15)
# end_point = (10,4)
end_point = (8,2)

# Visualize line
visualize_line(grid, start_point[0], start_point[1], end_point[0], end_point[1])