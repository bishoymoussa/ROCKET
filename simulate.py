import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import heapq
from scipy.ndimage import distance_transform_edt
import time
import matplotlib.animation as animation

class InverseCollisionPlanner:
    def __init__(self, grid_size=100, obstacle_radius=5, num_obstacles=10, seed=42):
        """
        Initialize the environment with grid size and obstacles
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            obstacle_radius: Radius of obstacles
            num_obstacles: Number of obstacles to generate
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.grid_size = grid_size
        self.obstacle_radius = obstacle_radius
        self.num_obstacles = num_obstacles
        
        # Create grid and place obstacles
        self.grid = np.zeros((grid_size, grid_size))
        self.obstacles = []
        self.start = (10, 10)
        self.goal = (grid_size - 10, grid_size - 10)
        
        self._place_obstacles()
        self._create_collision_probability_field()
        
    def _place_obstacles(self):
        """Place random obstacles in the grid"""
        # Ensure start and goal are not within obstacles
        min_dist_from_start_goal = 15
        
        for _ in range(self.num_obstacles):
            valid = False
            while not valid:
                x = np.random.randint(5, self.grid_size - 5)
                y = np.random.randint(5, self.grid_size - 5)
                
                # Check distance from start and goal
                dist_from_start = np.sqrt((x - self.start[0])**2 + (y - self.start[1])**2)
                dist_from_goal = np.sqrt((x - self.goal[0])**2 + (y - self.goal[1])**2)
                
                if dist_from_start > min_dist_from_start_goal and dist_from_goal > min_dist_from_start_goal:
                    valid = True
                    self.obstacles.append((x, y))
                    
                    # Mark grid cells within obstacle radius as obstacles (1)
                    for i in range(max(0, x - self.obstacle_radius), min(self.grid_size, x + self.obstacle_radius + 1)):
                        for j in range(max(0, y - self.obstacle_radius), min(self.grid_size, y + self.obstacle_radius + 1)):
                            if np.sqrt((i - x)**2 + (j - y)**2) <= self.obstacle_radius:
                                self.grid[j, i] = 1  # Mark as obstacle
    
    def _create_collision_probability_field(self):
        """
        Create a probability field representing likelihood of collision
        Higher values = higher probability of collision
        """
        # Create distance transform from obstacles
        self.collision_prob = distance_transform_edt(1 - self.grid)
        
        # Convert distances to probabilities (closer to obstacle = higher probability)
        max_dist = np.max(self.collision_prob)
        # Invert and normalize: p(collision) = 1 near obstacles, decreases with distance
        self.collision_prob = 1 - (self.collision_prob / max_dist)
        
        # Add buffer zone around obstacles (probability decreases with distance)
        buffer_size = 10  # Buffer size in grid cells
        
        # Adaptive threshold: start with a low threshold and gradually increase if needed
        threshold = 0.3  # Initial threshold (30% collision probability)
        self.safe_grid = self.collision_prob < threshold
        
        # Check if there's a valid path with this threshold
        # If not, gradually relax the threshold until a path exists
        while not self._path_exists_with_current_threshold() and threshold < 0.9:
            threshold += 0.1
            print(f"Increasing safety threshold to {threshold:.1f}")
            self.safe_grid = self.collision_prob < threshold
            
        print(f"Final safety threshold: {threshold:.1f} (higher = less safe)")
        
    def _path_exists_with_current_threshold(self):
        """
        Check if a path exists from start to goal with the current safety threshold
        Uses a simple breadth-first search
        
        Returns:
            Boolean indicating if a path exists
        """
        # Use breadth-first search to check for path existence
        queue = [self.start]
        visited = set([self.start])
        
        while queue:
            current = queue.pop(0)
            
            if current == self.goal:
                return True
                
            # Check 8 possible directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                # Check if within bounds
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue
                    
                # Check if in safe space
                if not self.safe_grid[ny, nx]:
                    continue
                    
                neighbor = (nx, ny)
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    
    def generate_paths_to_obstacles(self, num_samples=1000):
        """
        Generate paths that would collide with obstacles
        
        Args:
            num_samples: Number of sample paths to generate
            
        Returns:
            List of collision paths
        """
        collision_paths = []
        
        # For each obstacle, generate paths from start to obstacle
        for obs in self.obstacles:
            # Generate multiple paths with some randomness
            for _ in range(num_samples // self.num_obstacles):
                path = self._generate_path_to_point(self.start, obs, randomness=0.3)
                if path:
                    collision_paths.append(path)
        
        return collision_paths
    
    def _generate_path_to_point(self, start, end, randomness=0.0):
        """Generate a path from start to end with optional randomness"""
        path = [start]
        current = start
        
        max_iterations = 1000
        iterations = 0
        
        while current != end and iterations < max_iterations:
            iterations += 1
            
            # Calculate direction vector
            dx = end[0] - current[0]
            dy = end[1] - current[1]
            
            # Normalize
            length = max(1, np.sqrt(dx*dx + dy*dy))
            dx /= length
            dy /= length
            
            # Add randomness
            if randomness > 0:
                dx += np.random.normal(0, randomness)
                dy += np.random.normal(0, randomness)
            
            # Take a step
            next_x = int(round(current[0] + dx))
            next_y = int(round(current[1] + dy))
            
            # Ensure within grid bounds
            next_x = max(0, min(self.grid_size - 1, next_x))
            next_y = max(0, min(self.grid_size - 1, next_y))
            
            next_pos = (next_x, next_y)
            
            # Check if we've moved
            if next_pos == current:
                # If stuck, take a random step
                neighbors = [(current[0]+1, current[1]), (current[0]-1, current[1]), 
                            (current[0], current[1]+1), (current[0], current[1]-1)]
                valid_neighbors = [(x, y) for (x, y) in neighbors 
                                  if 0 <= x < self.grid_size and 0 <= y < self.grid_size]
                if valid_neighbors:
                    next_pos = valid_neighbors[np.random.randint(0, len(valid_neighbors))]
            
            path.append(next_pos)
            current = next_pos
            
            # If we reached the target point (or close enough)
            if abs(current[0] - end[0]) <= 1 and abs(current[1] - end[1]) <= 1:
                break
                
        return path
    
    def find_optimal_path(self):
        """
        Find the optimal path using A* search in the safe space
        
        Returns:
            Optimal path from start to goal
        """
        # A* search implementation
        heap = [(0, 0, self.start, [self.start])]  # (f_score, g_score, node, path)
        visited = set()
        
        while heap:
            f, g, current, path = heapq.heappop(heap)
            
            if current == self.goal:
                return path
                
            if current in visited:
                continue
                
            visited.add(current)
            
            # Check 8 possible directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                # Check if within bounds
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue
                    
                # Check if in safe space
                if not self.safe_grid[ny, nx]:
                    continue
                    
                neighbor = (nx, ny)
                
                # Cost to reach neighbor: diagonal movements cost more
                movement_cost = 1.4 if (dx != 0 and dy != 0) else 1.0
                
                # g_score is the cost from start to current node
                new_g = g + movement_cost
                
                # Use Euclidean distance as heuristic
                h = np.sqrt((nx - self.goal[0])**2 + (ny - self.goal[1])**2)
                
                # f_score = g_score + heuristic
                f = new_g + h
                
                # Only add if not visited or better path
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    heapq.heappush(heap, (f, new_g, neighbor, new_path))
        
        return None  # No path found
    
    def visualize(self, collision_paths=None, optimal_path=None):
        """
        Visualize the environment, collision paths, and optimal path
        
        Args:
            collision_paths: List of collision paths to visualize
            optimal_path: Optimal path to visualize
        """
        # Convert paths to numpy arrays if they aren't already
        if collision_paths is not None:
            collision_paths = [np.array(path) for path in collision_paths]
        if optimal_path is not None:
            optimal_path = np.array(optimal_path)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Collision probability field
        im = axs[0].imshow(self.collision_prob, origin='lower')
        plt.colorbar(im, ax=axs[0], label='Collision Probability')
        axs[0].set_title('Collision Probability Field')
        
        # Plot obstacles
        for x, y in self.obstacles:
            circle = Circle((x, y), self.obstacle_radius, color='gray')
            axs[0].add_patch(circle)
        
        # Plot 2: Collision paths
        if collision_paths is not None:
            # Plot collision probability field with low opacity
            axs[1].imshow(self.collision_prob, origin='lower', alpha=0.3, cmap='YlOrRd')
            
            for path in collision_paths:
                axs[1].plot(path[:, 0], path[:, 1], 'r-', alpha=0.1)
        
        # Plot obstacles
        for x, y in self.obstacles:
            circle = Circle((x, y), self.obstacle_radius, color='gray')
            axs[1].add_patch(circle)
        
        axs[1].plot(self.start[0], self.start[1], 'go', markersize=8, label='Start')
        axs[1].plot(self.goal[0], self.goal[1], 'ro', markersize=8, label='Goal')
        axs[1].set_title('Collision Paths')
        axs[1].legend()
        
        # Plot 3: Safe space and optimal path
        # Plot collision probability field with low opacity
        axs[2].imshow(self.collision_prob, origin='lower', alpha=0.3, cmap='YlOrRd')
        
        if optimal_path is not None:
            axs[2].plot(optimal_path[:, 0], optimal_path[:, 1], 'g-', linewidth=2, label='Optimal Path')
        
        # Plot obstacles
        for x, y in self.obstacles:
            circle = Circle((x, y), self.obstacle_radius, color='gray')
            axs[2].add_patch(circle)
        
        axs[2].plot(self.start[0], self.start[1], 'go', markersize=8, label='Start')
        axs[2].plot(self.goal[0], self.goal[1], 'ro', markersize=8, label='Goal')
        axs[2].set_title('Safe Space & Optimal Path')
        axs[2].legend()
        
        plt.tight_layout()
        plt.show()

    def create_animation(self, collision_paths, optimal_path, save_path='path_animation.mp4'):
        """
        Create and save an animation of the path planning process
        
        Args:
            collision_paths: List of collision paths to visualize
            optimal_path: Optimal path to visualize (as numpy array)
            save_path: Path to save the animation video
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        
        # Plot collision probability field
        probability_field = ax.imshow(self.collision_prob, 
                                    extent=[0, self.grid_size, 0, self.grid_size],
                                    origin='lower', cmap='YlOrRd', alpha=0.3)
        plt.colorbar(probability_field, ax=ax, label='Collision Probability')
        
        # Plot obstacles
        for x, y in self.obstacles:
            circle = Circle((x, y), self.obstacle_radius, color='gray', alpha=0.5)
            ax.add_patch(circle)
        
        # Plot start and goal
        ax.plot(self.start[0], self.start[1], 'go', markersize=8, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'ro', markersize=8, label='Goal')
        
        # Plot collision paths
        for path in collision_paths:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'r-', alpha=0.1)
        
        # Initialize optimal path line
        line, = ax.plot([], [], 'g-', linewidth=2, label='Optimal Path')
        point, = ax.plot([], [], 'bo', markersize=6)
        
        ax.legend()
        
        # Animation update function
        def update(frame):
            if frame < len(optimal_path):
                path_segment = optimal_path[:frame+1]
                line.set_data(path_segment[:, 0], path_segment[:, 1])
                point.set_data([optimal_path[frame, 0]], [optimal_path[frame, 1]])
            return line, point
        
        # Create animation
        print("Creating path planning animation...")
        anim = animation.FuncAnimation(fig, update, frames=len(optimal_path),
                                     interval=50, blit=True)
        
        # Save animation
        writer = animation.FFMpegWriter(fps=30, bitrate=2000)
        anim.save(save_path, writer=writer)
        print("Animation created!")
        
        plt.close()

def run_simulation(randomize=False):
    """
    Run a complete simulation of the inverse collision planner
    
    Args:
        randomize: If True, use a random seed for obstacle placement
    """
    print("Initializing Inverse Collision Planner...")
    if randomize:
        seed = np.random.randint(0, 10000)
        planner = InverseCollisionPlanner(grid_size=100, obstacle_radius=5, num_obstacles=8, seed=seed)
        print(f"Using random seed: {seed}")
    else:
        planner = InverseCollisionPlanner(grid_size=100, obstacle_radius=5, num_obstacles=8)
    
    print("Generating paths that would collide with obstacles...")
    start_time = time.time()
    collision_paths = planner.generate_paths_to_obstacles(num_samples=500)
    collision_time = time.time() - start_time
    print(f"Generated {len(collision_paths)} collision paths in {collision_time:.2f} seconds")
    
    print("Finding optimal path in the safe space...")
    start_time = time.time()
    optimal_path = planner.find_optimal_path()
    optimal_time = time.time() - start_time
    
    if optimal_path:
        print(f"Found optimal path of length {len(optimal_path)} in {optimal_time:.2f} seconds")
    else:
        print("No valid path found!")
    
    print("Visualizing results...")
    planner.visualize(collision_paths, optimal_path)
    
    # Compute path metrics if a path was found
    if optimal_path:
        # Calculate path length
        path_length = 0
        for i in range(1, len(optimal_path)):
            x1, y1 = optimal_path[i-1]
            x2, y2 = optimal_path[i]
            segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            path_length += segment_length
            
        print(f"Optimal path length: {path_length:.2f} units")
        
        # Calculate average distance from obstacles
        total_distance = 0
        for point in optimal_path:
            # Use collision probability as a proxy for distance from obstacles
            # Lower probability = further from obstacles
            prob = planner.collision_prob[point[1], point[0]]
            total_distance += (1 - prob)  # Convert to distance (higher = further)
            
        avg_distance = total_distance / len(optimal_path)
        print(f"Average safety margin (distance from obstacles): {avg_distance:.2f}")
        
        # Create animation
        optimal_path_array = np.array(optimal_path)
        collision_paths_array = [np.array(path) for path in collision_paths]
        planner.create_animation(collision_paths_array, optimal_path_array)
    
    return planner, collision_paths, optimal_path

if __name__ == "__main__":
    run_simulation(randomize=True)