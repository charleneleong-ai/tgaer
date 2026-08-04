"""Custom Key-Door Game - Simple Test Environment

A minimal key-door game for testing agent generalization.
The avatar (A) must navigate to the key (K), then to the door (D).

Level 1: Simple corridor
Level 2: L-shaped path
Level 3: Maze with walls
"""

import os
from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)

# Sprite definitions using color indices that match LS20_DEFAULT semantics
# avatar=12, keys=(0,1), door=9, walls=(4,11)
sprites = {
    "avatar": Sprite(
        pixels=[[12, 12], [12, 12]],
        name="avatar",
        tags=["avatar"],
    ),
    "key": Sprite(
        pixels=[[0]],  # Key color 0
        name="key",
        tags=["key"],
    ),
    "key2": Sprite(
        pixels=[[1]],  # Key color 1
        name="key2",
        tags=["key"],
    ),
    "door": Sprite(
        pixels=[[9]],
        name="door",
        tags=["door"],
    ),
    "wall": Sprite(
        pixels=[[4]],
        name="wall",
        blocking=True,
        tags=["wall"],
    ),
    "wall2": Sprite(
        pixels=[[11]],
        name="wall2",
        blocking=True,
        tags=["wall"],
    ),
}

# Level definitions using grid notation
# A = avatar, K = key (color 0), L = key (color 1), D = door, W = wall, . = empty
level_grids = [
    # Level 1: Simple corridor with key and door
    """
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . A . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . K . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . D . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
""",
    # Level 2: L-shaped path with key and door
    """
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . A . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . K . .
. . . . . . . . . . . . . .
. . . . . . . . . . . D . .
""",
    # Level 3: Maze with walls, key, and door
    """
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . A . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
. . . . . . . . . . . K . .
. . . . . . . . . . . . . .
. . . . . . . . . . . D . .
""",
]


class Cust(ARCBaseGame):
    """Simple key-door game for testing."""
    
    def __init__(self, game_id: str = "custom_kd01-test", **kwargs):
        # Parse levels first
        levels = []
        for grid_str in level_grids:
            level = self._create_level(grid_str)
            levels.append(level)
        
        super().__init__(game_id=game_id, levels=levels, **kwargs)
        self._title = "Custom Key-Door 01"
        self._tags = ["keyboard"]
        self._default_fps = 10
        
    def _parse_grid(self, grid_str: str) -> list[list[str]]:
        """Parse grid string into 2D list."""
        rows = []
        for line in grid_str.strip().split('\n'):
            row = [cell.strip() for cell in line.split()]
            rows.append(row)
        return rows
    
    def _create_level(self, grid_str: str) -> Level:
        """Create a level from grid notation."""
        grid = self._parse_grid(grid_str)
        height = len(grid)
        width = len(grid[0]) if grid else 0
        
        sprites_list = []
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == 'A':
                    s = sprites["avatar"].clone()
                    s.set_position(x, y)
                    sprites_list.append(s)
                elif cell == 'K':
                    s = sprites["key"].clone()
                    s.set_position(x, y)
                    sprites_list.append(s)
                elif cell == 'L':
                    s = sprites["key2"].clone()
                    s.set_position(x, y)
                    sprites_list.append(s)
                elif cell == 'D':
                    s = sprites["door"].clone()
                    s.set_position(x, y)
                    sprites_list.append(s)
                elif cell == 'W':
                    s = sprites["wall"].clone()
                    s.set_position(x, y)
                    sprites_list.append(s)
                elif cell == 'V':
                    s = sprites["wall2"].clone()
                    s.set_position(x, y)
                    sprites_list.append(s)
        
        return Level(
            sprites=sprites_list,
            grid_size=(width, height),
        )
    
    def _setup_levels(self) -> None:
        """Set up all levels."""
        self._levels = []
        for grid_str in level_grids:
            level = self._create_level(grid_str)
            self._levels.append(level)
    
    def step(self) -> None:
        """Handle game logic."""
        # Get current level sprites
        level = self.current_level
        avatar = None
        keys = []
        doors = []
        walls = []
        
        for sprite in level._sprites:
            if 'avatar' in sprite.tags:
                avatar = sprite
            elif 'key' in sprite.tags:
                keys.append(sprite)
            elif 'door' in sprite.tags:
                doors.append(sprite)
            elif 'wall' in sprite.tags:
                walls.append(sprite)
        
        if avatar is None:
            self.complete_action()
            return
        
        # Handle movement based on action
        dx, dy = 0, 0
        match self.action.id:
            case GameAction.ACTION1:  # Up
                dy = -1
            case GameAction.ACTION2:  # Down
                dy = 1
            case GameAction.ACTION3:  # Left
                dx = -1
            case GameAction.ACTION4:  # Right
                dx = 1
        
        # Try to move avatar
        if dx != 0 or dy != 0:
            # Check wall collisions
            can_move = True
            for wall in walls:
                # Simple bounding box check
                if (avatar.x + dx == wall.x and avatar.y + dy == wall.y):
                    can_move = False
                    break
            
            if can_move:
                avatar.move(dx, dy)
        
        # Check key collisions (collect keys)
        for key in keys[:]:  # Copy list to allow removal
            if avatar.collides_with(key):
                level._sprites.remove(key)
        
        # Check door collisions (win level)
        for door in doors:
            if avatar.collides_with(door):
                self.next_level()
                break
        
        self.complete_action()


if __name__ == "__main__":
    # Test the game
    game = CustomKD01()
    print(f"Game: {game._game_id}")
    print(f"Title: {game._title}")
    print(f"Levels: {len(level_grids)}")
