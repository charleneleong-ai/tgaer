from __future__ import annotations

"""
Vendored local game definitions for PrimeIntellect evaluation.

These are copied from the ArcEngine example games because the published
`arcengine` package currently ships the core engine module but does not ship
the `examples/` package path needed to import `SimpleMaze` / `ComplexMaze`
directly at runtime.
"""

from arcengine import ARCBaseGame, Camera, GameAction, InteractionMode, Level, Sprite, ToggleableUserDisplay


def build_simple_maze() -> ARCBaseGame:
    sprites = {
        "player": Sprite(
            pixels=[[8]],
            name="player",
            visible=True,
            collidable=True,
        ),
        "exit": Sprite(
            pixels=[[9]],
            name="exit",
            visible=True,
            collidable=True,
        ),
        "maze_1": Sprite(
            pixels=[
                [5, 5, 5, 5, 5, 5, 5, 5],
                [5, -1, -1, -1, 5, -1, -1, 5],
                [5, -1, 5, -1, 5, -1, 5, 5],
                [5, -1, 5, -1, -1, -1, -1, 5],
                [5, -1, 5, 5, 5, 5, -1, 5],
                [5, -1, -1, -1, -1, 5, -1, 5],
                [5, 5, 5, 5, -1, -1, -1, 5],
                [5, 5, 5, 5, 5, 5, 5, 5],
            ],
            name="maze_1",
            visible=True,
            collidable=True,
            layer=-1,
        ),
        "maze_2": Sprite(
            pixels=[
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, 5],
                [5, -1, 5, -1, 5, -1, 5, 5, 5, 5, -1, 5],
                [5, -1, 5, -1, -1, -1, -1, -1, -1, 5, -1, 5],
                [5, -1, 5, 5, 5, 5, 5, 5, -1, 5, -1, 5],
                [5, -1, -1, -1, -1, -1, -1, 5, -1, 5, -1, 5],
                [5, 5, 5, 5, 5, 5, -1, 5, -1, 5, -1, 5],
                [5, -1, -1, -1, -1, 5, -1, 5, -1, 5, -1, 5],
                [5, -1, 5, 5, -1, 5, -1, 5, -1, 5, -1, 5],
                [5, -1, 5, -1, -1, 5, -1, -1, -1, 5, -1, 5],
                [5, -1, -1, -1, 5, 5, 5, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ],
            name="maze_2",
            visible=True,
            collidable=True,
            layer=-1,
        ),
    }

    levels = [
        Level(
            sprites=[
                sprites["maze_1"].clone(),
                sprites["player"].clone().set_position(1, 1),
                sprites["exit"].clone().set_position(6, 6),
            ],
            grid_size=(8, 8),
        ),
        Level(
            sprites=[
                sprites["maze_2"].clone(),
                sprites["player"].clone().set_position(1, 1),
                sprites["exit"].clone().set_position(10, 10),
            ],
            grid_size=(12, 12),
        ),
    ]

    class SimpleMaze(ARCBaseGame):
        def __init__(self) -> None:
            super().__init__(
                game_id="simple_maze",
                levels=levels,
                camera=Camera(background=0, letter_box=3),
            )

        def step(self) -> None:
            dx = 0
            dy = 0
            if self.action.id == GameAction.ACTION1:
                dy = -1
            elif self.action.id == GameAction.ACTION2:
                dy = 1
            elif self.action.id == GameAction.ACTION3:
                dx = -1
            elif self.action.id == GameAction.ACTION4:
                dx = 1

            collided = self.try_move("player", dx, dy)
            if collided and any(sprite.name == "exit" for sprite in collided):
                if self.is_last_level():
                    self.win()
                else:
                    self.next_level()
            self.complete_action()

    return SimpleMaze()


def build_complex_maze() -> ARCBaseGame:
    sprites = {
        "block_orange": Sprite(pixels=[[12]], name="block_orange", visible=True, collidable=True, tags=["fixed"]),
        "block_orange_flex": Sprite(pixels=[[12]], name="block_orange_flex", visible=True, collidable=True, tags=["floating"]),
        "energy_pill": Sprite(pixels=[[6, 6], [6, 6]], name="energy_pill", visible=True, collidable=False, tags=["energy"]),
        "energy_pill_off": Sprite(pixels=[[3, 3], [3, 3]], name="energy_pill_off", visible=False, collidable=False),
        "exit": Sprite(pixels=[[9]], name="exit", visible=True, collidable=True, tags=["fixed"]),
        "maze_1": Sprite(
            pixels=[
                [5, 5, 5, 5, 5, 5, 5, 5],
                [5, -1, -1, -1, 5, -1, -1, 5],
                [5, -1, 5, -1, 5, -1, 5, 5],
                [5, -1, 5, -1, -1, -1, -1, 5],
                [5, -1, 5, 5, 5, 5, -1, 5],
                [5, -1, -1, -1, -1, 5, -1, 5],
                [5, 5, 5, 5, -1, 5, -1, 5],
                [5, 5, 5, 5, 5, 5, 5, 5],
            ],
            name="maze_1",
            visible=True,
            collidable=True,
            layer=-1,
        ),
        "maze_2": Sprite(
            pixels=[
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, 5],
                [5, -1, 5, -1, 5, -1, 5, 5, 5, 5, -1, 5],
                [5, -1, 5, -1, -2, -1, 5, 5, -1, 5, -1, 5],
                [5, -1, 5, -1, -2, -1, 5, 5, -1, 5, -1, 5],
                [5, -1, 5, -1, -2, -1, 5, 5, -1, 5, -1, 5],
                [5, -1, 5, -1, -2, -1, 5, 5, -1, 5, -1, 5],
                [5, -1, 5, -1, -1, -1, 5, 5, -1, 5, -1, 5],
                [5, -1, 5, 5, 5, 5, 5, 5, -1, 5, -1, 5],
                [5, -1, 5, 5, 5, 5, 4, 4, -1, 5, -1, 5],
                [5, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ],
            name="maze_2",
            visible=True,
            collidable=True,
            layer=-1,
        ),
        "maze_3": Sprite(
            pixels=[
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, 5],
                [5, 5, 5, 5, -1, 5, -1, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, -1, 5, -1, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, -1, 5, -1, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, -1, 5, -1, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, -1, -2, -1, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, -1, -2, -1, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, -1, -2, -1, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, -1, -1, -1, 5, -1, -1, -1, 5],
                [5, 5, 5, 5, -1, -2, -2, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ],
            name="maze_3",
            visible=True,
            collidable=True,
            layer=-1,
        ),
        "maze_4": Sprite(
            pixels=[
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, -1, 5, -1, -1, -1, -1, -1, -1, 5, -1, 5],
                [5, -1, 5, 5, 5, 5, 5, -1, 5, 5, -1, 5],
                [5, -1, 5, 5, 5, 5, 5, -1, 5, 5, -1, 5],
                [5, -1, 5, 5, 5, 5, 5, -1, -2, -2, -1, 5],
                [5, -1, 5, 5, 5, 5, 5, -1, 5, 5, -1, 5],
                [5, -1, -1, -1, 5, 5, 5, -1, 5, 5, -1, 5],
                [5, -1, -1, -1, 5, 5, 5, -1, 5, 5, -1, 5],
                [5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ],
            name="maze_4",
            visible=True,
            collidable=True,
            layer=-1,
        ),
        "maze_5": Sprite(
            pixels=[
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                [5, -1, 5, 5, 5, 5, 5, 5, 5, 5, -1, 5],
                [5, -1, 5, 5, 5, 5, 5, 5, 5, 5, -1, 5],
                [5, -1, 5, 5, 5, 5, 5, 5, 5, 5, -1, 5],
                [5, -1, 5, 5, 5, 5, 5, 5, 5, 5, -1, 5],
                [5, -1, 5, 5, 5, 5, 5, 5, 5, 5, -1, 5],
                [5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, 5],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            ],
            name="maze_5",
            visible=True,
            collidable=True,
            layer=-1,
        ),
        "player": Sprite(pixels=[[8]], name="player", visible=True, collidable=True, tags=["fixed"]),
    }

    levels = [
        Level(
            sprites=[sprites["exit"].clone().set_position(6, 6), sprites["maze_1"].clone(), sprites["player"].clone().set_position(1, 1)],
            grid_size=(8, 8),
            data={"move_maze": False},
        ),
        Level(
            sprites=[sprites["exit"].clone().set_position(10, 10), sprites["maze_2"].clone(), sprites["player"].clone().set_position(1, 1)],
            grid_size=(12, 12),
            data={"move_maze": False},
        ),
        Level(
            sprites=[
                sprites["block_orange"].clone().set_position(4, 2),
                sprites["block_orange"].clone().set_position(10, 2),
                sprites["exit"].clone().set_position(8, 9),
                sprites["maze_3"].clone(),
                sprites["player"].clone().set_position(1, 1),
            ],
            grid_size=(12, 12),
            data={"move_maze": False},
        ),
        Level(
            sprites=[
                sprites["block_orange"].clone().set_position(3, 8),
                sprites["block_orange"].clone().set_position(10, 8),
                sprites["exit"].clone().set_position(10, 1),
                sprites["maze_4"].clone(),
                sprites["player"].clone().set_position(1, 1),
            ],
            grid_size=(12, 12),
            data={"move_maze": False},
        ),
        Level(
            sprites=[
                sprites["block_orange"].clone().set_position(10, 6),
                sprites["block_orange_flex"].clone().set_position(5, 8),
                sprites["exit"].clone().set_position(10, 1),
                sprites["maze_5"].clone(),
                sprites["player"].clone().set_position(1, 1),
            ],
            grid_size=(12, 12),
            data={"move_maze": True, "max_delta_x": 0, "max_delta_y": 2},
        ),
    ]

    class ComplexMaze(ARCBaseGame):
        _ui: ToggleableUserDisplay

        def __init__(self) -> None:
            sprite_pairs = []
            for i in range(32):
                sprite_pairs.append((sprites["energy_pill"].clone().set_position(i * 2, 0), sprites["energy_pill_off"].clone().set_position(i * 2, 0)))
            for i in range(31):
                sprite_pairs.append((sprites["energy_pill"].clone().set_position(62, i * 2 + 2), sprites["energy_pill_off"].clone().set_position(62, i * 2 + 2)))
            self._ui = ToggleableUserDisplay(sprite_pairs)
            camera = Camera(width=8, height=8, background=0, letter_box=0, interfaces=[self._ui])
            super().__init__(game_id="complex_maze", levels=levels, camera=camera)

        def step(self) -> None:
            dx = 0
            dy = 0
            if self.action.id == GameAction.ACTION1:
                dy = -1
            elif self.action.id == GameAction.ACTION2:
                dy = 1
            elif self.action.id == GameAction.ACTION3:
                dx = -1
            elif self.action.id == GameAction.ACTION4:
                dx = 1

            self._try_pushing_move(dx, dy)

            if self.action.id != GameAction.RESET and not self._ui.disabled_first_by_tag("energy"):
                self.lose()

            self.complete_action()

        def on_set_level(self, level: Level) -> None:
            self._ui.enable_all_by_tag("energy")

        def _try_pushing_move(self, dx: int, dy: int) -> None:
            collided = self.try_move("player", dx, dy)

            if collided and any(sprite.name == "exit" for sprite in collided):
                self.next_level()

            if collided:
                for sprite in collided:
                    if sprite.name.startswith("block"):
                        block_collided = self.try_move_sprite(sprite, dx, dy)
                        if block_collided:
                            for other_sprite in block_collided:
                                if sprite.name.startswith(other_sprite.name):
                                    sprite.set_interaction(InteractionMode.REMOVED)
                                    other_sprite.set_interaction(InteractionMode.REMOVED)
                                    break
                        else:
                            self.try_move("player", dx, dy)

                    if self.current_level.get_data("move_maze") and sprite.name.startswith("maze"):
                        sprite_dx = abs(sprite.x + dx - 0)
                        sprite_dy = abs(sprite.y + dy - 0)
                        if sprite_dx <= self.current_level.get_data("max_delta_x") and sprite_dy <= self.current_level.get_data("max_delta_y"):
                            sprite.move(dx, dy)
                            fixed_sprites = self.current_level.get_sprites_by_tag("fixed")
                            for fixed_sprite in fixed_sprites:
                                fixed_sprite.move(dx, dy)

    return ComplexMaze()
