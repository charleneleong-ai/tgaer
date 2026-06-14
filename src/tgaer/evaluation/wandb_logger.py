from __future__ import annotations

from typing import Any

# ARC colour palette: grid index 0-15 -> RGB. Index 0 is the background/black.
ARC_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (170, 170, 170),
    (240, 18, 190),
    (255, 133, 27),
    (127, 219, 255),
    (135, 12, 37),
    (60, 60, 60),
    (0, 80, 160),
    (180, 40, 30),
    (30, 140, 40),
    (180, 150, 0),
    (90, 90, 90),
]


def build_logger(
    wandb_cfg: dict[str, Any] | None, run_config: dict[str, Any] | None = None
):
    """Construct a WandbRunLogger from a config's ``wandb:`` block, or return
    None when logging is disabled (the default) — keeps wandb fully opt-in and
    out of the test path."""
    wandb_cfg = wandb_cfg or {}
    if not wandb_cfg.get("enabled"):
        return None
    return WandbRunLogger(
        project=wandb_cfg.get("project", "tgaer-arc-agi3"),
        run_name=wandb_cfg.get("run_name"),
        run_config=run_config or {},
        log_images=wandb_cfg.get("log_images", True),
        image_every=int(wandb_cfg.get("image_every", 1)),
    )


def grid_to_rgb(frame: list[list[list[int]]] | None):
    """Render the most recent 64x64 grid in an ARC-AGI-3 frame to an HxWx3 uint8
    RGB array via the palette. Returns None if the frame is empty/malformed."""
    import numpy as np

    if not frame:
        return None
    arr = np.asarray(frame[-1], dtype=int)
    if arr.ndim != 2:
        return None
    palette = np.asarray(ARC_PALETTE, dtype=np.uint8)
    return palette[np.clip(arr, 0, len(palette) - 1)]


class WandbRunLogger:
    """Per-step metric + frame-image logger for one ARC-AGI-3 episode. Imports
    wandb lazily so the dependency is only touched when logging is enabled."""

    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        run_config: dict[str, Any] | None = None,
        log_images: bool = True,
        image_every: int = 1,
    ) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, config=run_config or {})
        self._log_images = log_images
        self._image_every = max(1, image_every)
        self._rows: list[list[Any]] = []  # trajectory table (incl. reasoning)

    @property
    def url(self) -> str:
        return getattr(self._run, "url", "") or ""

    def log_step(
        self,
        *,
        step: int,
        action_id: int | None,
        reward: float,
        score: float,
        levels_completed: int | None,
        guard_fired: bool,
        frame: list[list[list[int]]] | None = None,
        reasoning: str | None = None,
        reply: str | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "step": step,
            "reward": reward,
            "score": score,
            "levels_completed": levels_completed,
            "action_id": action_id,
            "guard_fired": int(guard_fired),
        }
        if self._log_images and step % self._image_every == 0:
            rgb = grid_to_rgb(frame)
            if rgb is not None:
                data["frame"] = self._wandb.Image(rgb, caption=f"step {step}")
        self._run.log(data)
        self._rows.append(
            [
                step,
                action_id,
                reward,
                score,
                levels_completed,
                int(guard_fired),
                (reasoning or "")[:4000],
                (reply or "")[:1000],
            ]
        )

    TRAJECTORY_COLUMNS = [
        "step",
        "action_id",
        "reward",
        "score",
        "levels",
        "guard_fired",
        "reasoning",
        "reply",
    ]

    def finish(self, summary: dict[str, Any]) -> None:
        if self._rows:
            table = self._wandb.Table(columns=self.TRAJECTORY_COLUMNS, data=self._rows)
            self._run.log({"trajectory": table})
        self._run.summary.update(summary)
        self._run.finish()
