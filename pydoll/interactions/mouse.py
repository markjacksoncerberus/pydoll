"""
Mouse interaction API for browser automation with human-like cursor movement.

Provides methods for natural mouse movement, clicking, and dragging using
bezier curve trajectories that simulate realistic human cursor behavior.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Literal, Optional

from pydoll.commands import InputCommands
from pydoll.interactions.movement import (
    calculate_distance,
    generate_human_mouse_trajectory,
)
from pydoll.protocol.input.types import MouseButton, MouseEventType

if TYPE_CHECKING:
    from pydoll.browser.tab import Tab

logger = logging.getLogger(__name__)


class MouseAPI:
    """
    API for controlling mouse movement and interactions with human-like behavior.

    Provides methods for cursor movement using bezier curve trajectories,
    clicking, dragging, and other mouse actions through CDP Input domain.
    Movement simulates natural human cursor behavior with acceleration,
    deceleration, and subtle randomization.
    """

    def __init__(self, tab: Tab):
        """
        Initialize the MouseAPI with a tab instance.

        Args:
            tab: Tab instance to execute mouse commands on.
        """
        logger.debug(f'Initializing MouseAPI for tab: {tab}')
        self._tab = tab
        self._current_x: float = 0.0
        self._current_y: float = 0.0

    async def move_to(
        self,
        x: float,
        y: float,
        duration: float = 0.5,
        steps_per_second: int = 60,
        knots_count: int = 2,
        offset_boundary_x: float = 80.0,
        offset_boundary_y: float = 80.0,
    ):
        """
        Move mouse cursor to absolute coordinates using bezier curve trajectory.

        Generates a smooth, human-like path from current position to target
        coordinates using bezier curves with randomized control points.

        Args:
            x: Target x coordinate in CSS pixels.
            y: Target y coordinate in CSS pixels.
            duration: Movement duration in seconds.
            steps_per_second: Number of movement steps per second (affects smoothness).
            knots_count: Number of bezier curve control points (affects curve complexity).
            offset_boundary_x: Horizontal boundary offset for control point randomization.
            offset_boundary_y: Vertical boundary offset for control point randomization.

        Example:
            await tab.mouse.move_to(500, 300, duration=0.7)
        """
        logger.info(f'Moving mouse to ({x}, {y}) from ({self._current_x}, {self._current_y})')

        distance = calculate_distance(self._current_x, self._current_y, x, y)
        if distance < 1:
            logger.debug('Target position too close to current position, skipping movement')
            return

        target_points = max(int(duration * steps_per_second), 2)

        trajectory = generate_human_mouse_trajectory(
            from_point=(self._current_x, self._current_y),
            to_point=(x, y),
            knots_count=knots_count,
            distortion_mean=1.0,
            distortion_stdev=1.0,
            distortion_frequency=0.5,
            target_points=target_points,
            offset_boundary_x=offset_boundary_x,
            offset_boundary_y=offset_boundary_y,
        )

        await self._execute_trajectory(trajectory, duration)

        self._current_x = x
        self._current_y = y

    async def move_by(
        self,
        delta_x: float,
        delta_y: float,
        duration: float = 0.5,
        steps_per_second: int = 60,
        knots_count: int = 2,
    ):
        """
        Move mouse cursor by relative offset from current position.

        Args:
            delta_x: Horizontal offset in CSS pixels.
            delta_y: Vertical offset in CSS pixels.
            duration: Movement duration in seconds.
            steps_per_second: Number of movement steps per second.
            knots_count: Number of bezier curve control points.

        Example:
            await tab.mouse.move_by(100, -50, duration=0.3)
        """
        target_x = self._current_x + delta_x
        target_y = self._current_y + delta_y
        logger.info(f'Moving mouse by offset ({delta_x}, {delta_y}) to ({target_x}, {target_y})')

        await self.move_to(
            target_x,
            target_y,
            duration=duration,
            steps_per_second=steps_per_second,
            knots_count=knots_count,
        )

    async def click(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        button: MouseButton = MouseButton.LEFT,
        click_count: int = 1,
        move_duration: float = 0.5,
        hold_duration: float = 0.1,
    ):
        """
        Click at specified coordinates or current position.

        If coordinates are provided, moves to position first using bezier
        curve trajectory, then performs click action.

        Args:
            x: Target x coordinate (uses current position if None).
            y: Target y coordinate (uses current position if None).
            button: Mouse button to click.
            click_count: Number of clicks (1 for single, 2 for double).
            move_duration: Duration of movement to target (if x, y provided).
            hold_duration: Duration to hold button down before release.

        Example:
            await tab.mouse.click(500, 300)
            await tab.mouse.click(button=MouseButton.RIGHT)
        """
        if x is not None and y is not None:
            await self.move_to(x, y, duration=move_duration)

        logger.info(
            f'Clicking at ({self._current_x}, {self._current_y}) '
            f'with {button.value} button, count={click_count}'
        )

        press_command = InputCommands.dispatch_mouse_event(
            type=MouseEventType.MOUSE_PRESSED,
            x=int(self._current_x),
            y=int(self._current_y),
            button=button,
            click_count=click_count,
        )

        release_command = InputCommands.dispatch_mouse_event(
            type=MouseEventType.MOUSE_RELEASED,
            x=int(self._current_x),
            y=int(self._current_y),
            button=button,
            click_count=click_count,
        )

        await self._tab._execute_command(press_command)
        await asyncio.sleep(hold_duration)
        await self._tab._execute_command(release_command)

    async def double_click(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        button: MouseButton = MouseButton.LEFT,
        move_duration: float = 0.5,
    ):
        """
        Double-click at specified coordinates or current position.

        Args:
            x: Target x coordinate (uses current position if None).
            y: Target y coordinate (uses current position if None).
            button: Mouse button to click.
            move_duration: Duration of movement to target (if x, y provided).

        Example:
            await tab.mouse.double_click(500, 300)
        """
        await self.click(
            x=x,
            y=y,
            button=button,
            click_count=2,
            move_duration=move_duration,
            hold_duration=0.05,
        )

    async def drag(
        self,
        from_x: float,
        from_y: float,
        to_x: float,
        to_y: float,
        button: MouseButton = MouseButton.LEFT,
        move_to_start_duration: float = 0.5,
        drag_duration: float = 0.7,
    ):
        """
        Perform drag and drop operation from one position to another.

        Moves to start position, presses mouse button, drags to end position
        using bezier curve, then releases button.

        Args:
            from_x: Starting x coordinate.
            from_y: Starting y coordinate.
            to_x: Ending x coordinate.
            to_y: Ending y coordinate.
            button: Mouse button to use for dragging.
            move_to_start_duration: Duration to move to start position.
            drag_duration: Duration of the drag movement.

        Example:
            await tab.mouse.drag(100, 200, 500, 400, drag_duration=1.0)
        """
        logger.info(f'Drag operation: ({from_x}, {from_y}) -> ({to_x}, {to_y})')

        await self.move_to(from_x, from_y, duration=move_to_start_duration)

        press_command = InputCommands.dispatch_mouse_event(
            type=MouseEventType.MOUSE_PRESSED,
            x=int(from_x),
            y=int(from_y),
            button=button,
            click_count=1,
        )
        await self._tab._execute_command(press_command)

        await self.move_to(to_x, to_y, duration=drag_duration)

        release_command = InputCommands.dispatch_mouse_event(
            type=MouseEventType.MOUSE_RELEASED,
            x=int(to_x),
            y=int(to_y),
            button=button,
            click_count=1,
        )
        await self._tab._execute_command(release_command)

    async def scroll_wheel(
        self,
        delta_x: float = 0,
        delta_y: float = 0,
    ):
        """
        Scroll mouse wheel at current position.

        Args:
            delta_x: Horizontal scroll delta in CSS pixels.
            delta_y: Vertical scroll delta in CSS pixels (positive scrolls down).

        Example:
            await tab.mouse.scroll_wheel(delta_y=100)
        """
        logger.info(f'Scrolling wheel at ({self._current_x}, {self._current_y}): dx={delta_x}, dy={delta_y}')

        scroll_command = InputCommands.dispatch_mouse_event(
            type=MouseEventType.MOUSE_WHEEL,
            x=int(self._current_x),
            y=int(self._current_y),
            delta_x=delta_x,
            delta_y=delta_y,
        )
        await self._tab._execute_command(scroll_command)

    def get_position(self) -> tuple[float, float]:
        """
        Get current mouse cursor position.

        Returns:
            Tuple of (x, y) coordinates.

        Example:
            x, y = tab.mouse.get_position()
        """
        return (self._current_x, self._current_y)

    def set_position(self, x: float, y: float):
        """
        Set current mouse position without movement.

        Updates internal position tracking without dispatching mouse events.
        Useful for synchronizing position state after page navigation or
        external position changes.

        Args:
            x: New x coordinate.
            y: New y coordinate.
        """
        logger.debug(f'Setting mouse position to ({x}, {y})')
        self._current_x = x
        self._current_y = y

    async def _execute_trajectory(
        self,
        trajectory: list[tuple[float, float]],
        duration: float,
    ):
        """
        Execute a movement trajectory by dispatching mouse move events.

        Args:
            trajectory: List of (x, y) coordinates to move through.
            duration: Total duration for the complete trajectory.
        """
        if not trajectory:
            return

        step_delay = duration / max(len(trajectory) - 1, 1)

        for i, (x, y) in enumerate(trajectory):
            move_command = InputCommands.dispatch_mouse_event(
                type=MouseEventType.MOUSE_MOVED,
                x=int(x),
                y=int(y),
            )
            await self._tab._execute_command(move_command)

            if i < len(trajectory) - 1:
                await asyncio.sleep(step_delay)
