"""
Mouse interaction API for browser automation with human-like cursor movement.

Provides methods for natural mouse movement, clicking, and dragging using
bezier curve trajectories that simulate realistic human cursor behavior.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Optional

from pydoll.commands import InputCommands
from pydoll.constants import MouseMovement
from pydoll.interactions.movement import (
    calculate_distance,
    generate_human_mouse_trajectory,
)
from pydoll.protocol.input.types import MouseButton, MouseEventType

if TYPE_CHECKING:
    from pydoll.browser.tab import Tab

logger = logging.getLogger(__name__)


def _generate_random_duration(min_duration: float = 0.6, max_duration: float = 1.4) -> float:
    """
    Generate randomized movement duration for natural variation.

    Args:
        min_duration: Minimum duration in seconds.
        max_duration: Maximum duration in seconds.

    Returns:
        Random duration between min and max.
    """
    return random.uniform(min_duration, max_duration)


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
        self._humanize: bool = getattr(tab._browser, 'humanize_mouse_movement', True)

    async def move_to(
        self,
        x: float,
        y: float,
        duration: Optional[float] = None,
    ):
        """
        Move mouse cursor to absolute coordinates using bezier curve trajectory.

        Generates a smooth, human-like path from current position to target
        coordinates using bezier curves with randomized control points, distortion,
        and easing functions. If humanize_mouse_movement is disabled, instantly
        teleports to position.

        All trajectory parameters (knots, distortion, easing) are randomly selected
        per movement following HumanCursor best practices for maximum realism.
        Timing between points varies based on distance to create natural
        acceleration and deceleration.

        Args:
            x: Target x coordinate in CSS pixels.
            y: Target y coordinate in CSS pixels.
            duration: Movement duration in seconds. If None, uses random duration (0.6-1.4s).
            steps_per_second: Number of movement steps per second (affects smoothness).

        Example:
            await tab.mouse.move_to(500, 300)
            await tab.mouse.move_to(500, 300, duration=1.5)
        """
        logger.info(f'Moving mouse to ({x}, {y}) from ({self._current_x}, {self._current_y})')

        distance = calculate_distance(self._current_x, self._current_y, x, y)
        if distance < 1:
            logger.debug('Target position too close to current position, skipping movement')
            return

        if not self._humanize:
            # Teleport mode: instant movement with single mouseMoved event
            await self._teleport_to(x, y)
            self._current_x = x
            self._current_y = y
            return

        # Humanize mode: bezier curve trajectory with randomized parameters
        if duration is None:
            duration = _generate_random_duration()

        target_points = max(int(duration * MouseMovement.STEPS_PER_SECOND), 2)

        # Generate trajectory with all random parameters (HumanCursor approach)
        trajectory = generate_human_mouse_trajectory(
            from_point=(self._current_x, self._current_y),
            to_point=(x, y),
            target_points=target_points,
            # All other parameters will be randomly generated
        )

        await self._execute_trajectory(trajectory, duration)

        self._current_x = x
        self._current_y = y

    async def move_by(
        self,
        delta_x: float,
        delta_y: float,
        duration: Optional[float] = None,
    ):
        """
        Move mouse cursor by relative offset from current position.

        Args:
            delta_x: Horizontal offset in CSS pixels.
            delta_y: Vertical offset in CSS pixels.
            duration: Movement duration in seconds. If None, uses random duration (0.6-1.4s).

        Example:
            await tab.mouse.move_by(100, -50)
            await tab.mouse.move_by(100, -50, duration=0.3)
        """
        target_x = self._current_x + delta_x
        target_y = self._current_y + delta_y
        logger.info(f'Moving mouse by offset ({delta_x}, {delta_y}) to ({target_x}, {target_y})')

        await self.move_to(
            target_x,
            target_y,
            duration=duration,
        )

    async def click(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        button: MouseButton = MouseButton.LEFT,
        click_count: int = 1,
        move_duration: Optional[float] = None,
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
                          If None and humanize_mouse_movement is enabled,
                          uses random duration (0.3-0.7s).
            hold_duration: Duration to hold button down before release.

        Example:
            await tab.mouse.click(500, 300)
            await tab.mouse.click(button=MouseButton.RIGHT)
        """
        if x is not None and y is not None:
            # Use random duration if humanize enabled and no duration specified
            if move_duration is None and self._humanize:
                move_duration = _generate_random_duration()
            await self.move_to(x, y, duration=move_duration)

        logger.info(
            f'Clicking at ({self._current_x}, {self._current_y}) '
            f'with {button.value} button, count={click_count}'
        )

        press_command = InputCommands.dispatch_mouse_event(
            type=MouseEventType.MOUSE_PRESSED,
            x=self._current_x,
            y=self._current_y,
            button=button,
            click_count=click_count,
        )

        release_command = InputCommands.dispatch_mouse_event(
            type=MouseEventType.MOUSE_RELEASED,
            x=self._current_x,
            y=self._current_y,
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
        move_duration: Optional[float] = None,
    ):
        """
        Double-click at specified coordinates or current position.

        Args:
            x: Target x coordinate (uses current position if None).
            y: Target y coordinate (uses current position if None).
            button: Mouse button to click.
            move_duration: Duration of movement to target (if x, y provided).
                          If None and humanize_mouse_movement is enabled,
                          uses random duration (0.3-0.7s).

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

    async def left_click(
        self,
        x: float,
        y: float,
        move_duration: Optional[float] = None,
        hold_duration: float = 0.1,
    ):
        """
        Left-click at specified coordinates.

        Convenience method that moves to coordinates and performs left click.

        Args:
            x: Target x coordinate in CSS pixels.
            y: Target y coordinate in CSS pixels.
            move_duration: Duration of movement to target.
                          If None and humanize_mouse_movement is enabled,
                          uses random duration (0.3-0.7s).
            hold_duration: Duration to hold button down before release.

        Example:
            await tab.mouse.left_click(500, 300)
        """
        await self.click(
            x=x,
            y=y,
            button=MouseButton.LEFT,
            click_count=1,
            move_duration=move_duration,
            hold_duration=hold_duration,
        )

    async def right_click(
        self,
        x: float,
        y: float,
        move_duration: Optional[float] = None,
        hold_duration: float = 0.1,
    ):
        """
        Right-click at specified coordinates.

        Convenience method that moves to coordinates and performs right click.

        Args:
            x: Target x coordinate in CSS pixels.
            y: Target y coordinate in CSS pixels.
            move_duration: Duration of movement to target.
                          If None and humanize_mouse_movement is enabled,
                          uses random duration (0.3-0.7s).
            hold_duration: Duration to hold button down before release.

        Example:
            await tab.mouse.right_click(500, 300)
        """
        await self.click(
            x=x,
            y=y,
            button=MouseButton.RIGHT,
            click_count=1,
            move_duration=move_duration,
            hold_duration=hold_duration,
        )

    async def middle_click(
        self,
        x: float,
        y: float,
        move_duration: Optional[float] = None,
        hold_duration: float = 0.1,
    ):
        """
        Middle-click at specified coordinates.

        Convenience method that moves to coordinates and performs middle click.

        Args:
            x: Target x coordinate in CSS pixels.
            y: Target y coordinate in CSS pixels.
            move_duration: Duration of movement to target.
                          If None and humanize_mouse_movement is enabled,
                          uses random duration (0.3-0.7s).
            hold_duration: Duration to hold button down before release.

        Example:
            await tab.mouse.middle_click(500, 300)
        """
        await self.click(
            x=x,
            y=y,
            button=MouseButton.MIDDLE,
            click_count=1,
            move_duration=move_duration,
            hold_duration=hold_duration,
        )

    async def drag(
        self,
        from_x: float,
        from_y: float,
        to_x: float,
        to_y: float,
        button: MouseButton = MouseButton.LEFT,
        move_to_start_duration: Optional[float] = None,
        drag_duration: Optional[float] = None,
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
                                   If None and humanize_mouse_movement is
                                   enabled, uses random duration (0.3-0.7s).
            drag_duration: Duration of the drag movement.
                          If None and humanize_mouse_movement is enabled,
                          uses random duration (0.3-0.7s).

        Example:
            await tab.mouse.drag(100, 200, 500, 400, drag_duration=1.0)
        """
        logger.info(f'Drag operation: ({from_x}, {from_y}) -> ({to_x}, {to_y})')

        # Use random durations if humanize enabled and no durations specified
        if move_to_start_duration is None and self._humanize:
            move_to_start_duration = _generate_random_duration()
        if drag_duration is None and self._humanize:
            drag_duration = _generate_random_duration()

        await self.move_to(from_x, from_y, duration=move_to_start_duration)

        press_command = InputCommands.dispatch_mouse_event(
            type=MouseEventType.MOUSE_PRESSED,
            x=from_x,
            y=from_y,
            button=button,
            click_count=1,
        )
        await self._tab._execute_command(press_command)

        await self.move_to(to_x, to_y, duration=drag_duration)

        release_command = InputCommands.dispatch_mouse_event(
            type=MouseEventType.MOUSE_RELEASED,
            x=to_x,
            y=to_y,
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
        logger.info(
            f'Scrolling wheel at ({self._current_x}, {self._current_y}): dx={delta_x}, dy={delta_y}'
        )

        scroll_command = InputCommands.dispatch_mouse_event(
            type=MouseEventType.MOUSE_WHEEL,
            x=self._current_x,
            y=self._current_y,
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

        Uses non-uniform timing based on distance between consecutive points
        to create smooth acceleration/deceleration matching the easing function.

        Args:
            trajectory: List of (x, y) coordinates to move through.
            duration: Total duration for the complete trajectory.
        """
        if not trajectory:
            return

        # Calculate total path distance
        total_distance = 0.0
        for i in range(len(trajectory) - 1):
            x1, y1 = trajectory[i]
            x2, y2 = trajectory[i + 1]
            segment_distance = calculate_distance(x1, y1, x2, y2)
            total_distance += segment_distance

        # Dispatch first point immediately
        if trajectory:
            x, y = trajectory[0]
            move_command = InputCommands.dispatch_mouse_event(
                type=MouseEventType.MOUSE_MOVED,
                x=x,  # Keep float precision
                y=y,
            )
            await self._tab._execute_command(move_command)

        # Execute remaining points with timing proportional to distance
        for i in range(1, len(trajectory)):
            x1, y1 = trajectory[i - 1]
            x2, y2 = trajectory[i]

            # Calculate delay based on segment distance relative to total distance
            segment_distance = calculate_distance(x1, y1, x2, y2)
            if total_distance > 0:
                delay = (segment_distance / total_distance) * duration
            else:
                delay = duration / max(len(trajectory) - 1, 1)

            await asyncio.sleep(delay)

            move_command = InputCommands.dispatch_mouse_event(
                type=MouseEventType.MOUSE_MOVED,
                x=x2,  # Keep float precision
                y=y2,
            )
            await self._tab._execute_command(move_command)

    async def _teleport_to(self, x: float, y: float):
        """
        Instantly move mouse to position without trajectory (teleport mode).

        Args:
            x: Target x coordinate in CSS pixels.
            y: Target y coordinate in CSS pixels.
        """
        move_command = InputCommands.dispatch_mouse_event(
            type=MouseEventType.MOUSE_MOVED,
            x=x,  # Keep float precision
            y=y,
        )
        await self._tab._execute_command(move_command)
