"""Unit tests for mouse movement algorithms and MouseAPI."""

import math

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from pydoll.interactions.mouse import MouseAPI
from pydoll.interactions.movement import (
    apply_distortion,
    apply_easing,
    calculate_distance,
    generate_bezier_curve_points,
    generate_human_mouse_trajectory,
    generate_internal_knots,
)
from pydoll.protocol.input.types import MouseButton, MouseEventType


class TestMovementAlgorithms:
    """Test bezier curve and trajectory generation algorithms."""

    def test_calculate_distance(self):
        """Test Euclidean distance calculation."""
        distance = calculate_distance(0, 0, 3, 4)
        assert distance == 5.0

        distance = calculate_distance(100, 100, 100, 100)
        assert distance == 0.0

        distance = calculate_distance(0, 0, 100, 0)
        assert distance == 100.0

    def test_generate_internal_knots(self):
        """Test control point generation."""
        knots = generate_internal_knots(
            from_point=(0.0, 0.0),
            to_point=(100.0, 100.0),
            knots_count=2,
            offset_boundary_x=50.0,
            offset_boundary_y=50.0,
        )

        assert len(knots) == 2
        assert all(isinstance(k, tuple) and len(k) == 2 for k in knots)

    def test_generate_internal_knots_zero_count(self):
        """Test control point generation with zero knots."""
        knots = generate_internal_knots(
            from_point=(0.0, 0.0),
            to_point=(100.0, 100.0),
            knots_count=0,
            offset_boundary_x=50.0,
            offset_boundary_y=50.0,
        )

        assert len(knots) == 0

    def test_generate_bezier_curve_points(self):
        """Test bezier curve point generation."""
        points = generate_bezier_curve_points(
            from_point=(0.0, 0.0),
            to_point=(100.0, 100.0),
            internal_knots=[(50.0, 25.0)],
        )

        assert len(points) > 0
        assert points[0] == (0.0, 0.0)
        assert points[-1] == (100.0, 100.0)

    def test_apply_distortion(self):
        """Test distortion application to curve points."""
        original_points = [(0.0, 0.0), (50.0, 50.0), (100.0, 100.0)]

        distorted = apply_distortion(
            points=original_points,
            distortion_mean=0.0,
            distortion_stdev=1.0,
            distortion_frequency=1.0,
        )

        assert len(distorted) == len(original_points)
        assert distorted[0] == original_points[0]
        assert distorted[-1] == original_points[-1]

    def test_apply_distortion_zero_frequency(self):
        """Test distortion with zero frequency produces identical points."""
        original_points = [(0.0, 0.0), (50.0, 50.0), (100.0, 100.0)]

        distorted = apply_distortion(
            points=original_points,
            distortion_mean=0.0,
            distortion_stdev=1.0,
            distortion_frequency=0.0,
        )

        assert distorted == original_points

    def test_apply_easing(self):
        """Test easing function application."""
        original_points = [(i * 10.0, i * 10.0) for i in range(11)]

        eased = apply_easing(points=original_points, target_points=20)

        assert len(eased) == 20
        assert eased[0] == original_points[0]
        assert eased[-1] == original_points[-1]

    def test_apply_easing_minimum_points(self):
        """Test easing with minimum target points."""
        original_points = [(0.0, 0.0), (50.0, 50.0), (100.0, 100.0)]

        eased = apply_easing(points=original_points, target_points=1)

        assert len(eased) == 2

    def test_generate_human_mouse_trajectory(self):
        """Test complete trajectory generation."""
        trajectory = generate_human_mouse_trajectory(
            from_point=(0.0, 0.0),
            to_point=(500.0, 300.0),
            knots_count=2,
            target_points=50,
        )

        assert len(trajectory) == 50
        assert trajectory[0] == (0.0, 0.0)
        assert trajectory[-1] == (500.0, 300.0)

    def test_generate_human_mouse_trajectory_short_distance(self):
        """Test trajectory generation for short distance."""
        trajectory = generate_human_mouse_trajectory(
            from_point=(100.0, 100.0),
            to_point=(105.0, 102.0),
            knots_count=1,
            target_points=10,
        )

        assert len(trajectory) == 10
        assert trajectory[0] == (100.0, 100.0)
        assert trajectory[-1] == (105.0, 102.0)


@pytest_asyncio.fixture
async def mock_tab():
    """Mock Tab instance for MouseAPI tests."""
    tab = MagicMock()
    tab._execute_command = AsyncMock()
    return tab


@pytest_asyncio.fixture
async def mouse_api(mock_tab):
    """Create MouseAPI instance with mocked tab."""
    return MouseAPI(mock_tab)


class TestMouseAPIInitialization:
    """Test MouseAPI initialization."""

    def test_initialization(self, mock_tab):
        """Test MouseAPI is properly initialized with tab."""
        mouse_api = MouseAPI(mock_tab)
        assert mouse_api._tab == mock_tab
        assert mouse_api._current_x == 0.0
        assert mouse_api._current_y == 0.0

    def test_get_position_initial(self, mouse_api):
        """Test initial position retrieval."""
        x, y = mouse_api.get_position()
        assert x == 0.0
        assert y == 0.0

    def test_set_position(self, mouse_api):
        """Test position setting."""
        mouse_api.set_position(100.0, 200.0)
        x, y = mouse_api.get_position()
        assert x == 100.0
        assert y == 200.0


class TestMouseAPIMoveTo:
    """Test MouseAPI move_to method."""

    @pytest.mark.asyncio
    async def test_move_to_basic(self, mouse_api, mock_tab):
        """Test basic move_to operation."""
        await mouse_api.move_to(100.0, 200.0, duration=0.1, steps_per_second=10)

        assert mock_tab._execute_command.called
        assert mouse_api._current_x == 100.0
        assert mouse_api._current_y == 200.0

    @pytest.mark.asyncio
    async def test_move_to_generates_mouse_moved_events(self, mouse_api, mock_tab):
        """Test that move_to generates mouseMoved CDP events."""
        await mouse_api.move_to(50.0, 50.0, duration=0.05, steps_per_second=20)

        calls = mock_tab._execute_command.call_args_list
        assert len(calls) > 0

        for call in calls:
            command = call[0][0]
            assert command['method'] == 'Input.dispatchMouseEvent'
            assert command['params']['type'] == MouseEventType.MOUSE_MOVED

    @pytest.mark.asyncio
    async def test_move_to_skip_if_distance_too_small(self, mouse_api, mock_tab):
        """Test move_to skips if distance is negligible."""
        mouse_api.set_position(100.0, 100.0)
        await mouse_api.move_to(100.1, 100.1, duration=0.1)

        assert not mock_tab._execute_command.called

    @pytest.mark.asyncio
    async def test_move_to_updates_position(self, mouse_api):
        """Test move_to updates internal position tracking."""
        await mouse_api.move_to(300.0, 400.0, duration=0.05)

        x, y = mouse_api.get_position()
        assert x == 300.0
        assert y == 400.0


class TestMouseAPIMoveBy:
    """Test MouseAPI move_by method."""

    @pytest.mark.asyncio
    async def test_move_by_basic(self, mouse_api, mock_tab):
        """Test basic move_by operation."""
        mouse_api.set_position(100.0, 100.0)
        await mouse_api.move_by(50.0, -30.0, duration=0.05)

        assert mock_tab._execute_command.called
        assert mouse_api._current_x == 150.0
        assert mouse_api._current_y == 70.0

    @pytest.mark.asyncio
    async def test_move_by_from_origin(self, mouse_api):
        """Test move_by from origin position."""
        await mouse_api.move_by(200.0, 300.0, duration=0.05)

        x, y = mouse_api.get_position()
        assert x == 200.0
        assert y == 300.0


class TestMouseAPIClick:
    """Test MouseAPI click method."""

    @pytest.mark.asyncio
    async def test_click_at_current_position(self, mouse_api, mock_tab):
        """Test clicking at current position."""
        mouse_api.set_position(100.0, 200.0)
        await mouse_api.click(hold_duration=0.01)

        assert mock_tab._execute_command.call_count == 2

        press_call = mock_tab._execute_command.call_args_list[0]
        release_call = mock_tab._execute_command.call_args_list[1]

        press_command = press_call[0][0]
        assert press_command['params']['type'] == MouseEventType.MOUSE_PRESSED
        assert press_command['params']['x'] == 100
        assert press_command['params']['y'] == 200
        assert press_command['params']['button'] == MouseButton.LEFT

        release_command = release_call[0][0]
        assert release_command['params']['type'] == MouseEventType.MOUSE_RELEASED

    @pytest.mark.asyncio
    async def test_click_with_coordinates(self, mouse_api, mock_tab):
        """Test clicking with specific coordinates."""
        await mouse_api.click(x=300.0, y=400.0, move_duration=0.05, hold_duration=0.01)

        assert mock_tab._execute_command.called
        assert mouse_api._current_x == 300.0
        assert mouse_api._current_y == 400.0

    @pytest.mark.asyncio
    async def test_click_with_right_button(self, mouse_api, mock_tab):
        """Test right-click."""
        await mouse_api.click(button=MouseButton.RIGHT, hold_duration=0.01)

        press_call = mock_tab._execute_command.call_args_list[0]
        press_command = press_call[0][0]
        assert press_command['params']['button'] == MouseButton.RIGHT

    @pytest.mark.asyncio
    async def test_click_count(self, mouse_api, mock_tab):
        """Test click with custom click count."""
        await mouse_api.click(click_count=2, hold_duration=0.01)

        press_call = mock_tab._execute_command.call_args_list[0]
        press_command = press_call[0][0]
        assert press_command['params']['clickCount'] == 2


class TestMouseAPIDoubleClick:
    """Test MouseAPI double_click method."""

    @pytest.mark.asyncio
    async def test_double_click(self, mouse_api, mock_tab):
        """Test double-click operation."""
        await mouse_api.double_click(x=100.0, y=100.0, move_duration=0.05)

        assert mock_tab._execute_command.called

        press_call = mock_tab._execute_command.call_args_list[-2]
        press_command = press_call[0][0]
        assert press_command['params']['clickCount'] == 2


class TestMouseAPIDrag:
    """Test MouseAPI drag method."""

    @pytest.mark.asyncio
    async def test_drag_operation(self, mouse_api, mock_tab):
        """Test drag and drop operation."""
        await mouse_api.drag(
            from_x=100.0,
            from_y=100.0,
            to_x=300.0,
            to_y=200.0,
            move_to_start_duration=0.05,
            drag_duration=0.05,
        )

        assert mock_tab._execute_command.called

        calls = [call[0][0] for call in mock_tab._execute_command.call_args_list]

        press_events = [c for c in calls if c['params']['type'] == MouseEventType.MOUSE_PRESSED]
        release_events = [c for c in calls if c['params']['type'] == MouseEventType.MOUSE_RELEASED]

        assert len(press_events) == 1
        assert len(release_events) == 1

        assert mouse_api._current_x == 300.0
        assert mouse_api._current_y == 200.0


class TestMouseAPIScrollWheel:
    """Test MouseAPI scroll_wheel method."""

    @pytest.mark.asyncio
    async def test_scroll_wheel(self, mouse_api, mock_tab):
        """Test mouse wheel scroll."""
        mouse_api.set_position(500.0, 500.0)
        await mouse_api.scroll_wheel(delta_y=100.0)

        assert mock_tab._execute_command.called

        call = mock_tab._execute_command.call_args_list[0]
        command = call[0][0]

        assert command['params']['type'] == MouseEventType.MOUSE_WHEEL
        assert command['params']['deltaY'] == 100.0
        assert command['params']['x'] == 500
        assert command['params']['y'] == 500

    @pytest.mark.asyncio
    async def test_scroll_wheel_horizontal(self, mouse_api, mock_tab):
        """Test horizontal mouse wheel scroll."""
        await mouse_api.scroll_wheel(delta_x=50.0)

        call = mock_tab._execute_command.call_args_list[0]
        command = call[0][0]

        assert command['params']['deltaX'] == 50.0
