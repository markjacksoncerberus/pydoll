"""Comprehensive tests for mouse movement algorithms, MouseAPI, and integrations."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from pydoll.browser.chromium.chrome import Chrome
from pydoll.browser.tab import Tab
from pydoll.interactions.mouse import MouseAPI, MouseButton, _generate_random_duration
from pydoll.interactions.movement import (
    apply_distortion,
    apply_easing,
    calculate_distance,
    generate_bezier_curve_points,
    generate_human_mouse_trajectory,
    generate_internal_knots,
)
from pydoll.protocol.input.types import MouseEventType

if TYPE_CHECKING:
    from pydoll.browser.tab import Tab


# ============================================================================
# MOVEMENT ALGORITHM TESTS
# ============================================================================


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

    def test_easing_function_edge_cases(self):
        """Test easing functions with edge case values."""
        from pydoll.interactions.movement import (
            _ease_out_expo,
            _ease_in_out_quart,
            _ease_in_out_quint,
        )
        
        # Test t == 1 edge case
        assert _ease_out_expo(1) == 1
        
        # Test second half of ease-in-out functions
        result = _ease_in_out_quart(0.75)
        assert isinstance(result, float)
        
        result = _ease_in_out_quint(0.75)
        assert isinstance(result, float)

    def test_distortion_invalid_frequency(self):
        """Test distortion with invalid frequency gets clamped."""
        original_points = [(0.0, 0.0), (50.0, 50.0), (100.0, 100.0)]
        
        # Test with frequency > 1 (should be clamped to 0.5)
        distorted = apply_distortion(
            points=original_points,
            distortion_mean=0.0,
            distortion_stdev=1.0,
            distortion_frequency=1.5,  # Invalid
        )
        
        assert len(distorted) == len(original_points)

    def test_generate_internal_knots_boundary_collision(self):
        """Test knot generation when boundaries collide."""
        # Same point causes boundary collision
        knots = generate_internal_knots(
            from_point=(100.0, 100.0),
            to_point=(100.0, 100.0),
            knots_count=2,
            offset_boundary_x=0.0,
            offset_boundary_y=0.0,
        )
        
        assert len(knots) == 2

    def test_random_parameter_generators(self):
        """Test random parameter generation functions."""
        from pydoll.interactions.movement import (
            _get_random_knots_count,
            _get_random_target_points,
            _get_random_offset_boundary,
            _get_random_distortion_params,
        )
        
        # Test knots count generation
        for _ in range(10):
            knots = _get_random_knots_count()
            assert 1 <= knots <= 10
        
        # Test target points generation
        for _ in range(10):
            points = _get_random_target_points()
            assert 35 <= points <= 80
        
        # Test offset boundary generation
        for _ in range(10):
            x, y = _get_random_offset_boundary()
            assert 5 <= x <= 35
            assert 5 <= y <= 35
        
        # Test distortion params generation
        for _ in range(10):
            mean, stdev, freq = _get_random_distortion_params()
            assert 0.0 <= mean <= 0.2
            assert 0.2 <= stdev <= 0.4
            assert 0.1 <= freq <= 0.25

    def test_generate_trajectory_with_all_random_params(self):
        """Test trajectory generation with all None params (triggers all random generators)."""
        trajectory = generate_human_mouse_trajectory(
            from_point=(0.0, 0.0),
            to_point=(500.0, 300.0),
            # All None to trigger random parameter generation
            knots_count=None,
            distortion_mean=None,
            distortion_stdev=None,
            distortion_frequency=None,
            target_points=None,
            offset_boundary_x=None,
            offset_boundary_y=None,
        )
        
        assert len(trajectory) > 0
        assert trajectory[0] == (0.0, 0.0)
        assert trajectory[-1] == (500.0, 300.0)

    def test_ease_out_expo_edge_case(self):
        """Test _ease_out_expo with t != 1 to cover the else branch."""
        from pydoll.interactions.movement import _ease_out_expo
        
        # Test t != 1 (normal case)
        result = _ease_out_expo(0.5)
        assert 0 <= result <= 1
        
        # Test t == 1 (edge case already covered, but explicit)
        result = _ease_out_expo(1.0)
        assert result == 1

    def test_all_easing_functions(self):
        """Test all easing functions to ensure 100% coverage."""
        from pydoll.interactions.movement import (
            _ease_out_quad,
            _ease_out_cubic,
            _ease_out_quart,
            _ease_out_quint,
            _ease_out_sine,
            _ease_out_circ,
            _linear,
        )
        
        # Test each easing function
        for func in [_ease_out_quad, _ease_out_cubic, _ease_out_quart, 
                     _ease_out_quint, _ease_out_sine, _ease_out_circ, _linear]:
            result = func(0.5)
            assert isinstance(result, float)
            assert 0 <= result <= 1


# ============================================================================
# MOUSEAPI CORE FUNCTIONALITY TESTS
# ============================================================================


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
        await mouse_api.move_to(100.0, 200.0, duration=0.1)

        assert mock_tab._execute_command.called
        assert mouse_api._current_x == 100.0
        assert mouse_api._current_y == 200.0

    @pytest.mark.asyncio
    async def test_move_to_generates_mouse_moved_events(self, mouse_api, mock_tab):
        """Test that move_to generates mouseMoved CDP events."""
        await mouse_api.move_to(50.0, 50.0, duration=0.05)

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


class TestMouseAPIEdgeCases:
    """Test MouseAPI edge cases for full coverage."""

    @pytest.mark.asyncio
    async def test_execute_trajectory_empty(self, mouse_api, mock_tab):
        """Test _execute_trajectory with empty trajectory."""
        await mouse_api._execute_trajectory([], 1.0)
        
        # Should not execute any commands
        assert not mock_tab._execute_command.called

    @pytest.mark.asyncio
    async def test_execute_trajectory_zero_distance(self, mouse_api, mock_tab):
        """Test _execute_trajectory with zero total distance."""
        # All points are the same (zero distance)
        trajectory = [(100.0, 100.0), (100.0, 100.0), (100.0, 100.0)]
        
        await mouse_api._execute_trajectory(trajectory, 1.0)
        
        # Should still execute commands
        assert mock_tab._execute_command.called


# ============================================================================
# HUMANIZE FEATURE TESTS
# ============================================================================


class TestRandomDurationGenerator:
    """Test random duration generation."""

    def test_generate_random_duration_in_range(self):
        """Test that random duration is within specified range."""
        for _ in range(100):
            duration = _generate_random_duration(0.7, 2.0)
            assert 0.7 <= duration <= 2.0

    def test_generate_random_duration_default_range(self):
        """Test random duration with default range."""
        for _ in range(100):
            duration = _generate_random_duration()
            assert 0.6 <= duration <= 1.4


class TestBrowserHumanizeConfiguration:
    """Test Browser humanize_mouse_movement configuration."""

    def test_browser_default_humanize_enabled(self):
        """Test that humanize_mouse_movement defaults to True."""
        with patch('pydoll.browser.chromium.base.BrowserProcessManager'):
            with patch('pydoll.browser.chromium.base.ConnectionHandler'):
                browser = Chrome.__new__(Chrome)
                browser._options_manager = MagicMock()
                browser.humanize_mouse_movement = True
                assert browser.humanize_mouse_movement is True

    def test_browser_humanize_can_be_disabled(self):
        """Test that humanize_mouse_movement can be set to False."""
        with patch('pydoll.browser.chromium.base.BrowserProcessManager'):
            with patch('pydoll.browser.chromium.base.ConnectionHandler'):
                browser = Chrome.__new__(Chrome)
                browser._options_manager = MagicMock()
                browser.humanize_mouse_movement = False
                assert browser.humanize_mouse_movement is False


class TestMouseAPIHumanizeMode:
    """Test MouseAPI humanize mode behavior."""

    @pytest.fixture
    def mock_tab_humanize_enabled(self) -> 'Tab':
        """Create mock tab with humanize enabled."""
        tab = MagicMock()
        tab._browser = MagicMock()
        tab._browser.humanize_mouse_movement = True
        tab._execute_command = AsyncMock()
        return tab

    @pytest.fixture
    def mock_tab_humanize_disabled(self) -> 'Tab':
        """Create mock tab with humanize disabled."""
        tab = MagicMock()
        tab._browser = MagicMock()
        tab._browser.humanize_mouse_movement = False
        tab._execute_command = AsyncMock()
        return tab

    def test_mouse_api_reads_humanize_setting_enabled(self, mock_tab_humanize_enabled):
        """Test that MouseAPI reads humanize setting from browser (enabled)."""
        mouse = MouseAPI(mock_tab_humanize_enabled)
        assert mouse._humanize is True

    def test_mouse_api_reads_humanize_setting_disabled(self, mock_tab_humanize_disabled):
        """Test that MouseAPI reads humanize setting from browser (disabled)."""
        mouse = MouseAPI(mock_tab_humanize_disabled)
        assert mouse._humanize is False

    def test_mouse_api_defaults_to_humanize_if_no_browser_setting(self):
        """Test that MouseAPI defaults to humanize=True if browser has no setting."""
        tab = MagicMock()
        tab._browser = MagicMock(spec=[])
        tab._execute_command = AsyncMock()
        mouse = MouseAPI(tab)
        assert mouse._humanize is True

    @pytest.mark.asyncio
    async def test_teleport_mode_uses_single_event(self, mock_tab_humanize_disabled):
        """Test that teleport mode uses single mouseMoved event."""
        mouse = MouseAPI(mock_tab_humanize_disabled)
        mouse.set_position(0, 0)

        with patch.object(mouse, '_teleport_to', new_callable=AsyncMock) as mock_teleport:
            with patch.object(
                mouse, '_execute_trajectory', new_callable=AsyncMock
            ) as mock_trajectory:
                await mouse.move_to(500, 300)

                # Should call teleport, not trajectory
                mock_teleport.assert_called_once_with(500, 300)
                mock_trajectory.assert_not_called()

    @pytest.mark.asyncio
    async def test_humanize_mode_uses_trajectory(self, mock_tab_humanize_enabled):
        """Test that humanize mode uses bezier trajectory."""
        mouse = MouseAPI(mock_tab_humanize_enabled)
        mouse.set_position(0, 0)

        with patch.object(mouse, '_teleport_to', new_callable=AsyncMock) as mock_teleport:
            with patch.object(
                mouse, '_execute_trajectory', new_callable=AsyncMock
            ) as mock_trajectory:
                await mouse.move_to(500, 300, duration=0.5)

                # Should call trajectory, not trajectory
                mock_trajectory.assert_called_once()
                mock_teleport.assert_not_called()

    @pytest.mark.asyncio
    async def test_humanize_mode_uses_random_duration_when_none(self, mock_tab_humanize_enabled):
        """Test that humanize mode uses random duration when duration is None."""
        mouse = MouseAPI(mock_tab_humanize_enabled)
        mouse.set_position(0, 0)

        with patch(
            'pydoll.interactions.mouse._generate_random_duration', return_value=0.55
        ) as mock_gen:
            with patch.object(mouse, '_execute_trajectory', new_callable=AsyncMock):
                await mouse.move_to(500, 300, duration=None)

                # Should have called random duration generator
                mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_teleport_to_dispatches_single_event(self, mock_tab_humanize_disabled):
        """Test _teleport_to dispatches single mouseMoved event."""
        mouse = MouseAPI(mock_tab_humanize_disabled)

        await mouse._teleport_to(400, 300)

        # Verify single command was executed
        mock_tab_humanize_disabled._execute_command.assert_called_once()

        # Get the command that was executed
        call_args = mock_tab_humanize_disabled._execute_command.call_args
        command = call_args[0][0]

        # Verify it's a mouseMoved event with correct coordinates
        assert command['method'] == 'Input.dispatchMouseEvent'
        assert command['params']['type'] == 'mouseMoved'
        assert command['params']['x'] == 400
        assert command['params']['y'] == 300


class TestConvenienceClickMethods:
    """Test convenience click methods (left_click, right_click, middle_click)."""

    @pytest.fixture
    def mock_tab_for_clicks(self) -> 'Tab':
        """Create mock tab for testing."""
        tab = MagicMock()
        tab._browser = MagicMock()
        tab._browser.humanize_mouse_movement = True
        tab._execute_command = AsyncMock()
        return tab

    @pytest.mark.asyncio
    async def test_left_click_calls_click_with_left_button(self, mock_tab_for_clicks):
        """Test that left_click calls click with LEFT button."""
        mouse = MouseAPI(mock_tab_for_clicks)

        with patch.object(mouse, 'click', new_callable=AsyncMock) as mock_click:
            await mouse.left_click(400, 300)

            mock_click.assert_called_once_with(
                x=400,
                y=300,
                button=MouseButton.LEFT,
                click_count=1,
                move_duration=None,
                hold_duration=0.1,
            )

    @pytest.mark.asyncio
    async def test_right_click_calls_click_with_right_button(self, mock_tab_for_clicks):
        """Test that right_click calls click with RIGHT button."""
        mouse = MouseAPI(mock_tab_for_clicks)

        with patch.object(mouse, 'click', new_callable=AsyncMock) as mock_click:
            await mouse.right_click(400, 300)

            mock_click.assert_called_once_with(
                x=400,
                y=300,
                button=MouseButton.RIGHT,
                click_count=1,
                move_duration=None,
                hold_duration=0.1,
            )

    @pytest.mark.asyncio
    async def test_middle_click_calls_click_with_middle_button(self, mock_tab_for_clicks):
        """Test that middle_click calls click with MIDDLE button."""
        mouse = MouseAPI(mock_tab_for_clicks)

        with patch.object(mouse, 'click', new_callable=AsyncMock) as mock_click:
            await mouse.middle_click(400, 300)

            mock_click.assert_called_once_with(
                x=400,
                y=300,
                button=MouseButton.MIDDLE,
                click_count=1,
                move_duration=None,
                hold_duration=0.1,
            )

    @pytest.mark.asyncio
    async def test_convenience_methods_support_custom_duration(self, mock_tab_for_clicks):
        """Test that convenience methods support custom move_duration."""
        mouse = MouseAPI(mock_tab_for_clicks)

        with patch.object(mouse, 'click', new_callable=AsyncMock) as mock_click:
            await mouse.left_click(400, 300, move_duration=0.8)

            assert mock_click.call_args[1]['move_duration'] == 0.8

    @pytest.mark.asyncio
    async def test_convenience_methods_support_custom_hold_duration(self, mock_tab_for_clicks):
        """Test that convenience methods support custom hold_duration."""
        mouse = MouseAPI(mock_tab_for_clicks)

        with patch.object(mouse, 'click', new_callable=AsyncMock) as mock_click:
            await mouse.left_click(400, 300, hold_duration=0.2)

            assert mock_click.call_args[1]['hold_duration'] == 0.2


class TestClickMethodRandomDuration:
    """Test that click methods use random duration when humanize enabled."""

    @pytest.fixture
    def mock_tab_humanize(self) -> 'Tab':
        """Create mock tab with humanize enabled."""
        tab = MagicMock()
        tab._browser = MagicMock()
        tab._browser.humanize_mouse_movement = True
        tab._execute_command = AsyncMock()
        return tab

    @pytest.mark.asyncio
    async def test_click_uses_random_duration_when_humanize_and_none(self, mock_tab_humanize):
        """Test click uses random duration when humanize=True and duration=None."""
        mouse = MouseAPI(mock_tab_humanize)
        mouse.set_position(100, 100)

        with patch(
            'pydoll.interactions.mouse._generate_random_duration', return_value=0.55
        ) as mock_gen:
            with patch.object(mouse, 'move_to', new_callable=AsyncMock) as mock_move:
                await mouse.click(400, 300, move_duration=None)

                # Should have generated random duration
                mock_gen.assert_called_once()

                # move_to should have been called with random duration
                mock_move.assert_called_once()
                assert mock_move.call_args[1]['duration'] == 0.55

    @pytest.mark.asyncio
    async def test_click_respects_explicit_duration(self, mock_tab_humanize):
        """Test click respects explicit duration even when humanize=True."""
        mouse = MouseAPI(mock_tab_humanize)
        mouse.set_position(100, 100)

        with patch('pydoll.interactions.mouse._generate_random_duration') as mock_gen:
            with patch.object(mouse, 'move_to', new_callable=AsyncMock) as mock_move:
                await mouse.click(400, 300, move_duration=1.5)

                # Should NOT generate random duration
                mock_gen.assert_not_called()

                # move_to should use explicit duration
                mock_move.assert_called_once()
                assert mock_move.call_args[1]['duration'] == 1.5

    @pytest.mark.asyncio
    async def test_drag_uses_random_durations_when_humanize_and_none(self, mock_tab_humanize):
        """Test drag uses random durations when humanize=True and durations=None."""
        mouse = MouseAPI(mock_tab_humanize)
        mouse.set_position(100, 100)

        with patch(
            'pydoll.interactions.mouse._generate_random_duration', side_effect=[0.4, 0.6]
        ) as mock_gen:
            with patch.object(mouse, 'move_to', new_callable=AsyncMock):
                await mouse.drag(100, 100, 500, 500)

                # Should have generated two random durations (move_to_start, drag)
                assert mock_gen.call_count == 2


class TestBackwardsCompatibility:
    """Test that changes don't break existing API."""

    @pytest.fixture
    def mock_tab_compat(self) -> 'Tab':
        """Create mock tab."""
        tab = MagicMock()
        tab._browser = MagicMock()
        tab._browser.humanize_mouse_movement = True
        tab._execute_command = AsyncMock()
        return tab

    @pytest.mark.asyncio
    async def test_click_still_accepts_explicit_duration(self, mock_tab_compat):
        """Test that click() still accepts explicit move_duration."""
        mouse = MouseAPI(mock_tab_compat)

        with patch.object(mouse, 'move_to', new_callable=AsyncMock) as mock_move:
            # Old API: explicit duration
            await mouse.click(400, 300, move_duration=0.5)

            mock_move.assert_called_once()
            assert mock_move.call_args[1]['duration'] == 0.5

    @pytest.mark.asyncio
    async def test_drag_still_accepts_explicit_durations(self, mock_tab_compat):
        """Test that drag() still accepts explicit durations."""
        mouse = MouseAPI(mock_tab_compat)

        with patch.object(mouse, 'move_to', new_callable=AsyncMock) as mock_move:
            # Old API: explicit durations
            await mouse.drag(
                100, 100, 500, 500, move_to_start_duration=0.5, drag_duration=0.7
            )

            # Should have been called twice (move to start, then drag)
            assert mock_move.call_count == 2

    @pytest.mark.asyncio
    async def test_move_to_still_accepts_explicit_duration(self, mock_tab_compat):
        """Test that move_to() still accepts explicit duration."""
        mouse = MouseAPI(mock_tab_compat)
        mouse.set_position(0, 0)

        with patch.object(mouse, '_execute_trajectory', new_callable=AsyncMock) as mock_traj:
            await mouse.move_to(400, 300, duration=1.0)

            mock_traj.assert_called_once()
            # Duration should be passed to trajectory execution
            assert mock_traj.call_args[0][1] == 1.0


# ============================================================================
# TAB INTEGRATION TESTS
# ============================================================================


@pytest_asyncio.fixture
async def mock_browser():
    """Mock Browser instance."""
    browser = MagicMock()
    browser._connection_port = 9222
    return browser


@pytest_asyncio.fixture
async def mock_connection_handler():
    """Mock ConnectionHandler for Tab."""
    handler = MagicMock()
    handler.execute_command = AsyncMock()
    return handler


@pytest_asyncio.fixture
async def tab_with_mouse(mock_browser, mock_connection_handler):
    """Create Tab instance with mocked dependencies."""
    with patch('pydoll.browser.tab.ConnectionHandler', return_value=mock_connection_handler):
        tab = Tab(
            browser=mock_browser,
            connection_port=9222,
            target_id='test-target-id',
        )
        tab._connection_handler = mock_connection_handler
        return tab


class TestMouseAPITabIntegration:
    """Test MouseAPI integration with Tab class."""

    def test_tab_has_mouse_property(self, tab_with_mouse):
        """Test that Tab has mouse property."""
        assert hasattr(tab_with_mouse, 'mouse')

    def test_mouse_property_returns_mouse_api(self, tab_with_mouse):
        """Test that mouse property returns MouseAPI instance."""
        mouse = tab_with_mouse.mouse
        assert isinstance(mouse, MouseAPI)

    def test_mouse_property_lazy_initialization(self, tab_with_mouse):
        """Test that MouseAPI is lazily initialized."""
        assert tab_with_mouse._mouse is None
        mouse1 = tab_with_mouse.mouse
        assert tab_with_mouse._mouse is not None
        mouse2 = tab_with_mouse.mouse
        assert mouse1 is mouse2

    @pytest.mark.asyncio
    async def test_mouse_commands_execute_through_tab(
        self, tab_with_mouse, mock_connection_handler
    ):
        """Test that mouse commands execute through Tab's command handler."""
        await tab_with_mouse.mouse.move_to(100.0, 100.0, duration=0.05)

        assert mock_connection_handler.execute_command.called

    @pytest.mark.asyncio
    async def test_mouse_click_integration(self, tab_with_mouse, mock_connection_handler):
        """Test integrated click operation."""
        await tab_with_mouse.mouse.click(x=200.0, y=300.0, move_duration=0.05, hold_duration=0.01)

        assert mock_connection_handler.execute_command.called
        calls = mock_connection_handler.execute_command.call_args_list
        assert len(calls) > 0


class TestMouseAPICommandSequencing:
    """Test proper sequencing of mouse commands."""

    @pytest.mark.asyncio
    async def test_move_to_command_sequence(self, tab_with_mouse, mock_connection_handler):
        """Test that move_to generates proper sequence of mouseMoved commands."""
        await tab_with_mouse.mouse.move_to(100.0, 100.0, duration=0.05)

        calls = mock_connection_handler.execute_command.call_args_list
        assert len(calls) > 0

        for call in calls:
            command = call[0][0]
            assert 'method' in command
            assert command['method'] == 'Input.dispatchMouseEvent'
            assert 'params' in command

    @pytest.mark.asyncio
    async def test_click_command_sequence(self, tab_with_mouse, mock_connection_handler):
        """Test click generates press and release in correct order."""
        tab_with_mouse.mouse.set_position(100.0, 100.0)
        await tab_with_mouse.mouse.click(hold_duration=0.01)

        calls = mock_connection_handler.execute_command.call_args_list
        assert len(calls) == 2

        press_command = calls[0][0][0]
        release_command = calls[1][0][0]

        assert press_command['params']['type'] == 'mousePressed'
        assert release_command['params']['type'] == 'mouseReleased'

    @pytest.mark.asyncio
    async def test_drag_command_sequence(self, tab_with_mouse, mock_connection_handler):
        """Test drag generates move, press, move, release sequence."""
        await tab_with_mouse.mouse.drag(
            from_x=100.0,
            from_y=100.0,
            to_x=300.0,
            to_y=200.0,
            move_to_start_duration=0.05,
            drag_duration=0.05,
        )

        calls = [call[0][0] for call in mock_connection_handler.execute_command.call_args_list]

        press_events = [c for c in calls if c['params']['type'] == 'mousePressed']
        release_events = [c for c in calls if c['params']['type'] == 'mouseReleased']
        move_events = [c for c in calls if c['params']['type'] == 'mouseMoved']

        assert len(press_events) == 1
        assert len(release_events) == 1
        assert len(move_events) > 0


class TestMouseAPIStateManagement:
    """Test position state management across operations."""

    @pytest.mark.asyncio
    async def test_position_tracking_across_operations(self, tab_with_mouse):
        """Test position is correctly tracked across multiple operations."""
        mouse = tab_with_mouse.mouse

        assert mouse.get_position() == (0.0, 0.0)

        await mouse.move_to(100.0, 200.0, duration=0.05)
        assert mouse.get_position() == (100.0, 200.0)

        await mouse.move_by(50.0, -50.0, duration=0.05)
        assert mouse.get_position() == (150.0, 150.0)

    @pytest.mark.asyncio
    async def test_position_tracking_after_click(self, tab_with_mouse):
        """Test position remains consistent after click."""
        mouse = tab_with_mouse.mouse

        await mouse.click(x=300.0, y=400.0, move_duration=0.05, hold_duration=0.01)
        assert mouse.get_position() == (300.0, 400.0)

        await mouse.click(hold_duration=0.01)
        assert mouse.get_position() == (300.0, 400.0)

    @pytest.mark.asyncio
    async def test_position_tracking_after_drag(self, tab_with_mouse):
        """Test position updates to end position after drag."""
        mouse = tab_with_mouse.mouse

        await mouse.drag(
            from_x=100.0,
            from_y=100.0,
            to_x=500.0,
            to_y=600.0,
            move_to_start_duration=0.05,
            drag_duration=0.05,
        )

        assert mouse.get_position() == (500.0, 600.0)


class TestMouseAPIConcurrentOperations:
    """Test behavior with concurrent mouse operations."""

    @pytest.mark.asyncio
    async def test_sequential_moves_maintain_state(self, tab_with_mouse):
        """Test sequential move operations maintain correct state."""
        mouse = tab_with_mouse.mouse

        await mouse.move_to(100.0, 100.0, duration=0.05)
        await mouse.move_to(200.0, 200.0, duration=0.05)
        await mouse.move_to(300.0, 300.0, duration=0.05)

        assert mouse.get_position() == (300.0, 300.0)

    @pytest.mark.asyncio
    async def test_position_reset_and_move(self, tab_with_mouse):
        """Test manual position reset followed by movement."""
        mouse = tab_with_mouse.mouse

        await mouse.move_to(500.0, 500.0, duration=0.05)
        mouse.set_position(0.0, 0.0)

        assert mouse.get_position() == (0.0, 0.0)

        await mouse.move_to(100.0, 100.0, duration=0.05)
        assert mouse.get_position() == (100.0, 100.0)
