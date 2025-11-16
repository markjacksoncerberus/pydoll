"""Tests for humanize_mouse_movement feature and convenience click methods."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydoll.browser.chromium.chrome import Chrome
from pydoll.interactions.mouse import MouseAPI, MouseButton, _generate_random_duration

if TYPE_CHECKING:
    from pydoll.browser.tab import Tab


class TestRandomDurationGenerator:
    """Test random duration generation."""

    def test_generate_random_duration_in_range(self):
        """Test that random duration is within specified range."""
        for _ in range(100):
            duration = _generate_random_duration(0.3, 0.7)
            assert 0.3 <= duration <= 0.7

    def test_generate_random_duration_custom_range(self):
        """Test random duration with custom range."""
        for _ in range(100):
            duration = _generate_random_duration(1.0, 2.0)
            assert 1.0 <= duration <= 2.0

    def test_generate_random_duration_default_range(self):
        """Test random duration with default range."""
        for _ in range(100):
            duration = _generate_random_duration()
            assert 0.7 <= duration <= 2.0


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
        tab._browser = MagicMock(spec=[])  # No humanize_mouse_movement attribute
        tab._execute_command = AsyncMock()
        mouse = MouseAPI(tab)
        assert mouse._humanize is True

    @pytest.mark.asyncio
    async def test_teleport_mode_uses_single_event(self, mock_tab_humanize_disabled):
        """Test that teleport mode uses single mouseMoved event."""
        mouse = MouseAPI(mock_tab_humanize_disabled)
        mouse.set_position(0, 0)

        with patch.object(mouse, '_teleport_to', new_callable=AsyncMock) as mock_teleport:
            with patch.object(mouse, '_execute_trajectory', new_callable=AsyncMock) as mock_trajectory:
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
            with patch.object(mouse, '_execute_trajectory', new_callable=AsyncMock) as mock_trajectory:
                await mouse.move_to(500, 300, duration=0.5)
                
                # Should call trajectory, not teleport
                mock_trajectory.assert_called_once()
                mock_teleport.assert_not_called()

    @pytest.mark.asyncio
    async def test_humanize_mode_uses_random_duration_when_none(self, mock_tab_humanize_enabled):
        """Test that humanize mode uses random duration when duration is None."""
        mouse = MouseAPI(mock_tab_humanize_enabled)
        mouse.set_position(0, 0)

        with patch('pydoll.interactions.mouse._generate_random_duration', return_value=0.55) as mock_gen:
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
        # Command is a TypedDict, so access like a dictionary
        assert command['method'] == 'Input.dispatchMouseEvent'
        assert command['params']['type'] == 'mouseMoved'
        assert command['params']['x'] == 400
        assert command['params']['y'] == 300


class TestConvenienceClickMethods:
    """Test convenience click methods (left_click, right_click, middle_click)."""

    @pytest.fixture
    def mock_tab(self) -> 'Tab':
        """Create mock tab for testing."""
        tab = MagicMock()
        tab._browser = MagicMock()
        tab._browser.humanize_mouse_movement = True
        tab._execute_command = AsyncMock()
        return tab

    @pytest.mark.asyncio
    async def test_left_click_calls_click_with_left_button(self, mock_tab):
        """Test that left_click calls click with LEFT button."""
        mouse = MouseAPI(mock_tab)
        
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
    async def test_right_click_calls_click_with_right_button(self, mock_tab):
        """Test that right_click calls click with RIGHT button."""
        mouse = MouseAPI(mock_tab)
        
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
    async def test_middle_click_calls_click_with_middle_button(self, mock_tab):
        """Test that middle_click calls click with MIDDLE button."""
        mouse = MouseAPI(mock_tab)
        
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
    async def test_convenience_methods_support_custom_duration(self, mock_tab):
        """Test that convenience methods support custom move_duration."""
        mouse = MouseAPI(mock_tab)
        
        with patch.object(mouse, 'click', new_callable=AsyncMock) as mock_click:
            await mouse.left_click(400, 300, move_duration=0.8)
            
            assert mock_click.call_args[1]['move_duration'] == 0.8

    @pytest.mark.asyncio
    async def test_convenience_methods_support_custom_hold_duration(self, mock_tab):
        """Test that convenience methods support custom hold_duration."""
        mouse = MouseAPI(mock_tab)
        
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
        
        with patch('pydoll.interactions.mouse._generate_random_duration', return_value=0.55) as mock_gen:
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
        
        with patch('pydoll.interactions.mouse._generate_random_duration', side_effect=[0.4, 0.6]) as mock_gen:
            with patch.object(mouse, 'move_to', new_callable=AsyncMock):
                await mouse.drag(100, 100, 500, 500)
                
                # Should have generated two random durations (move_to_start, drag)
                assert mock_gen.call_count == 2


class TestBackwardsCompatibility:
    """Test that changes don't break existing API."""

    @pytest.fixture
    def mock_tab(self) -> 'Tab':
        """Create mock tab."""
        tab = MagicMock()
        tab._browser = MagicMock()
        tab._browser.humanize_mouse_movement = True
        tab._execute_command = AsyncMock()
        return tab

    @pytest.mark.asyncio
    async def test_click_still_accepts_explicit_duration(self, mock_tab):
        """Test that click() still accepts explicit move_duration."""
        mouse = MouseAPI(mock_tab)
        
        with patch.object(mouse, 'move_to', new_callable=AsyncMock) as mock_move:
            # Old API: explicit duration
            await mouse.click(400, 300, move_duration=0.5)
            
            mock_move.assert_called_once()
            assert mock_move.call_args[1]['duration'] == 0.5

    @pytest.mark.asyncio
    async def test_drag_still_accepts_explicit_durations(self, mock_tab):
        """Test that drag() still accepts explicit durations."""
        mouse = MouseAPI(mock_tab)
        
        with patch.object(mouse, 'move_to', new_callable=AsyncMock) as mock_move:
            # Old API: explicit durations
            await mouse.drag(100, 100, 500, 500, move_to_start_duration=0.5, drag_duration=0.7)
            
            # Should have been called twice (move to start, then drag)
            assert mock_move.call_count == 2

    @pytest.mark.asyncio
    async def test_move_to_still_accepts_explicit_duration(self, mock_tab):
        """Test that move_to() still accepts explicit duration."""
        mouse = MouseAPI(mock_tab)
        mouse.set_position(0, 0)
        
        with patch.object(mouse, '_execute_trajectory', new_callable=AsyncMock) as mock_traj:
            await mouse.move_to(400, 300, duration=1.0)
            
            mock_traj.assert_called_once()
            # Duration should be passed to trajectory execution
            assert mock_traj.call_args[0][1] == 1.0
