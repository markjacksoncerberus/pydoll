"""Integration tests for MouseAPI with Tab and ConnectionHandler."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from pydoll.browser.tab import Tab
from pydoll.interactions.mouse import MouseAPI


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
    async def test_mouse_commands_execute_through_tab(self, tab_with_mouse, mock_connection_handler):
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
