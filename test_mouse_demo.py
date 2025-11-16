"""
Demo script for pydoll mouse movement features.

Demonstrates:
1. Humanized mouse movement (bezier curves with random duration)
2. Teleport mode (instant movement)
3. Convenience click methods (left_click, right_click, middle_click)
4. Browser-level humanize_mouse_movement configuration
"""

import asyncio
from pydoll.browser.chromium.chrome import Chrome


async def test_humanized_movement():
    """Test humanized mouse movement with bezier curves."""
    print("\n=== Testing Humanized Mouse Movement ===")
    
    async with Chrome(humanize_mouse_movement=True) as browser:
        tab = await browser.start()
        await tab.go_to('https://freeonlinepaint.com/paint')
        print("Moving with bezier curves and random durations...")
        
        for _ in range(3):
            # Drag also uses random duration when not specified
            await tab.mouse.drag(500, 500, 900, 900)
            print("Performed drag with humanized movement")


async def test_teleport_mode():
    """Test instant mouse movement (teleport mode)."""
    print("\n=== Testing Teleport Mode (Instant Movement) ===")
    
    async with Chrome(humanize_mouse_movement=False) as browser:
        tab = await browser.start()
        await tab.go_to('https://webutility.io/mouse-tester')
        
        print("Moving with instant teleport (no bezier curves)...")
        
        # All movements are instant - single mouseMoved event
        await tab.mouse.move_to(300, 200)
        pos = tab.mouse.get_position()
        print(f"Teleported to: {pos}")
        
        await tab.mouse.left_click(400, 300)
        print("Performed instant left_click (no movement animation)")
        
        await tab.mouse.move_to(600, 400)
        print("Instant teleport to new position")


async def test_convenience_methods():
    """Test all convenience click methods."""
    print("\n=== Testing Convenience Click Methods ===")
    
    async with Chrome(humanize_mouse_movement=True) as browser:
        tab = await browser.start()
        
        await tab.go_to('https://webutility.io/mouse-tester')
        await asyncio.sleep(3)
        
        # Left click (most common)
        await tab.mouse.left_click(300, 200)
        print("Left-clicked at (300, 200)")
        
        # Right click (context menu)
        await tab.mouse.right_click(400, 250)
        print("Right-clicked at (400, 250)")
        
        # Middle click (less common, but supported)
        await tab.mouse.middle_click(500, 300)
        print("Middle-clicked at (500, 300)")
        
        # All convenience methods support custom durations
        await tab.mouse.left_click(600, 350, move_duration=0.8)
        print("Left-clicked with custom 0.8s movement duration")


async def test_original_api_still_works():
    """Verify original click() and drag() API still works."""
    print("\n=== Testing Original API (Backwards Compatibility) ===")
    
    async with Chrome() as browser:  # humanize=True by default
        tab = await browser.start()
        
        await tab.go_to('https://webutility.io/mouse-tester')
        await asyncio.sleep(3)
        
        # Original click() with coordinates
        await tab.mouse.click(300, 200)
        print("Original click() method works")
        
        # Original click() at current position
        await tab.mouse.click()
        print("Click at current position works")
        
        # Original drag()
        await tab.mouse.drag(200, 200, 400, 400)
        print("Original drag() method works")
        
        # Double-click still works
        await tab.mouse.double_click(500, 300)
        print("Double-click works")


async def main():
    """Run all demo tests."""
    try:
        await test_humanized_movement()
        # await test_teleport_mode()
        # await test_convenience_methods()
        # await test_original_api_still_works()
        print("\n✅ All demos completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(main())
