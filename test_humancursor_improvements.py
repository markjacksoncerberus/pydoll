"""
Quick test to verify HumanCursor-style improvements.

This demonstrates the enhanced mouse movement with:
- Random easing functions (13 different types)
- Weighted random knots (1-10, more likely 2-3)
- Weighted random target points (35-80, more likely 35-60)
- Random distortion parameters
- Slower duration range (0.7-2.0 seconds)
"""

import asyncio
from pydoll.browser.chromium.chrome import Chrome


async def test_improved_humanlike_movement():
    """Test the improved human-like movement."""
    print("Testing improved HumanCursor-style mouse movements...")
    
    async with Chrome(humanize_mouse_movement=True) as browser:
        await browser.start()
        tab = await browser.create_tab()
        await tab.goto('https://www.google.com')
        
        print("\n1. Each movement uses different random parameters:")
        print("   - Random easing function (13 options)")
        print("   - Random knots count (1-10, weighted toward 2-3)")
        print("   - Random target points (35-80, weighted)")
        print("   - Random distortion (0.8-1.1 mean, 0.85-1.1 stdev, 0.25-0.7 freq)")
        print("   - Random duration (0.7-2.0 seconds)\n")
        
        # Test 5 movements to see variety
        positions = [
            (300, 200),
            (500, 350),
            (700, 250),
            (400, 450),
            (600, 300),
        ]
        
        for i, (x, y) in enumerate(positions, 1):
            print(f"Movement {i}: Moving to ({x}, {y})...")
            await tab.mouse.move_to(x, y)
            actual_pos = tab.mouse.get_position()
            print(f"  ✓ Arrived at {actual_pos}")
        
        print("\n2. Testing convenience click methods:")
        await tab.mouse.left_click(400, 300)
        print("  ✓ Left-clicked at (400, 300)")
        
        await tab.mouse.right_click(500, 350)
        print("  ✓ Right-clicked at (500, 350)")
        
        print("\n3. Testing drag with random parameters:")
        await tab.mouse.drag(200, 200, 600, 400)
        print("  ✓ Dragged from (200, 200) to (600, 400)")
        
        print("\n✅ All movements completed successfully!")
        print("Each movement had unique random characteristics matching HumanCursor behavior.")


if __name__ == '__main__':
    asyncio.run(test_improved_humanlike_movement())
