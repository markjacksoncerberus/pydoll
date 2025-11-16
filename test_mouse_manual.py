"""Manual test for mouse movement - to be deleted after testing."""

import asyncio
from pydoll.browser import Chrome


async def main():
    async with Chrome() as browser:
        tab = await browser.start()
        
        await tab.go_to('https://cps-check.com/mouse-acceleration')
        await asyncio.sleep(3)
        
        print("Moving to point 1: (200, 200)")
        await tab.mouse.move_to(200, 200, duration=1.0)
        pos = tab.mouse.get_position()
        print(f"Current position: {pos}")
        assert pos == (200.0, 200.0), f"Expected (200, 200), got {pos}"
        await asyncio.sleep(0.5)
        
        print("Moving to point 2: (600, 400)")
        await tab.mouse.move_to(600, 400, duration=1.0)
        pos = tab.mouse.get_position()
        print(f"Current position: {pos}")
        assert pos == (600.0, 400.0), f"Expected (600, 400), got {pos}"
        await asyncio.sleep(0.5)
        
        print("Moving to point 3: (300, 600)")
        await tab.mouse.move_to(300, 600, duration=1.0)
        pos = tab.mouse.get_position()
        print(f"Current position: {pos}")
        assert pos == (300.0, 600.0), f"Expected (300, 600), got {pos}"
        await asyncio.sleep(0.5)
        
        print("Moving to point 4: (800, 200)")
        await tab.mouse.move_to(800, 200, duration=1.0)
        pos = tab.mouse.get_position()
        print(f"Current position: {pos}")
        assert pos == (800.0, 200.0), f"Expected (800, 200), got {pos}"
        await asyncio.sleep(0.5)
        
        print("Moving to center: (500, 350)")
        await tab.mouse.move_to(500, 350, duration=1.0)
        pos = tab.mouse.get_position()
        print(f"Current position: {pos}")
        assert pos == (500.0, 350.0), f"Expected (500, 350), got {pos}"
        await asyncio.sleep(0.5)
        
        print("\nAll movements completed successfully!")
        print("Clicking at current position...")
        await tab.mouse.click()
        
        await asyncio.sleep(2)


if __name__ == '__main__':
    asyncio.run(main())
