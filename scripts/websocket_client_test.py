import asyncio
import json
import websockets

async def consume():
    uri = "ws://localhost:6467"
    async with websockets.connect(uri) as ws:
        async for message in ws:
            data = json.loads(message)
            print(data)  # you'll see od + speed here

asyncio.run(consume())