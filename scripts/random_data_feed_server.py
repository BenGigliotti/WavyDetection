import asyncio
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
import numpy as np

sample_rate = 2400
mean_od = 0.5
od_deviation = 0.4

async def output(websocket):
    sleep_time = 1.0 / sample_rate
    while True:
        measurement = np.random.normal(mean_od, od_deviation)
        try: 
            await websocket.send(f"{measurement}")
        except ConnectionClosed:
            break
        await asyncio.sleep(sleep_time)


async def main():
    async with serve(output, "localhost", 6467) as server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())