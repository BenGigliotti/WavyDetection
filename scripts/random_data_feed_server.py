import asyncio
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
import numpy as np
import json
import random

sample_rate = 2400
mean_od = 0.5
od_deviation = 0.4

async def generate_random_measurements(od, speed):
    if random.random() > 0.99:
        if speed != 0:
            speed = 0
        else:
            speed = random.randint(15,25)
    
    if speed != 0:
        od = np.random.normal(mean_od, od_deviation)

    return {
        "od": od,
        "speed": speed
    }

async def send_output(websocket):
    sleep_time = 1.0 / sample_rate
    current_od = mean_od
    current_speed = 0
    while True:
        measurement = await generate_random_measurements(current_od, current_speed)
        current_od = measurement['od']
        current_speed = measurement['speed']
        try: 
            await websocket.send(json.dumps(measurement))
        except ConnectionClosed:
            break
        await asyncio.sleep(sleep_time)


async def main():
    async with serve(send_output, "localhost", 6467) as server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())