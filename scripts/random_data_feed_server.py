import asyncio
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
import numpy as np
import json
import random

sample_rate = 2400
mean_od = 0.5
od_deviation = 0.4

port = 6467

current_od = mean_od
current_speed = 0
state_lock = asyncio.Lock()

async def generate_random_measurements():
    global current_od, current_speed

    if random.random() > 0.999:
        if current_speed != 0:
            current_speed = 0
        else:
            current_speed = random.randint(15,25)
    
    if current_speed != 0:
        new_od = np.random.normal(mean_od, od_deviation)
        current_od = 0.95 * current_od + 0.05 * new_od

    return {
        "od": current_od,
        "speed": current_speed
    }

async def measurement_generator():
    sleep_time = 1.0 / sample_rate
    while True:
        async with state_lock:
            await generate_random_measurements()
        await asyncio.sleep(sleep_time)

async def send_output(websocket):
    sleep_time = 1.0 / sample_rate
    try:
        while True:
            async with state_lock:
                measurement = {
                    "od": current_od,
                    "speed": current_speed
                }
            await websocket.send(json.dumps(measurement))
            await asyncio.sleep(sleep_time)
    except ConnectionClosed:
        pass


async def main():
    generator_task = asyncio.create_task(measurement_generator())
    
    async with serve(send_output, "localhost", port) as server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())