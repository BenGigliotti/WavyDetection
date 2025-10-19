import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_excel_data():    
    num_rows = int(input("Enter the number of rows/samples: "))
    mean_od = float(input("Enter the mean OD value: "))
    od_deviation = float(input("Enter the average deviation from mean OD: "))
    
    # generate timestamps
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i) for i in range(num_rows)]
    
    # generate speed
    speed_data = []
    i = 0
    while i < num_rows:
        # random length of non-zero speed period (10-50 samples)
        active_period = random.randint(10, 50)
        # random length of zero speed period (5-20 samples)
        zero_period = random.randint(5, 20)
        
        # add non-zero speeds (random between 10-30 fpm)
        for _ in range(min(active_period, num_rows - i)):
            speed_data.append(random.uniform(10, 30))
            i += 1
            if i >= num_rows:
                break
        
        # add zero speeds
        for _ in range(min(zero_period, num_rows - i)):
            speed_data.append(0)
            i += 1
            if i >= num_rows:
                break
    
    # generate OD data based on speed
    od_data = []
    current_od = mean_od  # start at mean
    
    for speed in speed_data:
        if speed == 0:
            od_data.append(current_od)
        else:
            current_od = np.random.normal(mean_od, od_deviation)
            od_data.append(current_od)
    
    df_od = pd.DataFrame({
        't_stamp': timestamps,
        'Tag_value': od_data
    })
    
    df_speed = pd.DataFrame({
        't_stamp': timestamps,
        'Tag_value': speed_data
    })
    
    # write to Excel file with two sheets
    output_file = f'dummy_m{mean_od}_d{od_deviation}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_od.to_excel(writer, sheet_name='NDC_System_OD_Value', index=False)
        df_speed.to_excel(writer, sheet_name='YS_Pullout1_Act_Speed_fpm', index=False)
        
        for sheet_name in ['NDC_System_OD_Value', 'YS_Pullout1_Act_Speed_fpm']:
            worksheet = writer.sheets[sheet_name]
            for row in range(2, num_rows + 2):
                cell = worksheet.cell(row=row, column=1)
                cell.number_format = 'M/D/YYYY H:MM:SS'
    
    print(f"\nExcel file '{output_file}' generated successfully")
    print(f"- Sheet 1: NDC_System_OD_Value ({len(df_od)} rows)")
    print(f"- Sheet 2: YS_Pullout1_Act_Speed_fpm ({len(df_speed)} rows)")
    print(f"\nData Summary:")
    print(f"- OD Mean: {np.mean(od_data):.5f}")
    print(f"- OD Std Dev: {np.std(od_data):.5f}")
    print(f"- Zero Speed Samples: {sum(1 for s in speed_data if s == 0)}")
    print(f"- Non-Zero Speed Samples: {sum(1 for s in speed_data if s > 0)}")

if __name__ == "__main__":
    generate_excel_data()