import pandas as pd
from sjvisualizer import Canvas, DataHandler, PieRace

# File paths
csv_file = "emotion_data.csv"
xlsx_file = "emotion_data.xlsx"

# Step 1: Load CSV, convert 'Time' to datetime, set as index
df_csv = pd.read_csv(csv_file)
df_csv["Time"] = pd.to_datetime(df_csv["Time"], format='%H:%M:%S', errors="coerce")
df_csv = df_csv.dropna(subset=["Time"])
df_csv.set_index("Time", inplace=True)

# Step 2: Ensure emotion columns are numeric
emotion_cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
df_csv[emotion_cols] = df_csv[emotion_cols].apply(pd.to_numeric, errors='coerce')

# Step 3: Drop rows where sum of emotion values is zero or NaN
df_csv = df_csv[df_csv[emotion_cols].sum(axis=1) > 0]

# Step 4: Create continuous time index with 1-second frequency
full_time_index = pd.date_range(start=df_csv.index.min(), end=df_csv.index.max(), freq='S')

# Step 5: Reindex dataframe to full time index, fill missing rows with zeros
df_csv = df_csv.reindex(full_time_index)
df_csv[emotion_cols] = df_csv[emotion_cols].fillna(0)

# Step 6: Save cleaned and continuous data to Excel
df_csv.to_excel(xlsx_file)

# Step 7: Load Excel with DataHandler for visualization
print("loading new data frame")
df = DataHandler.DataHandler(excel_file=xlsx_file, number_of_frames=3600).df

# Step 8: Setup visualization
canvas = Canvas.canvas()
pie_race = PieRace.pie_plot(canvas=canvas.canvas, df=df)
canvas.add_sub_plot(pie_race)

# Step 9: Play animation
canvas.play()
