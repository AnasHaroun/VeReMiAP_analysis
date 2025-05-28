"""
This script analyzes vehicular hazard data collected from simulation CSV files.
It covers data loading, preprocessing, analysis, and visualization of hazard events.
"""

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Define the dataset directory and gather CSV files
data_dir = "/content/drive/MyDrive/Prediction Datasets/Csv6.3(0.2,10,5,70)_sol"
csv_files = glob.glob(data_dir + "/*.csv")

# Read all CSV files into a single DataFrame
df_list = (pd.read_csv(file) for file in csv_files)
df = pd.concat(df_list, ignore_index=True)


"""
Fill missing values and create backup of original DataFrame.
"""

# Fill NA values with domain-specific defaults
df['hazardAttack'] = df['hazardAttack'].fillna(0)
df['hazardOccurrence'] = df['hazardOccurrence'].fillna(-1)
df['messageID'] = df['messageID'].fillna(0)
df['EventID'] = df['EventID'].fillna(0)

# Backup the original DataFrame for safe reference
df2 = df.copy()


"""
Identify vehicles that encountered hazards and those that did not.
"""

# Select type 3 hazard messages
type3_df = df[df.type == 3]

# Examine hazard activity for a specific vehicle
percHazard = df[df.vehicleId == 9].sort_values(by=['sendTime'])

# Identify vehicle IDs that never experienced a hazard
attacks = df.groupby('vehicleId')['hazardOccurrence'].any()
never_attacked = attacks[~attacks].index.tolist()

# Create DataFrame of unimpacted vehicles
df_never_attacked = pd.DataFrame({'vehicle_id': never_attacked})
df_never_attacked.to_csv('data.csv', index=False)

print(f"There are {len(df['vehicleId'].unique())} cars in the dataset.")


"""
Aggregate hazard occurrences and vehicle counts by lane position.
Save and print the summary statistics.
"""

road_summary = type3_df.groupby('lanePosition').agg({
    'hazardOccurrence': 'sum',
    'vehicleId': 'count'
}).rename(columns={'hazardOccurrence': 'hazard_count', 'vehicleId': 'car_count'})

# Export summary statistics to CSV
road_summary.to_csv('data.csv', index=False)
print(road_summary)

"""
Visualize the number of vehicles and hazard occurrences by lane using Plotly.
"""

# Filter rows where hazards occurred
df_filtered = df[df['hazardOccurrence'] > 0]

# Group by lane position and summarize
grouped = df_filtered.groupby('lanePosition').agg({
    'vehicleId': 'nunique',
    'hazardOccurrence': 'sum'
}).head(40)

# Create interactive plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=grouped.index, y=grouped['vehicleId'], mode='lines', name='Vehicle Count'))
fig.add_trace(go.Bar(x=grouped.index, y=grouped['hazardOccurrence'], name='Hazard Count'))

fig.update_layout(
    title='Lane Hazards and Vehicle Numbers',
    xaxis_title='Lane Position',
    yaxis_title='Number of Vehicles / Sum of Hazards',
    barmode='overlay',
    font=dict(size=18)
)
fig.show()


"""
Show class distributions and plot hazard attacks over time intervals.
"""

# Class distribution (count + percentage)
class_counts = {}
for col in ['hazardOccurrence', 'hazardAttack', 'type']:
    counts = df[col].value_counts()
    percentages = counts / df.shape[0] * 100
    class_counts[col] = pd.concat([counts, percentages], axis=1, keys=['count', 'percentage'])

for col, summary in class_counts.items():
    print(f"Class Distribution for {col}:\n{summary}\n")

# Create attackTime based on rcvTime where hazardAttack == 1
df['attackTime'] = pd.to_datetime(df['rcvTime'].where(df['hazardAttack'] == 1), unit='s')
df['attackTime'] = df['attackTime'].ffill()

# Compute speed from X and Y components
df['speed'] = np.sqrt(df['x_spd']**2 + df['y_spd']**2)

# Convert rcvTime to datetime format
df['rcvTime'] = pd.to_datetime(df['rcvTime'], unit='s')

# Define reusable plotting function
def plot_attacks_vs_speed(freq='500S'):
    """
    Plot number of attacks and average speed over time intervals.

    Args:
        freq (str): Time interval (e.g. '500S' for 500 seconds).
    """
    attacks_interval = df.groupby(pd.Grouper(key='attackTime', freq=freq))['hazardAttack'].sum()
    speed_interval = df.groupby(pd.Grouper(key='rcvTime', freq=freq))['speed'].mean()

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xlabel('Time Interval')
    ax1.set_ylabel('Number of Attacks', color='tab:red')
    ax1.plot(attacks_interval.index, attacks_interval.values, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Speed', color='tab:blue')
    ax2.plot(speed_interval.index, speed_interval.values, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    correlation = speed_interval.corr(attacks_interval)
    print(f"Correlation between speed and attacks (freq={freq}): {correlation:.4f}")

    fig.tight_layout()
    plt.show()

# Plot across multiple time frequencies
for interval in ['500S', '700S', '1500S', '2000S']:
    plot_attacks_vs_speed(freq=interval)


"""
Compare number of attacks and average speed per vehicle.
Visualize and compute correlation.
"""

# Compute per-vehicle metrics
attacks_per_vehicle = df.groupby('vehicleId')['hazardAttack'].sum()
speed_per_vehicle = df.groupby('vehicleId')['speed'].mean()

# Filter to show vehicles with >1 attack
filtered_attacks = attacks_per_vehicle[attacks_per_vehicle > 1]
filtered_speeds = speed_per_vehicle[filtered_attacks.index]

# Plot attack count and average speed side-by-side
fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_xlabel('Vehicle ID')
ax1.set_ylabel('Number of Attacks', color='tab:red')
ax1.bar(filtered_attacks.index, filtered_attacks.values, color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Average Speed (m/s)', color='tab:blue')
ax2.plot(filtered_speeds.index, filtered_speeds.values, color='tab:blue', marker='o')
ax2.tick_params(axis='y', labelcolor='tab:blue')

correlation = filtered_speeds.corr(filtered_attacks)
print(f"Correlation between average speed and attacks: {correlation:.4f}")
plt.tight_layout()
plt.show()


"""
Create scatter plots using Plotly and Seaborn to analyze
the relationship between vehicle speed and hazard attacks.
"""

# Filter and group vehicle data
df_filtered = df[df['hazardAttack'] > 0]
grouped = df_filtered.groupby('vehicleId').agg({'hazardAttack': 'sum', 'speed': 'mean'})
grouped = grouped.sort_values(by='hazardAttack', ascending=False).head(30)

# Plot with Plotly
fig = go.Figure()
fig.add_trace(go.Bar(x=grouped.index, y=grouped['hazardAttack'], name='Hazard Attack'))
fig.add_trace(go.Bar(x=grouped.index, y=grouped['speed'], name='Average Speed'))

fig.update_layout(
    title='Top 30 Vehicles: Hazard Attacks vs. Avg Speed',
    xaxis_title='Vehicle ID',
    yaxis_title='Count / Speed',
    barmode='group'
)
fig.show()

# Scatter plot with linear regression using Seaborn
import seaborn as sns
corr_coeff = grouped['speed'].corr(grouped['hazardAttack'])

sns.lmplot(x='hazardAttack', y='speed', data=grouped.reset_index(), height=8, aspect=1.5)
plt.title('Correlation: Speed vs. Hazard Attacks', fontsize=20)
plt.xlabel('Number of Hazard Attacks', fontsize=16)
plt.ylabel('Average Speed (km/h)', fontsize=16)
plt.text(0.1, 0.9, f'Correlation: {corr_coeff:.2f}', transform=plt.gca().transAxes, fontsize=14,
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
plt.show()


"""
Visual analysis and counts of lane positions, hazard types, and event IDs.
"""

# Basic distributions
print(df['lanePosition'].value_counts(dropna=False))
print(df['type'].value_counts(dropna=False))
print(df['hazardAttack'].value_counts(normalize=True))
print(df['hazardOccurrence'].value_counts(normalize=True))

# Time-based plotting
df.index = df['rcvTime']
mpl.rcParams['agg.path.chunksize'] = 10000

df.plot(figsize=(20, 10), y='hazardAttack', title='Hazard Attacks Over Time')
df.plot(figsize=(20, 10), y='hazardOccurrence', title='Hazard Occurrence Over Time')

# Focused time window plot
focus = df[(df['rcvTime'] >= pd.to_datetime(1500, unit='s')) &
           (df['rcvTime'] <= pd.to_datetime(3400, unit='s'))]
focus.plot(figsize=(20, 10), y='hazardOccurrence', title='Focused Hazard Occurrence View')
