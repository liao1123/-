# Function to get memory usage
def get_memory_usage():
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024 * 1024)  # Convert to MB
    used_memory = memory_info.used / (1024 * 1024)  # Convert to MB
    available_memory = memory_info.available / (1024 * 1024)  # Convert to MB
    return used_memory, available_memory, total_memory


# Function to update memory data and return a DataFrame for the bar chart
def update_memory():
    used_memory, available_memory, total_memory = get_memory_usage()
    used_percent = used_memory / total_memory
    available_percent = available_memory / total_memory

    # Create a DataFrame for the bar chart
    data = {
        'Memory Type': ['Used', 'Available'],
        'Percentage': [used_percent, available_percent]
    }
    df = pd.DataFrame(data)

    # Return the memory info and the DataFrame
    return df


def get_GPU_usage():
    gpu_usage = GPUtil.getGPUs()[0].load * 100

    x_gpu_data.append(time.time() - start_time)
    ydata_gpu.append(gpu_usage)

    # Keep only the latest 100 data points
    xdata_recent = x_gpu_data[-50:]  # Ensure this is the same length as CPU
    ydata_gpu_recent = ydata_gpu[-50:]

    # Return the data for plotting
    return pd.DataFrame({
        "time": xdata_recent,
        "GPU Usage": ydata_gpu_recent
    })

# Function to create and save the bar chart as an image
def create_bar_chart():
    df = update_memory()
    fig, ax = plt.subplots(figsize=(6, 4))

    # Softer colors for the bars
    colors = ['#ff9999', '#99ff99']  # Light red and light green

    # Create the bar chart with thinner bars
    ax.bar(df['Memory Type'], df['Percentage'], color=colors, width=0.5)

    # Set title and labels with a larger font size for clarity
    ax.set_title('Memory Usage', fontsize=16)
    ax.set_xlabel('Memory Type', fontsize=14)
    ax.set_ylabel('Percentage', fontsize=14)

    # Set y-axis limit from 0 to 1 (since it's a percentage)
    ax.set_ylim(0, 1)

    # Adding percentage labels on top of each bar
    for i, v in enumerate(df['Percentage']):
        ax.text(i, v + 0.02, f"{v * 100:.1f}%", ha='center', fontsize=12)

    # Save the figure as a file
    fig.tight_layout()
    fig.savefig('memory_usage.png')

    return 'memory_usage.png'