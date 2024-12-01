import psutil
import GPUtil
import gradio as gr
import time
import pandas as pd

# Initialize data
x_cpu_data = []
x_gpu_data = []
ydata_cpu = []
ydata_gpu = []
start_time = time.time()

def get_CPU_usage():
    cpu_usage = psutil.cpu_percent(interval=0.01)

    x_cpu_data.append(time.time() - start_time)
    ydata_cpu.append(cpu_usage)

    # Keep only the latest 100 data points
    xdata_recent = x_cpu_data[-50:]
    ydata_cpu_recent = ydata_cpu[-50:]

    # Return the data for plotting
    return pd.DataFrame({
        "time": xdata_recent,
        "CPU Usage": ydata_cpu_recent
    })

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

with gr.Blocks() as demo:
    timer = gr.Timer(0.01)

    plot1 = gr.LinePlot(
        get_CPU_usage,
        x="time",
        y="CPU Usage",
        every=timer,
        y_lim=[0, 100],
        title="CPU Usage",
        x_axis_labels_visible=False
    )
    plot2 = gr.LinePlot(
        get_GPU_usage,
        x='time',
        y="GPU Usage",
        every=timer,
        y_lim=[0, 100],  # Set y-axis range
        title="GPU Usage",
        x_axis_labels_visible=False
    )

demo.launch()
