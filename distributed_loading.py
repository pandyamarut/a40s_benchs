from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json
import pynvml
import matplotlib.pyplot as plt
import threading
import numpy as np
import psutil

accelerator = Accelerator()


# load a base model and tokenizer
# model_path="NousResearch/Llama-2-7b-hf"
# start = time.time()
# model = AutoModelForCausalLM.from_pretrained(v
#     model_path,    
#     device_map={"": accelerator.process_index},
#     # device_map="auto",
#     torch_dtype=torch.bfloat16,
# )
# end = time.time()
# print(f"Loading time = {end - start}")
  

# funtion for model loading 
def init_pynvml():
    try:
        pynvml.nvmlInit()
        return True
    except pynvml.NVMLError as err:
        print(f"Failed to initialize NVML: {err}")
        return False

# def get_pci_throughput(gpu_handles):
#     stats = {"pci_tx": [], "pci_rx": []}
#     for handle in gpu_handles:
#         tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) * 1024
#         rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) * 1024
#         stats["pci_tx"].append(tx)
#         stats["pci_rx"].append(rx)
#     return stats
def get_pci_throughput(gpu_handles):
    stats = {"pci_tx": [], "pci_rx": []}
    bytes_to_gb = 1024 * 1024 * 1024

    for handle in gpu_handles: 
        tx = (pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) * 1024) / bytes_to_gb
        rx = (pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) * 1024) / bytes_to_gb
        stats["pci_tx"].append(tx)
        stats["pci_rx"].append(rx)
    return stats
    
def load_model():
    model_path="mistralai/Mistral-7B-v0.1"
    print(f"model. Name: {model_path}")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map={"": accelerator.process_index},
        # device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    end = time.time()
    print(f"Loading time = {end - start}")
    return model

def get_gpu_handles():
    try:
        ngpus = pynvml.nvmlDeviceGetCount()
        return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(ngpus)]
    except pynvml.NVMLError as err:
        print(f"Error getting GPU handles: {err}")
        return []
    
def get_disk_io_speed():
    io1 = psutil.disk_io_counters()
    time.sleep(1)
    io2 = psutil.disk_io_counters()
    read_speed = (io2.read_bytes - io1.read_bytes) / 1024 / 1024 / 1024  # GB/s
    write_speed = (io2.write_bytes - io1.write_bytes) / 1024 / 1024 / 1024  # GB/s
    return read_speed, write_speed


def monitor_and_plot(gpu_handles):
    tx_rates, rx_rates, read_speeds, write_speeds, time_points = [], [], [], [], []

    model_thread = threading.Thread(target=load_model)
    model_thread.start()

    start_time = time.time()
    interval = 1  # seconds

    while model_thread.is_alive():
        pci_stats = get_pci_throughput(gpu_handles)
        tx_rates.append(sum(pci_stats["pci_tx"]))
        rx_rates.append(sum(pci_stats["pci_rx"]))

        read_speed, write_speed = get_disk_io_speed()
        read_speeds.append(read_speed)
        write_speeds.append(write_speed)

        current_time = time.time()
        time_points.append(current_time - start_time)
        time.sleep(interval)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot PCIe throughput
    ax1.plot(time_points, tx_rates, label='PCIe TX (GB/s)')
    ax1.plot(time_points, rx_rates, label='PCIe RX (GB/s)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Throughput (GB/s)')
    ax1.set_title('PCIe Throughput Over Time During Model Loading-70b-8xA40')
    ax1.legend()

    # Plot Disk IO speeds
    ax2.plot(time_points, read_speeds, label='Disk Read (GB/s)')
    ax2.plot(time_points, write_speeds, label='Disk Write (GB/s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed (GB/s)')
    ax2.set_title('Disk Read/Write Speed Over Time During Model Loading -70b- 8xA40')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('70b_disk_pcie')
    
       
def main():
    if not init_pynvml():
        return    
    gpu_handles = get_gpu_handles()
    if not gpu_handles:
        return
    monitor_and_plot(gpu_handles)



if __name__ == "__main__":
    main()    
