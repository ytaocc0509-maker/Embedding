import torch
import time

# 创建大矩阵测试 GPU 性能
if torch.cuda.is_available():
    print("=== GPU 性能测试 ===")

    # 创建两个 5000x5000 的随机矩阵
    size = 5000
    a = torch.randn(size, size).cuda()
    b = torch.randn(size, size).cuda()

    # GPU 矩阵乘法
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # 等待 GPU 完成
    gpu_time = time.time() - start

    print(f"GPU 计算 {size}x{size} 矩阵乘法: {gpu_time:.4f} 秒")

    # CPU 对比测试（如果内存足够）
    try:
        a_cpu = a.cpu()
        b_cpu = b.cpu()

        start = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start

        print(f"CPU 计算 {size}x{size} 矩阵乘法: {cpu_time:.4f} 秒")
        print(f"GPU 加速比: {cpu_time / gpu_time:.2f}x")
    except:
        print("CPU 测试跳过（内存可能不足）")

    print("\n=== GPU 信息 ===")
    print(f"GPU 内存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"GPU 内存已用: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(
        f"GPU 内存剩余: {torch.cuda.get_device_properties(0).total_memory / 1e9 - torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
else:
    print("GPU 不可用")