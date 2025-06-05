import numpy as np
import matplotlib.pyplot as plt
from lab1_ndpcm_library import init, prepare_params_for_prediction, predict, calculate_error, reconstruct

def format_large_number(value, precision=2):
    """格式化特大值为科学计数法或常规浮点数"""
    if abs(value) > 1e6:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"

def run_h_depth_experiment():
    """运行历史深度(h_depth)参数实验"""
    # 固定参数
    n_bits = 14
    n = 100  
    frequency = 2.0 

    h_depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]

    # 存储结果
    results = {
        'h_depths': h_depths,
        'distortions': [],
        'avg_errors': [],
        'max_errors': [],
        'bit_rates': []
    }

    print("Starting h_depth parameter experiment...")
    print(f"{'h_depth':<10} {'Distortion (%)':<25} {'Avg Error':<25} {'Max Error':<25} {'Bit Rate (kbps)':<15}")
    print("-" * 100)

    for h_depth in h_depths:
        # 生成正弦信号
        x = np.linspace(0, 2*np.pi, n)
        f = (np.sin(frequency * x) + 1) * pow(2, n_bits - 1)

        # 初始化发送端和接收端
        tx_data = init(n, h_depth, n_bits)
        rx_data = init(n, h_depth, n_bits)

        # NADPCM处理过程
        for k in range(n):
            # 发送端处理
            prepare_params_for_prediction(tx_data, k)
            y_hat = predict(tx_data, k)
            eq = calculate_error(tx_data, k, f[k])
            y_rec = reconstruct(tx_data, k)

            # 传输到接收端
            rx_data.eq[k] = tx_data.eq[k]

            # 接收端处理
            prepare_params_for_prediction(rx_data, k)
            y_hat_rx = predict(rx_data, k)
            y_rec_rx = reconstruct(rx_data, k)

        # 计算性能指标
        abs_errors = np.abs(rx_data.y_recreated - f)
        distortion = np.mean(abs_errors / (np.abs(f) + 1e-10)) * 100  # 避免除以零
        avg_error = np.mean(abs_errors)
        max_error = np.max(abs_errors)

        # 计算比特率 (假设采样率1kHz)
        bit_rate = n_bits * 1000 / 1000  # kbps

        # 存储结果
        results['distortions'].append(distortion)
        results['avg_errors'].append(avg_error)
        results['max_errors'].append(max_error)
        results['bit_rates'].append(bit_rate)

        # 格式化特大值为科学计数法
        distortion_str = format_large_number(distortion)
        avg_error_str = format_large_number(avg_error)
        max_error_str = format_large_number(max_error)

        # 打印当前结果
        print(f"{h_depth:<10} {distortion_str:<25} {avg_error_str:<25} {max_error_str:<25} {bit_rate:<15.2f}")

    return results
def plot_h_depth_results(results):
    """绘制历史深度实验结果 - 4个子图显示在一起"""
    plt.figure(figsize=(16, 12))

    # 第1个子图：失真度曲线 (对数坐标)
    plt.subplot(2, 2, 1)
    plt.semilogy(results['h_depths'], results['distortions'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('History Depth (h_depth)')
    plt.ylabel('Distortion (%) - Log Scale')
    plt.title('Distortion vs History Depth (Log Scale)')
    plt.grid(True, alpha=0.3, which='both')
    plt.xticks(results['h_depths'])

    # 添加数据点标注
    for i, (h, d) in enumerate(zip(results['h_depths'], results['distortions'])):
        plt.annotate(format_large_number(d, 1),
                    (h, d),
                    textcoords="offset points",
                    xytext=(0, 10 if i % 2 == 0 else -15),
                    ha='center',
                    fontsize=8)

    # 第2个子图：平均误差曲线 (对数坐标)
    plt.subplot(2, 2, 2)
    plt.semilogy(results['h_depths'], results['avg_errors'], 'ro-', linewidth=2, markersize=8)
    plt.xlabel('History Depth (h_depth)')
    plt.ylabel('Average Error - Log Scale')
    plt.title('Average Error vs History Depth (Log Scale)')
    plt.grid(True, alpha=0.3, which='both')
    plt.xticks(results['h_depths'])

    # 添加数据点标注
    for i, (h, e) in enumerate(zip(results['h_depths'], results['avg_errors'])):
        plt.annotate(format_large_number(e, 1),
                    (h, e),
                    textcoords="offset points",
                    xytext=(0, 10 if i % 2 == 0 else -15),
                    ha='center',
                    fontsize=8)

    # 第3个子图：最大误差曲线 (对数坐标)
    plt.subplot(2, 2, 3)
    plt.semilogy(results['h_depths'], results['max_errors'], 'go-', linewidth=2, markersize=8)
    plt.xlabel('History Depth (h_depth)')
    plt.ylabel('Maximum Error - Log Scale')
    plt.title('Maximum Error vs History Depth (Log Scale)')
    plt.grid(True, alpha=0.3, which='both')
    plt.xticks(results['h_depths'])

    # 添加数据点标注
    for i, (h, e) in enumerate(zip(results['h_depths'], results['max_errors'])):
        plt.annotate(format_large_number(e, 1),
                    (h, e),
                    textcoords="offset points",
                    xytext=(0, 10 if i % 2 == 0 else -15),
                    ha='center',
                    fontsize=8)

    # 第4个子图：综合性能对比图 (对数坐标)
    plt.subplot(2, 2, 4)
    plt.semilogy(results['h_depths'], results['distortions'], 'bo-', label='Distortion (%)', linewidth=2, markersize=6)
    plt.semilogy(results['h_depths'], results['avg_errors'], 'ro-', label='Avg Error', linewidth=2, markersize=6)
    plt.semilogy(results['h_depths'], results['max_errors'], 'go-', label='Max Error', linewidth=2, markersize=6)

    plt.xlabel('History Depth (h_depth)')
    plt.ylabel('Logarithmic Scale')
    plt.title('Performance Metrics Comparison (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.xticks(results['h_depths'])

    # 调整子图间距
    plt.tight_layout(pad=3.0)
    
    # 保存图片
    plt.savefig('nadpcm_h_depth_4plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_optimal_h_depth(results):
    """打印最优历史深度结果"""
    # 找到最小失真的索引
    min_dist_idx = np.argmin(results['distortions'])
    min_avg_idx = np.argmin(results['avg_errors'])
    min_max_idx = np.argmin(results['max_errors'])

    print("\n" + "="*100)
    print("OPTIMAL HISTORY DEPTH ANALYSIS")
    print("="*100)
    
    # 格式化输出
    min_dist_str = format_large_number(results['distortions'][min_dist_idx])
    min_avg_str = format_large_number(results['avg_errors'][min_avg_idx])
    min_max_str = format_large_number(results['max_errors'][min_max_idx])
    
    print(f"Minimum distortion ({min_dist_str}%) at h_depth = {results['h_depths'][min_dist_idx]}")
    print(f"Minimum average error ({min_avg_str}) at h_depth = {results['h_depths'][min_avg_idx]}")
    print(f"Minimum maximum error ({min_max_str}) at h_depth = {results['h_depths'][min_max_idx]}")

    # 综合推荐值 (平均三个最优值的索引)
    avg_idx = np.mean([min_dist_idx, min_avg_idx, min_max_idx])
    rec_idx = int(round(avg_idx))
    rec_h_depth = results['h_depths'][rec_idx]

    # 格式化推荐值
    rec_dist_str = format_large_number(results['distortions'][rec_idx])
    rec_avg_str = format_large_number(results['avg_errors'][rec_idx])
    rec_max_str = format_large_number(results['max_errors'][rec_idx])
    
    print("\n" + "-"*100)
    print(f"RECOMMENDED optimal h_depth: {rec_h_depth}")
    print(f"  Distortion: {rec_dist_str}%")
    print(f"  Average Error: {rec_avg_str}")
    print(f"  Maximum Error: {rec_max_str}")
    print("-"*100)

if __name__ == "__main__":
    # 运行历史深度参数实验
    h_depth_results = run_h_depth_experiment()

    # 打印并绘制结果
    print_optimal_h_depth(h_depth_results)
    plot_h_depth_results(h_depth_results)

    print("\n历史深度参数实验完成！结果已保存为图片文件。")
