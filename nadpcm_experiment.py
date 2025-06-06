# Lab 1: Nonlinear Adaptive Pulse Coded Modulation-Based Compression (NADPCMC)

import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple
from lab1_library import quantize
from lab1_ndpcm_library import init, prepare_params_for_prediction, predict, calculate_error, reconstruct
import matplotlib.ticker as mticker

def run_nadpcm_experiment():
    
    n_bits_list = [8, 10, 12, 14, 16]  
    n = 200  
    h_depth = 3  
    
    original_signal = 12
    
    x = np.linspace(0, 2*np.pi, n)
    
    # 测试不同类型的信号
    #Type,Frequency
    signals = {
        'slow_sine': 1,  # 慢变正弦
        'fast_sine': 3,  # 快变正弦
        'very_fast_sine': 5,  # 很快变正弦
    }
    results = {}
    for signal_name, freq in signals.items():
        results[signal_name] = {}
        
        for n_bits in n_bits_list:
            f = (np.sin(freq*x) + 1) * pow(2, 12 - 1)
            # 初始化发送端和接收端
            tx_data = init(n, h_depth, n_bits)
            rx_data = init(n, h_depth, n_bits)
            
            # NADPCM编码和解码过程
            for k in range(n):
                # 发送端
                prepare_params_for_prediction(tx_data, k)
                y_hat = predict(tx_data, k)
                eq = calculate_error(tx_data, k, f[k])
                y_rec = reconstruct(tx_data, k)
                
                # 传输量化误差到接收端
                rx_data.eq[k] = tx_data.eq[k]
                
                # 接收端
                prepare_params_for_prediction(rx_data, k)
                y_hat_rx = predict(rx_data, k)
                y_rec_rx = reconstruct(rx_data, k)
            
            # 计算性能指标
            reconstruction_error = np.abs(rx_data.y_recreated - f)
            avg_error = np.mean(reconstruction_error)
            max_error = np.max(reconstruction_error)
            
            # 计算失真度
            distortion = np.mean(np.abs((rx_data.y_recreated - f) / (f + 1e-10)) * 100)
            
            # 计算压缩比
            original_bits = 12 * n  
            compressed_bits = n_bits * n
            compression_ratio = original_bits / compressed_bits
            
            results[signal_name][n_bits] = {
                'avg_error': avg_error,
                'max_error': max_error,
                'distortion': distortion,
                'compression_ratio': compression_ratio,
                'original_signal': f,
                'reconstructed_signal': rx_data.y_recreated.copy(),
                'quantized_error': rx_data.eq.copy()
            }
    
    return results

def plot_results(results):
    """绘制实验结果，显示所有比特数的重建信号和误差，使用科学计数法显示关键指标"""
    for signal_name, signal_results in results.items():
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 2)
        fig.suptitle(f'NADPCM Performance - {signal_name}', fontsize=20, y=0.98)

        # 获取所有可用比特数（排序）
        n_bits_list = sorted(signal_results.keys())

        # ==================== 子图1: 原始信号和所有比特数的重建信号 ====================
        ax1 = fig.add_subplot(gs[0, :])

        # 获取原始信号（使用第一个可用的比特数）
        first_bits = n_bits_list[0]
        original_signal = signal_results[first_bits]['original_signal']
        n = len(original_signal)
        t = np.arange(n)

        # 绘制原始信号
        ax1.plot(t, original_signal, 'k-', label='Original-12bit', linewidth=3, alpha=0.8)

        # 绘制所有比特数的重建信号
        colors = plt.cm.viridis(np.linspace(0, 1, len(n_bits_list)))
        for i, bits in enumerate(n_bits_list):
            reconstructed = signal_results[bits]['reconstructed_signal']
            ax1.plot(t, reconstructed, '--',
                     color=colors[i],
                     linewidth=1.5,
                     alpha=0.7,
                     label=f'Recon {bits}-bit')

        # 设置Y轴范围基于原始信号振幅
        y_min = np.min(original_signal)
        y_max = np.max(original_signal)
        y_padding = (y_max - y_min) * 0.15  # 15%的填充
        ax1.set_ylim([y_min - y_padding, y_max + y_padding])

        ax1.set_xlabel('Sample Index', fontsize=14)
        ax1.set_ylabel('Amplitude', fontsize=14)
        ax1.set_title('Signal Reconstruction (All Bit Depths)', fontsize=16)
        ax1.legend(loc='upper right', fontsize=11, ncol=2)
        ax1.grid(True, alpha=0.3)

        # ==================== 子图2: 所有比特数的重建误差 ====================
        ax2 = fig.add_subplot(gs[1, :])

        # 计算并绘制所有比特数的重建误差
        max_error = 0
        min_error = 0

        for i, bits in enumerate(n_bits_list):
            reconstructed = signal_results[bits]['reconstructed_signal']
            reconstruction_error = reconstructed - original_signal
            ax2.plot(t, reconstruction_error,'--',
                     color=colors[i],
                     linewidth=1.2,
                     alpha=0.8,
                     label=f'Error {bits}-bit')

            # 更新最大最小误差值
            current_max = np.max(reconstruction_error)
            current_min = np.min(reconstruction_error)
            if current_max > max_error:
                max_error = current_max
            if current_min < min_error:
                min_error = current_min

        # 设置Y轴范围基于原始信号振幅
        y_min = np.min(original_signal)
        y_max = np.max(original_signal)
        y_padding = (y_max - y_min) * 0.15  # 15%的填充
        ax2.set_ylim([-y_max-y_min-y_padding, y_max + y_min+y_padding])

        ax2.set_xlabel('Sample Index', fontsize=14)
        ax2.set_ylabel('Error Magnitude', fontsize=14)
        ax2.set_title('Reconstruction Errors (All Bit Depths)', fontsize=16)
        ax2.legend(loc='upper right', fontsize=11, ncol=2)
        ax2.grid(True, alpha=0.3)

        # ==================== 子图3: 压缩比vs失真度（对数y轴）====================
        ax3 = fig.add_subplot(gs[2, 0])
        compression_ratios = [signal_results[nb]['compression_ratio'] for nb in n_bits_list]
        distortions = [signal_results[nb]['distortion'] for nb in n_bits_list]

        # 验证数据一致性
        if len(compression_ratios) != len(distortions) or len(compression_ratios) != len(n_bits_list):
            print(f"警告: {signal_name} 的数据长度不一致")
            continue

        # 创建主Y轴（对数坐标）
        ax3.set_xlabel('Number of Encoding Bits', fontsize=14)
        ax3.set_ylabel('Distortion (%) - Log Scale', fontsize=14, color='b')
        ax3.set_title('Compression Ratio vs Distortion (Log Scale)', fontsize=16)
        ax3.set_yscale('log')  # 设置y轴为对数坐标
        ax3.grid(True, alpha=0.3, which='both')  # 添加对数网格

        # 绘制失真度（对数坐标）
        line1 = ax3.plot(n_bits_list, distortions, 'bo-', linewidth=2, markersize=8,
                         label='Distortion')

        # 添加压缩比作为次Y轴（线性坐标）
        ax3b = ax3.twinx()
        ax3b.set_ylabel('Compression Ratio', fontsize=14, color='r')
        line2 = ax3b.plot(n_bits_list, compression_ratios, 'rs-', linewidth=2, markersize=8,
                          label='Compression Ratio')

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='best', fontsize=10)

        # 添加数据点标签
        for i, nb in enumerate(n_bits_list):
            # 使用科学计数法显示失真度
            dist_label = f'D:{distortions[i]:.2e}' if distortions[i] != 0 else '0'
            comp_label = f'C:{compression_ratios[i]:.2f}'

            ax3.annotate(f'{dist_label}\n{comp_label}',
                         (nb, distortions[i]),
                         xytext=(0, 15),
                         textcoords='offset points',
                         ha='center',
                         fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # ==================== 子图4: 平均误差vs编码位数（对数y轴）====================
        ax4 = fig.add_subplot(gs[2, 1])
        avg_errors = [signal_results[nb]['avg_error'] for nb in n_bits_list]

        # 验证数据一致性
        if len(avg_errors) != len(n_bits_list):
            print(f"警告: {signal_name} 的平均误差数据长度不一致")
            continue

        # 创建散点图（对数y轴）
        ax4.set_xlabel('Number of Encoding Bits', fontsize=14)
        ax4.set_ylabel('Average Error - Log Scale', fontsize=14)
        ax4.set_title('Average Error vs Encoding Bits (Log Scale)', fontsize=16)
        ax4.set_yscale('log')  # 设置y轴为对数坐标
        ax4.grid(True, alpha=0.3, which='both')  # 添加对数网格

        scatter = ax4.scatter(n_bits_list, avg_errors, c=n_bits_list,
                             cmap='viridis', s=150, alpha=0.8)

        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax4)
        cbar.set_label('Encoding Bits', fontsize=12)

        # 添加数据点标签
        for i, nb in enumerate(n_bits_list):
            # 使用科学计数法显示平均误差
            label = f'{avg_errors[i]:.2e}'
            
            ax4.annotate(label,
                         (nb, avg_errors[i]),
                         xytext=(0, 12),
                         textcoords='offset points',
                         ha='center',
                         fontsize=10,
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # 添加整体统计信息
        min_distortion = min(distortions) if distortions else 0
        max_distortion = max(distortions) if distortions else 0
        min_avg_error = min(avg_errors) if avg_errors else 0
        max_avg_error = max(avg_errors) if avg_errors else 0
        min_comp_ratio = min(compression_ratios) if compression_ratios else 0
        max_comp_ratio = max(compression_ratios) if compression_ratios else 0
        
        stats_text = (
            f"Statistics for {signal_name}:\n"
            f"Distortion Range: {min_distortion:.2e} - {max_distortion:.2e}\n"
            f"Avg Error Range: {min_avg_error:.2e} - {max_avg_error:.2e}\n"
            f"Compression Ratios: {min_comp_ratio:.2f} - {max_comp_ratio:.2f}"
        )
        fig.text(0.5, 0.01, stats_text, fontsize=12, ha='center',
                 bbox=dict(facecolor='white', alpha=0.8))

        # 调整布局并保存
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # 为总标题和统计信息留空间
        plt.savefig(f'nadpcm_results_{signal_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

def print_performance_table(results):
    """打印性能对比表"""
    print("\n" + "="*80)
    print("NADPCM PERFORMANCE COMPARISON TABLE")
    print("="*80)
    
    # 辅助函数：智能格式化数值
    def smart_format(num):
        if abs(num) > 1e4 or (abs(num) < 1e-2 and num != 0):
            return "{:.2e}".format(num)
        else:
            return "{:.2f}".format(num)
    
    for signal_name, signal_results in results.items():
        print(f"\n{signal_name.upper()} Signal:")
        print("-" * 60)
        print(f"{'Bits':<6} {'Compression':<12} {'Avg Error':<12} {'Max Error':<12} {'Distortion':<12}")
        print(f"{'':6} {'Ratio':<12} {'':12} {'':12} {'(%)':<12}")
        print("-" * 60)

        for n_bits in sorted(signal_results.keys()):
            result = signal_results[n_bits]
            # 使用智能格式化处理大数值
            avg_error_fmt = smart_format(result['avg_error'])
            max_error_fmt = smart_format(result['max_error'])
            distortion_fmt = smart_format(result['distortion'])
            
            print(f"{n_bits:<6} {result['compression_ratio']:<12.2f} {avg_error_fmt:<12} "
                  f"{max_error_fmt:<12} {distortion_fmt:<12}")

# ======================= 测试代码 =======================
def find_20_percent_threshold(results):
    """找到失真度超过20%的信号类型"""
    print("\n" + "="*60)
    print("SIGNALS WITH >20% DISTORTION:")
    print("="*60)
    
    # 辅助函数：智能格式化数值
    def smart_format(num):
        if abs(num) > 1e4 or (abs(num) < 1e-2 and num != 0):
            return "{:.2e}".format(num)
        else:
            return "{:.2f}".format(num)
    
    for signal_name, signal_results in results.items():
        for n_bits, result in signal_results.items():
            if result['distortion'] > 20:
                # 使用智能格式化处理大数值
                distortion_fmt = smart_format(result['distortion'])
                print(f"{signal_name} with {n_bits}-bit encoding: {distortion_fmt}% distortion")

def print_distortion_table(results):
    """打印不同比特数和信号类型下的失真率表格"""
    print("\n" + "="*60)
    print("DISTORTION RATE PER SIGNAL AND BIT DEPTH")
    print("="*60)
    
    # 获取所有比特数并排序
    n_bits_list = sorted(next(iter(results.values())).keys())
    
    # 表头
    header = f"{'Signal/Bits':<20}" + "".join([f"{f'{bits}-bit':<15}" for bits in n_bits_list])
    print(header)
    print("-" * (20 + 15*len(n_bits_list)))
    
    # 每行数据
    for signal_name in results.keys():
        row = f"{signal_name:<20}"
        for n_bits in n_bits_list:
            distortion = results[signal_name][n_bits]['distortion']
            # 格式化显示：科学计数法用于极大值，否则保留2位小数
            if distortion > 1e5:
                distortion_str = f"{distortion:.2e}%"
            else:
                distortion_str = f"{distortion:.2f}%"
            row += f"{distortion_str:<15}"
        print(row)

if __name__ == "__main__":
    # 运行主实验
    print("\nstart NADPCM experiment...")
    results = run_nadpcm_experiment()
    
    # 显示结果
    print_performance_table(results)
    print_distortion_table(results)  # 新增的失真率表格
    find_20_percent_threshold(results)
    
    # 绘制图表
    plot_results(results)
    
    print("\n实验完成！结果已保存为图片文件。")

