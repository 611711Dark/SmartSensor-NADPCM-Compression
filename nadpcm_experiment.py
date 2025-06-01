# 完整的NADPCM实验解决方案
# Lab 1: Nonlinear Adaptive Pulse Coded Modulation-Based Compression (NADPCMC)

import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple
from lab1_library import quantize
from lab1_ndpcm_library import init, prepare_params_for_prediction, predict, calculate_error, reconstruct

# ======================= 主实验代码 =======================
def run_nadpcm_experiment():
    """运行NADPCM实验"""
    
    # 实验参数
    n_bits_list = [4, 6, 8, 10, 13]  # 不同的编码位数
    n = 100  # 迭代次数
    h_depth = 3  # 历史深度
    
    # 生成测试信号
    x = np.linspace(0, 2*np.pi, n)
    
    # 测试不同类型的信号
    signals = {
        'slow_sine': (np.sin(x) + 1) * 1000,  # 慢变正弦
        'fast_sine': (np.sin(5*x) + 1) * 1000,  # 快变正弦
        'very_fast_sine': (np.sin(10*x) + 1) * 1000,  # 很快变正弦
    }
    
    results = {}
    
    for signal_name, f in signals.items():
        results[signal_name] = {}
        
        for n_bits in n_bits_list:
            print(f"\n处理信号: {signal_name}, 编码位数: {n_bits}")
            
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
            original_bits = 16 * n  # 假设原始数据16位
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
    """绘制实验结果"""
    
    # 为每个信号类型创建图表
    for signal_name, signal_results in results.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'NADPCM Performance - {signal_name}', fontsize=16)
        
        # 子图1: 原始信号vs重构信号 (使用8bit作为示例)
        if 8 in signal_results:
            ax1 = axes[0, 0]
            n = len(signal_results[8]['original_signal'])
            t = np.arange(n)
            ax1.plot(t, signal_results[8]['original_signal'], 'b-', label='Original', linewidth=2)
            ax1.plot(t, signal_results[8]['reconstructed_signal'], 'r--', label='Reconstructed (8-bit)', linewidth=1)
            ax1.set_xlabel('Sample')
            ax1.set_ylabel('Amplitude')
            ax1.set_title('Signal Reconstruction (8-bit)')
            ax1.legend()
            ax1.grid(True)
        
        # 子图2: 重构误差
        if 8 in signal_results:
            ax2 = axes[0, 1]
            reconstruction_error = signal_results[8]['reconstructed_signal'] - signal_results[8]['original_signal']
            ax2.plot(t, reconstruction_error, 'g-', linewidth=1)
            ax2.set_xlabel('Sample')
            ax2.set_ylabel('Error')
            ax2.set_title('Reconstruction Error (8-bit)')
            ax2.grid(True)
        
        # 子图3: 压缩比vs失真度
        ax3 = axes[1, 0]
        n_bits_list = list(signal_results.keys())
        compression_ratios = [signal_results[nb]['compression_ratio'] for nb in n_bits_list]
        distortions = [signal_results[nb]['distortion'] for nb in n_bits_list]
        
        ax3.plot(compression_ratios, distortions, 'o-', linewidth=2, markersize=8)
        for i, nb in enumerate(n_bits_list):
            ax3.annotate(f'{nb}-bit', (compression_ratios[i], distortions[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax3.set_xlabel('Compression Ratio')
        ax3.set_ylabel('Distortion (%)')
        ax3.set_title('Compression Ratio vs Distortion')
        ax3.grid(True)
        
        # 子图4: 平均误差vs编码位数
        ax4 = axes[1, 1]
        avg_errors = [signal_results[nb]['avg_error'] for nb in n_bits_list]
        ax4.plot(n_bits_list, avg_errors, 's-', linewidth=2, markersize=8, color='orange')
        ax4.set_xlabel('Number of Bits')
        ax4.set_ylabel('Average Error')
        ax4.set_title('Average Error vs Encoding Bits')
        ax4.grid(True)
        
        plt.tight_layout()
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

if __name__ == "__main__":
    # 运行主实验
    print("\n开始NADPCM实验...")
    results = run_nadpcm_experiment()
    
    # 显示结果
    print_performance_table(results)
    find_20_percent_threshold(results)
    
    # 绘制图表
    plot_results(results)
    
    print("\n实验完成！结果已保存为图片文件。")
