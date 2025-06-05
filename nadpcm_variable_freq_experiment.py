import numpy as np
import matplotlib.pyplot as plt
from lab1_ndpcm_library import init, prepare_params_for_prediction, predict, calculate_error, reconstruct

def run_variable_frequency_experiment():
    """运行变频率实验，直到失真度超过20%"""
    # 固定参数
    n_bits = 16
    n = 100  # 采样点数
    h_depth = 3  # 历史深度
    
    # 存储结果
    results = {
        'frequencies': [],
        'distortions': [],
        'avg_errors': [],
        'critical_freq': None
    }
    
    # 初始频率和步长
    frequency = 1.0
    freq_step = 0.1
    max_freq = 10.0  # 最大频率限制
    
    print("Starting variable frequency experiment...")
    print(f"{'Frequency':<10} {'Distortion (%)':<15} {'Avg Error':<15}")
    print("-" * 40)
    
    while frequency <= max_freq:
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
        
        # 计算失真度
        distortion = np.mean(np.abs((rx_data.y_recreated - f) / (np.abs(f) + 1e-10)) * 100)
        avg_error = np.mean(np.abs(rx_data.y_recreated - f))
        
        # 存储结果
        results['frequencies'].append(frequency)
        results['distortions'].append(distortion)
        results['avg_errors'].append(avg_error)
        
        # 打印当前结果
        print(f"{frequency:<10.2f} {distortion:<15.2f} {avg_error:<15.2f}")
        
        # 检查是否超过20%失真度
        if distortion > 20 and results['critical_freq'] is None:
            results['critical_freq'] = frequency
            print(f"\n⚠️ Critical point reached at frequency {frequency:.2f} Hz with {distortion:.2f}% distortion")
        
        # 增加频率
        frequency += freq_step
    
    return results
def plot_variable_frequency_results(results):
    """绘制变频率实验结果（y轴对数显示）"""
    plt.figure(figsize=(12, 8))

    # 失真度曲线（对数y轴）
    plt.subplot(2, 1, 1)
    plt.plot(results['frequencies'], results['distortions'], 'bo-', linewidth=2, markersize=6)
    plt.axhline(y=20, color='r', linestyle='--', label='20% Distortion Threshold')
    
    # 设置y轴为对数坐标
    plt.yscale('log')
    
    # 标记临界点
    if results['critical_freq'] is not None:
        plt.axvline(x=results['critical_freq'], color='g', linestyle=':',
                   label=f'Critical Freq: {results["critical_freq"]:.2f} Hz')

    plt.xlabel('Signal Frequency (Hz)')
    plt.ylabel('Distortion (%) - Log Scale')
    plt.title('Distortion vs Signal Frequency (Log Scale)')
    plt.grid(True, alpha=0.3, which='both')  # 添加对数网格线
    plt.legend()

    # 平均误差曲线（对数y轴）
    plt.subplot(2, 1, 2)
    plt.plot(results['frequencies'], results['avg_errors'], 'ro-', linewidth=2, markersize=6)
    
    # 设置y轴为对数坐标
    plt.yscale('log')
    
    # 标记临界点
    if results['critical_freq'] is not None:
        plt.axvline(x=results['critical_freq'], color='g', linestyle=':')

    plt.xlabel('Signal Frequency (Hz)')
    plt.ylabel('Average Error - Log Scale')
    plt.title('Average Error vs Signal Frequency (Log Scale)')
    plt.grid(True, alpha=0.3, which='both')  # 添加对数网格线

    plt.tight_layout()
    plt.savefig('nadpcm_variable_frequency_results_log.png', dpi=300)  # 修改文件名以区分
    plt.show()

def print_critical_frequency(results):
    """打印临界频率结果"""
    if results['critical_freq'] is None:
        print("\nNo critical frequency found below 20% distortion within tested range.")
    else:
        idx = results['frequencies'].index(results['critical_freq'])
        print("\n" + "="*60)
        print("CRITICAL FREQUENCY ANALYSIS")
        print("="*60)
        print(f"Critical frequency: {results['critical_freq']:.2f} Hz")
        print(f"Distortion at critical point: {results['distortions'][idx]:.2f}%")
        print(f"Average error at critical point: {results['avg_errors'][idx]:.2f}")
        
        # 找到低于临界点的最大频率
        below_threshold = [f for f, d in zip(results['frequencies'], results['distortions']) 
                          if d <= 20 and f < results['critical_freq']]
        if below_threshold:
            max_safe_freq = max(below_threshold)
            print(f"\nMaximum safe frequency below 20% distortion: {max_safe_freq:.2f} Hz")
        else:
            print("\nNo safe frequencies found below 20% distortion.")

if __name__ == "__main__":
    # 运行变频率实验
    freq_results = run_variable_frequency_experiment()
    
    # 打印并绘制结果
    print_critical_frequency(freq_results)
    plot_variable_frequency_results(freq_results)
    
    print("\n变频率实验完成！结果已保存为图片文件。")
