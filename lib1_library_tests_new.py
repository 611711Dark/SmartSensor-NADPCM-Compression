import lab1_library    # The code to test
import unittest   # The test framework
import lab1_ndpcm_library
import numpy as np

## Example of quantization function tests
class Test_TestQuantizationOfErrors(unittest.TestCase):
    def test_quantize_returns_zero_for_1bit(self):
        self.assertEqual(lab1_library.quantize(0, 1), 0)
    def test_quantize_returns_zero_for_10bit(self):
        self.assertEqual(lab1_library.quantize(0, 10), 0)
    def test_quantize_rounds_to_integer_for_10bit(self):
        self.assertEqual(lab1_library.quantize(11.4, 10), 11)
    def test_quantize_saturates_at_minimum_for_8bit(self):
        self.assertEqual(lab1_library.quantize(-129, 8), -128)
    def test_quantize_saturates_at_maximum_for_9bit(self):
        self.assertEqual(lab1_library.quantize(40000, 9), +255)
    def test_predictor_initialization(self):
        n = 100
        h_depth = 3 
        n_bits = 12
        data = lab1_ndpcm_library.init(n, h_depth, n_bits)
        
        # 验证初始值设置
        self.assertTrue(np.array_equal(data.phi[0], np.zeros(h_depth)))
        self.assertTrue(np.array_equal(data.theta[0], np.zeros(h_depth)))
        self.assertEqual(data.y_recreated[0], 0)
        
        # 验证数组维度
        self.assertEqual(data.phi.shape, (n, h_depth))
        self.assertEqual(data.theta.shape, (n, h_depth))
        self.assertEqual(len(data.y_hat), n)

    def test_predictor_update(self):
        n = 10
        h_depth = 3
        n_bits = 8
        data = lab1_ndpcm_library.init(n, h_depth, n_bits)
        
        # 模拟前几个时间步的数据
        data.y_recreated[0] = 100
        data.y_recreated[1] = 105
        data.y_recreated[2] = 110
        data.eq[1] = 5
        data.eq[2] = 3
        
        # 测试k=1时的更新
        lab1_ndpcm_library.prepare_params_for_prediction(data, 1)
        self.assertTrue(np.array_equal(data.phi[0], [0, 0, 0]))
        
        # 测试k=2时的更新
        lab1_ndpcm_library.prepare_params_for_prediction(data, 2)
        self.assertTrue(np.array_equal(data.phi[1], [105, 100, 0]))
        
        # 测试k=3时的更新
        lab1_ndpcm_library.prepare_params_for_prediction(data, 3)
        self.assertTrue(np.array_equal(data.phi[2], [110, 105, 100]))
        
        # 验证权重更新
        alpha = lab1_ndpcm_library.calculate_alpha_max(data.phi[1])
        expected_theta = data.theta[0] + alpha * data.eq[1] * data.phi[0]
        self.assertTrue(np.allclose(data.theta[1], expected_theta))


    def test_quantizer_boundaries(self):
        # 测试不同比特数下的量化边界
        self.assertEqual(lab1_library.quantize(127.4, 8), 127)
        self.assertEqual(lab1_library.quantize(127.6, 8), 127)  # 饱和
        self.assertEqual(lab1_library.quantize(-128.5, 8), -128)  # 饱和

        # 测试高比特数
        self.assertEqual(lab1_library.quantize(32767.2, 16), 32767)
        self.assertEqual(lab1_library.quantize(-32768.7, 16), -32768)

        # 测试低比特数
        self.assertEqual(lab1_library.quantize(1.4, 2), 1)
        self.assertEqual(lab1_library.quantize(2.1, 2), 1)  # 最大值为1 (2^2-1=3范围是-2到1)
        self.assertEqual(lab1_library.quantize(-3.2, 2), -2)  # 最小值-2
    def test_end_to_end_system(self):
        n = 50
        h_depth = 3
        n_bits = 10

        # 创建慢速变化信号
        x = np.linspace(0, 2*np.pi, n)
        signal = (np.sin(x) + 1) * 5000

        tx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
        rx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)

        reconstruction_errors = []

        for k in range(1, n-1):
            # 发送端处理
            lab1_ndpcm_library.prepare_params_for_prediction(tx_data, k)
            y_hat = lab1_ndpcm_library.predict(tx_data, k)
            eq = lab1_ndpcm_library.calculate_error(tx_data, k, signal[k])
            y_rec = lab1_ndpcm_library.reconstruct(tx_data, k)

            # 模拟完美信道传输
            rx_data.eq[k] = tx_data.eq[k]

            # 接收端处理
            lab1_ndpcm_library.prepare_params_for_prediction(rx_data, k)
            y_hat_rx = lab1_ndpcm_library.predict(rx_data, k)
            y_rec_rx = lab1_ndpcm_library.reconstruct(rx_data, k)

            # 计算并记录重建误差
            error = abs(rx_data.y_recreated[k] - signal[k]) / signal[k] * 100
            reconstruction_errors.append(error)

        avg_error = np.mean(reconstruction_errors)

        # 验证重建误差在预期范围内
        self.assertLess(avg_error, 5.0)  # 慢速信号应小于5%

        # 验证发送端和接收端重建结果一致
        self.assertTrue(np.allclose(tx_data.y_recreated, rx_data.y_recreated, atol=1e-5))

    def test_fast_changing_signal(self):
        n = 100
        h_depth = 3
        n_bits = 8

        # 创建快速变化信号 (20Hz)
        x = np.linspace(0, 4*np.pi, n)  # 2个周期
        signal = (np.sin(20*x) + 1) * 10000

        tx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
        rx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)

        reconstruction_errors = []

        for k in range(1, n-1):
            # 发送端处理
            lab1_ndpcm_library.prepare_params_for_prediction(tx_data, k)
            y_hat = lab1_ndpcm_library.predict(tx_data, k)
            eq = lab1_ndpcm_library.calculate_error(tx_data, k, signal[k])
            y_rec = lab1_ndpcm_library.reconstruct(tx_data, k)

            # 模拟完美信道传输
            rx_data.eq[k] = tx_data.eq[k]

            # 接收端处理
            lab1_ndpcm_library.prepare_params_for_prediction(rx_data, k)
            y_hat_rx = lab1_ndpcm_library.predict(rx_data, k)
            y_rec_rx = lab1_ndpcm_library.reconstruct(rx_data, k)

            # 计算并记录重建误差
            error = abs(rx_data.y_recreated[k] - signal[k]) / signal[k] * 100
            reconstruction_errors.append(error)

        avg_error = np.mean(reconstruction_errors)

        # 验证重建误差超过20%
        self.assertGreater(avg_error, 20.0)

        # 验证高频区域误差更大
        peak_errors = reconstruction_errors[n//4:3*n//4]  # 中间部分（变化最快）
        self.assertGreater(np.mean(peak_errors), avg_error)

    def test_compression_ratio(self):
        n = 1000
        h_depth = 3

        # 测试不同比特数下的压缩比
        for n_bits in [8, 12, 16]:
            # 初始化系统
            tx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)

            # 计算压缩比
            original_bits = 16 * n  # 假设原始16位ADC
            compressed_bits = n_bits * n
            compression_ratio = original_bits / compressed_bits

            # 验证压缩比计算正确
            self.assertAlmostEqual(compression_ratio, 16 / n_bits, delta=0.01)

            # 验证比特率计算
            sample_rate = 1000  # 1kSamples/sec
            expected_bitrate = n_bits * sample_rate
            self.assertEqual(tx_data.n_bits * sample_rate, expected_bitrate)
if __name__ == '__main__':
    unittest.main()
