"""
图像压缩编码模块

实现四种经典的无损压缩编码算法：
1. 哈夫曼编码 (Huffman Coding)
2. 算术编码 (Arithmetic Coding)
3. LZW编码 (Lempel-Ziv-Welch)
4. 位平面编码 (Bit-Plane Coding with Run-Length Encoding)
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import heapq


# ==================== 哈夫曼编码 ====================

class HuffmanNode:
    """哈夫曼树节点"""
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanCoding:
    """哈夫曼编码类"""

    def __init__(self):
        self.codes = {}
        self.reverse_codes = {}

    def build_frequency_table(self, data: np.ndarray) -> Dict[int, int]:
        """构建频率表"""
        flat_data = data.flatten()
        return dict(Counter(flat_data))

    def build_huffman_tree(self, freq_table: Dict[int, int]) -> HuffmanNode:
        """构建哈夫曼树"""
        heap = [HuffmanNode(symbol=symbol, freq=freq)
                for symbol, freq in freq_table.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            parent = HuffmanNode(
                symbol=None,
                freq=left.freq + right.freq,
                left=left,
                right=right
            )
            heapq.heappush(heap, parent)

        return heap[0]

    def generate_codes(self, node: HuffmanNode, code: str = ""):
        """生成哈夫曼编码表"""
        if node is None:
            return

        if node.symbol is not None:
            self.codes[node.symbol] = code if code else "0"
            self.reverse_codes[code if code else "0"] = node.symbol
            return

        self.generate_codes(node.left, code + "0")
        self.generate_codes(node.right, code + "1")

    def encode(self, data: np.ndarray) -> Tuple[str, Dict]:
        """编码"""
        # 构建频率表
        freq_table = self.build_frequency_table(data)

        # 构建哈夫曼树
        root = self.build_huffman_tree(freq_table)

        # 生成编码表
        self.generate_codes(root)

        # 编码数据
        flat_data = data.flatten()
        encoded = ''.join(self.codes[symbol] for symbol in flat_data)

        # 返回编码结果和元数据
        metadata = {
            'codes': self.codes,
            'shape': data.shape,
            'dtype': str(data.dtype)
        }

        return encoded, metadata

    def decode(self, encoded: str, metadata: Dict) -> np.ndarray:
        """解码"""
        self.codes = metadata['codes']
        self.reverse_codes = {v: k for k, v in self.codes.items()}

        # 解码
        decoded = []
        current_code = ""

        for bit in encoded:
            current_code += bit
            if current_code in self.reverse_codes:
                decoded.append(self.reverse_codes[current_code])
                current_code = ""

        # 恢复形状
        result = np.array(decoded, dtype=metadata['dtype'])
        result = result.reshape(metadata['shape'])

        return result

    def calculate_compression_ratio(self, original: np.ndarray, encoded: str) -> float:
        """计算压缩比"""
        original_bits = original.size * 8  # 假设每个像素8位
        compressed_bits = len(encoded)
        return original_bits / compressed_bits


# ==================== 算术编码 ====================

class ArithmeticCoding:
    """算术编码类"""

    def __init__(self):
        self.precision = 32  # 精度位数

    def build_probability_table(self, data: np.ndarray) -> Dict[int, float]:
        """构建概率表"""
        freq_table = dict(Counter(data.flatten()))
        total = sum(freq_table.values())
        prob_table = {symbol: freq / total for symbol, freq in freq_table.items()}
        return prob_table

    def build_cumulative_table(self, prob_table: Dict[int, float]) -> Dict[int, Tuple[float, float]]:
        """构建累积概率表"""
        cumulative = {}
        cum_prob = 0.0

        for symbol in sorted(prob_table.keys()):
            prob = prob_table[symbol]
            cumulative[symbol] = (cum_prob, cum_prob + prob)
            cum_prob += prob

        return cumulative

    def encode(self, data: np.ndarray) -> Tuple[float, Dict]:
        """编码"""
        # 构建概率表和累积概率表
        prob_table = self.build_probability_table(data)
        cumulative = self.build_cumulative_table(prob_table)

        # 算术编码
        low = 0.0
        high = 1.0

        for symbol in data.flatten():
            range_size = high - low
            symbol_low, symbol_high = cumulative[symbol]

            high = low + range_size * symbol_high
            low = low + range_size * symbol_low

        # 选择区间中点作为编码值
        code = (low + high) / 2

        metadata = {
            'prob_table': prob_table,
            'cumulative': cumulative,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'length': data.size
        }

        return code, metadata

    def decode(self, code: float, metadata: Dict) -> np.ndarray:
        """解码"""
        cumulative = metadata['cumulative']
        length = metadata['length']

        # 反向查找符号
        reverse_cumulative = {v: k for k, v in cumulative.items()}
        sorted_ranges = sorted(reverse_cumulative.keys())

        decoded = []

        for _ in range(length):
            # 找到当前code对应的符号
            for i, (low, high) in enumerate(sorted_ranges):
                if low <= code < high:
                    symbol = reverse_cumulative[sorted_ranges[i]]
                    decoded.append(symbol)

                    # 更新code
                    range_size = high - low
                    code = (code - low) / range_size
                    break

        result = np.array(decoded, dtype=metadata['dtype'])
        result = result.reshape(metadata['shape'])

        return result


# ==================== LZW编码 ====================

class LZWCoding:
    """LZW编码类"""

    def __init__(self):
        self.dict_size = 256  # 初始字典大小（0-255为单字节）

    def encode(self, data: np.ndarray) -> Tuple[List[int], Dict]:
        """编码"""
        # 初始化字典
        dictionary = {bytes([i]): i for i in range(self.dict_size)}

        flat_data = data.flatten().tobytes()

        result = []
        current = b""

        for byte in flat_data:
            byte_val = bytes([byte])
            combined = current + byte_val

            if combined in dictionary:
                current = combined
            else:
                result.append(dictionary[current])
                dictionary[combined] = len(dictionary)
                current = byte_val

        if current:
            result.append(dictionary[current])

        metadata = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'dict_size': len(dictionary)
        }

        return result, metadata

    def decode(self, encoded: List[int], metadata: Dict) -> np.ndarray:
        """解码"""
        # 初始化字典
        dictionary = {i: bytes([i]) for i in range(self.dict_size)}

        result = []

        # 第一个编码
        code = encoded[0]
        current = dictionary[code]
        result.append(current)

        for code in encoded[1:]:
            if code in dictionary:
                entry = dictionary[code]
            else:
                # 特殊情况：code不在字典中
                entry = current + bytes([current[0]])

            result.append(entry)

            # 添加新条目到字典
            dictionary[len(dictionary)] = current + bytes([entry[0]])
            current = entry

        # 合并结果
        decoded_bytes = b''.join(result)
        decoded_array = np.frombuffer(decoded_bytes, dtype=metadata['dtype'])
        decoded_array = decoded_array.reshape(metadata['shape'])

        return decoded_array

    def calculate_compression_ratio(self, original: np.ndarray, encoded: List[int]) -> float:
        """计算压缩比"""
        original_size = original.size
        compressed_size = len(encoded)
        return original_size / compressed_size


# ==================== 位平面编码（游程编码）====================

class BitPlaneCoding:
    """位平面编码类（结合游程编码）"""

    def __init__(self):
        self.num_bits = 8  # 8位灰度图像

    def decompose_bit_planes(self, image: np.ndarray) -> List[np.ndarray]:
        """分解位平面"""
        bit_planes = []

        for bit in range(self.num_bits):
            # 提取第bit位
            plane = (image >> bit) & 1
            bit_planes.append(plane.astype(np.uint8))

        return bit_planes

    def run_length_encode(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """游程编码"""
        flat_data = data.flatten()

        if len(flat_data) == 0:
            return []

        runs = []
        current_value = flat_data[0]
        count = 1

        for value in flat_data[1:]:
            if value == current_value:
                count += 1
            else:
                runs.append((int(current_value), count))
                current_value = value
                count = 1

        runs.append((int(current_value), count))

        return runs

    def run_length_decode(self, runs: List[Tuple[int, int]], length: int) -> np.ndarray:
        """游程解码"""
        result = []

        for value, count in runs:
            result.extend([value] * count)

        return np.array(result[:length], dtype=np.uint8)

    def encode(self, image: np.ndarray) -> Tuple[List[List[Tuple[int, int]]], Dict]:
        """编码"""
        if len(image.shape) != 2:
            raise ValueError("位平面编码仅支持灰度图像")

        # 分解位平面
        bit_planes = self.decompose_bit_planes(image)

        # 对每个位平面进行游程编码
        encoded_planes = []
        for plane in bit_planes:
            runs = self.run_length_encode(plane)
            encoded_planes.append(runs)

        metadata = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'num_bits': self.num_bits
        }

        return encoded_planes, metadata

    def decode(self, encoded_planes: List[List[Tuple[int, int]]], metadata: Dict) -> np.ndarray:
        """解码"""
        shape = metadata['shape']
        length = shape[0] * shape[1]

        # 解码每个位平面
        bit_planes = []
        for runs in encoded_planes:
            plane = self.run_length_decode(runs, length)
            plane = plane.reshape(shape)
            bit_planes.append(plane)

        # 重组图像
        result = np.zeros(shape, dtype=np.uint8)
        for bit, plane in enumerate(bit_planes):
            result += (plane << bit)

        return result

    def calculate_compression_ratio(self, original: np.ndarray,
                                   encoded_planes: List[List[Tuple[int, int]]]) -> float:
        """计算压缩比"""
        original_bits = original.size * 8

        # 计算编码后的大小（每个游程需要存储值和长度）
        compressed_bits = 0
        for plane_runs in encoded_planes:
            for value, count in plane_runs:
                compressed_bits += 1 + 32  # 1位值 + 32位计数（假设）

        return original_bits / compressed_bits


# ==================== 压缩算法比较 ====================

class CompressionComparison:
    """压缩算法比较类"""

    def __init__(self):
        self.huffman = HuffmanCoding()
        self.arithmetic = ArithmeticCoding()
        self.lzw = LZWCoding()
        self.bitplane = BitPlaneCoding()

    def compare_all(self, image: np.ndarray) -> Dict:
        """比较所有压缩算法"""
        results = {}

        # 哈夫曼编码
        try:
            encoded_huff, meta_huff = self.huffman.encode(image)
            ratio_huff = self.huffman.calculate_compression_ratio(image, encoded_huff)
            results['Huffman'] = {
                'ratio': ratio_huff,
                'encoded_size': len(encoded_huff),
                'status': 'success'
            }
        except Exception as e:
            results['Huffman'] = {'status': 'error', 'error': str(e)}

        # 算术编码
        try:
            encoded_arith, meta_arith = self.arithmetic.encode(image)
            results['Arithmetic'] = {
                'code': encoded_arith,
                'status': 'success'
            }
        except Exception as e:
            results['Arithmetic'] = {'status': 'error', 'error': str(e)}

        # LZW编码
        try:
            encoded_lzw, meta_lzw = self.lzw.encode(image)
            ratio_lzw = self.lzw.calculate_compression_ratio(image, encoded_lzw)
            results['LZW'] = {
                'ratio': ratio_lzw,
                'encoded_size': len(encoded_lzw),
                'status': 'success'
            }
        except Exception as e:
            results['LZW'] = {'status': 'error', 'error': str(e)}

        # 位平面编码
        if len(image.shape) == 2:  # 仅灰度图像
            try:
                encoded_bp, meta_bp = self.bitplane.encode(image)
                ratio_bp = self.bitplane.calculate_compression_ratio(image, encoded_bp)
                results['BitPlane'] = {
                    'ratio': ratio_bp,
                    'num_planes': len(encoded_bp),
                    'status': 'success'
                }
            except Exception as e:
                results['BitPlane'] = {'status': 'error', 'error': str(e)}

        return results


def main():
    """演示程序"""
    print("图像压缩编码演示程序")
    print("=" * 50)

    # 创建测试图像
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    cv2.imwrite('compression_test_image.png', test_image)

    print("\n测试图像: 64x64 灰度图像")
    print(f"原始大小: {test_image.size} 字节")

    # 1. 哈夫曼编码
    print("\n1. 哈夫曼编码")
    huffman = HuffmanCoding()
    encoded_huff, meta_huff = huffman.encode(test_image)
    decoded_huff = huffman.decode(encoded_huff, meta_huff)
    ratio_huff = huffman.calculate_compression_ratio(test_image, encoded_huff)

    print(f"   压缩后大小: {len(encoded_huff)} 位")
    print(f"   压缩比: {ratio_huff:.2f}:1")
    print(f"   重建成功: {np.array_equal(test_image, decoded_huff)}")

    # 2. LZW编码
    print("\n2. LZW编码")
    lzw = LZWCoding()
    encoded_lzw, meta_lzw = lzw.encode(test_image)
    decoded_lzw = lzw.decode(encoded_lzw, meta_lzw)
    ratio_lzw = lzw.calculate_compression_ratio(test_image, encoded_lzw)

    print(f"   压缩后大小: {len(encoded_lzw)} 个码字")
    print(f"   压缩比: {ratio_lzw:.2f}:1")
    print(f"   重建成功: {np.array_equal(test_image, decoded_lzw)}")

    # 3. 位平面编码
    print("\n3. 位平面编码（游程编码）")
    bitplane = BitPlaneCoding()
    encoded_bp, meta_bp = bitplane.encode(test_image)
    decoded_bp = bitplane.decode(encoded_bp, meta_bp)
    ratio_bp = bitplane.calculate_compression_ratio(test_image, encoded_bp)

    print(f"   位平面数: {len(encoded_bp)}")
    print(f"   压缩比: {ratio_bp:.2f}:1")
    print(f"   重建成功: {np.array_equal(test_image, decoded_bp)}")

    # 4. 比较所有算法
    print("\n4. 算法比较")
    comparison = CompressionComparison()
    results = comparison.compare_all(test_image)

    for method, result in results.items():
        if result['status'] == 'success':
            if 'ratio' in result:
                print(f"   {method:12s}: 压缩比 {result['ratio']:.2f}:1")
        else:
            print(f"   {method:12s}: {result.get('error', 'Unknown error')}")

    print("\n✅ 演示完成！")


if __name__ == '__main__':
    main()
