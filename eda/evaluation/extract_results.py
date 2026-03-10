#!/usr/bin/env python3
"""
提取 EAGLE 评测结果并生成 CSV 文件
用法: python extract_results.py <results_dir> [output_csv]
"""

import os
import re
import sys
import csv
from pathlib import Path


def extract_accept_length(log_file):
    """从 log 文件中提取 accept length 统计信息"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # 提取统计数据
        avg_match = re.search(r'Average accept length:\s+([\d.]+)', content)
        min_match = re.search(r'Min accept length:\s+([\d.]+)', content)
        max_match = re.search(r'Max accept length:\s+([\d.]+)', content)
        total_match = re.search(r'Total turns evaluated:\s+(\d+)', content)
        
        if not avg_match:
            return None
        
        return {
            'avg': float(avg_match.group(1)),
            'min': float(min_match.group(1)) if min_match else None,
            'max': float(max_match.group(1)) if max_match else None,
            'total_turns': int(total_match.group(1)) if total_match else None
        }
    except Exception as e:
        print(f"Error processing {log_file}: {e}")
        return None


def extract_epoch_from_filename(filename):
    """从文件名中提取 epoch 编号"""
    match = re.search(r'_(\d+)\.log$', filename)
    return int(match.group(1)) if match else None


def main():
    if len(sys.argv) < 2:
        print("用法: python extract_results.py <results_dir> [output_csv]")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"错误: 目录 {results_dir} 不存在")
        sys.exit(1)
    
    # 确定输出 CSV 文件名
    if len(sys.argv) >= 3:
        csv_file = sys.argv[2]
    else:
        csv_file = results_dir / f"{results_dir.name}.csv"
    
    # 收集所有 log 文件
    log_files = sorted(results_dir.glob("*.log"))
    if not log_files:
        print(f"警告: 在 {results_dir} 中没有找到 .log 文件")
        sys.exit(0)
    
    print(f"找到 {len(log_files)} 个 log 文件")
    
    # 提取数据
    results = []
    for log_file in log_files:
        epoch = extract_epoch_from_filename(log_file.name)
        if epoch is None:
            continue
        
        data = extract_accept_length(log_file)
        if data:
            results.append({
                'epoch': epoch,
                **data
            })
            print(f"Epoch {epoch}: avg={data['avg']:.4f}, min={data['min']:.4f}, max={data['max']:.4f}")
    
    if not results:
        print("错误: 没有提取到任何数据")
        sys.exit(1)
    
    # 按 epoch 排序
    results.sort(key=lambda x: x['epoch'])
    
    # 写入 CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'avg', 'min', 'max', 'total_turns'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ CSV 文件已生成: {csv_file}")
    print(f"✓ 总共 {len(results)} 个 epoch 的数据")


if __name__ == '__main__':
    main()
