"""Split a txt data info file into train/val sets - Debug Version."""

import argparse
import numpy as np
import pdb  # 导入调试器

parser = argparse.ArgumentParser(description='Split dataset info file')
parser.add_argument('--info_file', required=True, type=str)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--cat_loc', type=int, default=1)
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()

print(f"[DEBUG] 参数解析完成: info_file={args.info_file}, val_ratio={args.val_ratio}, cat_loc={args.cat_loc}")

# 设置断点
if args.debug:
    pdb.set_trace()

print("[DEBUG] 开始读取文件...")
with open(args.info_file, 'r') as f:
    all_path = [line.strip() for line in f.readlines()]
print(f"[DEBUG] 读取了 {len(all_path)} 条路径")
print(f"[DEBUG] 前3条路径示例: {all_path[:3]}")

if args.debug:
    pdb.set_trace()

print("[DEBUG] 开始提取类别...")
all_cat = np.unique([line.split('/')[args.cat_loc] for line in all_path])
print(f'[DEBUG] 检测到的类别: {all_cat}\n[DEBUG] 类别数量: {len(all_cat)}')

if args.debug:
    pdb.set_trace()

print("[DEBUG] 构建类别到路径的映射...")
cat2path = {
    cat: [path for path in all_path if cat == path.split('/')[args.cat_loc]]
    for cat in all_cat
}
print(f"[DEBUG] 每个类别的路径数量:")
for cat, paths in cat2path.items():
    print(f"  {cat}: {len(paths)} 条路径")

train_paths, val_paths = [], []
print("[DEBUG] 开始分割数据集...")
for cat, paths in cat2path.items():
    print(f"[DEBUG] 处理类别: {cat}, 路径数: {len(paths)}")
    
    if args.debug:
        pdb.set_trace()
    
    np.random.shuffle(paths)
    n_val = len(paths) * args.val_ratio
    print(f"[DEBUG]   计算验证集数量: {len(paths)} * {args.val_ratio} = {n_val}")
    
    if n_val <= 1:
        n_val = 1
    else:
        n_val = int(n_val)
    
    print(f"[DEBUG]   最终验证集数量: {n_val}, 训练集数量: {len(paths) - n_val}")
    train_paths.extend(paths[n_val:])
    val_paths.extend(paths[:n_val])

print(f'[DEBUG] 分割完成: 总共 {len(all_path)} 条数据')
print(f'[DEBUG] 训练集: {len(train_paths)} 条')
print(f'[DEBUG] 验证集: {len(val_paths)} 条')

# save to {}.train.txt and {}.val.txt
train_file = args.info_file.replace('.txt', '.train.txt')
val_file = args.info_file.replace('.txt', '.val.txt')

print(f"[DEBUG] 保存训练集到: {train_file}")
with open(train_file, 'w') as f:
    f.write('\n'.join(train_paths))

print(f"[DEBUG] 保存验证集到: {val_file}")
with open(val_file, 'w') as f:
    f.write('\n'.join(val_paths))

print("[DEBUG] 完成！")


