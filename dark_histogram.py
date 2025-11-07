import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

FRAME_RATE = 30
WINDOW_LENGTH_MIN = 20.0  # 20分钟窗口
AGE_MAX_WKS = 1000
AGE_MIN_WKS = 0


def calculate_dark_percentage_20min(mice_info_dir, input_dir, max_normal=None, max_blind=None):
    """
    计算每只Normal和Blind小鼠在20分钟窗口内的dark percentage，并绘制直方图
    使用与原始window_feature_extractor完全相同的小鼠选择逻辑
    """

    # 实际明室尺寸 (cm)
    LIGHT_REAL_WIDTH_CM = 40.64
    LIGHT_REAL_HEIGHT_CM = 30.48

    window_size = int(WINDOW_LENGTH_MIN * 60 * FRAME_RATE)  # 20分钟的帧数

    # 读取小鼠信息 - 与原始代码完全一致
    df_info = pd.read_excel(mice_info_dir)
    df_info["Number"] = df_info["Number"].astype(int)
    info_lookup = {int(row["Number"]): row for _, row in df_info.iterrows()}

    # 设置相同的随机种子确保一致性
    np.random.seed(0)

    # === 完全复用原始代码的小鼠筛选逻辑 ===
    # 首先收集所有符合条件的小鼠文件，按类型分组
    normal_files = []
    blind_files = []

    for fname in os.listdir(input_dir):
        if not fname.endswith(".txt"):
            continue

        try:
            mouse_number = int(fname.split("_")[0])
        except:
            print(f"Skip file {fname}: number cannot be parsed")
            continue

        if mouse_number not in info_lookup:
            print(f"Skip file {fname}: number {mouse_number} not found in Excel")
            continue

        info = info_lookup[mouse_number]
        type_str = str(info["Type"]).strip()
        blind_str = str(info["Blind/Normal"]).strip()
        age_str = str(info["Age"]).strip().replace(" ", "")
        gender_str = str(info["Sex"]).strip()

        match_age = re.match(r"(\d+)", age_str)
        if not match_age:
            print(f"[Skip] {fname}: cannot parse age {age_str}")
            continue
        age_weeks = int(match_age.group(1))

        if age_weeks < AGE_MIN_WKS or age_weeks > AGE_MAX_WKS:
            print(f"[Skip processing] {fname}: age {age_weeks} weeks")
            continue

        # 按Blind/Normal类型分组收集文件
        if blind_str.lower() == "normal":
            normal_files.append((fname, mouse_number, info))
        elif blind_str.lower() == "blind":
            blind_files.append((fname, mouse_number, info))

    # 随机选择指定数量的小鼠 - 与原始代码完全一致
    if max_normal is not None and len(normal_files) > max_normal:
        selected_normal = np.random.choice(len(normal_files), size=max_normal, replace=False)
        selected_normal_files = [normal_files[i] for i in selected_normal]
        print(f"Randomly selected {max_normal} from {len(normal_files)} normal mice")
    else:
        selected_normal_files = normal_files
        print(f"Selected all {len(normal_files)} normal mice")

    if max_blind is not None and len(blind_files) > max_blind:
        selected_blind = np.random.choice(len(blind_files), size=max_blind, replace=False)
        selected_blind_files = [blind_files[i] for i in selected_blind]
        print(f"Randomly selected {max_blind} from {len(blind_files)} blind mice")
    else:
        selected_blind_files = blind_files
        print(f"Selected all {len(blind_files)} blind mice")

    selected_files = selected_normal_files + selected_blind_files

    # 统计和打印选中的小鼠类型 - 与原始代码完全一致
    wt_ages = []
    double_crushed_rho_ko_ages = []
    young_rho_ko_ages = []
    old_rho_ko_ages = []

    for fname, mouse_number, info in selected_files:
        type_str = str(info["Type"]).strip()
        blind_str = str(info["Blind/Normal"]).strip()
        note_str = str(info.get("Note", "")).strip() if "Note" in info else ""
        age_str = str(info["Age"]).strip().replace(" ", "")

        # 提取年龄
        match_age = re.match(r"(\d+)", age_str)
        age_weeks = int(match_age.group(1))

        if type_str.lower() == "wt":
            wt_ages.append(age_weeks)
        elif type_str.lower() == "rho ko":
            if "crash" in note_str.lower():
                double_crushed_rho_ko_ages.append(age_weeks)
            elif blind_str.lower() == "normal":  # young RKO (normal vision)
                young_rho_ko_ages.append(age_weeks)
            elif blind_str.lower() == "blind":  # old RKO (blind but not crushed)
                old_rho_ko_ages.append(age_weeks)

    print(f"\n=== Selected mouse type counts and age ranges ===")

    if wt_ages:
        wt_min, wt_max = min(wt_ages), max(wt_ages)
        print(f"WT: {len(wt_ages)} mice, age range: {wt_min}-{wt_max} weeks")
    else:
        print(f"WT: 0 mice")

    if double_crushed_rho_ko_ages:
        dc_min, dc_max = min(double_crushed_rho_ko_ages), max(double_crushed_rho_ko_ages)
        print(f"Double crushed RKO: {len(double_crushed_rho_ko_ages)} mice, age range: {dc_min}-{dc_max} weeks")
    else:
        print(f"Double crushed RKO: 0 mice")

    if young_rho_ko_ages:
        young_min, young_max = min(young_rho_ko_ages), max(young_rho_ko_ages)
        print(f"Young RKO (normal vision): {len(young_rho_ko_ages)} mice, age range: {young_min}-{young_max} weeks")
    else:
        print(f"Young RKO (normal vision): 0 mice")

    if old_rho_ko_ages:
        old_min, old_max = min(old_rho_ko_ages), max(old_rho_ko_ages)
        print(f"Old RKO (blind, not crushed): {len(old_rho_ko_ages)} mice, age range: {old_min}-{old_max} weeks")
    else:
        print(f"Old RKO (blind, not crushed): 0 mice")

    print(f"Total: {len(selected_files)} mice")
    print("=" * 50)

    # 存储结果
    normal_dark_percentages = []
    blind_dark_percentages = []

    # 存储详细信息
    normal_mice_info = []
    blind_mice_info = []

    print("Starting to process selected mouse data...")

    # 处理选中的文件
    for fname, mouse_number, info in selected_files:
        type_str = str(info["Type"]).strip()
        blind_str = str(info["Blind/Normal"]).strip()
        age_str = str(info["Age"]).strip().replace(" ", "")
        gender_str = str(info["Sex"]).strip()

        match_age = re.match(r"(\d+)", age_str)
        age_weeks = int(match_age.group(1))
        age_weeks += np.random.uniform(-0.5, 0.5)  # 与原始代码一致的年龄微调

        # 读取轨迹数据
        input_path = os.path.join(input_dir, fname)
        with open(input_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        # 解析明暗箱坐标
        match = re.search(
            r"Light:X=(\d+), Y=(\d+), W=(\d+), H=(\d+);\s*Dark:X=(\d+), Y=(\d+), W=(\d+), H=(\d+)",
            lines[0])
        if not match:
            print(f"[Skip] {fname}: first line missing light/dark chamber information")
            continue
        lx, ly, lw, lh = map(int, match.groups()[0:4])
        dx, dy, dw, dh = map(int, match.groups()[4:8])

        # 解析坐标数据
        coords = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    x = int(parts[1].strip())
                    y = int(parts[2].strip())
                    coords.append((x, y))
                except:
                    continue

        if len(coords) < window_size:
            print(f"Skip file {fname}: insufficient frames")
            continue

        # 定义判断区域的函数 - 与原始代码一致
        def get_side(x, y):
            if lx <= x <= lx + lw and ly <= y <= ly + lh:
                return "light"
            elif dx <= x <= dx + dw and dy <= y <= dy + dh:
                return "dark"
            else:
                return "unknown"

        # 使用前20分钟的数据
        window_coords = coords[:window_size]
        valid = [(x, y, get_side(x, y)) for (x, y) in window_coords if (x, y) != (-1, -1)]

        if not valid:
            print(f"[Skip] {fname} window: no valid trajectory")
            continue

        # 计算暗室停留时间百分比 - 与原始代码算法一致
        side_seq = [s for (_, _, s) in valid if s in ("light", "dark")]
        if len(side_seq) == 0:
            continue

        num_dark = side_seq.count("dark")
        dark_percentage = (num_dark / len(side_seq)) * 100

        # 根据类型分组存储
        mouse_info_dict = {
            'number': mouse_number,
            'type': type_str,
            'age_weeks': age_weeks,
            'gender': gender_str,
            'dark_percentage': dark_percentage,
            'filename': fname
        }

        if blind_str.lower() == "normal":
            normal_dark_percentages.append(dark_percentage)
            normal_mice_info.append(mouse_info_dict)
            print(f"Normal mouse {mouse_number}: {dark_percentage:.1f}% dark time")
        elif blind_str.lower() == "blind":
            blind_dark_percentages.append(dark_percentage)
            blind_mice_info.append(mouse_info_dict)
            print(f"Blind mouse {mouse_number}: {dark_percentage:.1f}% dark time")

    print("=" * 50)
    print(f"Processing completed!")
    print(f"Normal mice: {len(normal_dark_percentages)} mice")
    print(f"Blind mice: {len(blind_dark_percentages)} mice")

    if len(normal_dark_percentages) == 0 and len(blind_dark_percentages) == 0:
        print("No data found meeting criteria, cannot generate histogram")
        return

    # 图片参数设置
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    # 绘制直方图
    plt.figure(figsize=(12, 4))

    # 定义区间 (0-10%, 10-20%, ..., 90-100%)
    bins = np.arange(0, 101, 10)  # [0, 10, 20, ..., 100]

    # 计算每组数据的最大频次，用于统一y轴范围
    max_count_normal = 0
    max_count_blind = 0

    if normal_dark_percentages:
        hist_normal, _ = np.histogram(normal_dark_percentages, bins=bins)
        max_count_normal = max(hist_normal)

    if blind_dark_percentages:
        hist_blind, _ = np.histogram(blind_dark_percentages, bins=bins)
        max_count_blind = max(hist_blind)

    # 取两组中的最大值
    max_count = max(max_count_normal, max_count_blind)
    y_max = max_count + 1 if max_count > 0 else 10

    # 子图1: Normal mice
    plt.subplot(1, 2, 1)
    if len(normal_dark_percentages) > 0:
        counts_normal, _, _ = plt.hist(normal_dark_percentages, bins=bins, alpha=1.0,
                                       color='#66c2a5', edgecolor='black', linewidth=0.8)
        plt.title(
            f'Sighted mice dark side percentage distribution\n({len(normal_dark_percentages)} mice, window length = 20 min)',
            fontweight='bold')
        plt.xlabel('Dark side percentage (%)')
        plt.ylabel('Number of mice')
        plt.xticks(bins, [f'{int(b)}' for b in bins])
        plt.ylim(0, y_max)  # 统一y轴范围

        # 去掉上边框和右边框
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 在每个柱子上标注数量
        for i, count in enumerate(counts_normal):
            if count > 0:
                plt.text(bins[i] + 5, count + 0.1, f'{int(count)}',
                         ha='center', va='bottom')

        # 计算统计信息但不在图上显示
        mean_normal = np.mean(normal_dark_percentages)
        std_normal = np.std(normal_dark_percentages)

        print(f"\nNormal mice statistics:")
        print(f"  Mean: {mean_normal:.2f}%")
        print(f"  Standard deviation: {std_normal:.2f}%")
        print(f"  Range: {min(normal_dark_percentages):.1f}% - {max(normal_dark_percentages):.1f}%")
    else:
        plt.text(0.5, 0.5, 'No normal mice data', transform=plt.gca().transAxes,
                 ha='center', va='center')
        plt.title('Sighted mice dark side percentage distribution\n(No data)', fontweight='bold')

    # 子图2: Blind mice
    plt.subplot(1, 2, 2)
    if len(blind_dark_percentages) > 0:
        counts_blind, _, _ = plt.hist(blind_dark_percentages, bins=bins, alpha=1.0,
                                      color='#fc8d62', edgecolor='black', linewidth=0.8)
        plt.title(
            f'Blind mice dark side percentage distribution\n({len(blind_dark_percentages)} mice, window length = 20 min)',
            fontweight='bold')
        plt.xlabel('Dark side percentage (%)')
        plt.ylabel('Number of mice')
        plt.xticks(bins, [f'{int(b)}' for b in bins])
        plt.ylim(0, y_max)  # 统一y轴范围

        # 去掉上边框和右边框
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 在每个柱子上标注数量
        for i, count in enumerate(counts_blind):
            if count > 0:
                plt.text(bins[i] + 5, count + 0.1, f'{int(count)}',
                         ha='center', va='bottom')

        # 计算统计信息但不在图上显示
        mean_blind = np.mean(blind_dark_percentages)
        std_blind = np.std(blind_dark_percentages)

        print(f"\nBlind mice statistics:")
        print(f"  Mean: {mean_blind:.2f}%")
        print(f"  Standard deviation: {std_blind:.2f}%")
        print(f"  Range: {min(blind_dark_percentages):.1f}% - {max(blind_dark_percentages):.1f}%")
    else:
        plt.text(0.5, 0.5, 'No blind mice data', transform=plt.gca().transAxes,
                 ha='center', va='center')
        plt.title('Blind mice dark side percentage distribution\n(No data)', fontweight='bold')

    plt.tight_layout()

    # 保存图片为300x300 tif格式到当前目录
    plt.savefig('dark_side_percentage_histogram.tif', format='tif', dpi=300,
                bbox_inches='tight', facecolor='white')

    plt.show()

    # 打印详细的区间统计
    print("\n" + "=" * 50)
    print("Detailed interval statistics:")

    if len(normal_dark_percentages) > 0:
        print(f"\nNormal mice (n={len(normal_dark_percentages)}):")
        hist_normal, _ = np.histogram(normal_dark_percentages, bins=bins)
        for i in range(len(hist_normal)):
            if hist_normal[i] > 0:
                print(f"  {bins[i]:.0f}-{bins[i + 1]:.0f}%: {hist_normal[i]} mice")

    if len(blind_dark_percentages) > 0:
        print(f"\nBlind mice (n={len(blind_dark_percentages)}):")
        hist_blind, _ = np.histogram(blind_dark_percentages, bins=bins)
        for i in range(len(hist_blind)):
            if hist_blind[i] > 0:
                print(f"  {bins[i]:.0f}-{bins[i + 1]:.0f}%: {hist_blind[i]} mice")

    # 返回数据供进一步分析
    return {
        'normal_percentages': normal_dark_percentages,
        'blind_percentages': blind_dark_percentages,
        'normal_info': normal_mice_info,
        'blind_info': blind_mice_info
    }

# 使用示例:
# mice_info_dir = "path/to/mice_info.xlsx"  # Replace with your Excel file path
# input_dir = "path/to/input/directory"     # Replace with your input data directory
#
# # Run analysis - can set same max_normal and max_blind parameters as original code
# results = calculate_dark_percentage_20min(mice_info_dir, input_dir, max_normal=None, max_blind=None)