import matplotlib.pyplot as plt

def plot_curve(para_list, vacc_list, savename):
    vaccs = [100*vacc for vacc in vacc_list]
    plt.figure()
    plt.plot(para_list, vaccs, marker='o')
    plt.xlabel("Parameter Number (M)")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Parameter Number vs. Average Accuracy")
    plt.xticks(range(0, 100, 10))
    plt.yticks(range(0, 101, 10))
    plt.grid(True)
    # 在坐标点旁边标记数值
    for x, y in zip(para_list, vaccs):
        # plt.text(x, y, f'{y:.1f}', fontsize=12, ha='left', va='top')
        offset_x, offset_y = -2.5, 3
        plt.text(x + offset_x, y + offset_y, f'{y:.2f}', fontsize=9, ha='center', va='center')

    plt.savefig(savename, dpi=300, bbox_inches='tight')  # PNG 格式，300 DPI
    plt.close()

