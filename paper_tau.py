import random
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

# ==========================绘图设置======================================
# 设置线条的颜色
color_list = ['#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF',
              '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF',
              '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00',
              '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C',
              '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9',
              '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00',
              '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B',
              '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF']
# 线条标志
line_mark = ['.', ',', 'o', 'v', '^', '<', '>',
             '1', '2', '3', '4', 's', 'p', '*',
             'h', 'H', '+', 'x', 'D', 'd', '|', '_']
# 线条类型
line_style = ['-', '--',
              '-.', ':']
# ==========================================================================


class SRSO:
    # 初始化数值
    def __init__(self, name):
        self.name = name
        # self.list_1 = []
        self.num_relay_min = 3
        self.num_relay_max = 20
        self.num_relay_step = 1
        self.num_relay_list = np.arange(self.num_relay_min, self.num_relay_max + 1, self.num_relay_step)

        self.num_remote_min = 5
        self.num_remote_max = 25
        self.num_remote_step = 1
        self.num_remote_list = np.arange(self.num_remote_min, self.num_remote_max, self.num_remote_step)

        self.C_l, self.P_l = 600, 50  # 50mW
        self.w = 1  # 1Mhz
        self.N_0 = 10 ** (-13)  # 噪声
        self.V, self.D = 10000, 200  # 任务和任务完成一位所需要的比特值 1G bits
        self.x_01, self.x_1i = 0, 0
        self.sigma_1 = 10 ** (-13)
        self.alpha_1 = 3

        self.C_k_min, self.C_k_max = 4000, 6000  # 计算能力
        self.P_k_min, self.P_k_max = 150, 200  # 计算功率
        self.h_k_min, self.h_k_max = 0.00025, 0.00075  # 损失函数
        self.d_k_min, self.d_k_max = 1500, 3500
        self.t_limit = (self.V * (self.P_k_min + self.P_k_max) / 2) / ((self.C_k_min + self.C_k_max) / 2)  # 服务器截止时间设定

        self.P_t, self.P_t2 = 50, 50  # 传输功率为确定值 mW
        self.P_d = 17  # 信道检测功率
        self.tau = 0.02
        self.tau_2 = 0.04
        self.tau_3 = 0.06

        self.C_k_list = SRSO.ra_sample(self, self.C_k_min, self.C_k_max, self.num_relay_max)
        self.P_k_list = SRSO.ra_sample(self, self.P_k_min, self.P_k_max, self.num_relay_max)
        self.h_k_list = SRSO.ra_sample_float(self, self.h_k_min, self.h_k_max, self.num_relay_max)
        self.d_k_list = SRSO.ra_sample(self, self.d_k_min, self.d_k_max, self.num_relay_max)

        self.C_d_k_list = SRSO.ra_sample(self, self.C_k_min, self.C_k_max, self.num_remote_max)
        self.P_d_k_list = SRSO.ra_sample(self, self.P_k_min, self.P_k_max, self.num_remote_max)
        self.h_d_k_list = SRSO.ra_sample_float(self, self.h_k_min, self.h_k_max, self.num_remote_max)
        self.d_d_k_list = SRSO.ra_sample(self, self.d_k_min, self.d_k_max, self.num_remote_max)

        self.i_ratio_list = []  # 表示relay可用的概率
        self.i_ratio_list_d = []  # 表示remote可用的概率

        self.enable_ratio = 0.9  # 每个服务器可用概率的阈值
        self.result_sequence = []
        self.result_matrix = []
        self.result_matrix_rs = []
        self.result_matrix_gsp = []
    # 根据给定的范围[min_value, max_value]，生成num_server个随机 整数
    def ra_sample(self, min_value, max_value, num_server):
        begin = min_value
        end = max_value
        need_count = num_server
        sample_list = random.sample(range(begin, end), need_count)
        return sample_list
    # 根据给定的范围[min_value, max_value]，生成num_server个随机 小数
    def ra_sample_float(self, min_value, max_value, num_server):
        sample_list = []
        for i in range(num_server):
            rad_num = random.uniform(min_value, max_value)
            sample_list.append(round(rad_num, 5))
        return sample_list
    # 计算上传到Relay端所消耗的能量（返回值为：传输能耗 and 计算能耗）    C_r de E: 1000~125    C_l de E: 1000
    def relay_energy_1(self, x_01, x_1i, h_k, P_k, C_k, d_k):
        v_1i = self.w * np.log2(1 + self.P_t * (np.abs(h_k)) ** 2 / ((1 + d_k ** self.alpha_1) * self.sigma_1 ** 2))
        # print(v_1i)
        E_relay_tran = (1 - x_01) * (self.P_t * self.D / v_1i)
        E_relay_cal = x_1i * (self.V * P_k / C_k)
        return E_relay_tran, E_relay_cal
    # 计算上传到Remote端所消耗的能量（返回值为：传输能耗 and 计算能耗）    1000~125
    def remote_energy_1(self, x_01, x_1i, h_d_k, P_d_k, C_d_k, d_d_k):
        v_ij = self.w * np.log2(1 + self.P_t2 * (np.abs(h_d_k)) ** 2 / ((1 + d_d_k ** self.alpha_1) * self.sigma_1 ** 2))
        # print(v_ij)
        E_remote_tran = (1 - x_01 - x_1i) * (self.P_t2 * self.D / v_ij)
        E_remote_cal = (1 - x_01 - x_1i) * (self.V * P_d_k / C_d_k)
        return E_remote_tran, E_remote_cal
    # 把检测可用概率的放入列表中(小于0.8为可用)
    def available_prob_list(self):
        for x in range(self.num_relay_max + 1):
            i_ratio = random.random()
            self.i_ratio_list.append(i_ratio)
        # print('i_ratio_list:', self.i_ratio_list)
        return self.i_ratio_list
    # 把可用的放入remote列表中
    def available_pro_list_d(self):
        for x in range(self.num_remote_max + 1):
            i_ratio = random.random()
            self.i_ratio_list_d.append(i_ratio)
        # print('i_ratio_list_d:', self.i_ratio_list_d)
        return self.i_ratio_list_d

    def SRSO_optimal_test(self):
        for average_time in range(1, 1500, 1):
            self.C_k_list = SRSO.ra_sample(self, self.C_k_min, self.C_k_max, self.num_relay_max)
            self.P_k_list = SRSO.ra_sample(self, self.P_k_min, self.P_k_max, self.num_relay_max)
            self.h_k_list = SRSO.ra_sample_float(self, self.h_k_min, self.h_k_max, self.num_relay_max)
            self.d_k_list = SRSO.ra_sample(self, self.d_k_min, self.d_k_max, self.num_relay_max)

            self.C_d_k_list = SRSO.ra_sample(self, self.C_k_min, self.C_k_max, self.num_remote_max)
            self.P_d_k_list = SRSO.ra_sample(self, self.P_k_min, self.P_k_max, self.num_remote_max)
            self.h_d_k_list = SRSO.ra_sample_float(self, self.h_k_min, self.h_k_max, self.num_remote_max)
            self.d_d_k_list = SRSO.ra_sample(self, self.d_k_min, self.d_k_max, self.num_remote_max)

            self.i_ratio_list = []
            self.i_ratio_list_d = []
            self.result_sequence = []
            self.x_01, self.x_1i = 0, 0

            SRSO.available_prob_list(self)  # 服务器是否可用de概率
            SRSO.available_pro_list_d(self)  # 服务器是否可用de概率
            for num_relay in self.num_relay_list:
                E_detect = self.tau * self.P_d
                E_1_tran, E_1_cal, E_2_tran, E_2_cal = 0, 0, 0, 0
                can_use_list = []
                E_r = []
                E_d = []
                flag_x1 = 0
                for x in range(0, num_relay):
                    if self.i_ratio_list[x] <= self.enable_ratio:
                        can_use_list.append(1)  # 将可用的标记为1，计算可用的能耗，进行比较
                        flag_x1 = flag_x1 + 1
                    else:
                        can_use_list.append(0)
                        flag_x1 = flag_x1
                while 1:
                    if flag_x1 == 0:     # print('no_server_can_use')
                        E_1_tran = 0
                        E_1_cal = 0
                        self.x_01 = 1
                        break  # 若都不可用，则跳出循环，只能在本地计算
                    for i in range(0, num_relay):
                        if i == 0:
                            # if 可用 -> Step2判断
                            # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                            if self.i_ratio_list[i] <= self.enable_ratio:
                                # print("服务器可用")
                                # self.x_01, self.x_1i = 0, 1
                                self.x_1i = (self.t_limit * self.C_k_list[i]) / (self.V * self.P_k_list[i])
                                if self.x_1i > 1:  # 如果完全卸载到relay，没有超过时间限制，则就完全卸载到该处。
                                    self.x_1i = 1
                                E_1_tran, E_1_cal = SRSO.relay_energy_1(self, self.x_01, self.x_1i, self.h_k_list[i], self.P_k_list[i], self.C_k_list[i], self.d_k_list[i])
                                E_final_1 = E_detect + E_1_tran + E_1_cal
                                E_r.append(E_final_1)
                                #print('1', self.x_01, self.x_1i)
                            else:
                                # print("服务器不可用")
                                self.x_01, self.x_1i = 1, 0
                                E_local = self.x_01 * self.P_l * self.V / self.C_l
                                E_final_1 = E_detect + E_local
                                E_r.append(E_final_1)
                                #print('2', self.x_01, self.x_1i)
                        elif i > 0 and i <= (num_relay - 2):
                            # if 可用 -> Step2判断
                            # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                            if self.i_ratio_list[i] <= self.enable_ratio:
                                # print("服务器可用")
                                # self.x_01, self.x_1i = 0, 1
                                self.x_1i = (self.t_limit * self.C_k_list[i]) / (self.V * self.P_k_list[i])
                                if self.x_1i > 1:  # 如果完全卸载到relay，没有超过时间限制，则就完全卸载到该处。
                                    self.x_1i = 1
                                E_1_tran, E_1_cal = SRSO.relay_energy_1(self, self.x_01, self.x_1i, self.h_k_list[i],
                                                                        self.P_k_list[i], self.C_k_list[i],
                                                                        self.d_k_list[i])
                                sum_value = 0
                                # for i_11 in range(0, i):
                                #     sum_value = sum_value + E_r[i_11]
                                E_final_1 = E_detect + min(E_1_tran + E_1_cal, E_r[i - 1])
                                E_r.append(E_final_1)
                                if (E_1_tran + E_1_cal) <= sum_value:
                                    for xx in range(0, i):
                                        E_r[xx] = 0
                                #print('3', self.x_01, self.x_1i)
                            else:
                                # print("服务器不可用")
                                E_r.append(E_detect + E_r[i - 1])
                                #print('4', self.x_01, self.x_1i)
                        elif i == num_relay - 1:
                            # if 可用 -> Step2判断
                            # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                            if self.i_ratio_list[i] <= self.enable_ratio:
                                # print("服务器可用")
                                # self.x_01, self.x_1i = 0, 1
                                self.x_1i = (self.t_limit * self.C_k_list[i]) / (self.V * self.P_k_list[i])
                                if self.x_1i > 1:  # 如果完全卸载到relay，没有超过时间限制，则就完全卸载到该处。
                                    self.x_1i = 1
                                E_1_tran, E_1_cal = SRSO.relay_energy_1(self, self.x_01, self.x_1i, self.h_k_list[i],
                                                                        self.P_k_list[i], self.C_k_list[i],
                                                                        self.d_k_list[i])
                                sum_value = 0
                                # for i_11 in range(0, i):
                                #     sum_value = sum_value + E_r[i_11]
                                E_final_1 = E_detect + min(E_1_tran + E_1_cal, E_r[i - 1])
                                E_r.append(E_final_1)
                                if (E_1_tran + E_1_cal) <= sum_value:
                                    for xx in range(0, i):
                                        E_r[xx] = 0
                                #print('5', self.x_01, self.x_1i)
                            else:
                                # print("服务器不可用")
                                E_r.append(E_detect + E_r[i - 1])
                    break
                if self.x_01 == 1:
                    E_local = self.x_01 * self.P_l * self.V / self.C_l
                    E_final = E_local + E_detect * num_relay
                    # print(E_r)
                    # E_final = E_r[len(E_r)-1]
                elif (self.x_01 + self.x_1i) != 1:

                    len_remote = random.choice(self.num_remote_list)
                    flag_x2 = 0
                    for x in range(0, len_remote):
                        if self.i_ratio_list_d[x] <= self.enable_ratio:
                            flag_x2 = flag_x2 + 1
                        else:
                            flag_x2 = flag_x2
                    while 1:
                        if flag_x2 == 0:
                            self.x_01 = 1 - self.x_1i
                            break
                        for j in range(0, len_remote):
                            if j == 0:
                                # if 可用 -> Step2判断
                                # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                                if self.i_ratio_list_d[j] <= self.enable_ratio:
                                    # print("服务器可用")
                                    E_2_tran, E_2_cal = SRSO.remote_energy_1(self, self.x_01, self.x_1i,
                                                                            self.h_d_k_list[j], self.P_d_k_list[j],
                                                                            self.C_d_k_list[j], self.d_d_k_list[j])
                                    E_final_2 = E_detect + E_2_tran + E_2_cal
                                    E_d.append(E_final_2)
                                    # print('1', self.x_01, self.x_1i)
                                else:
                                    # print("服务器不可用")
                                    E_local_2 = (1 - self.x_1i) * self.V * self.P_l / self.C_l
                                    E_final_2 = E_detect + E_r[len(E_r)-1] + E_local_2
                                    E_d.append(E_final_2)
                                    # print('2', self.x_01, self.x_1i)
                            elif j > 0 and j <= (len_remote - 2):
                                # if 可用 -> Step2判断
                                # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                                if self.i_ratio_list_d[j] <= self.enable_ratio:
                                    # print("服务器可用")
                                    E_2_tran, E_2_cal = SRSO.remote_energy_1(self, self.x_01, self.x_1i,
                                                                            self.h_d_k_list[j],
                                                                            self.P_d_k_list[j], self.C_d_k_list[j],
                                                                            self.d_d_k_list[j])
                                    # sum_value = 0
                                    # for i_11 in range(0, i):
                                    #     sum_value = sum_value + E_r[i_11]
                                    E_final_2 = E_detect + min(E_2_tran + E_2_cal, E_d[j - 1])
                                    E_d.append(E_final_2)
                                    # if (E_1_tran + E_1_cal) <= sum_value:
                                    #     for xx in range(0, i):
                                    #         E_r[xx] = 0
                                    # print('3', self.x_01, self.x_1i)
                                else:
                                    # print("服务器不可用")
                                    E_d.append(E_detect + E_d[j - 1])
                                    # print('4', self.x_01, self.x_1i)
                            elif j == len_remote - 1:
                                # if 可用 -> Step2判断
                                # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                                if self.i_ratio_list_d[j] <= self.enable_ratio:
                                    # print("服务器可用")
                                    E_2_tran, E_2_cal = SRSO.remote_energy_1(self, self.x_01, self.x_1i,
                                                                            self.h_d_k_list[j],
                                                                            self.P_d_k_list[j], self.C_d_k_list[j],
                                                                            self.d_d_k_list[j])
                                    # sum_value = 0
                                    # for i_11 in range(0, i):
                                    #     sum_value = sum_value + E_r[i_11]
                                    E_final_2 = E_detect + min(E_2_tran + E_1_cal, E_d[j - 1])
                                    E_d.append(E_final_2)
                                    # if (E_2_tran + E_2_cal) <= sum_value:
                                    #     for xx in range(0, j):
                                    #         E_r[xx] = 0
                                    # print('5', self.x_01, self.x_1i)
                                else:
                                    # print("服务器不可用")
                                    E_d.append(E_detect + E_d[j - 1])
                        break

                    if self.x_01 != 0:  # 说明有剩下的任务还是只能在本地算
                        E_local = self.x_01 * self.P_l * self.V / self.C_l
                        E_final = E_r[len(E_r)-1] + E_local
                    else:
                        E_final = E_r[len(E_r)-1] + E_d[len(E_d)-1]
                else:
                    E_final = E_r[len(E_r)-1]
                self.result_sequence.append(E_final)
                # 为了计算optimal循环后的均方值
            self.result_matrix.append(self.result_sequence)
        return self.result_matrix

    def SRSO_RS_test(self):
        for average_time in range(1, 1500, 1):
            self.C_k_list = SRSO.ra_sample(self, self.C_k_min, self.C_k_max, self.num_relay_max)
            self.P_k_list = SRSO.ra_sample(self, self.P_k_min, self.P_k_max, self.num_relay_max)
            self.h_k_list = SRSO.ra_sample_float(self, self.h_k_min, self.h_k_max, self.num_relay_max)
            self.d_k_list = SRSO.ra_sample(self, self.d_k_min, self.d_k_max, self.num_relay_max)

            self.C_d_k_list = SRSO.ra_sample(self, self.C_k_min, self.C_k_max, self.num_remote_max)
            self.P_d_k_list = SRSO.ra_sample(self, self.P_k_min, self.P_k_max, self.num_remote_max)
            self.h_d_k_list = SRSO.ra_sample_float(self, self.h_k_min, self.h_k_max, self.num_remote_max)
            self.d_d_k_list = SRSO.ra_sample(self, self.d_k_min, self.d_k_max, self.num_remote_max)

            self.i_ratio_list = []
            self.i_ratio_list_d = []
            self.result_sequence = []
            self.x_01, self.x_1i = 0, 0

            SRSO.available_prob_list(self)  # 服务器是否可用de概率
            SRSO.available_pro_list_d(self)  # 服务器是否可用de概率
            for num_relay in self.num_relay_list:
                E_detect = self.tau_2 * self.P_d
                E_1_tran, E_1_cal, E_2_tran, E_2_cal = 0, 0, 0, 0
                can_use_list = []
                E_r = []
                E_d = []
                flag_x1 = 0
                for x in range(0, num_relay):
                    if self.i_ratio_list[x] <= self.enable_ratio:
                        can_use_list.append(1)  # 将可用的标记为1，计算可用的能耗，进行比较
                        flag_x1 = flag_x1 + 1
                    else:
                        can_use_list.append(0)
                        flag_x1 = flag_x1
                while 1:
                    if flag_x1 == 0:     # print('no_server_can_use')
                        E_1_tran = 0
                        E_1_cal = 0
                        self.x_01 = 1
                        break  # 若都不可用，则跳出循环，只能在本地计算
                    for i in range(0, num_relay):
                        if i == 0:
                            # if 可用 -> Step2判断
                            # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                            if self.i_ratio_list[i] <= self.enable_ratio:
                                # print("服务器可用")
                                # self.x_01, self.x_1i = 0, 1
                                self.x_1i = (self.t_limit * self.C_k_list[i]) / (self.V * self.P_k_list[i])
                                if self.x_1i > 1:  # 如果完全卸载到relay，没有超过时间限制，则就完全卸载到该处。
                                    self.x_1i = 1
                                E_1_tran, E_1_cal = SRSO.relay_energy_1(self, self.x_01, self.x_1i, self.h_k_list[i], self.P_k_list[i], self.C_k_list[i], self.d_k_list[i])
                                E_final_1 = E_detect + E_1_tran + E_1_cal
                                E_r.append(E_final_1)
                                #print('1', self.x_01, self.x_1i)
                            else:
                                # print("服务器不可用")
                                self.x_01, self.x_1i = 1, 0
                                E_local = self.x_01 * self.P_l * self.V / self.C_l
                                E_final_1 = E_detect + E_local
                                E_r.append(E_final_1)
                                #print('2', self.x_01, self.x_1i)
                        elif i > 0 and i <= (num_relay - 2):
                            # if 可用 -> Step2判断
                            # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                            if self.i_ratio_list[i] <= self.enable_ratio:
                                # print("服务器可用")
                                # self.x_01, self.x_1i = 0, 1
                                self.x_1i = (self.t_limit * self.C_k_list[i]) / (self.V * self.P_k_list[i])
                                if self.x_1i > 1:  # 如果完全卸载到relay，没有超过时间限制，则就完全卸载到该处。
                                    self.x_1i = 1
                                E_1_tran, E_1_cal = SRSO.relay_energy_1(self, self.x_01, self.x_1i, self.h_k_list[i],
                                                                        self.P_k_list[i], self.C_k_list[i],
                                                                        self.d_k_list[i])
                                sum_value = 0
                                # for i_11 in range(0, i):
                                #     sum_value = sum_value + E_r[i_11]
                                E_final_1 = E_detect + min(E_1_tran + E_1_cal, E_r[i - 1])
                                E_r.append(E_final_1)
                                if (E_1_tran + E_1_cal) <= sum_value:
                                    for xx in range(0, i):
                                        E_r[xx] = 0
                                #print('3', self.x_01, self.x_1i)
                            else:
                                # print("服务器不可用")
                                E_r.append(E_detect + E_r[i - 1])
                                #print('4', self.x_01, self.x_1i)
                        elif i == num_relay - 1:
                            # if 可用 -> Step2判断
                            # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                            if self.i_ratio_list[i] <= self.enable_ratio:
                                # print("服务器可用")
                                # self.x_01, self.x_1i = 0, 1
                                self.x_1i = (self.t_limit * self.C_k_list[i]) / (self.V * self.P_k_list[i])
                                if self.x_1i > 1:  # 如果完全卸载到relay，没有超过时间限制，则就完全卸载到该处。
                                    self.x_1i = 1
                                E_1_tran, E_1_cal = SRSO.relay_energy_1(self, self.x_01, self.x_1i, self.h_k_list[i],
                                                                        self.P_k_list[i], self.C_k_list[i],
                                                                        self.d_k_list[i])
                                sum_value = 0
                                # for i_11 in range(0, i):
                                #     sum_value = sum_value + E_r[i_11]
                                E_final_1 = E_detect + min(E_1_tran + E_1_cal, E_r[i - 1])
                                E_r.append(E_final_1)
                                if (E_1_tran + E_1_cal) <= sum_value:
                                    for xx in range(0, i):
                                        E_r[xx] = 0
                                #print('5', self.x_01, self.x_1i)
                            else:
                                # print("服务器不可用")
                                E_r.append(E_detect + E_r[i - 1])
                    break
                if self.x_01 == 1:
                    E_local = self.x_01 * self.P_l * self.V / self.C_l
                    E_final = E_local + E_detect * num_relay
                    # print(E_r)
                    # E_final = E_r[len(E_r)-1]
                elif (self.x_01 + self.x_1i) != 1:

                    len_remote = random.choice(self.num_remote_list)
                    flag_x2 = 0
                    for x in range(0, len_remote):
                        if self.i_ratio_list_d[x] <= self.enable_ratio:
                            flag_x2 = flag_x2 + 1
                        else:
                            flag_x2 = flag_x2
                    while 1:
                        if flag_x2 == 0:
                            self.x_01 = 1 - self.x_1i
                            break
                        for j in range(0, len_remote):
                            if j == 0:
                                # if 可用 -> Step2判断
                                # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                                if self.i_ratio_list_d[j] <= self.enable_ratio:
                                    # print("服务器可用")
                                    E_2_tran, E_2_cal = SRSO.remote_energy_1(self, self.x_01, self.x_1i,
                                                                            self.h_d_k_list[j], self.P_d_k_list[j],
                                                                            self.C_d_k_list[j], self.d_d_k_list[j])
                                    E_final_2 = E_detect + E_2_tran + E_2_cal
                                    E_d.append(E_final_2)
                                    # print('1', self.x_01, self.x_1i)
                                else:
                                    # print("服务器不可用")
                                    E_local_2 = (1 - self.x_1i) * self.V * self.P_l / self.C_l
                                    E_final_2 = E_detect + E_r[len(E_r)-1] + E_local_2
                                    E_d.append(E_final_2)
                                    # print('2', self.x_01, self.x_1i)
                            elif j > 0 and j <= (len_remote - 2):
                                # if 可用 -> Step2判断
                                # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                                if self.i_ratio_list_d[j] <= self.enable_ratio:
                                    # print("服务器可用")
                                    E_2_tran, E_2_cal = SRSO.remote_energy_1(self, self.x_01, self.x_1i,
                                                                            self.h_d_k_list[j],
                                                                            self.P_d_k_list[j], self.C_d_k_list[j],
                                                                            self.d_d_k_list[j])
                                    # sum_value = 0
                                    # for i_11 in range(0, i):
                                    #     sum_value = sum_value + E_r[i_11]
                                    E_final_2 = E_detect + min(E_2_tran + E_2_cal, E_d[j - 1])
                                    E_d.append(E_final_2)
                                    # if (E_1_tran + E_1_cal) <= sum_value:
                                    #     for xx in range(0, i):
                                    #         E_r[xx] = 0
                                    # print('3', self.x_01, self.x_1i)
                                else:
                                    # print("服务器不可用")
                                    E_d.append(E_detect + E_d[j - 1])
                                    # print('4', self.x_01, self.x_1i)
                            elif j == len_remote - 1:
                                # if 可用 -> Step2判断
                                # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                                if self.i_ratio_list_d[j] <= self.enable_ratio:
                                    # print("服务器可用")
                                    E_2_tran, E_2_cal = SRSO.remote_energy_1(self, self.x_01, self.x_1i,
                                                                            self.h_d_k_list[j],
                                                                            self.P_d_k_list[j], self.C_d_k_list[j],
                                                                            self.d_d_k_list[j])
                                    # sum_value = 0
                                    # for i_11 in range(0, i):
                                    #     sum_value = sum_value + E_r[i_11]
                                    E_final_2 = E_detect + min(E_2_tran + E_1_cal, E_d[j - 1])
                                    E_d.append(E_final_2)
                                    # if (E_2_tran + E_2_cal) <= sum_value:
                                    #     for xx in range(0, j):
                                    #         E_r[xx] = 0
                                    # print('5', self.x_01, self.x_1i)
                                else:
                                    # print("服务器不可用")
                                    E_d.append(E_detect + E_d[j - 1])
                        break

                    if self.x_01 != 0:  # 说明有剩下的任务还是只能在本地算
                        E_local = self.x_01 * self.P_l * self.V / self.C_l
                        E_final = E_r[len(E_r)-1] + E_local
                    else:
                        E_final = E_r[len(E_r)-1] + E_d[len(E_d)-1]
                else:
                    E_final = E_r[len(E_r)-1]
                self.result_sequence.append(E_final)
                # 为了计算optimal循环后的均方值
            self.result_matrix.append(self.result_sequence)
        return self.result_matrix

    def SRSO_GSP_test(self):
        for average_time in range(1, 1500, 1):
            self.C_k_list = SRSO.ra_sample(self, self.C_k_min, self.C_k_max, self.num_relay_max)
            self.P_k_list = SRSO.ra_sample(self, self.P_k_min, self.P_k_max, self.num_relay_max)
            self.h_k_list = SRSO.ra_sample_float(self, self.h_k_min, self.h_k_max, self.num_relay_max)
            self.d_k_list = SRSO.ra_sample(self, self.d_k_min, self.d_k_max, self.num_relay_max)

            self.C_d_k_list = SRSO.ra_sample(self, self.C_k_min, self.C_k_max, self.num_remote_max)
            self.P_d_k_list = SRSO.ra_sample(self, self.P_k_min, self.P_k_max, self.num_remote_max)
            self.h_d_k_list = SRSO.ra_sample_float(self, self.h_k_min, self.h_k_max, self.num_remote_max)
            self.d_d_k_list = SRSO.ra_sample(self, self.d_k_min, self.d_k_max, self.num_remote_max)

            self.i_ratio_list = []
            self.i_ratio_list_d = []
            self.result_sequence = []
            self.x_01, self.x_1i = 0, 0

            SRSO.available_prob_list(self)  # 服务器是否可用de概率
            SRSO.available_pro_list_d(self)  # 服务器是否可用de概率
            for num_relay in self.num_relay_list:
                E_detect = self.tau_3 * self.P_d
                E_1_tran, E_1_cal, E_2_tran, E_2_cal = 0, 0, 0, 0
                can_use_list = []
                E_r = []
                E_d = []
                flag_x1 = 0
                for x in range(0, num_relay):
                    if self.i_ratio_list[x] <= self.enable_ratio:
                        can_use_list.append(1)  # 将可用的标记为1，计算可用的能耗，进行比较
                        flag_x1 = flag_x1 + 1
                    else:
                        can_use_list.append(0)
                        flag_x1 = flag_x1
                while 1:
                    if flag_x1 == 0:     # print('no_server_can_use')
                        E_1_tran = 0
                        E_1_cal = 0
                        self.x_01 = 1
                        break  # 若都不可用，则跳出循环，只能在本地计算
                    for i in range(0, num_relay):
                        if i == 0:
                            # if 可用 -> Step2判断
                            # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                            if self.i_ratio_list[i] <= self.enable_ratio:
                                # print("服务器可用")
                                # self.x_01, self.x_1i = 0, 1
                                self.x_1i = (self.t_limit * self.C_k_list[i]) / (self.V * self.P_k_list[i])
                                if self.x_1i > 1:  # 如果完全卸载到relay，没有超过时间限制，则就完全卸载到该处。
                                    self.x_1i = 1
                                E_1_tran, E_1_cal = SRSO.relay_energy_1(self, self.x_01, self.x_1i, self.h_k_list[i], self.P_k_list[i], self.C_k_list[i], self.d_k_list[i])
                                E_final_1 = E_detect + E_1_tran + E_1_cal
                                E_r.append(E_final_1)
                                #print('1', self.x_01, self.x_1i)
                            else:
                                # print("服务器不可用")
                                self.x_01, self.x_1i = 1, 0
                                E_local = self.x_01 * self.P_l * self.V / self.C_l
                                E_final_1 = E_detect + E_local
                                E_r.append(E_final_1)
                                #print('2', self.x_01, self.x_1i)
                        elif i > 0 and i <= (num_relay - 2):
                            # if 可用 -> Step2判断
                            # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                            if self.i_ratio_list[i] <= self.enable_ratio:
                                # print("服务器可用")
                                # self.x_01, self.x_1i = 0, 1
                                self.x_1i = (self.t_limit * self.C_k_list[i]) / (self.V * self.P_k_list[i])
                                if self.x_1i > 1:  # 如果完全卸载到relay，没有超过时间限制，则就完全卸载到该处。
                                    self.x_1i = 1
                                E_1_tran, E_1_cal = SRSO.relay_energy_1(self, self.x_01, self.x_1i, self.h_k_list[i],
                                                                        self.P_k_list[i], self.C_k_list[i],
                                                                        self.d_k_list[i])
                                sum_value = 0
                                # for i_11 in range(0, i):
                                #     sum_value = sum_value + E_r[i_11]
                                E_final_1 = E_detect + min(E_1_tran + E_1_cal, E_r[i - 1])
                                E_r.append(E_final_1)
                                if (E_1_tran + E_1_cal) <= sum_value:
                                    for xx in range(0, i):
                                        E_r[xx] = 0
                                #print('3', self.x_01, self.x_1i)
                            else:
                                # print("服务器不可用")
                                E_r.append(E_detect + E_r[i - 1])
                                #print('4', self.x_01, self.x_1i)
                        elif i == num_relay - 1:
                            # if 可用 -> Step2判断
                            # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                            if self.i_ratio_list[i] <= self.enable_ratio:
                                # print("服务器可用")
                                # self.x_01, self.x_1i = 0, 1
                                self.x_1i = (self.t_limit * self.C_k_list[i]) / (self.V * self.P_k_list[i])
                                if self.x_1i > 1:  # 如果完全卸载到relay，没有超过时间限制，则就完全卸载到该处。
                                    self.x_1i = 1
                                E_1_tran, E_1_cal = SRSO.relay_energy_1(self, self.x_01, self.x_1i, self.h_k_list[i],
                                                                        self.P_k_list[i], self.C_k_list[i],
                                                                        self.d_k_list[i])
                                sum_value = 0
                                # for i_11 in range(0, i):
                                #     sum_value = sum_value + E_r[i_11]
                                E_final_1 = E_detect + min(E_1_tran + E_1_cal, E_r[i - 1])
                                E_r.append(E_final_1)
                                if (E_1_tran + E_1_cal) <= sum_value:
                                    for xx in range(0, i):
                                        E_r[xx] = 0
                                #print('5', self.x_01, self.x_1i)
                            else:
                                # print("服务器不可用")
                                E_r.append(E_detect + E_r[i - 1])
                    break
                if self.x_01 == 1:
                    E_local = self.x_01 * self.P_l * self.V / self.C_l
                    E_final = E_local + E_detect * num_relay
                    # print(E_r)
                    # E_final = E_r[len(E_r)-1]
                elif (self.x_01 + self.x_1i) != 1:

                    len_remote = random.choice(self.num_remote_list)
                    flag_x2 = 0
                    for x in range(0, len_remote):
                        if self.i_ratio_list_d[x] <= self.enable_ratio:
                            flag_x2 = flag_x2 + 1
                        else:
                            flag_x2 = flag_x2
                    while 1:
                        if flag_x2 == 0:
                            self.x_01 = 1 - self.x_1i
                            break
                        for j in range(0, len_remote):
                            if j == 0:
                                # if 可用 -> Step2判断
                                # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                                if self.i_ratio_list_d[j] <= self.enable_ratio:
                                    # print("服务器可用")
                                    E_2_tran, E_2_cal = SRSO.remote_energy_1(self, self.x_01, self.x_1i,
                                                                            self.h_d_k_list[j], self.P_d_k_list[j],
                                                                            self.C_d_k_list[j], self.d_d_k_list[j])
                                    E_final_2 = E_detect + E_2_tran + E_2_cal
                                    E_d.append(E_final_2)
                                    # print('1', self.x_01, self.x_1i)
                                else:
                                    # print("服务器不可用")
                                    E_local_2 = (1 - self.x_1i) * self.V * self.P_l / self.C_l
                                    E_final_2 = E_detect + E_r[len(E_r)-1] + E_local_2
                                    E_d.append(E_final_2)
                                    # print('2', self.x_01, self.x_1i)
                            elif j > 0 and j <= (len_remote - 2):
                                # if 可用 -> Step2判断
                                # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                                if self.i_ratio_list_d[j] <= self.enable_ratio:
                                    # print("服务器可用")
                                    E_2_tran, E_2_cal = SRSO.remote_energy_1(self, self.x_01, self.x_1i,
                                                                            self.h_d_k_list[j],
                                                                            self.P_d_k_list[j], self.C_d_k_list[j],
                                                                            self.d_d_k_list[j])
                                    # sum_value = 0
                                    # for i_11 in range(0, i):
                                    #     sum_value = sum_value + E_r[i_11]
                                    E_final_2 = E_detect + min(E_2_tran + E_2_cal, E_d[j - 1])
                                    E_d.append(E_final_2)
                                    # if (E_1_tran + E_1_cal) <= sum_value:
                                    #     for xx in range(0, i):
                                    #         E_r[xx] = 0
                                    # print('3', self.x_01, self.x_1i)
                                else:
                                    # print("服务器不可用")
                                    E_d.append(E_detect + E_d[j - 1])
                                    # print('4', self.x_01, self.x_1i)
                            elif j == len_remote - 1:
                                # if 可用 -> Step2判断
                                # else 不可用 -> 取 i-1阶段的最优并加上检测de E
                                if self.i_ratio_list_d[j] <= self.enable_ratio:
                                    # print("服务器可用")
                                    E_2_tran, E_2_cal = SRSO.remote_energy_1(self, self.x_01, self.x_1i,
                                                                            self.h_d_k_list[j],
                                                                            self.P_d_k_list[j], self.C_d_k_list[j],
                                                                            self.d_d_k_list[j])
                                    # sum_value = 0
                                    # for i_11 in range(0, i):
                                    #     sum_value = sum_value + E_r[i_11]
                                    E_final_2 = E_detect + min(E_2_tran + E_1_cal, E_d[j - 1])
                                    E_d.append(E_final_2)
                                    # if (E_2_tran + E_2_cal) <= sum_value:
                                    #     for xx in range(0, j):
                                    #         E_r[xx] = 0
                                    # print('5', self.x_01, self.x_1i)
                                else:
                                    # print("服务器不可用")
                                    E_d.append(E_detect + E_d[j - 1])
                        break

                    if self.x_01 != 0:  # 说明有剩下的任务还是只能在本地算
                        E_local = self.x_01 * self.P_l * self.V / self.C_l
                        E_final = E_r[len(E_r)-1] + E_local
                    else:
                        E_final = E_r[len(E_r)-1] + E_d[len(E_d)-1]
                else:
                    E_final = E_r[len(E_r)-1]
                self.result_sequence.append(E_final)
                # 为了计算optimal循环后的均方值
            self.result_matrix.append(self.result_sequence)
        return self.result_matrix


srso = SRSO('SRSO')
print('SRSO')
rand_sel = srso.SRSO_optimal_test()
result_mean_optimal = list(np.mean(rand_sel, axis=0))
print("result_mean:", result_mean_optimal)
value_stage_mean_optimal = OrderedDict()
for i in range(3, 20, 3):
    value_stage_mean_optimal[str(i)] = result_mean_optimal[i - 3]
d_time_optimal = np.array([int(x) for x in value_stage_mean_optimal.keys()])
e_consu_optimal = value_stage_mean_optimal.values()

srso_rs = SRSO('SRSO_RS')
print('SRSO_RS')
rand_sel_rs = srso.SRSO_RS_test()
result_mean_rs = list(np.mean(rand_sel_rs, axis=0))
print("result_mean:", result_mean_rs)
value_stage_mean_rs = OrderedDict()
for i in range(3, 20, 3):
    value_stage_mean_rs[str(i)] = result_mean_rs[i - 3]
d_time_rs = np.array([int(x) for x in value_stage_mean_rs.keys()])
e_consu_rs = value_stage_mean_rs.values()

srso_gsp = SRSO('SRSO_GSP')
print('SRSO_GSP')
rand_sel_gsp = srso.SRSO_GSP_test()
result_mean_gsp = list(np.mean(rand_sel_gsp, axis=0))
print("result_mean:", result_mean_gsp)
value_stage_mean_gsp = OrderedDict()
for i in range(3, 20, 3):
    value_stage_mean_gsp[str(i)] = result_mean_gsp[i - 3]
d_time_gsp = np.array([int(x) for x in value_stage_mean_gsp.keys()])
e_consu_gsp = value_stage_mean_gsp.values()


plt.figure(figsize=(8, 5))
font2 = {'size': 15,
         }
plt.xlabel('Number of relay servers')
plt.ylabel('Average energy consumption')
plt.xlim([2, 20])
# 设置坐标轴刻度
my_x_ticks = np.arange(3, 20, 3)
plt.xticks(my_x_ticks)
plt.plot(d_time_optimal, e_consu_optimal, color='blue', alpha=0.7,
         marker=line_mark[13], linestyle='-',
         label=r'$\tau$=2ms', markersize=13, linewidth=2, clip_on=False)
plt.plot(d_time_rs, e_consu_rs, color='red', alpha=0.7,
         marker=line_mark[3], linestyle='-',
         label=r'$\tau$=4ms', markersize=8, linewidth=2, clip_on=False)
plt.plot(d_time_gsp, e_consu_gsp, color='green', alpha=0.5,
         marker=line_mark[12], linestyle='-',
         label=r'$\tau$=6ms', markersize=8, linewidth=2, clip_on=False)
plt.xlim(3, 18)
# plt.ylim(630, 880)
plt.legend(fontsize=10)
plt.grid(linestyle='-.')     # 添加网格
# plt.savefig("Figure_result_33.png", dpi=500, bbox_inches='tight')    # 解决图片不清晰，不完整的问题
plt.show()
