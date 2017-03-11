#!/usr/bin/env python
'''
:mod:`sar.viz` is a module containing classes for visualizing sar logs.
'''

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np


class Visualization(object):
    PDF_OUTPUT = 0
    PNG_OUTPUT = 1
    SAR_TYPES = ['cpu', 'mem', 'io', 'net', 'paging', 'dev']
    PLT_XTICK_LABEL_ROTATION = 'horizontal'

    def __init__(self, sar_data, cpu=False, mem=False, io=False,
                 net=False, paging=False, dev=False):
        """Create a sar log visualization.

        Only CPU and memory usage charts are enabled by default.

        Args:
            sar_data (dict): Processed sar log from Parser
            cpu (:obj:`bool`, optional): Enable CPU usage charts
            mem (:obj:`bool`, optional): Enable memory usage charts
            io (:obj:`bool`, optional): Enable disk usage charts
            net (:obj:`bool`, optional): Enable network usage charts
            paging (:obj:`bool`, optional): Enable paging usage charts
            dev (:obj:`bool`, optional): Enable device usage charts
        """

        if not isinstance(sar_data, dict):
            raise Exception('Incompatible sar_data type: {}'.format(
                type(sar_data).__name__))

        self.sar_data = sar_data
        """dict: Processed sar logs"""

        self.enable_cpu = cpu
        self.enable_mem = mem
        self.enable_io  = io
        self.enable_net = net
        self.enable_paging = paging
        self.enable_dev = dev

        self.time_points = []
        """(:obj:`list` of :obj:`str`): time points which system activity was
            recorded"""

        self.x_data = []
        """(:obj:`list` of :obj:`int`): x axis data"""

        self.xticks = []
        """(:obj:`list` of :obj:`int`): x axis ticks"""

        self.xtick_labels = []
        """(:obj:`list` of :obj:`str`) x axis tick labels"""

        self.cpu_usage_usr = []
        self.cpu_usage_sys = []
        self.page_faults_per_sec = []
        self.major_page_faults_per_sec = []
        self.page_ins_per_sec = []
        self.page_outs_per_sec = []
        self.pct_mem_used = []
        self.mem_used_gb = []
        self.mem_cached_gb = []
        self.mem_buffer_gb = []
        self.rcv_per_sec = {}
        self.trans_per_sec = {}
        self.breads_per_sec = []
        self.bwrites_per_sec = []
        self.rd_per_sec = {}
        self.wr_per_sec = {}
        self.fig_height = 0
        self.num_plots = 0

        self._calculate_plot_height()
        self._preprocess_sar_data()

    def _calculate_plot_height(self):
        num_plots = 0

        if self.enable_cpu:
            num_plots += 1

        if self.enable_io:
            num_plots += 1

        if self.enable_mem:
            num_plots += 1

        if self.enable_net:
            num_plots += 1

        if self.enable_paging:
            num_plots += 2

        if self.enable_dev:
            num_plots += 1

        self.num_plots = num_plots
        self.fig_height = num_plots * 4

    def _preprocess_sar_data(self):
        self.host_name = self.sar_data['host']
        for t in Visualization.SAR_TYPES:
            if t in self.sar_data \
            and self.sar_data[t] \
            and 'time_list' in self.sar_data[t] \
            and self.sar_data[t]['time_list']:
                time_points = self.sar_data[t]['time_list']
                self.time_points = time_points
                break

        tp_count = len(time_points)
        xtick_label_stepsize = tp_count / 15
        if xtick_label_stepsize == 0:
            xtick_label_stepsize = 1
        self.x_data = range(tp_count)
        self.xticks = np.arange(0, tp_count, xtick_label_stepsize)
        self.xtick_labels = [self.x_data[i] for i in self.xticks]

        for i in range(len(self.xticks)):
            if i % 2 == 1:
                self.xtick_labels[i] = ""

        if self.enable_cpu:
            self.cpu_usage_sys = [self.sar_data['cpu'][tp]['all']['sys']
                                  for tp in self.time_points]
            self.cpu_usage_usr = [self.sar_data['cpu'][tp]['all']['usr']
                                  for tp in self.time_points]

        if self.enable_mem:
            factor = 1024 * 1024
            self.pct_mem_used = [self.sar_data['mem'][tp]['memusedpercent']
                                 for tp in self.time_points]
            self.mem_used_gb = [(self.sar_data['mem'][tp]['memused'] - (
                                 self.sar_data['mem'][tp]['memcache'] + self.sar_data['mem']
                                 [tp]['membuffer'])) / factor for tp in self.time_points]
            self.mem_cached_gb = [self.sar_data['mem'][tp]['memcache'] / factor
                                  for tp in self.time_points]
            self.mem_buffer_gb = [self.sar_data['mem'][tp]['membuffer'] / factor
                                  for tp in self.time_points]

        if self.enable_paging:
            self.page_faults_per_sec = [self.sar_data['paging'][tp]['fault']
                                        for tp in self.time_points]
            self.major_page_faults_per_sec = [self.sar_data['paging'][tp]['majflt']
                                              for tp in self.time_points]
            self.page_ins_per_sec = [self.sar_data['paging'][tp]['pgpgin']
                                     for tp in self.time_points]
            self.page_outs_per_sec = [self.sar_data['paging'][tp]['pgpgout']
                                      for tp in self.time_points]

        if self.enable_io:
            factor = 1000
            self.breads_per_sec = [self.sar_data['io'][tp]['bread'] / factor for tp in self.time_points]
            self.bwrites_per_sec = [self.sar_data['io'][tp]['bwrite'] / factor for tp in self.time_points]

        if self.enable_net:
            factor = 1024
            net_data = self.sar_data['net']
            dp = {}
            for tp in self.time_points:
                dp = net_data[tp]
                for iface in dp.keys():
                    if iface not in self.rcv_per_sec:
                        self.rcv_per_sec[iface] = [dp[iface]['rxkB'] / factor]
                    else:
                        self.rcv_per_sec[iface].append(dp[iface]['rxkB'] / factor)

                    if iface not in self.trans_per_sec:
                        self.trans_per_sec[iface] = [dp[iface]['txkB'] / factor]
                    else:
                        self.trans_per_sec[iface].append(dp[iface]['txkB'] / factor)

            max_rcv = 0
            max_trans = 0
            for iface in dp.keys():
                max_rcv = max(max_rcv, sum(self.rcv_per_sec[iface]))
                max_trans = max(max_trans, sum(self.trans_per_sec[iface]))

            for iface in dp.keys():
                if (sum(self.rcv_per_sec[iface]) < max_rcv / 100 and sum(self.trans_per_sec[iface]) < max_trans / 100) \
                or (sum(self.rcv_per_sec[iface]) < 1 and sum(self.trans_per_sec[iface]) < 1):
                    del(self.rcv_per_sec[iface])
                    del(self.trans_per_sec[iface])

        if self.enable_dev:
            factor = 1000
            dev_data = self.sar_data['dev']
            for tp in self.time_points:
                dp = dev_data[tp]
                for dev in dp.keys():
                    if dev not in self.rd_per_sec:
                        self.rd_per_sec[dev] = [dp[dev]['rd_sec'] / factor]
                    else:
                        self.rd_per_sec[dev].append(dp[dev]['rd_sec'] / factor)

                    if dev not in self.wr_per_sec:
                        self.wr_per_sec[dev] = [dp[dev]['wr_sec'] / factor]
                    else:
                        self.wr_per_sec[dev].append(dp[dev]['wr_sec'] / factor)

            max_rd = 0
            max_wr = 0
            for dev in dp.keys():
                max_rd = max(max_rd, sum(self.rd_per_sec[dev]))
                max_wr = max(max_wr, sum(self.wr_per_sec[dev]))

            for dev in dp.keys():
                if sum(self.rd_per_sec[dev]) < max_rd / 100 and sum(self.wr_per_sec[dev]) < max_wr / 100:
                    del(self.rd_per_sec[dev])
                    del(self.wr_per_sec[dev])

    def save(self, output_path, output_type=PDF_OUTPUT):
        plt_idx = 1
        fig = plt.figure()
        fig.set_figheight(self.fig_height)

        plt.clf()
        plt.subplots_adjust(wspace=1, hspace=1)

        if self.enable_cpu:
            plt.subplot(self.num_plots, 1, plt_idx)
            plt.xticks(self.xticks, self.xtick_labels,
                       rotation=Visualization.PLT_XTICK_LABEL_ROTATION)
            plt.plot(self.x_data, self.cpu_usage_usr, label='usr')
            plt.plot(self.x_data, self.cpu_usage_sys, label='sys')
            plt.xlabel('time (min)')
            plt.ylabel('% usage')
            plt.title('CPU Usage - {}'.format(self.host_name))
            lg = plt.legend(frameon=False, loc='upper right')
            if lg is not None:
                lg_txts = lg.get_texts()
                plt.setp(lg_txts, fontsize=10)
            plt_idx += 1

        if self.enable_mem:
            plt.subplot(self.num_plots, 1, plt_idx)
            plt.xticks(self.xticks, self.xtick_labels,
                       rotation=Visualization.PLT_XTICK_LABEL_ROTATION)
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            line, = ax2.plot(self.x_data, self.pct_mem_used, label='% mem used',
                     color='dodgerblue')
            ax2.set_ylabel('% mem used')

            ax1.stackplot(self.x_data, self.mem_cached_gb, self.mem_used_gb,
                          colors=['navajowhite', 'sandybrown'])
            ax1.set_ylabel('Mem Usage (GB)')

            ax1.set_xlabel('time (min)')
            plt.title('Memory Usage - {}'.format(self.host_name))

            lg = plt.legend([mpatches.Patch(color='navajowhite'),
                             mpatches.Patch(color='sandybrown'),
                             line],
                            ['Cached Memory', 'Used Memory', '% mem used'],
                            loc='upper right')
            if lg is not None:
                lg.get_frame().set_alpha(0.6)
                lg_txts = lg.get_texts()
                plt.setp(lg_txts, fontsize=10)
            plt_idx += 1

        if self.enable_paging:
            plt.subplot(self.num_plots, 1, plt_idx)
            plt.xticks(self.xticks, self.xtick_labels,
                       rotation=Visualization.PLT_XTICK_LABEL_ROTATION)
            plt.plot(self.x_data, self.page_faults_per_sec, label='faults/s')
            plt.plot(self.x_data, self.major_page_faults_per_sec, label='major faults/s')
            plt.xlabel('time (min)')
            plt.ylabel('faults/s')
            plt.title('Page Faults - {}'.format(self.host_name))
            lg = plt.legend(frameon=False)
            if lg is not None:
                lg_txts = lg.get_texts()
                plt.setp(lg_txts, fontsize=10)
            plt_idx += 1

            plt.subplot(self.num_plots, 1, plt_idx)
            plt.xticks(self.xticks, self.xtick_labels,
                       rotation=Visualization.PLT_XTICK_LABEL_ROTATION)
            plt.plot(self.x_data, self.page_ins_per_sec, label='page ins/s')
            plt.plot(self.x_data, self.page_outs_per_sec, label='page outs/s')
            plt.xlabel('time (min)')
            plt.ylabel('KB/s')
            plt.title('Page Ins and Outs - {}'.format(self.host_name))
            lg = plt.legend(frameon=False, loc='upper right')
            if lg is not None:
                lg_txts = lg.get_texts()
                plt.setp(lg_txts, fontsize=10)
            plt_idx += 1

        if self.enable_net:
            plt.subplot(self.num_plots, 1, plt_idx)
            plt.xticks(self.xticks, self.xtick_labels,
                       rotation=Visualization.PLT_XTICK_LABEL_ROTATION)
            for iface in self.rcv_per_sec.keys():
                plt.plot(self.x_data, self.rcv_per_sec[iface], label='{}-rx'.format(iface))
            for iface in self.trans_per_sec.keys():
                plt.plot(self.x_data, self.trans_per_sec[iface], label='{}-tx'.format(iface))
            plt.xlabel('time (min)')
            plt.ylabel('MB/s')
            plt.title('Network Usage - {}'.format(self.host_name))
            lg = plt.legend(ncol=len(self.rcv_per_sec.keys()), frameon=False, loc='upper right')
            if lg is not None:
                lg.get_frame().set_alpha(0)
                lg_txts = lg.get_texts()
                plt.setp(lg_txts, fontsize=10)
            plt_idx += 1

        if self.enable_io:
            plt.subplot(self.num_plots, 1, plt_idx)
            plt.xticks(self.xticks, self.xtick_labels,
                       rotation=Visualization.PLT_XTICK_LABEL_ROTATION)
            plt.plot(self.x_data, self.breads_per_sec, label='reads')
            plt.plot(self.x_data, self.bwrites_per_sec, label='writes')
            plt.xlabel('time (min)')
            plt.ylabel('kilo-blocks/s')
            plt.title('Disk IO - {}'.format(self.host_name))
            lg = plt.legend(frameon=False, loc = 'upper right')
            if lg is not None:
                lg_txts = lg.get_texts()
                plt.setp(lg_txts, fontsize=10)
            plt_idx += 1

        if self.enable_dev:
            plt.subplot(self.num_plots, 1, plt_idx)
            plt.xticks(self.xticks, self.xtick_labels,
                       rotation=Visualization.PLT_XTICK_LABEL_ROTATION)
            for dev in self.rd_per_sec.keys():
                plt.plot(self.x_data, self.rd_per_sec[dev], label='{}-read'.format(dev))
            for dev in self.wr_per_sec.keys():
                plt.plot(self.x_data, self.wr_per_sec[dev], label='{}-write'.format(dev))
            plt.xlabel('time (min)')
            plt.ylabel('kilo-sections/s')
            plt.title('Device Usage - {}'.format(self.host_name))
            lg = plt.legend(ncol=len(self.rd_per_sec.keys()), frameon=False, loc='upper right')
            if lg is not None:
                lg.get_frame().set_alpha(0)
                lg_txts = lg.get_texts()
                plt.setp(lg_txts, fontsize=10)
            plt_idx += 1

        fig.tight_layout()
        if output_type == Visualization.PDF_OUTPUT:
            pp = PdfPages(output_path)
            pp.savefig()
            pp.close()
        elif output_type == Visualization.PNG_OUTPUT:
            fig.savefig(output_path)
            plt.close(fig)
