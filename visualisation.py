import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import *
from utils import *

class NFCSECICIDS2018_Visualizer():
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.benign = df[df["Label"] == 0]
        self.attacks = df[df["Label"] == 1]

    def visualize(self):
        #self.scatter_out_bytes_out_pkts()
        #self.scatter_in_bytes_out_bytes()
        #self.density_out_bytes_label()
        #self.density_out_bytes_attack_type()
        #self.density_in_bytes_label()
        #self.count_label_tcp_flag()
        #self.count_attack_protocol()
        #self.density_flow_duration()
        #self.density_flow_duration_per_attack()
        #self.density_in_bytes_per_attack()
        #self.scatter_in_bytes_in_pkts()
        self.scatter_flow_duration_out_pkts()
        #self.scatter_flow_duration_out_bytes()
        #self.scatter_in_pkts_out_pkts()

    def scatter_in_bytes_out_bytes(self):
        label_0 = self.df[self.df["Label"] == 0]
        label_1 = self.df[self.df["Label"] == 1]

        fig, ax = plt.subplots()
        """sns.scatterplot(data=self.df[(self.df["OUT_BYTES"] < 2e7) & (self.df["OUT_PKTS"] < 10000)],
                        x='OUT_PKTS', y='OUT_BYTES',
                        hue='Label', palette=["blue", "red"], alpha=0.25, s=20)"""
        sns.scatterplot(data=label_0[(label_0["OUT_BYTES"] < 2e8) & (label_0["IN_BYTES"] < 24e6)],
                        x='IN_BYTES', y='OUT_BYTES', color="blue", s=10)
        sns.scatterplot(data=label_1[(label_1["OUT_BYTES"] < 2e8) & (label_1["IN_BYTES"] < 24e6)],
                        x='IN_BYTES', y='OUT_BYTES', color="red", s=10)
        ax.set_title("Scatter plot IN_BYTES : OUT_BYTES")
        ax.set_xlabel('IN_BYTES')
        ax.set_ylabel('OUT_BYTES')
        plt.subplots_adjust(left=0.135)
        plt.show()

        fig, ax = plt.subplots()
        """sns.scatterplot(data=self.df[(self.df["OUT_BYTES"] < 0.15e6) & (self.df["OUT_PKTS"] < 100)],
                        x='OUT_PKTS', y='OUT_BYTES',
                        hue='Label', palette=["blue", "red"], alpha=0.25, s=20)"""
        sns.scatterplot(data=label_0[(label_0["OUT_BYTES"] < 100e3) & (label_0["IN_BYTES"] < 28000)],
                        x='IN_BYTES', y='OUT_BYTES', color="blue", ax=ax, s=15)
        sns.scatterplot(data=label_1[(label_1["OUT_BYTES"] < 100e3) & (label_1["IN_BYTES"] < 28000)],
                        x='IN_BYTES', y='OUT_BYTES', color="red", ax=ax, s=15)
        ax.set_title("Scatter plot IN_BYTES : OUT_BYTES zoomed in")
        ax.set_xlabel('IN_BYTES')
        ax.set_ylabel('OUT_BYTES')
        plt.subplots_adjust(left=0.135)
        plt.show()

    def scatter_out_bytes_out_pkts(self):
        label_0 = self.df[self.df["Label"] == 0]
        label_1 = self.df[self.df["Label"] == 1]

        fig, ax = plt.subplots()
        """sns.scatterplot(data=self.df[(self.df["OUT_BYTES"] < 2e7) & (self.df["OUT_PKTS"] < 10000)],
                        x='OUT_PKTS', y='OUT_BYTES',
                        hue='Label', palette=["blue", "red"], alpha=0.25, s=20)"""
        sns.scatterplot(data=label_0[(label_0["OUT_BYTES"] < 2e7) & (label_0["OUT_PKTS"] < 10000)],
                        x='OUT_PKTS', y='OUT_BYTES', color="blue", s=10)
        sns.scatterplot(data=label_1[(label_1["OUT_BYTES"] < 2e7) & (label_1["OUT_PKTS"] < 10000)],
                        x='OUT_PKTS', y='OUT_BYTES', color="red", s=10)
        ax.set_title("Scatter plot OUT_PKTS : OUT_BYTES")
        ax.set_xlabel('OUT_PKTS')
        ax.set_ylabel('OUT_BYTES')
        plt.subplots_adjust(left=0.135)
        plt.show()

        fig, ax = plt.subplots()
        """sns.scatterplot(data=self.df[(self.df["OUT_BYTES"] < 0.15e6) & (self.df["OUT_PKTS"] < 100)],
                        x='OUT_PKTS', y='OUT_BYTES',
                        hue='Label', palette=["blue", "red"], alpha=0.25, s=20)"""
        sns.scatterplot(data=label_0[(label_0["OUT_BYTES"] < 0.15e6) & (label_0["OUT_PKTS"] < 100)],
                        x='OUT_PKTS', y='OUT_BYTES', color="blue", ax=ax, s=15)
        sns.scatterplot(data=label_1[(label_1["OUT_BYTES"] < 0.15e6) & (label_1["OUT_PKTS"] < 100)],
                        x='OUT_PKTS', y='OUT_BYTES', color="red", ax=ax, s=15)
        ax.set_title("Scatter plot OUT_PKTS : OUT_BYTES zoomed in")
        ax.set_xlabel('OUT_PKTS')
        ax.set_ylabel('OUT_BYTES')
        plt.subplots_adjust(left=0.135)
        plt.show()

    def scatter_in_bytes_in_pkts(self):
        label_0 = self.df[self.df["Label"] == 0]
        label_1 = self.df[self.df["Label"] == 1]

        fig, ax = plt.subplots()
        sns.scatterplot(data=label_0[(label_0["IN_BYTES"] < 24e6) & (label_0["IN_PKTS"] < 250000)],
                        x='IN_PKTS', y='IN_BYTES', color="blue", s=10)
        sns.scatterplot(data=label_1[(label_1["IN_BYTES"] < 24e6) & (label_1["IN_PKTS"] < 250000)],
                        x='IN_PKTS', y='IN_BYTES', color="red", s=10)
        ax.set_title("Scatter plot IN_PKTS : IN_BYTES")
        ax.set_xlabel('IN_PKTS')
        ax.set_ylabel('IN_BYTES')
        plt.subplots_adjust(left=0.135)
        plt.show()

        fig, ax = plt.subplots()
        sns.scatterplot(data=label_0[(label_0["IN_BYTES"] < 60e3) & (label_0["IN_PKTS"] < 250)],
                        x='IN_PKTS', y='IN_BYTES', color="blue", ax=ax, s=15)
        sns.scatterplot(data=label_1[(label_1["IN_BYTES"] < 60e3) & (label_1["IN_PKTS"] < 250)],
                        x='IN_PKTS', y='IN_BYTES', color="red", ax=ax, s=15)
        ax.set_title("Scatter plot IN_PKTS : IN_BYTES zoomed in")
        ax.set_xlabel('IN_PKTS')
        ax.set_ylabel('IN_BYTES')
        plt.subplots_adjust(left=0.135)
        plt.show()

    def scatter_flow_duration_out_pkts(self):
        label_0 = self.df[self.df["Label"] == 0]
        label_1 = self.df[self.df["Label"] == 1]

        fig, ax = plt.subplots()
        sns.scatterplot(data=self.df,
                        x='FLOW_DURATION_MILLISECONDS', y='OUT_PKTS',
                        hue='Label', palette=["blue", "red"], alpha=0.25, s=20)
        ax.set_title("Scatter plot FLOW_DURATION_MILLISECONDS : OUT_PKTS")
        ax.set_xlabel('FLOW_DURATION_MILLISECONDS')
        ax.set_ylabel('OUT_PKTS')
        plt.show()

        fig, ax = plt.subplots()
        sns.scatterplot(data=label_0[(label_0["FLOW_DURATION_MILLISECONDS"] > 4.17e6) & (label_0["OUT_PKTS"] < 250)],
                        x='FLOW_DURATION_MILLISECONDS', y='OUT_PKTS', color="blue", ax=ax, s=15)
        sns.scatterplot(data=label_1[(label_1["FLOW_DURATION_MILLISECONDS"] > 4.17e6) & (label_1["OUT_PKTS"] < 250)],
                        x='FLOW_DURATION_MILLISECONDS', y='OUT_PKTS', color="red", ax=ax, s=15)
        ax.set_title("Scatter plot FLOW_DURATION_MILLISECONDS : OUT_PKTS zoomed in")
        ax.set_xlabel('FLOW_DURATION_MILLISECONDS')
        ax.set_ylabel('OUT_PKTS')
        plt.show()

        """fig, ax = plt.subplots()
        sns.scatterplot(data=self.attacks[(self.attacks["FLOW_DURATION_MILLISECONDS"] > 4.17e6) & (self.attacks["OUT_PKTS"] < 0.25e3)],
                        x='FLOW_DURATION_MILLISECONDS', y='OUT_PKTS',
                        hue='Attack', palette=sns.color_palette("tab10"), alpha=0.5, s=40)
        ax.set_title("Scatter plot FLOW_DURATION_MILLISECONDS : OUT_PKTS zoomed in")
        ax.set_xlabel('FLOW_DURATION_MILLISECONDS')
        ax.set_ylabel('OUT_PKTS')
        plt.show()"""

    def scatter_in_pkts_out_pkts(self):
        fig, ax = plt.subplots()
        sns.scatterplot(data=self.df,
                        x='IN_PKTS', y='OUT_PKTS',
                        hue='Label', palette=["blue", "red"], alpha=0.25, s=20)
        ax.set_title("Scatter plot IN_PKTS : OUT_PKTS")
        ax.set_xlabel('IN_PKTS')
        ax.set_ylabel('OUT_PKTS')
        plt.show()

        fig, ax = plt.subplots()
        sns.scatterplot(data=self.df[(self.df["OUT_PKTS"] < 500) & (self.df["IN_PKTS"] < 250)],
                        x='IN_PKTS', y='OUT_PKTS',
                        hue='Label', palette=["blue", "red"], alpha=0.25, s=30)
        ax.set_title("Scatter plot IN_PKTS : OUT_PKTS zoomed in")
        ax.set_xlabel('IN_PKTS')
        ax.set_ylabel('OUT_PKTS')
        plt.show()

        for attack in self.attacks["Attack"].unique():
            attacks_df = self.attacks[self.attacks["Attack"] == attack]
            fig, ax = plt.subplots()
            sns.scatterplot(data=attacks_df[(attacks_df["OUT_PKTS"] < 500) & (attacks_df["IN_PKTS"] < 250)],
                            x='IN_PKTS', y='OUT_PKTS',
                            hue='Attack', alpha=0.5, s=40)
            ax.set_title(f"Scatter plot Attack type {attack} IN_PKTS : OUT_PKTS zoomed in")
            ax.set_xlabel('IN_PKTS')
            ax.set_ylabel('OUT_PKTS')
            ax.set_xlim(0, 250)
            ax.set_ylim(0, 500)
            plt.show()

    def scatter_flow_duration_out_bytes(self):
        fig, ax = plt.subplots()
        sns.scatterplot(data=self.df,
                        x='FLOW_DURATION_MILLISECONDS', y='OUT_BYTES',
                        hue='Label', palette=["blue", "red"], alpha=0.25, s=20)
        ax.set_title("Scatter plot FLOW_DURATION_MILLISECONDS : OUT_BYTES")
        ax.set_xlabel('FLOW_DURATION_MILLISECONDS')
        ax.set_ylabel('OUT_BYTES')
        plt.show()

        fig, ax = plt.subplots()
        sns.scatterplot(data=self.df[(self.df["FLOW_DURATION_MILLISECONDS"] > 4.17e6) & (self.df["OUT_BYTES"] < 0.6e6)],
                        x='FLOW_DURATION_MILLISECONDS', y='OUT_BYTES',
                        hue='Label', palette=["blue", "red"], alpha=0.25, s=30)
        ax.set_title("Scatter plot FLOW_DURATION_MILLISECONDS : OUT_BYTES zoomed in")
        ax.set_xlabel('FLOW_DURATION_MILLISECONDS')
        ax.set_ylabel('OUT_BYTES')
        plt.show()

        fig, ax = plt.subplots()
        sns.scatterplot(data=self.attacks[(self.attacks["FLOW_DURATION_MILLISECONDS"] > 4.17e6) & (self.attacks["OUT_BYTES"] < 0.6e6)],
                        x='FLOW_DURATION_MILLISECONDS', y='OUT_BYTES',
                        hue='Attack', palette=sns.color_palette("tab10"), alpha=0.5, s=40)
        ax.set_title("Scatter plot FLOW_DURATION_MILLISECONDS : OUT_BYTES zoomed in")
        ax.set_xlabel('FLOW_DURATION_MILLISECONDS')
        ax.set_ylabel('OUT_BYTES')
        plt.show()

    def density_in_bytes_label(self):
        fig, ax = plt.subplots()
        sns.kdeplot(clip=(0, 4000), data=self.attacks, x="IN_BYTES",
                    common_norm=False, fill=True, bw_adjust=0.01, ax=ax, color="red")
        ax.set_title('Attacks')
        ax.set_ylabel("Density")
        ax.set_xlabel("IN_BYTES")
        plt.show()

        fig, ax = plt.subplots()
        sns.kdeplot(clip=(0, 4000), data=self.benign, x="IN_BYTES",
                    common_norm=False, fill=True, bw_adjust=0.01, ax=ax, color="blue")
        ax.set_title('Benign')
        ax.set_ylabel("Density")
        ax.set_xlabel("IN_BYTES")
        plt.show()

    def density_out_bytes_label(self):
        fig, ax = plt.subplots()
        sns.kdeplot(clip=(0, 500), data=self.attacks, x="OUT_BYTES",
                    common_norm=False, fill=True, bw_adjust=0.05, ax=ax, color="red")
        ax.set_title('Attacks')
        ax.set_ylabel("Density")
        ax.set_xlabel("OUT_BYTES")
        plt.show()

        fig, ax = plt.subplots()
        sns.kdeplot(clip=(0, 10000), data=self.benign, x="OUT_BYTES",
                    common_norm=False, fill=True, bw_adjust=0.01, ax=ax, color="blue")
        ax.set_title('Benign')
        ax.set_ylabel("Density")
        ax.set_xlabel("OUT_BYTES")
        plt.show()

    def density_flow_duration(self):
        fig, ax = plt.subplots()
        sns.kdeplot(data=self.attacks, x="FLOW_DURATION_MILLISECONDS",
                    common_norm=False, fill=True, bw_adjust=0.1, ax=ax, color="red")
        ax.set_title('Attacks')
        ax.set_ylabel("Density")
        ax.set_xlabel("FLOW_DURATION_MILLISECONDS")
        plt.show()

        fig, ax = plt.subplots()
        sns.kdeplot(data=self.benign, x="FLOW_DURATION_MILLISECONDS",
                    common_norm=False, fill=True, bw_adjust=0.1, ax=ax, color="blue")
        ax.set_title('Benign')
        ax.set_ylabel("Density")
        ax.set_xlabel("FLOW_DURATION_MILLISECONDS")
        plt.show()

    def density_in_bytes_per_attack(self):
        print_line()
        print("density_in_bytes_per_attack")
        for attack in self.attacks["Attack"].unique():
            fig, ax = plt.subplots()
            print(attack)
            test = self.attacks[self.attacks["Attack"] == attack]["IN_BYTES"]
            print(test.dtype)
            if len(test.unique()) == 1:
                print("One unique value " + str(test.unique()[0]))
            print(f"Mean {test.mean()}")
            print(f"Variance {test.var()}")
            print("\n")
            if test.var() != 0:
                sns.kdeplot(data=self.attacks[self.attacks["Attack"] == attack], x="IN_BYTES",
                            common_norm=False, fill=True, bw_adjust=0.1, ax=ax, color="red")
                ax.set_title(attack)
                ax.set_ylabel("Density")
                ax.set_xlabel("IN_BYTES")
                plt.show()
        print_line()

    def density_flow_duration_per_attack(self):
        print_line()
        print("density_flow_duration_per_attack")
        for attack in self.attacks["Attack"].unique():
            fig, ax = plt.subplots()
            print(attack)
            test = self.attacks[self.attacks["Attack"] == attack]["FLOW_DURATION_MILLISECONDS"]
            print(test.dtype)
            if len(test.unique()) == 1:
                print("One unique value " + str(test.unique()[0]))
            print(f"Mean {test.mean()}")
            print(f"Variance {test.var()}")
            print("\n")
            if test.var() != 0:
                sns.kdeplot(data=self.attacks[self.attacks["Attack"] == attack], x="FLOW_DURATION_MILLISECONDS",
                            common_norm=False, fill=True, bw_adjust=0.1, ax=ax, color="red")
                ax.set_title(attack)
                ax.set_ylabel("Density")
                ax.set_xlabel("FLOW_DURATION_MILLISECONDS")
                plt.show()
        print_line()

    def density_out_bytes_attack_type(self):
        for attack in self.attacks["Attack"].unique():
            print(attack)
            fig, ax = plt.subplots()
            sns.kdeplot(data=self.attacks[self.attacks["Attack"] == attack], x="OUT_BYTES",
                        common_norm=False, fill=True, bw_adjust=0.5, ax=ax)
            ax.set_title('Attack ' + attack)
            ax.set_ylabel("Density")
            ax.set_xlabel("OUT_BYTES")
            plt.show()

    def count_label_tcp_flag(self):
        print_line()
        print("count_label_tcp_flag")
        counts = {"TCP_FLAG_0": [], "TCP_FLAG_1": [], "TCP_FLAG_2": [],
                  "TCP_FLAG_3": [], "TCP_FLAG_4": [], "TCP_FLAG_5": [],
                  "TCP_FLAG_6": [], "TCP_FLAG_7": []}

        sum_attacks = self.attacks.shape[0]
        print("Percentage of Attacks using the particular flag")
        for flag in counts.keys():
            counts[flag] = [self.attacks[flag].sum() / sum_attacks]
            print(f"{round(counts[flag][0], 4) * 100}% of the attacks had {flag}")

            for attack in self.attacks["Attack"].unique():
                has_flag = self.attacks[self.attacks[flag] == 1]
                sum_has_flag = has_flag[flag].sum()
                sum_attack_has_flag = has_flag[has_flag["Attack"] == attack].shape[0]

                print(f"Attacks with {flag} were {round(sum_attack_has_flag / sum_has_flag, 4) * 100}% {attack}")


        counts = pd.DataFrame(counts)

        fig, ax = plt.subplots()
        ax = counts.T.plot(kind='bar', rot=90)
        ax.set_title("Usage of TCP Flags in Attacks in percentage")
        ax.get_legend().remove()
        plt.subplots_adjust(bottom=0.25)
        plt.show()

        counts = {"TCP_FLAG_0": [], "TCP_FLAG_1": [], "TCP_FLAG_2": [],
                  "TCP_FLAG_3": [], "TCP_FLAG_4": [], "TCP_FLAG_5": [],
                  "TCP_FLAG_6": [], "TCP_FLAG_7": []}

        for attack in self.attacks["Attack"].unique():
            attack_group = self.attacks[self.attacks["Attack"] == attack]
            for flag in counts.keys():
                counts[flag] = [attack_group[flag].sum() / attack_group.shape[0]]
            counts = pd.DataFrame(counts)
            fig, ax = plt.subplots()
            ax = counts.T.plot(kind='bar', rot=90)
            ax.set_title(f"Usage of TCP Flags in Attack {attack} in percentage")
            ax.get_legend().remove()
            plt.subplots_adjust(bottom=0.25)
            plt.show()

        print_line()

    def count_attack_protocol(self):
        print_line()
        print("count_label_protocol")
        protocols = ["PROTOCOL_0", "PROTOCOL_1", "PROTOCOL_17", "PROTOCOL_2", "PROTOCOL_47", "PROTOCOL_58", "PROTOCOL_6"]
        protocols_used_by_attacks = []
        percentages = {}

        sum_attacks = self.attacks.shape[0]
        print("Percentage of Attacks using the particular protocol")
        for protocol in protocols:
            if self.attacks[protocol].sum() != 0:
                protocols_used_by_attacks.append(protocol)
            percentages[protocol] = [self.attacks[protocol].sum() / sum_attacks]

        percentages = pd.DataFrame(percentages)

        print(percentages.head())

        fig, ax = plt.subplots()
        ax = percentages.T.plot(kind='bar', rot=90)
        ax.set_title("Usage of Protocols in Attacks in percentage")
        ax.get_legend().remove()
        plt.subplots_adjust(bottom=0.25)
        plt.show()

        attacks = pd.DataFrame(columns=list(self.attacks["Attack"].unique()), index=protocols_used_by_attacks)

        print("Percentage of Attack types using the particular protocol")
        attack_groups = self.attacks.groupby("Attack")
        for protocol in protocols_used_by_attacks:
            for attack_type, group in attack_groups:
                attacks.loc[protocol, attack_type] = group[protocol].sum() / group.shape[0]

        print(attacks.head())

        fig, ax = plt.subplots()
        ax = attacks.T.plot(kind='bar', rot=90, stacked=True)
        ax.set_title("Usage of Protocols per Attack type in percentage")
        plt.subplots_adjust(bottom=0.5)
        plt.legend(bbox_to_anchor=(0, -1), loc='upper left')
        plt.show()

        print_line()
