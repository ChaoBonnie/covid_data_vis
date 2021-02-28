import pandas as pd
import seaborn as sns; sns.set(style='ticks', context='talk')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statannot import add_stat_annotation
from scipy import stats
import numpy as np
import itertools

# Set up the primary dataframe #

df_toronto = pd.read_excel(r'C:\Users\chaob\Documents\ETC Data.xlsx', sheet_name='Toronto')
df_turin = pd.read_excel(r'C:\Users\chaob\Documents\ETC Data.xlsx', sheet_name='Turin')
# df_toronto.rename(columns={"IL-6": "Log IL-6", "IL-8": "Log IL-8", "IL-10": "Log IL-10", "sTNFR1": "Log sTNFR1", "sTREM1": "Log sTREM1"})
# df_turin.rename(columns={"IL-6": "Log IL-6", "IL-8": "Log IL-8", "IL-10": "Log IL-10", "sTNFR1": "Log sTNFR1", "sTREM1": "Log sTREM1"})
df_toronto['Hospital Admission'] = df_toronto['Hospital Admission'].replace({'No': 'Discharged', 'Yes': 'Hospitalized', 'Yes (ICU)': 'Hospitalized'})
df_turin['Hospital Admission'] = df_turin['Hospital Admission'].replace({'No': 'Discharged', 'Yes': 'Hospitalized'})
df_toronto['Gender'] = df_toronto['Gender'].replace({'M': 'Male', 'F': 'Female'})
df_toronto['Age at Study Enrollment'] = pd.cut(df_toronto['Age at Study Enrollment'], bins=[0, 49, 70, 120], include_lowest=True, labels=['<50', '50-70', '>70'])
df_toronto['BMI'] = pd.cut(df_toronto['BMI'], bins=[0, 29.9, 100], include_lowest=True, labels=['<30', '≥30'])
print(df_toronto['Ethnicity'].value_counts()[:3])
df_toronto_3ethnicities = df_toronto[df_toronto['Ethnicity'].isin(['White or Caucasian', 'Black or African American', 'South Asian'])]
# df = pd.concat([df_toronto, df_turin])
print(df_toronto_3ethnicities.shape)

# Set up the secondary dataframe #

df_secondary = pd.read_excel(r'C:\Users\chaob\Documents\ETC Data v2.xlsx')
df_secondary['Temperature at Presentation (°C)'] = \
    df_secondary['Temperature at Presentation (°C)'].replace({'-': None})
df_secondary['Temperature at Presentation (°C)'] = pd.cut(df_secondary['Temperature at Presentation (°C)'], bins=[0, 37.9, 100], labels=['<38', '≥38'])
df_secondary['SpO2'] = pd.cut(df_secondary['SpO2'], bins=[0, 89, 95, 110], labels=['<90', '90-95', '>95'])
df_secondary['Heart Rate'] = pd.cut(df_secondary['Heart Rate'], bins=[0, 89, 200], labels=['<90', '≥90'])
df_secondary['CRB-65_3groups'] = df_secondary['CRB-65_3groups'].astype(str)
df_secondary['CRB-65_3groups'] = df_secondary['CRB-65_3groups'].replace({'0.0': '0', '1.0': '1 or 2', '2.0': '1 or 2', '3.0': '3 or 4', '4.0': '3 or 4'})
df_secondary['CRB-65_2groups'] = df_secondary['CRB-65_2groups'].astype(str)
df_secondary['CRB-65_2groups'] = df_secondary['CRB-65_2groups'].replace({'0.0': '0', '1.0': '1 to 4', '2.0': '1 to 4', '3.0': '1 to 4', '4.0': '1 to 4'})

# Set up the hospitalization and COVID severity dataframes #

df_hosp = pd.read_excel(r'C:\Users\chaob\Documents\ETC Hospitalization-COVID Data.xlsx', sheet_name='Hospitalization')
df_covid = pd.read_excel(r'C:\Users\chaob\Documents\ETC Hospitalization-COVID Data.xlsx', sheet_name='COVID-19')
df_hosp = df_hosp.rename(columns={"Subject Required ICU Care": "Hospitalization", "Total Hospitalization (Days)_3groups": "Hospital Length of Stay (Days)_3groups"})
df_hosp = df_hosp.rename(columns={"Subject Required ICU Care": "Hospitalization", "Total Hospitalization (Days)_2groups": "Hospital Length of Stay (Days)_2groups"})

df_hosp['Hospitalization'][~df_hosp['Hospitalization'].isnull()] = 'ICU'
df_hosp['Hospitalization'] = df_hosp['Hospitalization'].fillna('Ward')
df_hosp['Hospital Length of Stay (Days)_3groups'] = df_hosp['Hospital Length of Stay (Days)_3groups'].replace({'>28': 28})
df_hosp['Hospital Length of Stay (Days)_3groups'] = pd.cut(df_hosp['Hospital Length of Stay (Days)_3groups'], bins=[0, 3, 7, 29], labels=['≤3', '4-7', '>7'])
df_hosp['Hospital Length of Stay (Days)_2groups'] = df_hosp['Hospital Length of Stay (Days)_2groups'].replace({'>28': 28})
df_hosp['Hospital Length of Stay (Days)_2groups'] = pd.cut(df_hosp['Hospital Length of Stay (Days)_2groups'], bins=[0, 3, 29], labels=['≤3', '>3'])
df_hosp['28-Day Mortality'] = df_hosp['28-Day Mortality'].fillna('No')

# Set up the updated COVID dataframe #
df_covidV2= pd.read_excel(r'C:\Users\chaob\Documents\ETC Updated COVID Data.xlsx')
df_covidV2['COVID-19 Pneumonia'] = df_covidV2['COVID-19 Pneumonia'].replace({'N': 'No', 'Y': 'Yes'})
df_covidV2[['Bacteremia', 'ARDS', 'CAP', 'VAP']] = df_covidV2[['Bacteremia', 'ARDS', 'CAP', 'VAP']].replace({'N': 'No', 'Y': 'Yes'})

# Set up the COVID-v3 dataframe #
df_covidV3= pd.read_excel(r'C:\Users\chaob\Documents\ETC COVID Data v3.xlsx')
df_covidV3['COVID-19 Pneumonia'] = df_covidV3['COVID-19 Pneumonia'].replace({'N': 'No', 'Y': 'Yes'})

# Name subfigures and process NaN #
subfigures = ['IL-6 (pg/mL)', 'IL-8 (pg/mL)', 'IL-10 (pg/mL)', 'sTNFR1 (pg/mL)', 'sTREM1 (pg/mL)']
subfigures_log = [r'Log$_2$ IL-6 Concentration (pg/mL)', r'Log$_2$ IL-8 Concentration (pg/mL)', r'Log$_2$ IL-10 Concentration (pg/mL)',
                  r'Log$_2$ sTNFR1 Concentration (pg/mL)', r'Log$_2$ sTREM1 Concentration (pg/mL)']
dataframes = [df_toronto, df_toronto_3ethnicities, df_turin, df_secondary, df_hosp, df_covidV2, df_covidV3]
# categories = ['No, Toronto', 'Yes, Toronto', 'No, Turin', 'Yes, Turin']
df_toronto.loc[:, subfigures] = df_toronto[subfigures].fillna(0)
df_secondary.loc[:, subfigures] = df_secondary[subfigures].fillna(0)
df_hosp.loc[:, subfigures] = df_hosp[subfigures].fillna(0)
df_covidV2.loc[:, subfigures] = df_covidV2[subfigures].fillna(0)

# df_covid.loc[:, subfigures] = df_covid[subfigures].fillna(0)
# print(df_toronto.to_string())
# print(df_secondary.to_string())
# print(df_hosp.to_string())
# # print(df_covid.to_string())
# print(df_covidV2.to_string())

# Set zero-values to 0.5*LOD #
for dataframe in dataframes:
    dataframe['IL-6 (pg/mL)'] = dataframe['IL-6 (pg/mL)'].replace({0: 11.5})
    dataframe['IL-8 (pg/mL)'] = dataframe['IL-8 (pg/mL)'].replace({0: 13.5})
    dataframe['IL-10 (pg/mL)'] = dataframe['IL-10 (pg/mL)'].replace({0: 3.5})
    dataframe['sTNFR1 (pg/mL)'] = dataframe['sTNFR1 (pg/mL)'].replace({0: 8.5})
    dataframe['sTREM1 (pg/mL)'] = dataframe['sTREM1 (pg/mL)'].replace({0: 22})

# Check values #

def check_values (df, x, x1, x2):

    n_no = len(df[df[x] == x1])
    n_yes = len(df[df[x] == x2])
    print('No:', n_no, 'Yes:', n_yes)

    for cytokine in subfigures:
        # df[cytokine] = np.log2(df[cytokine], out=np.zeros_like(df[cytokine]), where=(df[cytokine] != 0))
        cytokine_med_no = np.median(df[cytokine][df[x] == x1])
        cytokine_25_no = np.percentile(df[cytokine][df[x] == x1], 25)
        cytokine_75_no = np.percentile(df[cytokine][df[x] == x1], 75)
        cytokine_med_yes = np.median(df[cytokine][df[x] == x2])
        cytokine_25_yes = np.percentile(df[cytokine][df[x] == x2], 25)
        cytokine_75_yes = np.percentile(df[cytokine][df[x] == x2], 75)
        print(cytokine, '\n', cytokine_med_no, cytokine_25_no, cytokine_75_no, '\n', cytokine_med_yes, cytokine_25_yes, cytokine_75_yes)

# check_values(df_toronto, 'Hospital Admission', 'Discharged', 'Hospitalized')
# check_values(df_hosp, 'Hospitalization', 'Ward', 'ICU')
# check_values(df_hosp, '28-Day Mortality', 'No', 'Yes')
# check_values(df_covidV2, 'Bacteremia', 'No', 'Yes')
# check_values(df_covidV2, 'ARDS', 'No', 'Yes')
# check_values(df_covidV2, 'CAP', 'No', 'Yes')

def annotate_anova(ax, data, y, anova_path, anova_sheet):
    df = pd.read_excel(anova_path, sheet_name=anova_sheet, index_col=0)
    df = df[y.split(' ')[0]]
    pvalues = []
    box_pairs = []
    for x in df.index:
        p = df[x]
        if p < 0.05:
            pvalues.append(p)
            box_pairs.append(((x, 'RU'), (x, 'LL')))
    add_stat_annotation(ax, data=data, x='EVLP ID', y=y, hue='Location',
                        box_pairs=box_pairs, pvalues=pvalues, perform_stat_test=False,
                        loc='outside', verbose=0)


def horizontal_lines(data, ax, x, y, **kwargs):
    data = data.copy()
    xticks = ax.get_xticklabels()
    xticks = {tick.get_text(): i for i, tick in enumerate(xticks)}
    data['xmin'] = data[x].apply(lambda xval: xticks[str(xval)] - 0.4)
    data['xmax'] = data[x].apply(lambda xval: xticks[str(xval)] + 0.4)
    ax.hlines(y=data[y], xmin=data['xmin'], xmax=data['xmax'], **kwargs)
    return ax


def make_graph(df, x, order, figname, figsize=None, box=True, show_xaxis=True, rotate_xaxis=False, anova_path=None, anova_sheet=None, show_fig=False):

    if figsize == None:
        fig = plt.figure(figsize=(24, 6))
    else:
        fig = plt.figure(figsize=figsize)

    gs = GridSpec(1, 5)
    axs = []
    for i in range(len(subfigures)):
        axs.append(fig.add_subplot(gs[0, i]))

    for ax, y, log in zip(axs, subfigures, subfigures_log):
        df.loc[:, log] = np.log2(df[y])
        ax.set_ylim(top=max(df[log]) * 1.2)
        if box == True:
            sns.boxplot(x=x, y=log, data=df, ax=ax, order=order, width=0.4, showfliers=False, saturation=0.6, palette='pastel')
            sns.stripplot(x=x, y=log, data=df, ax=ax, order=order, alpha=0.7)
        else:
            sns.stripplot(x=x, y=log, data=df, ax=ax, order=order, alpha=0.8)
            g = sns.pointplot(x=x, y=log, data=df, ax=ax, order=order, zorder=100,
                          join=False, ci=95, color='red', markers='x', errwidth=2, orient='v', capsize=0.15, estimator=np.median)
            plt.setp(g.lines, zorder=100)
            plt.setp(g.collections, zorder=100)
        if rotate_xaxis == True:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode='anchor')

        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 5
        # plt.yscale('log', base=2)
        # plt.ylim(0)
        # if df[log].min() == 0:
        #     ax.set_ylim(bottom=-0.2, top=max(df[log]) * 4)
        # else:
        #     ax.set_ylim(bottom=min(df[log]) / 2, top=max(df[log]) * 4)
        # ax.spines['bottom'].set_position('zero')
        # sns.despine(ax=ax, offset=0)
        if anova_path is not None:
            annotate_anova(ax, df, log, anova_path, anova_sheet)
        if len(order) == 2:
            xlabels = [l.get_text() for l in ax.get_xticklabels()]
            xlabels = list(itertools.combinations(xlabels, 2))
            add_stat_annotation(ax, data=df, x=x, y=log, order=order, width=0.4, box_pairs=xlabels,
                                perform_stat_test=True, test='Mann-Whitney',
                                loc='inside', verbose=0, no_ns=True, fontsize='large')
            # print(x, stats.mannwhitneyu(df[log][df[x] == order[0]], df[log][df[x] == order[1]], alternative='two-sided'))
        elif len(order) > 2:
            _, p = stats.kruskal(df[log][df[x] == order[0]], df[log][df[x] == order[1]], df[log][df[x] == order[2]])
            if p < 0.0001:
                ax.title.set_text('****')
            elif p < 0.001:
                ax.title.set_text('***')
            elif p < 0.01:
                ax.title.set_text('**')
            elif p < 0.05:
                ax.title.set_text('*')

        # ax.set_yscale('symlog', base=2)

        if show_xaxis == False:
            ax.set_xlabel('')

    fig.tight_layout()

    if show_fig:
        fig.show()
    else:
        fig.savefig(figname, dpi=200)
        plt.close()

# make_graph(df_toronto, 'Hospital Admission', ['Discharged', 'Hospitalized'], 'Fig1_Toronto', show_xaxis=False)
# make_graph(df_turin, 'Hospital Admission', ['Discharged', 'Hospitalized'], 'Fig1_Turin', show_xaxis=False)
# # # make_graph(df_toronto, 'COVID-19+', ['Negative', 'Positive'], 'Fig2_COVID-19')
# make_graph(df_toronto, 'Gender', ['Male', 'Female'], 'Fig2_Gender')
# make_graph(df_toronto, 'Age at Study Enrollment', ['<50', '50-70', '>70'], 'Fig2_Age')
# make_graph(df_toronto, 'BMI', ['<30', '≥30'], 'Fig2_BMI')
make_graph(df_toronto_3ethnicities, 'Ethnicity', ['White or Caucasian', 'Black or African American', 'South Asian'], 'Fig2_Ethnicity',
           figsize=(24, 8), show_xaxis=False, rotate_xaxis=True)
# make_graph(df_secondary, 'Temperature at Presentation (°C)', ['<38', '≥38'], 'Fig3_Temperature')
# make_graph(df_secondary, 'SpO2', ['<90', '90-95', '>95'], 'Fig3_SpO2')
# make_graph(df_secondary, 'Heart Rate', ['<90', '≥90'], 'Fig3_Heart Rate')
# make_graph(df_secondary, 'CRB-65_3groups', ['0', '1 or 2', '3 or 4'], 'Fig3_CRB-65_3groups')
# make_graph(df_secondary, 'CRB-65_2groups', ['0', '1 to 4'], 'Fig3_CRB-65_2groups')
# make_graph(df_hosp, 'Hospitalization', ['Ward', 'ICU'], 'Fig4_Hospitalization')
# make_graph(df_hosp, 'Hospital Length of Stay (Days)_3groups', ['≤3', '4-7', '>7'], 'Fig4_LOS_3groups')
# make_graph(df_hosp, 'Hospital Length of Stay (Days)_2groups', ['≤3', '>3'], 'Fig4_LOS_2groups')
# make_graph(df_hosp, '28-Day Mortality', ['No', 'Yes'], 'Fig4_Mortality')
# # # make_graph(df_covid, 'COVID-19 Severity', ['Mild', 'Moderate', 'Severe'], 'Fig4_COVID-19 Severity')
# make_graph(df_covidV2, 'COVID-19+', ['Negative', 'Positive'], 'Fig_COVID-19+')
# # make_graph(df_covidV2, 'COVID-19 Severity', ['Mild', 'Moderate', 'Severe'], 'Fig_COVID-19 Severity')
# # make_graph(df_covidV2, df_covidV2['COVID-19 Pneumonia'][df_covidV2['COVID-19+'] == 'Positive'], ['No', 'Yes'], 'Fig_COVID-19 Pneumonia')
# make_graph(df_covidV2, 'Bacteremia', ['No', 'Yes'], 'Fig_COVID-19 Bacteremia')
# make_graph(df_covidV2, 'ARDS', ['No', 'Yes'], 'Fig_COVID-19 ARDS')
# make_graph(df_covidV2, 'CAP', ['No', 'Yes'], 'Fig_COVID-19 CAP')
# make_graph(df_covidV2, 'VAP', ['No', 'Yes'], 'Fig_COVID-19 VAP')
# make_graph(df_covidV3, 'COVID-19 Severity', ['Mild', 'Moderate', 'Severe'], 'Fig_COVID-19 Severity')
# make_graph(df_covidV3, 'COVID-19 Pneumonia', ['No', 'Yes'], 'Fig_COVID-19 Pneumonia')


# make_graph(df_toronto, 'Hospital Admission', ['Discharged', 'Hospitalized'], 'Fig1_Toronto_95CI', box=False, show_xaxis=False)
# make_graph(df_turin, 'Hospital Admission', ['Discharged', 'Hospitalized'], 'Fig1_Turin_95CI', box=False, show_xaxis=False)
# make_graph(df_toronto, 'COVID-19+', ['Negative', 'Positive'], 'Fig2_COVID-19_95CI', box=False)
# make_graph(df_toronto, 'Gender', ['Male', 'Female'], 'Fig2_Gender_95CI', box=False)
# make_graph(df_toronto, 'Age at Study Enrollment', ['<50', '50-70', '>70'], 'Fig2_Age_95CI', box=False)
# make_graph(df_toronto, 'BMI', ['<30', '≥30'], 'Fig2_BMI_95CI', box=False)
# make_graph(df_toronto_3ethnicities, 'Ethnicity', ['White or Caucasian', 'Black or African American', 'South Asian'], 'Fig2_Ethnicity_95CI', box=False,
#            show_xaxis=False, rotate_xaxis=True)
# make_graph(df_secondary, 'Temperature at Presentation (°C)', ['<38', '≥38'], 'Fig3_Temperature_95CI', box=False)
# make_graph(df_secondary, 'SpO2', ['<90', '90-95', '>95'], 'Fig3_SpO2_95CI', box=False)
# make_graph(df_secondary, 'Heart Rate', ['<90', '≥90'], 'Fig3_Heart Rate_95CI', box=False)
# make_graph(df_secondary, 'CRB-65', ['0', '1 or 2', '3 or 4'], 'Fig3_CRB-65_95CI', box=False)
# make_graph(df_hosp, 'Hospitalization', ['Ward', 'ICU'], 'Fig4_Hospitalization_95CI', box=False)
# make_graph(df_hosp, 'Hospital Length of Stay (Days)', ['≤3', '4-7', '>7'], 'Fig4_LOS_95CI', box=False)
# make_graph(df_hosp, '28-Day Mortality', ['No', 'Yes'], 'Fig4_Mortality_95CI', box=False)
# make_graph(df_covid, 'COVID-19 Severity', ['Mild', 'Moderate', 'Severe'], 'Fig4_COVID-19 Severity_95CI', box=False)


