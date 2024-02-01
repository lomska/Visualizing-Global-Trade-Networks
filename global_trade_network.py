# IMPORTING PACKAGES ************************************************************

import pandas as pd
import numpy as np
import math

import comtradeapicall
import time

import pygraphviz as pgv

import networkx as nx

import dash_cytoscape as cyto

cyto.load_extra_layouts()
from plotly.colors import hex_to_rgb

from dash import Dash, html, Input, Output, callback, ctx
import dash_bootstrap_components as dbc
from datetime import datetime

import warnings

warnings.filterwarnings('ignore')

# COMTRADE SUBSCRIPTION KEY *****************************************************

subscription_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# CODES *************************************************************************

commodities = pd.read_csv('comtrade_codes/harmonized-system.csv')

reporters = pd.read_csv('comtrade_codes/reporterAreas.csv')
partners = pd.read_csv('comtrade_codes/partnerAreas.csv')

# EXTRACTING THE DATA ***********************************************************

comtrade_exp = comtradeapicall.getFinalData(subscription_key,
                                            typeCode='C',
                                            freqCode='A',
                                            clCode='HS',
                                            period='2022',
                                            reporterCode=None,
                                            cmdCode='1001',
                                            flowCode='X',
                                            partnerCode=None,
                                            partner2Code='0',
                                            customsCode='C00',
                                            motCode='0',
                                            maxRecords=250000)

time.sleep(10)

comtrade_imp = comtradeapicall.getFinalData(subscription_key,
                                            typeCode='C',
                                            freqCode='A',
                                            clCode='HS',
                                            period='2022',
                                            reporterCode=None,
                                            cmdCode='1001',
                                            flowCode='M',
                                            partnerCode=None,
                                            partner2Code='0',
                                            customsCode='C00',
                                            motCode='0',
                                            maxRecords=250000)

comtrade_imp = comtrade_imp[
    comtrade_imp['reporterCode'] != comtrade_imp['partnerCode']]

# GRAPH PARAMETERS **************************************************************

n_links = 2
chosen_color_parameter = 'region' # 'export_share' 'pagerank'
pic_width = 700
output_pic_name = 'wheat_and_meslin'

# DATA PROCESSING ***************************************************************

export_totals = comtrade_exp[comtrade_exp['partnerCode'] == 0][[
    'reporterCode', 'partnerCode', 'fobvalue'
]].groupby(['reporterCode', 'partnerCode']).agg('sum').reset_index()

import_totals = comtrade_imp[comtrade_imp['partnerCode'] == 0][[
    'reporterCode', 'partnerCode', 'cifvalue'
]].groupby(['reporterCode', 'partnerCode']).agg('sum').reset_index()

export_by_country = comtrade_exp[comtrade_exp['partnerCode'] != 0][[
    'reporterCode', 'partnerCode', 'fobvalue'
]].groupby(['reporterCode', 'partnerCode']).agg('sum').reset_index()

import_by_country = comtrade_imp[comtrade_imp['partnerCode'] != 0][[
    'reporterCode', 'partnerCode', 'cifvalue'
]].groupby(['reporterCode', 'partnerCode']).agg('sum').reset_index()

for dataset in [
        export_totals, import_totals, export_by_country, import_by_country
]:
    for col in ['reporterCode', 'partnerCode']:
        dataset[col] = [
            '00' + str(x) if len(str(x)) == 1 else '0' +
            str(x) if len(str(x)) == 2 else str(x)
            for x in dataset[col].tolist()
        ]

export_by_country.columns = ['exporter', 'importer', 'value']
import_by_country.columns = ['importer', 'exporter', 'value']

# NODES *************************************************************************

total_trade = export_totals[[
    'reporterCode', 'fobvalue'
]].set_index('reporterCode').join(
    import_totals[['reporterCode', 'cifvalue']].set_index('reporterCode'),
    how='outer').reset_index().rename(columns={
        'reporterCode': 'country_code',
        'fobvalue': 'export',
        'cifvalue': 'import'
    }).set_index('country_code')

for col in ['export', 'import']:
    total_trade[col + '_note'] = [
        1 if math.isnan(x) else 0 for x in total_trade[col].tolist()
    ]

imports_to_add = export_by_country[['exporter', 'importer',
                                    'value']].groupby('importer').agg('sum')
exports_to_add = import_by_country[['importer', 'exporter',
                                    'value']].groupby('exporter').agg('sum')

trade_to_add = exports_to_add.rename(columns={
    'value': 'export_by_partners'
}).join(imports_to_add.rename(columns={'value': 'import_by_partners'}),
        how='outer')

df_nodes = total_trade.join(trade_to_add, how='outer')[[
    'export', 'import', 'export_by_partners', 'import_by_partners'
]]

df_nodes[['export', 'import', 'export_by_partners',
          'import_by_partners']] = df_nodes[[
              'export', 'import', 'export_by_partners', 'import_by_partners'
          ]].fillna(0)

df_nodes['trade'] = [
    ex + im if ex != 0 and im != 0 else exp +
    im if ex == 0 and im != 0 else ex +
    imp if ex != 0 and im == 0 else exp + imp for ex, im, exp, imp in zip(
        df_nodes['export'], df_nodes['import'], df_nodes['export_by_partners'],
        df_nodes['import_by_partners'])
]

df_nodes = df_nodes[df_nodes['trade'] > 0]

df_nodes['trade_rescaled'] = df_nodes['trade'] / df_nodes['trade'].max() * 60
df_nodes['trade_rescaled'] = [
    0.15 if x <= 0.15 else x for x in df_nodes['trade_rescaled'].tolist()
]

df_nodes['diameter'] = [
    np.sqrt(x / np.pi) * 2 for x in df_nodes['trade_rescaled'].tolist()
]

diameter_dict = df_nodes['diameter'].to_dict()

# LINKS *************************************************************************

export_by_country = export_by_country[
    (export_by_country['exporter'].isin(df_nodes.index.tolist()))
    & (export_by_country['importer'].isin(df_nodes.index.tolist()))]
import_by_country = import_by_country[
    (import_by_country['exporter'].isin(df_nodes.index.tolist()))
    & (import_by_country['importer'].isin(df_nodes.index.tolist()))]

df_list = []

for dataset in [export_by_country, import_by_country]:

    dataset['source_target'] = [
        '_'.join(sorted([exporter, importer]))
        for exporter, importer in zip(dataset['exporter'], dataset['importer'])
    ]

    dataset['order'] = dataset.groupby('source_target').cumcount()

    df = dataset[dataset['order'] == 0].set_index('source_target')[[
        'exporter', 'importer', 'value'
    ]].rename(columns={
        'value': 'to'
    }).join(dataset[dataset['order'] == 1].set_index('source_target')[[
        'value'
    ]].rename(columns={'value': 'back'}))

    df_list.append(df)

source_target_df = df_list[0].combine_first(df_list[1])

source_target_df = source_target_df.fillna(0)

source_target_df[[
    'source', 'target'
]] = [[exporter, importer] if to - back >= 0 else [importer, exporter]
      for exporter, importer, to, back in
      zip(source_target_df['exporter'], source_target_df['importer'],
          source_target_df['to'], source_target_df['back'])]

df_links = source_target_df.reset_index()[['source', 'target']]

rank_df = export_by_country[['importer', 'exporter', 'value']].set_index([
    'importer', 'exporter'
]).combine_first(import_by_country[['importer', 'exporter', 'value'
                                    ]].set_index(['importer',
                                                  'exporter'])).reset_index()

rank_df['supplier_rank'] = rank_df.sort_values(
    by='value', ascending=False).groupby('importer').cumcount() + 1

supplier_rank_dict = rank_df.set_index(['importer', 'exporter'
                                        ])['supplier_rank'].to_dict()

df_links['supplier_rank_source'] = df_links.set_index(
    ['target', 'source']).index.map(supplier_rank_dict)

df_links['supplier_rank_target'] = df_links.set_index(
    ['source', 'target']).index.map(supplier_rank_dict)

link_dict = dict()

for e in export_by_country.exporter:
    link_dict[e] = []
for i in export_by_country.importer:
    if i not in link_dict.keys():
        link_dict[i] = []
for i in import_by_country.importer:
    if i not in link_dict.keys():
        link_dict[i] = []
for e in import_by_country.exporter:
    if e not in link_dict.keys():
        link_dict[e] = []

for e in export_by_country.exporter:
    data = export_by_country[export_by_country.exporter == e]
    importers_list = data['importer'].unique().tolist()
    for i in importers_list:
        link_dict[e].append(i)

for i in export_by_country.importer:
    data = export_by_country[export_by_country.importer == i]
    exporters_list = data['exporter'].unique().tolist()
    for e in exporters_list:
        link_dict[i].append(e)

for e in import_by_country.exporter:
    data = import_by_country[import_by_country.exporter == e]
    importers_list = data['importer'].unique().tolist()
    for i in importers_list:
        link_dict[e].append(i)

for i in import_by_country.importer:
    data = import_by_country[import_by_country.importer == i]
    exporters_list = data['exporter'].unique().tolist()
    for e in exporters_list:
        link_dict[i].append(e)

for key in link_dict.keys():
    link_dict[key] = len(list(dict.fromkeys(link_dict[key])))

# PYGRAPHVIZ LAYOUT *************************************************************

df = df_links[
    (df_links['supplier_rank_source'].isin(list(np.arange(n_links + 1))[1:])) |
    (df_links['supplier_rank_target'].isin(list(np.arange(n_links +
                                                          1))[1:]))].copy()

layout_dict = dict()

for s in df['source'].unique().tolist():
    s_list = df[df['source'] == s]['target'].unique().tolist()
    layout_dict[s] = s_list

G = pgv.AGraph(layout_dict)

for i, node in enumerate(G.iternodes()):
    node.attr['shape'] = 'circle'
    node.attr['width'] = diameter_dict[node]
    node.attr['height'] = diameter_dict[node]
    node.attr['fixedsize'] = True
    node.attr['fontsize'] = 1

G.layout(prog='fdp')

graph_width = int(G.graph_attr['bb'].split(',')[2])
graph_height = int(G.graph_attr['bb'].split(',')[3])

# CYTOSCAPE LAYOUT **************************************************************

# Nodes

node_dict = dict()

for node in G.nodes():
    node_dict[node] = dict()
    node_dict[node]['x'] = float(node.attr['pos'].split(',')[0])
    node_dict[node]['y'] = float(node.attr['pos'].split(',')[1])
    node_dict[node]['diameter'] = float(node.attr['width']) * 63.5

node_data = pd.DataFrame.from_dict(node_dict, orient='index')

for dataset in [reporters, partners]:
    dataset['id'] = [
        '00' + x if len(x) == 1 else '0' + x if len(x) == 2 else x
        for x in dataset['id'].tolist()
    ]

country_codes = reporters.set_index('id')['text'].to_dict()
country_codes.update(partners.set_index('id')['text'].to_dict())
country_codes['380'] = 'Italy'

node_data['country'] = node_data.index.map(country_codes)

# Coloring parameters: UN Region

ccode_to_region_dict = dict()

regions = [
    'Africa', 'Oceania', 'Antarctica', 'Americas', 'Asia', 'Europe',
    'Special categories and unspecified areas'
]

ccode_lists = [
    [
        '012', '024', '072', '086', '108', '120', '132', '140', '148', '174',
        '175', '178', '180', '204', '226', '231', '232', '260', '262', '266',
        '270', '288', '324', '384', '404', '426', '430', '434', '450', '454',
        '466', '478', '480', '504', '508', '516', '562', '566', '577', '624',
        '638', '646', '654', '678', '686', '690', '694', '706', '710', '716',
        '728', '729', '732', '736', '748', '768', '788', '800', '818', '834',
        '854', '894'
    ],
    [
        '016', '036', '090', '162', '166', '184', '242', '258', '296', '316',
        '334', '520', '527', '540', '548', '554', '570', '574', '580', '581',
        '583', '584', '585', '598', '612', '772', '776', '798', '876', '882'
    ], ['010'],
    [
        '028', '032', '044', '052', '060', '068', '074', '076', '084', '092',
        '124', '136', '152', '170', '188', '192', '212', '214', '218', '222',
        '238', '239', '254', '304', '308', '312', '320', '328', '332', '340',
        '388', '473', '474', '484', '500', '531', '533', '534', '535', '558',
        '591', '600', '604', '630', '636', '637', '652', '659', '660', '662',
        '663', '666', '670', '740', '780', '796', '840', '842', '850', '858',
        '862'
    ],
    [
        '004', '031', '048', '050', '051', '064', '096', '104', '116', '144',
        '156', '196', '268', '275', '344', '356', '360', '364', '368', '376',
        '392', '398', '400', '408', '410', '414', '417', '418', '422', '446',
        '458', '462', '490', '496', '512', '524', '586', '608', '626', '634',
        '682', '699', '702', '704', '760', '762', '764', '784', '792', '795',
        '860', '887'
    ],
    [
        '008', '020', '040', '056', '070', '100', '112', '191', '203', '208',
        '233', '234', '246', '248', '250', '251', '276', '292', '300', '336',
        '348', '352', '372', '380', '428', '438', '440', '442', '470', '492',
        '498', '499', '528', '568', '578', '579', '616', '620', '642', '643',
        '674', '680', '688', '703', '705', '724', '744', '752', '756', '757',
        '804', '807', '826', '831', '832', '833'
    ], ['837', '838', '839', '899']
]

for r, c in zip(regions, ccode_lists):
    ccode_to_region_dict.update(dict.fromkeys(c, r))

node_data = node_data[node_data.index.isin(
    [item for sublist in ccode_lists for item in sublist])]

node_data['region'] = node_data.index.map(ccode_to_region_dict)

# Coloring parameters: Export/Import Balance

node_data = node_data.join(
    df_nodes[['export', 'import', 'export_by_partners', 'import_by_partners']])

node_data['export_for_val'] = [
    ex if ex != 0 else exp
    for ex, exp in zip(node_data['export'], node_data['export_by_partners'])
]
node_data['import_for_val'] = [
    im if im != 0 else imp
    for im, imp in zip(node_data['import'], node_data['import_by_partners'])
]
node_data['export_share_val'] = node_data['export_for_val'] / node_data[[
    'import_for_val', 'export_for_val'
]].sum(axis=1)

node_data['export_share'] = pd.cut(node_data['export_share_val'],
                                   list(np.linspace(0, 1, 11)),
                                   labels=[
                                       '0' +
                                       str(n) if len(str(n)) == 1 else str(n)
                                       for n in np.arange(1, 11, 1)
                                   ],
                                   include_lowest=True)

# Coloring parameters: Pagerank

node_data['pagerank_val'] = node_data.index.map(
    nx.pagerank(nx.from_pandas_edgelist(df_links, 'source', 'target')))

node_data['pagerank'] = pd.cut(node_data['pagerank_val'],
                               list(
                                   np.linspace(node_data['pagerank_val'].min(),
                                               node_data['pagerank_val'].max(),
                                               11)),
                               labels=[
                                   '0' + str(n) if len(str(n)) == 1 else str(n)
                                   for n in np.arange(1, 11, 1)
                               ],
                               include_lowest=True)

node_data['trade_links'] = node_data.index.map(link_dict)

node_data = node_data.drop(
    ['export_for_val', 'import_for_val', 'export_share_val', 'pagerank_val'],
    axis=1)

# Edges

sources = []
targets = []

for edge in G.edges():
    sources.append(edge[0])
    targets.append(edge[1])

edge_data = pd.DataFrame([sources, targets]).T
edge_data.columns = ['source', 'target']

for side in ['source', 'target']:
    edge_data['country_' + side] = edge_data[side].map(
        node_data['country'].drop_duplicates().to_dict())

for parameter in ['region', 'export_share', 'pagerank']:
    for side in ['source', 'target']:
        edge_data[parameter + '_' + side] = edge_data[side].map(
            node_data[parameter].to_dict())

# Labels

label_dict = dict()

label_dict_short = {
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Brunei Darussalam': 'Brunei',
    'Central African Rep.': 'CAR',
    'China, Hong Kong SAR': 'Hong Kong',
    'China, Macao SAR': 'Macao',
    "Dem. People's Rep. of Korea": 'North Korea',
    'Dem. Rep. of the Congo': 'Congo-Kinshasa',
    'Falkland Isds (Malvinas)': 'Falkland Isds',
    'Holy See (Vatican City State)': 'Holy See',
    "Lao People's Dem. Rep.": 'Laos',
    'Neth. Antilles and Aruba': 'Neth. Antilles',
    'Rep. of Korea': 'Korea',
    'Rep. of Moldova': 'Moldova',
    'Russian Federation': 'Russia',
    'Saint Barthelemy': 'St Barthelemy',
    'Saint Helena': 'St Helena',
    'Saint Lucia': 'St Lucia',
    'Saint Maarten': 'St Maarten',
    'Saint Kitts and Nevis': 'St Kitts and Nevis',
    'Saint Kitts, Nevis and Anguilla': 'St Kitts and Nevis',
    'Saint Pierre and Miquelon': 'St Pierre and Miquelon',
    'Saint Vincent and the Grenadines': 'St Vincent',
    'So. African Customs Union': 'SACU',
    'United Arab Emirates': 'UAE',
    'United Rep. of Tanzania': 'Tanzania',
    'USA (before 1981)': 'USA',
    'Africa CAMEU region, nes': 'Africa CAMEU, nes',
    'Br. Antarctic Terr.': 'Br. Antarctic',
    'Br. Indian Ocean Terr.': 'Br. Indian Ocean',
    'Fr. South Antarctic Terr.': 'Fr. South Antarctic',
    'Heard Island and McDonald Islands': 'Heard and McDonald Isds',
    'North America and Central America, nes': 'North and Centr America, nes',
    'South Georgia and the South Sandwich Islands': 'SGSSI',
    'United States Minor Outlying Islands': 'US Minor Outlying Isds'
}

for n in node_data['country'].sort_values().unique().tolist():
    if len(n) > 10 and n in label_dict_short.keys():
        label_dict[n] = label_dict_short[n]
    else:
        label_dict[n] = n

nlinks_filter = []

for index in node_data.index.unique().tolist():
    if node_data.loc[index]['trade_links'] >= 50:
        nlinks_filter.append(index)

max_diameter = node_data['diameter'].max()

node_data['label'] = [
    label_dict[country] if diameter >= max_diameter * 0.25
    or index in nlinks_filter else np.nan for country, diameter, index in zip(
        node_data['country'], node_data['diameter'], node_data.index)
]

node_data['label_selected'] = [
    label_dict[country] for country in node_data['country']
]

# COLORING **********************************************************************

region_color_dict = {
    'Antarctica': '#7E6EBD',
    'Africa': '#FB9038',
    'Asia': '#D1085C',
    'Europe': '#08B0D1',
    'Americas': '#d4e2e8',
    'Oceania': '#134DD1',
    'Special categories and unspecified areas': '#F9F871'
}

export_share_color_dict = {
    '01': '#08b4d1',
    '02': '#39c3da',
    '03': '#6bd2e3',
    '04': '#9ce1ed',
    '05': '#cef0f6',
    '06': '#ffe5cd',
    '07': '#ffcc9b',
    '08': '#ffb268',
    '09': '#ff9936',
    '10': '#ff7f04'
}

pagerank_color_dict = {
    '01': '#3e26a8',
    '02': '#4746eb',
    '03': '#3e70ff',
    '04': '#2797eb',
    '05': '#08b4d1',
    '06': '#32c69f',
    '07': '#81cc59',
    '08': '#dbbd28',
    '09': '#fcd030',
    '10': '#f9fb15'
}

background_color = '#010103'


def rgb_to_hex(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def hex_gradient_list(hex_color1, hex_color2, n_colors):

    assert n_colors > 1
    color1_rgb = np.array(hex_to_rgb(hex_color1)) / 255
    color2_rgb = np.array(hex_to_rgb(hex_color2)) / 255
    ordered = np.linspace(0, 1, n_colors)
    gradient = [
        list(((1 - order) * color1_rgb + (order * color2_rgb)))
        for order in ordered
    ]
    gradient_transformed = [[int(round(val * 255)) for val in color]
                            for color in gradient]
    return [rgb_to_hex(color) for color in gradient_transformed]


def hex_gradient_str(hex_color1, hex_color2, n_colors):

    assert n_colors > 1
    color1_rgb = np.array(hex_to_rgb(hex_color1)) / 255
    color2_rgb = np.array(hex_to_rgb(hex_color2)) / 255
    ordered = np.linspace(0, 1, n_colors)
    gradient = [
        list(((1 - order) * color1_rgb + (order * color2_rgb)))
        for order in ordered
    ]
    gradient_transformed = [[int(round(val * 255)) for val in color]
                            for color in gradient]
    return " ".join([rgb_to_hex(color) for color in gradient_transformed])


# Node Edges

node_data['region_edge_color'] = node_data['region'].map(region_color_dict)
node_data['export_share_edge_color'] = node_data['export_share'].map(
    export_share_color_dict)
node_data['pagerank_edge_color'] = node_data['pagerank'].map(
    pagerank_color_dict)

# Nodes

node_data['region_color'] = [
    hex_gradient_list(color, background_color, 3)[1]
    for color in node_data['region_edge_color']
]
node_data['export_share_color'] = [
    hex_gradient_list(color, background_color, 3)[1]
    for color in node_data['export_share_edge_color']
]
node_data['pagerank_color'] = [
    hex_gradient_list(color, background_color, 3)[1]
    for color in node_data['pagerank_edge_color']
]

# Chosen Nodes

node_data['chosen_region_color'] = [
    hex_gradient_list(color, '#ffffff', 7)[3]
    for color in node_data['region_color']
]

node_data['chosen_export_share_color'] = [
    hex_gradient_list(color, '#ffffff', 7)[3]
    for color in node_data['export_share_color']
]

node_data['chosen_pagerank_color'] = [
    hex_gradient_list(color, '#ffffff', 7)[3]
    for color in node_data['pagerank_color']
]

# Node Labels

node_data['region_label_color'] = [
    hex_gradient_list(color, '#ffffff', 5)[3]
    for color in node_data['region_edge_color']
]
node_data['export_share_label_color'] = [
    hex_gradient_list(color, '#ffffff', 5)[3]
    for color in node_data['export_share_edge_color']
]
node_data['pagerank_label_color'] = [
    hex_gradient_list(color, '#ffffff', 5)[3]
    for color in node_data['pagerank_edge_color']
]

# Chosen Node Labels

node_data['chosen_region_label_color'] = [
    hex_gradient_list(color, '#ffffff', 6)[5]
    for color in node_data['region_edge_color']
]
node_data['chosen_export_share_label_color'] = [
    hex_gradient_list(color, '#ffffff', 6)[5]
    for color in node_data['export_share_edge_color']
]
node_data['chosen_pagerank_label_color'] = [
    hex_gradient_list(color, '#ffffff', 6)[5]
    for color in node_data['pagerank_edge_color']
]

# Edges

for parameter, d in zip(
    ['region', 'export_share', 'pagerank'],
    [region_color_dict, export_share_color_dict, pagerank_color_dict]):
    for side in ['source', 'target']:
        edge_data[parameter + '_color_' + side] = edge_data[parameter + '_' +
                                                            side].map(d)

for parameter in ['region', 'export_share', 'pagerank']:
    edge_data[parameter + '_colors_source'] = [
        hex_gradient_list(background_color, color, 11)[2]
        for color in edge_data[parameter + '_color_source']
    ]
    edge_data[parameter + '_colors_target'] = [
        hex_gradient_list(background_color, color, 11)[7]
        for color in edge_data[parameter + '_color_target']
    ]

for parameter in ['region', 'export_share', 'pagerank']:
    edge_data[parameter + '_colors'] = [
        hex_gradient_str(color_source, color_target, 10)
        for color_source, color_target in zip(
            edge_data[parameter +
                      '_colors_source'], edge_data[parameter +
                                                   '_colors_target'])
    ]

# BUILDING THE GRAPH ************************************************************

color_dict = {
    'region': {
        'color': 'region_color',
        'chosen_color': 'chosen_region_color',
        'label_color': 'region_label_color',
        'chosen_label_color': 'chosen_region_label_color',
        'edge_color': 'region_edge_color',
        'colors': 'region_colors'
    },
    'export_share': {
        'color': 'export_share_color',
        'chosen_color': 'chosen_export_share_color',
        'label_color': 'export_share_label_color',
        'chosen_label_color': 'chosen_export_share_label_color',
        'edge_color': 'export_share_edge_color',
        'colors': 'export_share_colors'
    },
    'pagerank': {
        'color': 'pagerank_color',
        'chosen_color': 'chosen_pagerank_color',
        'label_color': 'pagerank_label_color',
        'chosen_label_color': 'chosen_pagerank_label_color',
        'edge_color': 'pagerank_edge_color',
        'colors': 'pagerank_colors'
    }
}

elements = []

# Node parameters and positions

for i in node_data.index:
    el_dict = dict()
    el_dict['data'] = dict()
    el_dict['position'] = dict()

    el_dict['data']['id'] = i
    el_dict['data']['label'] = node_data.loc[i]['label']
    el_dict['data']['label_selected'] = node_data.loc[i]['label_selected']
    el_dict['data']['color'] = node_data.loc[i][color_dict[chosen_color_parameter]['label_color']]
    el_dict['data']['color_chosen'] = node_data.loc[i][color_dict[chosen_color_parameter][
        'chosen_label_color']]
    el_dict['data']['background_color'] = hex_gradient_list(
        node_data.loc[i][color_dict[chosen_color_parameter]['color']], background_color,
        4)[2] + ' ' + node_data.loc[i][color_dict[chosen_color_parameter]['color']] + ' ' + node_data.loc[
            i][color_dict[chosen_color_parameter]['edge_color']]
    el_dict['data']['background_color_chosen'] = hex_gradient_list(
        node_data.loc[i][color_dict[chosen_color_parameter]['chosen_color']], background_color, 4
    )[2] + ' ' + node_data.loc[i][color_dict[chosen_color_parameter]['edge_color']] + ' ' + node_data.loc[
        i][color_dict[chosen_color_parameter]['edge_color']]
    el_dict['data']['border_color'] = node_data.loc[i][color_dict[chosen_color_parameter]['edge_color']]
    el_dict['data']['size'] = node_data.loc[i]['diameter']
    el_dict['data']['opacity'] = 0.75
    el_dict['data']['font_size'] = 40

    el_dict['position']['x'] = node_data.loc[i]['x']
    el_dict['position']['y'] = node_data.loc[i]['y']

    elements.append(el_dict)

# Edge parameters

for i in edge_data[['source', 'target']].drop_duplicates().index:
    el_dict_e = dict()
    el_dict_e['data'] = dict()
    el_dict_e['data'][
        'id'] = edge_data.loc[i]['source'] + '_' + edge_data.loc[i]['target']
    el_dict_e['data']['source'] = edge_data.loc[i]['source']
    el_dict_e['data']['target'] = edge_data.loc[i]['target']
    el_dict_e['data']['colors'] = edge_data.loc[i][color_dict[chosen_color_parameter]['colors']]
    elements.append(el_dict_e)

# Final layout

app = Dash(__name__)

default_stylesheet = [{
    'selector': 'node',
    'style': {
        'label': 'data(label)',
        "background-fill": "radial-gradient",
        "background-gradient-stop-colors": 'data(background_color)',
        "background-gradient-stop-positions": '0, 80, 90, 100',
        'color': 'data(color)',
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': 'data(font_size)',
        'border-color': 'data(border_color)',
        'border-width': 1.5,
        "border-opacity": 1,
        'width': 'data(size)',
        'height': 'data(size)',
        'opacity': 0.98
    }
}, {
    'selector': 'edge',
    'style': {
        "line-fill": "linear-gradient",
        "line-gradient-stop-colors": 'data(colors)',
        "line-gradient-stop-positions": "10, 20, 30, 40, 50, 60, 70, 80, 90",
        'width': 2.5,
        'curve-style': 'bezier',
        'source-endpoint': 'outside-to-node',
        'target-endpoint': 'outside-to-node'
    }
}]

styles = {
    'Lab': {
        'height': '100%',
        'color': '#ffffff',
        'font-family': 'Courier New',
        'font-size': '80%',
        'padding-top': '10%',
        'padding-bottom': '10%',
        'padding-left': '10%'
    },
    'Output': {
        'height': '100%',
        'color': '#08B0D1',
        'font-family': 'Courier New',
        'font-size': '80%',
        'padding-top': '16%',
        'padding-bottom': '10%',
        'padding-left': '10%'
    },
    'Button': {
        'height': '100%',
        'width': '100%',
        'color': '#ffffff',
        'background-color': '#010103',
        'border-color': '#010103',
        'border-width': 0,
        'padding-left': '20%',
        'font-family': 'Courier New',
        'font-size': '80%',
        'text-align': 'left',
        'cursor': 'pointer'
    }
}

app.layout = dbc.Container([
    html.Div([
        html.Div([
            html.Div(html.P('Country Hovered:', style=styles['Lab']),
                     style={
                         'width': '15%',
                         'height': '100%',
                         'margin-left': '2%'
                     }),
            html.Div(html.Div(id='mouseoverNodeData', style=styles['Output']),
                     style={'width': '25%'}),
            html.Div(html.P('Country Highlighted:', style=styles['Lab']),
                     style={
                         'width': '15%',
                         'height': '100%',
                         'margin-left': '2%'
                     }),
            html.Div(html.Div(id='output-country', style=styles['Output']),
                     style={'width': '25%'}),
            html.Div(html.Button(
                "Save PNG", id="btn-get-png", style=styles['Button']),
                     style={
                         'width': '20%',
                         'background-color': '#010103',
                         'margin-left': '2%'
                     })
        ],
                 style={
                     'width': '100%',
                     'display': 'flex'
                 }),
        html.Div([
            cyto.Cytoscape(id='cytoscape',
                           layout={'name': 'cola'},
                           style={
                               'width': str(pic_width) + 'px',
                               'height':
                               str(graph_height * pic_width / graph_width) + 'px',
                               'background-color': '#010103'
                           },
                           elements=elements,
                           stylesheet=default_stylesheet)
        ],
                 style={
                     'width': '100%',
                     'background-color': '#010103'
                 })
    ],
             style={
                 'display': 'inline-block',
                 'width': str(pic_width) + 'px'
             })
],
                           style={
                               'display': 'flex',
                               'justify-content': 'center',
                               'margin-bottom': '50px'
                           })


@app.callback(Output('mouseoverNodeData', 'children'),
              Input('cytoscape', 'mouseoverNodeData'))
def display_hover_data(data):
    if data:
        return data['label_selected']


@app.callback(
    [Output('cytoscape', 'stylesheet'),
     Output('output-country', 'children')],
    [Input('cytoscape', 'tapNode'),
     Input('cytoscape', 'selectedNodeData')])
def generate_stylesheet(node, data_list):

    if not data_list:
        return default_stylesheet, 'No country selected'

    elif node:
        node_id = node['data']['id']

        stylesheet = [{
            "selector": 'node',
            'style': {
                'label': 'data(label)',
                "background-fill": "radial-gradient",
                "background-gradient-stop-colors": 'data(background_color)',
                "background-gradient-stop-positions": "0, 80, 90, 100",
                'color': 'data(color)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': 'data(font_size)',
                'border-color': 'data(border_color)',
                'border-width': 1.5,
                "border-opacity": 1,
                'width': 'data(size)',
                'height': 'data(size)',
                'opacity': 0.3
            }
        }, {
            'selector': 'edge',
            'style': {
                "line-fill": "linear-gradient",
                "line-gradient-stop-colors": 'data(colors)',
                "line-gradient-stop-positions":
                "10, 20, 30, 40, 50, 60, 70, 80, 90",
                'width': 2,
                'curve-style': 'bezier',
                'source-endpoint': 'outside-to-node',
                'target-endpoint': 'outside-to-node',
                'opacity': 0.2
            }
        }, {
            "selector": 'node[id = "{}"]'.format(node_id),
            "style": {
                'label': 'data(label_selected)',
                "background-fill": "radial-gradient",
                "background-gradient-stop-colors":
                'data(background_color_chosen)',
                "background-gradient-stop-positions": "0, 98, 99, 100",
                'color': 'data(color_chosen)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': 'data(font_size)',
                'border-color': 'data(border_color)',
                'border-width': 1.5,
                "border-opacity": 1,
                'width': 'data(size)',
                'height': 'data(size)',
                'opacity': 0.98,
                'z-index': 9999
            }
        }]

        for edge in node["edgesData"]:
            if edge['source'] == node_id:
                stylesheet.append({
                    "selector":
                    'node[id = "{}"]'.format(edge['target']),
                    "style": {
                        'label': 'data(label)',
                        "background-fill": "radial-gradient",
                        "background-gradient-stop-colors":
                        'data(background_color)',
                        "background-gradient-stop-positions": "0, 80, 90, 100",
                        'color': 'data(color)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': 'data(font_size)',
                        'border-color': 'data(border_color)',
                        'border-width': 1.5,
                        "border-opacity": 1,
                        'width': 'data(size)',
                        'height': 'data(size)',
                        'opacity': 0.98
                    }
                })
                stylesheet.append({
                    "selector":
                    'edge[id= "{}"]'.format(edge['id']),
                    "style": {
                        "line-fill": "linear-gradient",
                        "line-gradient-stop-colors": 'data(colors)',
                        "line-gradient-stop-positions":
                        "10, 20, 30, 40, 50, 60, 70, 80, 90",
                        'width': 7,
                        'curve-style': 'bezier',
                        'source-endpoint': 'outside-to-node',
                        'target-endpoint': 'outside-to-node',
                        'opacity': 0.98,
                        'z-index': 5000
                    }
                })

            if edge['target'] == node_id:
                stylesheet.append({
                    "selector":
                    'node[id = "{}"]'.format(edge['source']),
                    "style": {
                        'label': 'data(label)',
                        "background-fill": "radial-gradient",
                        "background-gradient-stop-colors":
                        'data(background_color)',
                        "background-gradient-stop-positions": "0, 80, 90, 100",
                        'color': 'data(color)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': 'data(font_size)',
                        'border-color': 'data(border_color)',
                        'border-width': 1.5,
                        "border-opacity": 1,
                        'width': 'data(size)',
                        'height': 'data(size)',
                        'opacity': 0.98,
                        'z-index': 9999
                    }
                })
                stylesheet.append({
                    "selector":
                    'edge[id= "{}"]'.format(edge['id']),
                    "style": {
                        "line-fill": "linear-gradient",
                        "line-gradient-stop-colors": 'data(colors)',
                        "line-gradient-stop-positions":
                        "10, 20, 30, 40, 50, 60, 70, 80, 90",
                        'width': 6,
                        'curve-style': 'bezier',
                        'source-endpoint': 'outside-to-node',
                        'target-endpoint': 'outside-to-node',
                        'opacity': 0.98,
                        'z-index': 5000
                    }
                })

        return stylesheet, node['data']['label_selected']


@callback(Output("cytoscape", "generateImage"),
          Input("btn-get-png", "n_clicks"),
          prevent_initial_call=True)
def get_image(get_png_clicks):

    if ctx.triggered_id == 'btn-get-png':
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return {
            'type': "png",
            'action': "download",
            'options': {
                'bg': '#010103'
            },
            'filename': f'{output_pic_name}_{now}'
        }


if __name__ == "__main__":
    app.run_server(debug=True)