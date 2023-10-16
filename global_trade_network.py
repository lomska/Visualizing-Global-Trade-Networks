# IMPORTING PACKAGES ************************************************************

import pandas as pd
import numpy as np
import math

import comtradeapicall
import time

import pygraphviz as pgv

import networkx as nx

import plotly.graph_objects as go
from plotly.colors import hex_to_rgb
import ast

import warnings

warnings.filterwarnings('ignore')

# COMTRADE SUBSCRIPTION KEY *****************************************************

subscription_key = "YOUR_SUBSCRIPTION_KEY"

# CODES *************************************************************************

commodities = pd.read_csv('harmonized-system.csv')

reporters = pd.read_csv('reporterAreas.csv')
partners = pd.read_csv('partnerAreas.csv')

# EXTRACTING THE DATA ***********************************************************

comtrade_exp = comtradeapicall.getFinalData(subscription_key,
                                            typeCode='C',
                                            freqCode='A',
                                            clCode='HS',
                                            period='2022',
                                            reporterCode=None,
                                            cmdCode='070920',
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
                                            cmdCode='070920',
                                            flowCode='M',
                                            partnerCode=None,
                                            partner2Code='0',
                                            customsCode='C00',
                                            motCode='0',
                                            maxRecords=250000)

comtrade_imp = comtrade_imp[
    comtrade_imp['reporterCode'] != comtrade_imp['partnerCode']]

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
        'according to partners' if math.isnan(x) else ''
        for x in total_trade[col].tolist()
    ]

imports_to_add = export_by_country[['exporter', 'importer',
                                    'value']].groupby('importer').agg('sum')
exports_to_add = import_by_country[['importer', 'exporter',
                                    'value']].groupby('exporter').agg('sum')

trade_to_add = exports_to_add.rename(columns={
    'value': 'export'
}).join(imports_to_add.rename(columns={'value': 'import'}), how='outer')

df_nodes = total_trade.combine_first(trade_to_add)[[
    'export', 'import', 'export_note', 'import_note'
]]
df_nodes[['export', 'import']] = df_nodes[['export', 'import']].fillna(0)

df_nodes[['export_note',
          'import_note']] = df_nodes[['export_note', 'import_note'
                                      ]].fillna('according to partners')

df_nodes['trade'] = df_nodes[['export', 'import']].sum(axis=1)

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

# PYGRAPHVIZ LAYOUT *************************************************************

n_links = 2

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

# PLOTLY FIGURE: EDGES DATA *****************************************************

edge_dict = dict()

for edge in G.edges():
    edge_dict[edge] = dict()
    edge_dict[edge]['x0'] = float(edge.attr['pos'].split(' ')[0].split(',')[0])
    edge_dict[edge]['y0'] = float(edge.attr['pos'].split(' ')[0].split(',')[1])
    edge_dict[edge]['x1'] = float(edge.attr['pos'].split(' ')[3].split(',')[0])
    edge_dict[edge]['y1'] = float(edge.attr['pos'].split(' ')[3].split(',')[1])

edge_data = pd.DataFrame.from_dict(edge_dict, orient='index')
edge_data.index = edge_data.index.rename(['source', 'target'])

edge_data['length_area_ratio'] = [
    math.dist([x0, y0], [x1, y1]) / np.sqrt(graph_width**2 + graph_height**2)
    for x0, y0, x1, y1 in zip(edge_data['x0'], edge_data['y0'],
                              edge_data['x1'], edge_data['y1'])
]

edge_data['x_points_list'] = [
    [x for i in np.linspace(x0, x1, 11) for x in [i] *
     5][4:] if ratio < 0.006229 else list(np.linspace(x0, x1, 51))
    for x0, x1, ratio in zip(edge_data['x0'], edge_data['x1'],
                             edge_data['length_area_ratio'])
]

edge_data['y_points_list'] = [
    [y for i in np.linspace(y0, y1, 11) for y in [i] *
     5][4:] if ratio < 0.006229 else list(np.linspace(y0, y1, 51))
    for y0, y1, ratio in zip(edge_data['y0'], edge_data['y1'],
                             edge_data['length_area_ratio'])
]

edge_data['x_lines_list'] = [[
    tuple([x_points_list[i], x_points_list[i + 1], None])
    for i in range(len(x_points_list) - 1)
] for x_points_list in edge_data['x_points_list']]

edge_data['y_lines_list'] = [[
    tuple([y_points_list[i], y_points_list[i + 1], None])
    for i in range(len(y_points_list) - 1)
] for y_points_list in edge_data['y_points_list']]

edge_data['path'] = [np.arange(0, 50, 1)] * len(edge_data.index)

edge_data = edge_data.explode(['x_lines_list', 'y_lines_list', 'path'])

edge_data['rows_to_keep'] = [
    1 if ratio >= 0.006229 or
    (ratio < 0.006229 and path in np.arange(0, 50, 5)) else 0
    for ratio, path in zip(edge_data['length_area_ratio'], edge_data['path'])
]

edge_data = edge_data[edge_data['rows_to_keep'] == 1][[
    'x_lines_list', 'y_lines_list', 'path'
]]

width_dict = dict()
for path, width in zip(edge_data['path'].sort_values().unique().tolist(),
                       list(np.linspace(0.5, 2.5, 50))):
    width_dict[path] = width
edge_data['link_width'] = edge_data['path'].map(width_dict)

edge_data = edge_data.reset_index()

# PLOTLY FIGURE: NODES DATA *****************************************************

node_dict = dict()

for node in G.nodes():
    node_dict[node] = dict()
    node_dict[node]['x'] = float(node.attr['pos'].split(',')[0])
    node_dict[node]['y'] = float(node.attr['pos'].split(',')[1])
    node_dict[node]['diameter'] = float(node.attr['width']) * 63.5

node_data = pd.DataFrame.from_dict(node_dict, orient='index')

node_data = node_data.join(
    df_nodes[['export', 'import', 'export_note', 'import_note']])

for dataset in [reporters, partners]:
    dataset['id'] = [
        '00' + x if len(x) == 1 else '0' + x if len(x) == 2 else x
        for x in dataset['id'].tolist()
    ]

country_codes = reporters.set_index('id')['text'].to_dict()
country_codes.update(partners.set_index('id')['text'].to_dict())
country_codes['380'] = 'Italy'

node_data['country'] = node_data.index.map(country_codes)

# PLOTLY FIGURE: NODE LABELS DATA ***********************************************

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
    targets = edge_data[edge_data['source'] == index]['target'].to_list()
    sources = edge_data[edge_data['target'] == index]['source'].to_list()
    if len(set(targets + sources)) >= 10:
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

# PLOTLY FIGURE: TOOLTIPS DATA **************************************************

for col in ['export', 'import']:
    node_data[col + '_t'] = [
        x / 1000000000 if x / 1000000000 >= 1 else x /
        1000000 if x / 1000000 >= 1 else x / 1000 if x / 1000 >= 1 else x
        for x in node_data[col].tolist()
    ]
    node_data[col + '_t_symbol'] = [
        'B' if x / 1000000000 >= 1 else
        'M' if x / 1000000 >= 1 else 'K' if x / 1000 >= 1 else ''
        for x in node_data[col].tolist()
    ]

top_suppliers_df = rank_df[rank_df['supplier_rank'].isin(
    list(np.arange(n_links + 1))[1:])].set_index('importer')
top_suppliers_df['importer_country'] = top_suppliers_df.index.map(
    country_codes)
top_suppliers_df['exporter_country'] = top_suppliers_df['exporter'].map(
    country_codes)

top_suppliers_dict = dict()
top_supplier_to_dict = dict()

for country in node_data.index.unique().tolist():

    if country in top_suppliers_df.index:
        try:
            top_suppliers = top_suppliers_df.loc[country][
                'exporter_country'].tolist()
        except:
            top_suppliers = [top_suppliers_df.loc[country]['exporter_country']]
        top_suppliers.sort()
        top_suppliers_str = ', '.join(top_suppliers)
        top_suppliers_dict[country] = top_suppliers_str

    else:
        top_suppliers_dict[country] = '-'

    if country in top_suppliers_df.exporter.tolist():
        try:
            top_supplier_to = top_suppliers_df[
                top_suppliers_df['exporter'] ==
                country]['importer_country'].tolist()
        except:
            top_suppliers = [
                top_suppliers_df[top_suppliers_df['exporter'] == country]
                ['importer_country']
            ]
        top_supplier_to.sort()

        # ************* For Plotly only: splitting the rows ***************

        cumlenths = np.cumsum(
            [len(top_supplier_to[i]) for i in range(len(top_supplier_to))])

        threshold_list = []
        n = 35

        while True:
            threshold = next((i for i in cumlenths if i > n), None)
            if threshold != None:
                threshold_list.append(threshold)
                n = threshold + 35
            else:
                break

        for i in range(len(cumlenths)):
            if cumlenths[i] in threshold_list and len(cumlenths) > (i + 1):
                top_supplier_to[i + 1] = '<br>' + top_supplier_to[i + 1]

        # *****************************************************************

        top_supplier_to_str = '<br>' + ', '.join(top_supplier_to)

        top_supplier_to_dict[country] = top_supplier_to_str

    else:
        top_supplier_to_dict[country] = '-'

node_data['top_suppliers'] = node_data.index.map(top_suppliers_dict)
node_data['top_supplier_to'] = node_data.index.map(top_supplier_to_dict)

node_data['export_note'] = [
    '' if export_t == 0 else export_note for export_t, export_note in zip(
        node_data['export_t'], node_data['export_note'])
]

node_data['import_note'] = [
    '' if import_t == 0 else import_note for import_t, import_note in zip(
        node_data['import_t'], node_data['import_note'])
]

for side in ['source', 'target']:
    edge_data['country_' + side] = edge_data[side].map(
        node_data['country'].drop_duplicates().to_dict())

for side in ['source', 'target']:
    edge_data['country_' + side + '_t'] = [
        (tuple([x] * 2) + tuple([None])) for x in edge_data['country_' + side]
    ]

# COLORING PARAMETERS ***********************************************************

ccode_to_region_dict = dict()

regions = [
    'Africa', 'Oceania', 'Antarctica', 'Americas', 'Asia', 'Europe',
    'Special categories and unspecified areas'
]

ccode_lists = [[
    12, 24, 72, 86, 108, 120, 132, 140, 148, 174, 175, 178, 180, 204, 226, 231,
    232, 260, 262, 266, 270, 288, 324, 384, 404, 426, 430, 434, 450, 454, 466,
    478, 480, 504, 508, 516, 562, 566, 577, 624, 638, 646, 654, 678, 686, 690,
    694, 706, 710, 716, 728, 729, 732, 736, 748, 768, 788, 800, 818, 834, 854,
    894
],
               [
                   16, 36, 90, 162, 166, 184, 242, 258, 296, 316, 334, 520,
                   527, 540, 548, 554, 570, 574, 580, 581, 583, 584, 585, 598,
                   612, 772, 776, 798, 876, 882
               ], [10],
               [
                   28, 32, 44, 52, 60, 68, 74, 76, 84, 92, 124, 136, 152, 170,
                   188, 192, 212, 214, 218, 222, 238, 239, 254, 304, 308, 312,
                   320, 328, 332, 340, 388, 473, 474, 484, 500, 531, 533, 534,
                   535, 558, 591, 600, 604, 630, 636, 637, 652, 659, 660, 662,
                   663, 666, 670, 740, 780, 796, 840, 842, 850, 858, 862
               ],
               [
                   4, 31, 48, 50, 51, 64, 96, 104, 116, 144, 156, 196, 268,
                   275, 344, 356, 360, 364, 368, 376, 392, 398, 400, 408, 410,
                   414, 417, 418, 422, 446, 458, 462, 490, 496, 512, 524, 586,
                   608, 626, 634, 682, 699, 702, 704, 760, 762, 764, 784, 792,
                   795, 860, 887
               ],
               [
                   8, 20, 40, 56, 70, 100, 112, 191, 203, 208, 233, 234, 246,
                   248, 250, 251, 276, 292, 300, 336, 348, 352, 372, 380, 428,
                   438, 440, 442, 470, 492, 498, 499, 528, 568, 578, 579, 616,
                   620, 642, 643, 674, 680, 688, 703, 705, 724, 744, 752, 756,
                   757, 804, 807, 826, 831, 832, 833
               ], [837, 838, 839, 899]]

for i in range(len(ccode_lists)):
    ccode_lists[i] = [
        '00' + str(x) if len(str(x)) == 1 else '0' +
        str(x) if len(str(x)) == 2 else str(x) for x in ccode_lists[i]
    ]

for r, c in zip(regions, ccode_lists):
    ccode_to_region_dict.update(dict.fromkeys(c, r))

node_data = node_data[node_data.index.isin(
    [item for sublist in ccode_lists for item in sublist])]

node_data['region'] = node_data.index.map(ccode_to_region_dict)

node_data['export_share_val'] = node_data['export'] / node_data[[
    'import', 'export'
]].sum(axis=1)
node_data['export_share'] = pd.cut(node_data['export_share_val'],
                                   list(np.linspace(0, 1, 11)),
                                   labels=[
                                       '0' +
                                       str(n) if len(str(n)) == 1 else str(n)
                                       for n in np.arange(1, 11, 1)
                                   ],
                                   include_lowest=True)

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

edge_data = edge_data[(edge_data['source'].isin(
    [item for sublist in ccode_lists
     for item in sublist])) & (edge_data['target'].isin(
         [item for sublist in ccode_lists for item in sublist]))]

for parameter in ['region', 'export_share', 'pagerank']:
    for side in ['source', 'target']:
        edge_data[parameter + '_' + side] = edge_data[side].map(
            node_data[parameter].to_dict())

# GRADIENT FUNCTIONS ************************************************************


def hex_to_rgba_gradient(hex_color1, hex_color2, alpha1, alpha2, n_colors):

    assert n_colors > 1
    color1_rgb = np.array(hex_to_rgb(hex_color1)) / 255
    color2_rgb = np.array(hex_to_rgb(hex_color2)) / 255
    ordered = np.linspace(0, 1, n_colors)
    gradient = [((1 - order) * color1_rgb + (order * color2_rgb))
                for order in ordered]
    alphas = np.linspace(alpha1, alpha2, n_colors)
    return [
        'rgba' +
        str(tuple([int(round(val * 255)) for val in color]) + tuple([alpha]))
        for color, alpha in zip(gradient, alphas)
    ]


def rgba_to_rgba_gradient(rgba_color1, hex_color2, alpha1, alpha2, n_colors):

    assert n_colors > 1
    color1_rgb = np.array(ast.literal_eval(str(rgba_color1[4:]))[:3]) / 255
    color2_rgb = np.array(hex_to_rgb(hex_color2)) / 255
    ordered = np.linspace(0, 1, n_colors)
    gradient = [((1 - order) * color1_rgb + (order * color2_rgb))
                for order in ordered]
    alphas = np.linspace(alpha1, alpha2, n_colors)
    return [
        'rgba' +
        str(tuple([int(round(val * 255)) for val in color]) + tuple([alpha]))
        for color, alpha in zip(gradient, alphas)
    ]


# COLOR DICTIONARY CREATING FUNCTION ********************************************


def create_color_dict(region_dict, export_share_dict, pagerank_dict):

    color_dict = dict()

    for key in [
            'node_edge_color', 'node_color', 'node_label_color', 'link_color'
    ]:
        color_dict[key] = dict()

    color_dict['node_edge_color'] = {
        'region': region_dict,
        'export_share': export_share_dict,
        'pagerank': pagerank_dict
    }

    gradient_vals = [['node_color', '#000000', 3, 1],
                     ['node_label_color', '#ffffff', 5, 3]]

    for val in gradient_vals:

        for color_scheme in ['region', 'export_share', 'pagerank']:
            color_dict[val[0]][color_scheme] = dict()
            for parameter in color_dict['node_edge_color'][color_scheme].keys(
            ):
                new_color = hex_to_rgba_gradient(
                    color_dict['node_edge_color'][color_scheme][parameter],
                    val[1], 1, 1, val[2])[val[3]]
                color_dict[val[0]][color_scheme][parameter] = new_color

    for color_scheme in ['region', 'export_share', 'pagerank']:
        color_dict['link_color'][color_scheme] = dict()
        for parameter1 in color_dict['node_edge_color'][color_scheme].keys():
            for parameter2 in color_dict['node_edge_color'][color_scheme].keys(
            ):
                color_range = hex_to_rgba_gradient(
                    color_dict['node_edge_color'][color_scheme][parameter1],
                    color_dict['node_edge_color'][color_scheme][parameter2],
                    0.3, 0.8, 50)
                for i in range(50):
                    color_dict['link_color'][color_scheme][(
                        parameter1, parameter2, i)] = color_range[i]

    for color_scheme in ['region', 'export_share', 'pagerank']:
        for parameter in color_dict['node_edge_color'][color_scheme].keys():
            new_color = hex_to_rgba_gradient(
                color_dict['node_edge_color'][color_scheme][parameter],
                color_dict['node_edge_color'][color_scheme][parameter], 1, 1,
                3)[1]
            color_dict['node_edge_color'][color_scheme][parameter] = new_color

    return color_dict


# COLORSCALE CREATING FUNCTION **************************************************


def create_colorbars(color_dict):

    values_array = np.linspace(0, 1, 11)
    colors_array = color_dict['node_edge_color']['export_share'].values()
    values = [
        item for sublist in [[values_array[i], values_array[i + 1]]
                             for i in range(len(values_array) - 1)]
        for item in sublist
    ]
    colors = [
        item for sublist in [[x, x] for x in colors_array] for item in sublist
    ]
    export_share_colorscale = []
    for value, color in zip(values, colors):
        export_share_colorscale.append([value, color])

    pagerank_colorscale = []
    for value, color in zip(np.linspace(
            0, 1, 10), color_dict['node_edge_color']['pagerank'].values()):
        pagerank_colorscale.append([value, color])

    pagerank_colorbar = {
        'x': 0.5,
        'y': 0.98,
        'orientation': 'h',
        'lenmode': 'pixels',
        'len': 300,
        'outlinecolor': '#000000',
        'outlinewidth': 1,
        'showticklabels': True,
        'thickness': 3,
        'tickcolor': '#000000',
        'thicknessmode': 'pixels',
        'tickvals': [0, 5, 10],
        'ticklen': 1,
        'ticktext': ['min', 'trade links number', 'max'],
        'tickfont': {
            'color': '#ffffff',
            'family': 'Raleway',
            'size': 12
        }
    }

    export_share_colorbar = pagerank_colorbar.copy()

    export_share_colorbar.update({
        'ticks':
        'inside',
        'tickvals':
        list(np.linspace(0, 10, 11)),
        'ticklen':
        10,
        'tickwidth':
        2,
        'ticktext': [
            '100% importer', '', '', '', '', '   50|50%', '', '', '', '',
            '100% exporter'
        ]
    })

    colorscales_dict = {
        'region': None,
        'export_share': export_share_colorscale,
        'pagerank': pagerank_colorscale
    }

    colorbars_dict = {
        'region': None,
        'export_share': export_share_colorbar,
        'pagerank': pagerank_colorbar
    }

    return colorscales_dict, colorbars_dict


# PLOTLY GRAPH CREATING FUNCTION ************************************************


def colored_network(coloring_parameter, country_chosen, node_data, edge_data,
                    graph_width, graph_height):

    edge_data['country_chosen'] = [
        1 if country_source == country_chosen
        or country_target == country_chosen or pd.isna(country_chosen) else 0
        for country_source, country_target in zip(edge_data['country_source'],
                                                  edge_data['country_target'])
    ]

    countries = list(
        set(edge_data[edge_data['country_chosen'] ==
                      1]['country_source'].unique().tolist() +
            edge_data[edge_data['country_chosen'] ==
                      1]['country_target'].unique().tolist()))

    node_data['country_chosen'] = [
        1 if country in countries else 0 for country in node_data['country']
    ]

    for col in ['node_color', 'node_edge_color', 'node_label_color']:

        node_data[col] = node_data[coloring_parameter].map(
            color_dict[col][coloring_parameter])

        node_data[col] = [
            'rgba' + str(ast.literal_eval(str(n[4:]))[:3] + tuple([0.1]))
            if country_chosen == 0 else n for n, country_chosen in zip(
                node_data[col], node_data['country_chosen'])
        ]

    node_data['node_color'] = [
        rgba_to_rgba_gradient(c, '#ffffff', 1, 1, 7)[3]
        if country == country_chosen else c
        for c, country in zip(node_data['node_color'], node_data['country'])
    ]

    node_data['node_label_color'] = [
        rgba_to_rgba_gradient(cl, '#ffffff', 1, 1, 5)[1]
        if country == country_chosen else c for cl, c, country in zip(
            node_data['node_edge_color'], node_data['node_label_color'],
            node_data['country'])
    ]

    node_data['label_to_show'] = [
        ls if country == country_chosen else l
        for ls, country, l in zip(node_data['label_selected'],
                                  node_data['country'], node_data['label'])
    ]

    edge_data['link_color'] = edge_data.set_index([
        coloring_parameter + '_source', coloring_parameter + '_target', 'path'
    ]).index.map(color_dict['link_color'][coloring_parameter])

    edge_data['link_color'] = [
        'rgba' + str(ast.literal_eval(str(n[4:]))[:3] + tuple([0.1]))
        if country_chosen == 0 else n for n, country_chosen in zip(
            edge_data['link_color'], edge_data['country_chosen'])
    ]

    coef = 900 / graph_width

    fig = go.Figure()

    for parameter1 in edge_data[coloring_parameter +
                                '_source'].unique().tolist():
        for parameter2 in edge_data[coloring_parameter +
                                    '_target'].unique().tolist():
            for path in edge_data['path'].unique().tolist():
                for country_chosen in edge_data['country_chosen'].unique(
                ).tolist():
                    sample_df = edge_data[
                        (edge_data[coloring_parameter +
                                   '_source'] == parameter1)
                        & (edge_data[coloring_parameter +
                                     '_target'] == parameter2)
                        & (edge_data['path'] == path) &
                        (edge_data['country_chosen'] == country_chosen)]

                    if not sample_df.empty:
                        fig.add_scatter(
                            x=tuple(sample_df['x_lines_list'].sum(axis=0)),
                            y=tuple(sample_df['y_lines_list'].sum(axis=0)),
                            line=dict(width=sample_df['link_width'].values[0],
                                      color=sample_df['link_color'].values[0]),
                            hoverinfo='none',
                            mode='lines',
                            customdata=np.stack(
                                (tuple(
                                    sample_df['country_source_t'].sum(axis=0)),
                                 tuple(sample_df['country_target_t'].sum(
                                     axis=0))),
                                axis=-1),
                            hovertemplate=
                            '<extra></extra>%{customdata[0]} - %{customdata[1]}'
                        )

    for parameter in node_data[coloring_parameter].sort_values().unique(
    ).tolist():
        sample_df = node_data[node_data[coloring_parameter] == parameter]

        fig.add_scatter(
            x=tuple(sample_df['x']),
            y=tuple(sample_df['y']),
            mode='markers',
            marker=dict(color=sample_df['node_color'],
                        colorscale=colorscales_dict[coloring_parameter],
                        cmin=0,
                        cmax=10,
                        size=sample_df['diameter'] * coef,
                        gradient=dict(
                            color='rgba(0,0,0,1)',
                            type="radial",
                        ),
                        line=dict(width=1,
                                  color=sample_df['node_edge_color'])),
            customdata=np.stack(
                (sample_df['country'], sample_df['export_t'],
                 sample_df['export_t_symbol'], sample_df['export_note'],
                 sample_df['import_t'], sample_df['import_t_symbol'],
                 sample_df['import_note'], sample_df['top_suppliers'],
                 sample_df['top_supplier_to']),
                axis=-1),
            hovertemplate='<extra></extra><b>%{customdata[0]}</b>\
            <br><br><b>Exported:</b> $%{customdata[1]:,.1f} %{customdata[2]} %{customdata[3]}\
            <br><b>Imported:</b> $%{customdata[4]:,.1f} %{customdata[5]} %{customdata[6]}\
            <br><br><b>Key suppliers:</b> %{customdata[7]}\
            <br><br><b>Key supplier to:</b> %{customdata[8]}',
            name=parameter)

    fig.add_scatter(x=node_data['x'],
                    y=node_data['y'],
                    mode='text',
                    text=node_data['label_to_show'],
                    textfont=dict(family='Raleway',
                                  size=10,
                                  color=node_data['node_label_color']),
                    hoverinfo='text')

    for trace in fig['data']:
        if trace['name'] not in color_dict['node_edge_color']['region'].keys():
            trace['showlegend'] = False

        if trace['name'] == '01':
            trace['marker']['colorbar'] = colorbars_dict[coloring_parameter]

    fig.update_layout(paper_bgcolor='#000000',
                      plot_bgcolor='#000000',
                      width=graph_width * coef,
                      height=graph_height * coef * 1.05,
                      hovermode='closest',
                      hoverlabel=dict(font=dict(family='Raleway', size=12)),
                      margin=dict(b=0, l=0, r=0, t=graph_height * coef * 0.05),
                      xaxis=dict(showgrid=False,
                                 zeroline=False,
                                 showticklabels=False),
                      yaxis=dict(showgrid=False,
                                 zeroline=False,
                                 showticklabels=False),
                      legend=dict(orientation='h',
                                  xanchor='center',
                                  yanchor='middle',
                                  x=0.5,
                                  y=1.02,
                                  font=dict(size=12,
                                            family='Raleway',
                                            color='#ffffff'),
                                  itemsizing='constant'))

    return fig

# SETTING THE PALETTES **********************************************************

region_color_dict = {
    'Antarctica': '#7a0c02',
    'Africa': '#fe9b2d',
    'Asia': '#d2e934',
    'Europe': '#62fa6b',
    'Americas': '#1bd0d5',
    'Oceania': '#4777ef',
    'Special categories and unspecified areas': '#db3a07'
}

export_share_color_dict = {
    '01': '#008ea8',
    '02': '#35a5aa',
    '03': '#69bbab',
    '04': '#99ccba',
    '05': '#c9ddc9',
    '06': '#f4d481',
    '07': '#f0b841',
    '08': '#eb9c00',
    '09': '#e57800',
    '10': '#de5400'
}

pagerank_color_dict = {
    '01': '#472f7d',
    '02': '#414487',
    '03': '#31688e',
    '04': '#23888e',
    '05': '#1f988b',
    '06': '#22a884',
    '07': '#35b779',
    '08': '#7ad151',
    '09': '#d2e21b',
    '10': '#fde725'
}

color_dict = create_color_dict(region_color_dict, export_share_color_dict,
                               pagerank_color_dict)

colorscales_dict, colorbars_dict = create_colorbars(color_dict)

# BUILDING THE GRAPH ************************************************************

colored_network('export_share', np.nan, node_data, edge_data, graph_width,
                graph_height).show()