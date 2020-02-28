import networkx as nx
from networkx.algorithms import isomorphism
import math
import re

ORIG_PPI = [["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 234) ('type_1' 'type_1') ('type_1' 234) (197 234) (234 197)]", 844],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 478) ('type_1' 'type_1') ('type_1' 478) (197 234) (197 478) (234 197) (478 197)]", 73],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 234) ('type_1' 'type_1') ('type_1' 234) (197 234) (234 197) (234 338) (338 234)]", 53],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 234) ('type_1' 'type_1') ('type_1' 234) (197 234) (234 197) (234 738) (234 800) (738 234) (800 234)]", 25],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 328) ('type_1' 'type_1') ('type_1' 328) (197 478) (478 197) (478 479) (328 479) (479 328) (479 478)]", 4],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 314) ('type_1' 'type_1') ('type_1' 314) (114 314) (314 114) (314 331) (314 1120) (294 331) (331 294) (331 314)]", 11],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 314) ('type_1' 'type_1') ('type_1' 314) (114 314) (314 114) (314 1120)]", 14],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 275) ('type_1' 'type_1') ('type_1' 275) (275 334) (275 375) (275 589) (275 1069) (334 275) (375 275) (589 275) (1069 275)]", 21],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 314) ('type_1' 'type_1') ('type_1' 314) (314 1120)]", 60],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 558) ('type_1' 'type_1') ('type_1' 558) (558 559) (558 571) (558 654) (558 1328) (558 1329) (559 558) (571 558) (654 558) (1328 558) (1329 558)]", 36],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 239) ('type_1' 'type_1') ('type_1' 239) (239 1056) (239 1107) (239 1331) (1056 239)]", 12],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 43) ('type_1' 'type_1') ('type_1' 43) (42 43)]", 7],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 1059) ('type_0' 925) ('type_1' 'type_1') ('type_1' 1059) ('type_1' 925) (925 1185) (1185 925) (1185 1059) (1059 1185)]", 12],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 1440) ('type_0' 749) ('type_0' 1551) ('type_1' 'type_1') ('type_1' 1440) ('type_1' 749) ('type_1' 1551) (749 750) (750 749) (750 1440) (750 1551) (1440 750) (1551 750)]", 4],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 1) ('type_0' 5) ('type_1' 'type_1') ('type_1' 1) ('type_1' 5) (1 12) (1 58) (1 188) (12 1) (12 5) (58 1) (58 5) (58 609) (188 1) (188 5) (5 12) (5 58) (5 188) (5 609) (609 5) (609 58)]", 1]]

CL_PPI =   [["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 215) ('type_1' 'type_1') ('type_1' 215) (215 807)]", 682],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 215) ('type_1' 'type_1') ('type_1' 215) (215 437) (215 807)]", 15],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 602) ('type_1' 'type_1') ('type_1' 602) (215 807) (602 215)]", 13],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 215) ('type_1' 'type_1') ('type_1' 215) (1064 215)]", 488],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 770) ('type_1' 'type_1') ('type_1' 76) (76 770)]", 392],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 64) ('type_1' 'type_1') ('type_1' 64) (5 64) (64 5)]", 56],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 30) ('type_1' 'type_1') ('type_1' 30) (30 503) (503 30) (777 30)]", 1],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 49) ('type_1' 'type_1') ('type_1' 9) (9 49) (49 9)]", 12],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 10) ('type_1' 'type_1') ('type_1' 10) ('type_1' 6) (6 10) (10 6)]", 2],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 9) ('type_0' 10) ('type_1' 'type_1') ('type_1' 9) ('type_1' 10) (9 10)]", 2],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 10) ('type_0' 50) ('type_1' 'type_1') ('type_1' 50) (10 50) (50 10)]", 4],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 20) ('type_0' 244) ('type_1' 'type_1') ('type_1' 33) ('type_1' 6) (3 6) (3 8) (3 33) (3 244) (6 8) (6 20) (8 20) (8 33) (8 244) (33 3) (33 6) (33 20) (244 3) (244 8) (244 20) (244 33) (20 8)]", 1]]

ER_PPI =   [["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 1480) ('type_1' 'type_1') ('type_1' 1281) (1281 1480)]", 1303],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 1643) ('type_1' 'type_1') ('type_1' 1643) (1643 1281)]", 127],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 1437) ('type_1' 'type_1') ('type_1' 1437) (1004 1437)]", 155],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 581) ('type_1' 'type_1') ('type_1' 736) (581 736)]", 102],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 650) ('type_1' 'type_1') ('type_1' 537) (537 650) (650 537)]", 2],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 650) ('type_1' 'type_1') ('type_1' 650) (537 650) (650 537)]", 6],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 32) ('type_1' 'type_1') ('type_1' 170) (4 32) (4 252) (170 4) (170 32) (252 4) (252 170)]", 1],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 17) ('type_0' 74) ('type_1' 'type_1') ('type_1' 8) ('type_1' 13) (1 8) (1 13) (1 17) (3 1) (3 74) (8 3) (8 17) (13 1) (13 3) (17 74) (74 3)]", 1]]

ORIG_BLOGS = [["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 896) ('type_1' 'type_1') ('type_1' 446) (896 446)]", 323],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 896) ('type_1' 'type_1') ('type_1' 446) (896 446) (913 446)]", 6],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 515) ('type_1' 'type_1') ('type_1' 515) (515 446)]", 127],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 772) ('type_1' 'type_1') ('type_1' 772) (446 772)]", 226],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 652) ('type_1' 'type_1') ('type_1' 446) (446 652) (652 446)]", 88],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 1068) ('type_1' 'type_1') ('type_1' 163) (92 163) (1066 163) (1068 163)]", 3],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 887) ('type_1' 'type_1') ('type_1' 887) (885 887) (887 885)]", 155],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 987) ('type_1' 'type_1') ('type_1' 871) (871 987)]", 99],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 193) ('type_0' 265) ('type_1' 'type_1') ('type_1' 265) (193 265) (265 193)]", 39],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 193) ('type_1' 'type_1') ('type_1' 193) (193 383) (193 679) (193 680)]", 8],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 209) ('type_1' 'type_1') ('type_1' 209) ('type_1' 210) (209 210)]", 14],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 4) ('type_0' 293) ('type_1' 'type_1') ('type_1' 4) (293 4)]", 21],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 4) ('type_1' 'type_1') ('type_1' 4) (405 4) (657 4)]", 9],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 135) ('type_1' 'type_1') ('type_1' 317) ('type_1' 135) (135 317) (317 135)]", 13],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 106) ('type_0' 44) ('type_1' 'type_1') ('type_1' 106) ('type_1' 44) (44 106) (106 44)]", 11],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 107) ('type_0' 44) ('type_1' 'type_1') ('type_1' 107) (107 44)]", 16],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 512) ('type_1' 'type_1') ('type_1' 512) ('type_1' 511) (511 512)]", 12],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 426) ('type_0' 427) ('type_1' 'type_1') ('type_1' 426) ('type_1' 427) (426 427)]", 9],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 64) ('type_0' 42) ('type_0' 4) ('type_1' 'type_1') ('type_1' 64) ('type_1' 2) (2 4) (2 42) (4 42) (4 64) (42 2) (42 4) (42 64) (64 2) (64 4) (64 42)]", 1],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 47) ('type_0' 7) ('type_1' 'type_1') ('type_1' 3) ('type_1' 47) (1 3) (1 7) (1 47) (3 1) (3 7) (3 47) (7 1) (7 3) (47 7)]", 1]]


CL_BLOGS = [["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 208) ('type_1' 'type_1') ('type_1' 208) (208 257)]", 191],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 208) ('type_1' 'type_1') ('type_1' 208) (208 257) (776 208)]", 7],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 208) ('type_1' 'type_1') ('type_1' 514) (514 208)]", 237],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 520) ('type_1' 'type_1') ('type_1' 208) (520 208)]", 321],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 208) ('type_1' 'type_1') ('type_1' 208) (776 208)]", 345],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 114) ('type_1' 'type_1') ('type_1' 114) ('type_1' 1155) (114 1155)]", 5],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 128) ('type_1' 'type_1') ('type_1' 70) (70 128) (128 70)]", 50],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 274) ('type_1' 'type_1') ('type_1' 274) (45 274) (274 45)]", 26],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 14) ('type_1' 'type_1') ('type_1' 68) ('type_1' 14) (14 68) (68 14)]", 7],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 29) ('type_0' 14) ('type_1' 'type_1') ('type_1' 14) (14 29) (29 14)]", 10],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 29) ('type_0' 46) ('type_1' 'type_1') ('type_1' 29) (29 46)]", 6],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 29) ('type_0' 14) ('type_1' 'type_1') ('type_1' 29) ('type_1' 14) (14 29) (29 14)]", 4],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 17) ('type_0' 19) ('type_1' 'type_1') ('type_1' 17) ('type_1' 19) (19 17)]", 4],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 1) ('type_0' 12) ('type_0' 5) ('type_1' 'type_1') ('type_1' 1) ('type_1' 5) (1 12) (1 26) (12 1) (12 5) (12 26) (26 1) (26 12) (5 12)]", 1]]

ER_BLOGS = [["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 877) ('type_1' 'type_1') ('type_1' 1121) (877 1121)]", 54],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 1075) ('type_1' 'type_1') ('type_1' 877) (877 1075)]", 1032],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 1109) ('type_1' 'type_1') ('type_1' 1109) (877 1109)]", 48],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 1152) ('type_1' 'type_1') ('type_1' 1152) (1152 1117)]", 18],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 141) ('type_1' 'type_1') ('type_1' 141) (141 1080) (1080 141)]", 4],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 406) ('type_1' 'type_1') ('type_1' 540) (406 540) (540 406)]", 57],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_0' 73) ('type_0' 6) ('type_1' 'type_1') ('type_1' 73) (73 6)]", 5],\
            ["[('type_0' 'type_0') ('type_0' 'type_1') ('type_1' 'type_1') (1 7) (1 23) (5 1) (5 7) (7 23) (10 1) (10 8) (23 1) (23 7) (23 8) (23 10) (8 5) (8 10)]", 1]]

def graph_from_string(the_string):
    pieces = re.split('\[\(| |\) \(|\)\]', the_string)
    pieces = pieces[1:(len(pieces)-1)]
    nodes = set()
    for piece in pieces:
        nodes.add(piece)
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node)
    idx = 0
    while idx < len(pieces):
        G.add_edge(pieces[idx], pieces[idx + 1])
        idx += 2
    return G

def make_graph_list(the_list):
    return [[graph_from_string(x[0]), x[1]] for x in the_list]

def merge_graph_lists(list_a, list_b, list_c):
    main_list = [[x[0], x[1], 0, 0] for x in list_a]
    for graph_and_count_b in list_b:
        found_match = False
        for i in range(0, len(main_list)):
            gm = isomorphism.DiGraphMatcher(graph_and_count_b[0], main_list[i][0])
            if gm.is_isomorphic():
                main_list[i][2] = graph_and_count_b[1]
                found_match = True
                break
        if not found_match:
            main_list.append([graph_and_count_b[0], 0, graph_and_count_b[1], 0])

    for graph_and_count_c in list_c:
        found_match = False
        for main_gc in main_list:
            gm = isomorphism.DiGraphMatcher(graph_and_count_c[0], main_gc[0])
            if gm.is_isomorphic():
                main_gc[3] = graph_and_count_c[1]
                found_match = True
                break
        if not found_match:
            main_list.append([graph_and_count_c[0], 0, 0, graph_and_count_c[1]])
    return main_list

def merged_probabilities(merged_lists):
    sum_a = 0.0
    sum_b = 0.0
    sum_c = 0.0
    for list_element in merged_lists:
        sum_a += list_element[1] + 1.0
        sum_b += list_element[2] + 1.0
        sum_c += list_element[3] + 1.0
    return [[x[0], (x[1] + 1.0) / sum_a, (x[2] + 1.0) / sum_b, (x[3] + 1.0) / sum_c] for x in merged_lists]

def ratios(prob_list, idx_1, idx_2):
    ratio_list = [[x[0], x[idx_1] / x[idx_2] if x[idx_1] > x[idx_2] else -1.0 * x[idx_2] / x[idx_1]] for x in prob_list]
    ratio_list.sort(key=(lambda x: -1.0 * abs(x[1])))
    return ratio_list

def kl_contributions(prob_list, idx_1, idx_2):
    contributions = [[x[0], -1.0 * x[idx_1] * math.log(x[idx_2] / x[idx_1])] for x in prob_list]
    contributions.sort(key=(lambda x: -1.0 * x[1]))
    return contributions

def KL_divergence(kl_contributions):
    kl = 0.0
    for item in kl_contributions:
        kl += item[1]
    return kl

def display_kl_contributions(some_kl, title):
    print(title)
    for i in range(0, min(len(some_kl), 3)):
        edges = list(some_kl[i][0].edges())
        edges.sort()
        print("%s %s" % (edges, some_kl[i][1]))

orig_ppi_list = make_graph_list(ORIG_PPI)
cl_ppi_list = make_graph_list(CL_PPI)
er_ppi_list = make_graph_list(ER_PPI)
merged_list = merge_graph_lists(orig_ppi_list, cl_ppi_list, er_ppi_list)
probs_list = merged_probabilities(merged_list)
orig_cl_kl_contributions = kl_contributions(probs_list, 1, 2)
orig_er_kl_contributions = kl_contributions(probs_list, 1, 3)
display_kl_contributions(orig_cl_kl_contributions, "PPI: Original vs CL")
display_kl_contributions(orig_er_kl_contributions, "PPI: Original vs ER")
orig_cl_kl = KL_divergence(orig_cl_kl_contributions)
print("orig_cl_kl: %s" % orig_cl_kl)
orig_er_kl = KL_divergence(orig_er_kl_contributions)
print("orig_er_kl: %s" % orig_er_kl)

orig_blogs_list = make_graph_list(ORIG_BLOGS)
cl_blogs_list = make_graph_list(CL_BLOGS)
er_blogs_list = make_graph_list(ER_BLOGS)
merged_list = merge_graph_lists(orig_blogs_list, cl_blogs_list, er_blogs_list)
probs_list = merged_probabilities(merged_list)
orig_cl_kl_contributions = kl_contributions(probs_list, 1, 2)
orig_er_kl_contributions = kl_contributions(probs_list, 1, 3)
display_kl_contributions(orig_cl_kl_contributions, "Blogs: Original vs CL")
display_kl_contributions(orig_er_kl_contributions, "Blogs: Original vs ER")
orig_cl_kl = KL_divergence(orig_cl_kl_contributions)
print("orig_cl_kl: %s" % orig_cl_kl)
orig_er_kl = KL_divergence(orig_er_kl_contributions)
print("orig_er_kl: %s" % orig_er_kl)
