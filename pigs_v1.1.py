#!/usr/bin/python -u

import sys
import os
import optparse
import networkx as nx
import numpy as np
import random
import time
import tempfile
import shutil

from copy import deepcopy
from itertools import combinations
from scipy.stats import norm


class IBD_Seg:
    def __init__(self, idv1, idv1_hap, idv2, idv2_hap, chrom, start, end, lod):
        self.idv1 = idv1
        self.idv1_hap = idv1_hap
        self.idv2 = idv2
        self.idv2_hap = idv2_hap
        self.chrom = chrom
        self.start = start
        self.end = end
        self.lod = lod
        self.prob = 0.0


def main():
    parser = optparse.OptionParser()
    parser.add_option('-b', '--bgl_ibd_file', dest='bgl_ibd_file', action='store', type='string', default='bgl.ibd', help="Refined IBD output file, REQUIRED")
    parser.add_option('-t', '--lod_threshold', dest='lod_threshold', action='store', type='string', default='0.01', help="Minimum LOD threshold for an edge to be added to graph, DEFAULT: 0.01")
    parser.add_option('-p', '--prior_prob', dest='prior_prob', action='store', type='string', default='0.004', help="Prior probability, DEFAULT: 0.004")
    parser.add_option('-o', '--output_file', dest='output_file', action='store', type='string', default='pigs_output.txt', help="Output file path, DEFAULT: pigs_output.txt")
    parser.add_option('-m', '--tmp_dir', dest='tmp_dir', action='store', type='string', default='.', help="Temp directory to use, DEFAULT: ./")
    parser.add_option('-l', '--list_of_idvs', dest='list_of_idvs', action='store', type='string', default='idvs.txt', help="List of sample IDs, REQUIRED")
    parser.add_option('-s', '--sampling_time', dest='sampling_time', action='store', type='string', default='-1', help="Sampling time limit in seconds, DEFAULT: None")

    (options, args) = parser.parse_args()

    global const_global_lod_threshold
    const_global_lod_threshold = float(options.lod_threshold)

    working_dir, working_bgl_ibd_file = copy_needed_files(options.bgl_ibd_file, options.tmp_dir)

    # print "initializing graph"
    G, samples = initialize_graph(options.list_of_idvs)

    # assign probabilities to all ibd segments
    prior_prob = float(options.prior_prob)

    sample_ibd_graphs_with_mcmc(G, samples, prior_prob, options.bgl_ibd_file, options.output_file, float(options.sampling_time))

    shutil.rmtree(working_dir)


def copy_needed_files(bgl_ibd_file, temp_dir):
    working_dir = tempfile.mkdtemp(dir=temp_dir)

    working_bgl_ibd_file = working_dir + '/' + os.path.basename(bgl_ibd_file)
    shutil.copyfile(bgl_ibd_file, working_bgl_ibd_file)

    return working_dir, working_bgl_ibd_file


def sample_ibd_graphs_with_mcmc(G, samples, prior_prob, bgl_ibd_file, output_file, total_time=-1):
    G.remove_edges_from(G.edges())

    # file lines we are interested in
    ibd_segments_to_analyze = {}

    i_file = open(bgl_ibd_file, 'rb')
    for line in i_file:
        ibd_data = line.strip().split('\t')
        ibd_seg = create_ibd_segment(ibd_data)

        key = tuple(sorted([ibd_seg.idv1 + '_' + ibd_seg.idv1_hap, ibd_seg.idv2 + '_' + ibd_seg.idv2_hap]))

        ibd_segments_to_analyze[key] = ibd_seg

    i_file.close()

    assign_ibd_probabilities(samples, ibd_segments_to_analyze, prior_prob)

    add_edges_to_graph(G, ibd_segments_to_analyze)

    edge_probability_map = create_edge_probability_map(samples, ibd_segments_to_analyze, prior_prob)

    connected_components = nx.connected_component_subgraphs(G)

    for cc in connected_components:
        pre_existing_edges = cc.edges()
        if len(cc.nodes()) < 2:
            continue
        if len(cc.nodes()) < 10:
            perform_mcmc_on_graph(cc, edge_probability_map, total_time=total_time)

            edge_probabilities = nx.get_edge_attributes(cc, 'prob')

            output_updated_ibd_segments(ibd_segments_to_analyze, pre_existing_edges, edge_probabilities, prior_prob, output_file)
        else:
            perform_mcmc_on_graph(cc, edge_probability_map, total_time=total_time)

            edge_probabilities = nx.get_edge_attributes(cc, 'prob')

            output_updated_ibd_segments(ibd_segments_to_analyze, pre_existing_edges, edge_probabilities, prior_prob, output_file)


def output_updated_ibd_segments(ibd_segments, pre_existing_edges, edge_probabilities, prior_prob, output_file):
    # get maximum overlap of pre existing edges
    max_start = 0
    min_end = 0
    chrom = 0

    for edge in pre_existing_edges:
        key = tuple(sorted(edge))
        ibd_seg = ibd_segments[key]

        if chrom == 0:
            chrom = ibd_seg.chrom

        if max_start == 0 or ibd_seg.start > max_start:
            max_start = ibd_seg.start

        if min_end == 0 or ibd_seg.end < min_end:
            min_end = ibd_seg.end

    o_file = open(output_file, 'wb')

    for edge in edge_probabilities:
        key = tuple(sorted(edge))
        if key in ibd_segments:
            ibd_seg = ibd_segments[key]

            edge_prob = edge_probabilities[edge]

            og_prob = convert_lod_to_prob(prior_prob, ibd_seg.lod)

            row_to_write = [ibd_seg.idv1, ibd_seg.idv1_hap, ibd_seg.idv2, ibd_seg.idv2_hap, ibd_seg.chrom, ibd_seg.start, ibd_seg.end, ibd_seg.lod, og_prob, edge_prob, False]
            row_to_write = map(str, row_to_write)

            o_file.write('\t'.join(row_to_write) + '\n')

        else:
            n1 = key[0]
            n2 = key[1]

            idv1 = n1.split('_')[0]
            idv1_hap = n1.split('_')[1]

            idv2 = n2.split('_')[0]
            idv2_hap = n2.split('_')[1]

            start = max_start
            end = min_end

            edge_prob = edge_probabilities[edge]
            if edge_prob < 0:
                edge_prob = 0

            row_to_write = [idv1, idv1_hap, idv2, idv2_hap, chrom, start, end, 0, 0, edge_prob, True]
            row_to_write = map(str, row_to_write)

            o_file.write('\t'.join(row_to_write) + '\n')

    o_file.close()


def convert_lod_to_prob(prior_prob, lod):
    posterior_prob = 0.99
    if lod < 3:
        prior_odds = prior_prob / (1 - prior_prob)
        posterior_odds = prior_odds * (10 ** lod) 

        scale_constant = (prior_prob * (10 ** 3) / .997) - (prior_prob * (10 ** 3))
        penalty = 0
        if lod <= 1:
            penalty =  ((1 - lod) ** 3) / 7.0
        else:
            penalty = -0.015

        posterior_prob = posterior_odds / (scale_constant + posterior_odds)
        posterior_prob = posterior_prob + penalty

    return posterior_prob


def create_ibd_segment(ibd_data):
    idv1 = ibd_data[0]
    idv1_hap = ibd_data[1]
    idv2 = ibd_data[2]
    idv2_hap = ibd_data[3]
    chrom = ibd_data[4]
    start = int(ibd_data[5])
    end = int(ibd_data[6])
    lod = float(ibd_data[7])   
    
    ibd_seg = IBD_Seg(idv1, idv1_hap, idv2, idv2_hap, chrom, start, end, lod)

    return ibd_seg


def perform_mcmc_on_graph(graph, edge_probability_map, total_time=-1):
    print "mcmc on connected component"
    convergence_count_end = 5000
    seen_again_count = 0

    graph_probs = []
    graph_state_keys = []
    graph_states = {}

    all_edges, edge_selection_probs = assign_edge_probabilities_weights(graph, edge_probability_map)

    added_edge_list = []
    graph.remove_edges_from(graph.edges())
    for edge in combinations(graph.nodes(), r=2):
        key = tuple(sorted(edge))
        if edge_probability_map[key] >= np.log(0.99, dtype=np.float128):
            graph.add_edge(key[0], key[1])

            added_edge_list.append(key)

    summed_graph_probabilities = -np.inf
    summed_graph_probs_by_edge = {}
    for pair in combinations(graph.nodes(), r=2):
        summed_graph_probs_by_edge[tuple(sorted(pair))] = -np.inf

    # we have to start at a valid state whenever we have p=1 edges
    graph_is_valid = check_graph_validity(graph)
    while not graph_is_valid:
        edge = added_edge_list.pop(0)

        make_graph_valid(graph, edge, True, edge_probability_map)

        graph_is_valid = check_graph_validity(graph)
        
        # if we have a valid graph or nothing to make it more valid, then move on to the next one
        if graph_is_valid:
            break

    if check_graph_validity(graph):
        save_graph_info(graph, edge_probability_map, graph_probs, graph_state_keys, graph_states)
        summed_graph_probabilities, summed_graph_probs_by_edge = sum_graph_probabilities(graph, summed_graph_probabilities, summed_graph_probs_by_edge, edge_probability_map)

    convergence_count = 0

    old_summed_graph_probs_by_edge = deepcopy(summed_graph_probs_by_edge)

    # the only way to break out of this loop is by convergence or time limit
    start_time = time.time()
    current_time = 0
    while True: 
        current_time = time.time() - start_time
        if total_time > 0 and current_time > total_time:
            break

        random_node1 = None
        random_node2 = None

        edge_selection_index = np.where(np.random.multinomial(1, edge_selection_probs) == 1)[0][0]
        edge_selected = all_edges[edge_selection_index]

        random_node1 = edge_selected[0]
        random_node2 = edge_selected[1]

        # check if there is an edge
        edge_exists = graph.has_edge(random_node1, random_node2)

        key = tuple(sorted([random_node1, random_node2]))

        p = np.exp(edge_probability_map[key])
        # note that this p is in log space
        while p == 1.0:
            r_nodes = random.sample(graph.nodes(), k=2)

            random_node1 = r_nodes[0]
            random_node2 = r_nodes[1]

            edge_exists = graph.has_edge(random_node1, random_node2)

            key = tuple(sorted([random_node1, random_node2]))
            p = np.exp(edge_probability_map[key])

        # check if we should turn the edge on or not
        added_edge_list = []
        removed_edge_list = []

        toggle_on = np.random.binomial(1, p, 1)[0]

        if toggle_on and not edge_exists:
            graph.add_edge(random_node1, random_node2)
            added_edge_list.append((random_node1, random_node2))
        elif toggle_on and edge_exists:
            convergence_count = update_convergence_count(summed_graph_probs_by_edge, old_summed_graph_probs_by_edge, summed_graph_probabilities, convergence_count)
            if convergence_count > convergence_count_end:
                break

            continue
        elif not toggle_on and edge_exists:
            graph.remove_edge(random_node1, random_node2)
            removed_edge_list.append((random_node1, random_node2))
        elif not toggle_on and not edge_exists:
            convergence_count = update_convergence_count(summed_graph_probs_by_edge, old_summed_graph_probs_by_edge, summed_graph_probabilities, convergence_count)
            if convergence_count > convergence_count_end:
                break

            continue

        # check if the graph is valid after adding/removing an edge
        graph_is_valid = check_graph_validity(graph)
        if graph_is_valid:
            pass
            seen_again_count += 1
        else:
            attempts_at_validity = 0

            while True:
                edge = None
                if (toggle_on and not edge_exists) or (toggle_on and edge_exists):
                    edge = added_edge_list.pop(0)
                else:
                    edge = removed_edge_list.pop(0)

                make_graph_valid(graph, edge, toggle_on, edge_probability_map)

                attempts_at_validity += 1

                graph_is_valid = check_graph_validity(graph)

                # if we have a valid graph or nothing to make it more valid, then move on to the next one
                if graph_is_valid or attempts_at_validity > 5 or (len(added_edge_list) == 0 and len(removed_edge_list) == 0):
                    break

            if not graph_is_valid:
                continue

        # check if it is a graph we have seen before
        # if it isn't save the graph info
        graph_key = str(nx.adjacency_matrix(graph))
        if graph_key in graph_states:
            pass
        else:
            save_graph_info(graph, edge_probability_map, graph_probs, graph_state_keys, graph_states)
            summed_graph_probabilities, summed_graph_probs_by_edge = sum_graph_probabilities(graph, summed_graph_probabilities, summed_graph_probs_by_edge, edge_probability_map)

        convergence_count = update_convergence_count(summed_graph_probs_by_edge, old_summed_graph_probs_by_edge, summed_graph_probabilities, convergence_count)
        if convergence_count > convergence_count_end:
            break
            
        # old_summed_graph_probabilities = summed_graph_probabilities
        old_summed_graph_probs_by_edge = deepcopy(summed_graph_probs_by_edge)

    initialize_graph_with_updated_edge_probabilities(graph, summed_graph_probabilities, summed_graph_probs_by_edge)


def assign_edge_probabilities_weights(graph, edge_probability_map):
    all_edges = []
    edge_selection_probs = []

    for edge in combinations(graph.nodes(), r=2):
        key = tuple(sorted(edge))

        edge_prob = edge_probability_map[key]

        edge_selection_prob = 0.0
        if edge_prob < np.log(0.99, dtype=np.float128):
            edge_selection_prob = norm.cdf(np.exp(edge_prob, dtype=np.float64), loc=0.5, scale=0.234164)
            if edge_selection_prob > 0.5:
                edge_selection_prob = 1.0 - edge_selection_prob

        all_edges.append(key)
        edge_selection_probs.append(edge_selection_prob)

    # need to normalize the weights after they are assigned to use in multinomial
    sum_of_edge_selection_probs = sum(edge_selection_probs)
    if sum_of_edge_selection_probs > 0:
        for i in xrange(len(edge_selection_probs)):
            edge_selection_probs[i] = edge_selection_probs[i] / sum_of_edge_selection_probs
    else:
        prob = 1 / float(len(edge_selection_probs))
        for i in xrange(0, len(edge_selection_probs)):
            edge_selection_probs[i] = prob

    return all_edges, edge_selection_probs


def update_convergence_count(summed_graph_probs_by_edge, old_summed_graph_probs_by_edge, summed_graph_probabilities, convergence_count):
    convergent = check_edges_for_convergence(summed_graph_probs_by_edge, old_summed_graph_probs_by_edge, summed_graph_probabilities)

    if convergent:
        convergence_count += 1
    else:
        convergence_count = 0

    return convergence_count


def check_edges_for_convergence(summed_graph_probs_by_edge, old_summed_graph_probs_by_edge, summed_graph_probabilities):
    convergent = True

    for edge in summed_graph_probs_by_edge:
        old = np.exp(old_summed_graph_probs_by_edge[edge]) - np.exp(summed_graph_probabilities)
        new = np.exp(summed_graph_probs_by_edge[edge]) - np.exp(summed_graph_probabilities)

        diff = np.fabs(new - old)

        if diff < 0.00000000001:
            convergent = convergent and True
        else:
            convergent = False
            break

    return convergent


def set_graph_to_state(graph, adjacency_matrix, og_graph_nodes):
    node_mapping = {}
    node_index = 0
    for node in og_graph_nodes:
        node_mapping[node_index] = node

    graph = nx.from_numpy_matrix(adjacency_matrix)
    graph = nx.relabel_nodes(graph, node_mapping)


def normalize_graph_probabilities(graph_probs):
    normalized_graph_probs = []

    summa = sum(graph_probs)

    for graph_prob in graph_probs:
        normalized_prob = graph_prob / summa
        normalized_graph_probs.append(normalized_prob)

    return normalized_graph_probs


def save_graph_info(graph, edge_probability_map, graph_probs, graph_state_keys, graph_states):
    # save the graph state and probability of the empty graph
    graph_prob = compute_graph_probability(graph, edge_probability_map)
    graph_state = nx.adjacency_matrix(graph)

    key = str(graph_state)

    # save all the information that we need
    graph_probs.append(graph_prob)
    graph_state_keys.append(key)
    graph_states[key] = graph_state


def make_connected_subgraphs_dense(graph):
    for cc in nx.connected_component_subgraphs(graph):
        if len(cc.nodes()) > 2:
            for pair in combinations(cc.nodes(), r=2):
                graph.add_edge(pair[0], pair[1])
        else:
            break


def make_graph_valid(graph, edge, added, edge_probability_map):
    if added:
        make_connected_subgraphs_dense(graph)
    elif not added:
        n1 = edge[0]
        n2 = edge[1]

        if nx.has_path(graph, n1, n2):
            # if the two nodes are in the same connected component, we have to split them up
            nodes_in_connected_component = list(nx.node_connected_component(graph, n1))
            randomized_nodes_in_cc = np.random.choice(nodes_in_connected_component, size=len(nodes_in_connected_component), replace=False)

            graph.remove_edges_from(graph.edges())
            for edge in combinations(graph.nodes(), r=2):
                key = tuple(sorted(edge))
                edge_prob = edge_probability_map[key]
                if edge_prob >= np.log(0.99, dtype=np.float128):
                    graph.add_edge(key[0], key[1])

            n1_component_nodes = [n1]
            n2_component_nodes = [n2]

            for cc in nx.connected_component_subgraphs(graph):
                cc_nodes = cc.nodes()
                if len(cc_nodes) > 1:
                    tmp_n1_probs = []
                    tmp_n2_probs = []

                    for node in cc_nodes:
                        if n1 == node or n2 == node:
                            continue

                        n1_edge = tuple(sorted([n1, node]))
                        n1_edge_prob = edge_probability_map[n1_edge]
                        if n1_edge_prob >= np.log(0.35, dtype=np.float128):
                            tmp_n1_probs.append(n1_edge_prob)

                        n2_edge = tuple(sorted([n2, node]))
                        n2_edge_prob = edge_probability_map[n2_edge]
                        if n2_edge_prob >= np.log(0.35, dtype=np.float128):
                            tmp_n2_probs.append(n2_edge_prob)

                    avg_n1_prob = 0
                    if len(tmp_n1_probs) > 0:
                        avg_n1_prob = sum(map(np.exp, tmp_n1_probs)) / len(tmp_n1_probs)

                    avg_n2_prob = 0
                    if len(tmp_n2_probs) > 0:
                        avg_n2_prob = sum(map(np.exp, tmp_n2_probs)) / len(tmp_n2_probs)

                    if n1 in cc_nodes:
                        n1_component_nodes += cc_nodes
                    elif n2 in cc_nodes:
                        n2_component_nodes += cc_nodes

                    if avg_n1_prob > avg_n2_prob:
                        n1_component_nodes += cc_nodes
                    elif avg_n2_prob > avg_n2_prob:
                        n2_component_nodes += cc_nodes
                    else:
                        to_n1_component = np.random.binomial(1, 0.5, 1)[0]
                        if to_n1_component:
                            n1_component_nodes += cc_nodes
                        else:
                            n2_component_nodes += cc_nodes

            n1_component_nodes = list(set(n1_component_nodes))
            n2_component_nodes = list(set(n2_component_nodes))

            # choose to add edges in a greedy fashion
            for node in randomized_nodes_in_cc:
                if node == n1 or node == n2 or node in n1_component_nodes or node in n2_component_nodes:
                    continue

                tmp_n1_probs = []
                tmp_n2_probs = []

                for n1_node in n1_component_nodes:
                    n1_edge = tuple(sorted([n1, node]))
                    n1_edge_prob = edge_probability_map[n1_edge]
                    if n1_edge_prob >= np.log(0.35, dtype=np.float128):
                        tmp_n1_probs.append(n1_edge_prob)

                for n2_node in n2_component_nodes:
                    n2_edge = tuple(sorted([n2, node]))
                    n2_edge_prob = edge_probability_map[n2_edge]
                    if n2_edge_prob >= np.log(0.35, dtype=np.float128):
                        tmp_n2_probs.append(n2_edge_prob)

                avg_n1_prob = 0
                if len(tmp_n1_probs) > 0:
                    avg_n1_prob = sum(map(np.exp, tmp_n1_probs)) / len(tmp_n1_probs)

                avg_n2_prob = 0
                if len(tmp_n2_probs) > 0:
                    avg_n2_prob = sum(map(np.exp, tmp_n2_probs)) / len(tmp_n2_probs)

                if avg_n1_prob > avg_n2_prob:
                    n1_component_nodes.append(node)
                elif avg_n2_prob > avg_n2_prob:
                    n2_component_nodes.append(node)
                else:
                    to_n1_component = np.random.binomial(1, 0.5, 1)[0]
                    if to_n1_component:
                        n1_component_nodes.append(node)
                    else:
                        n2_component_nodes.append(node)

            # after we have seperated into two components, make them cliques
            # graph.remove_edges_from(graph.edges())
            for pair in combinations(n1_component_nodes, r=2):
                graph.add_edge(pair[0], pair[1])

            for pair in combinations(n2_component_nodes, r=2):
                graph.add_edge(pair[0], pair[1])

        else:
            # if the two nodes are in different connected components, make them dense
            make_connected_subgraphs_dense(graph)


def initialize_graph_with_updated_edge_probabilities(graph, summed_graph_probabilities, summed_graph_probs_by_edge):
    # clear out the graph
    graph.remove_edges_from(graph.edges())

    # add edges again with updated probabilities
    for edge in summed_graph_probs_by_edge:
        updated_prob = 0.0
        if np.exp(summed_graph_probabilities) == 0:
            updated_prob = 0.0
        else:
            updated_prob = np.exp(summed_graph_probs_by_edge[edge] - summed_graph_probabilities, dtype=np.float128)

            if np.isnan(updated_prob):
                updated_prob = 0.0

        graph.add_edge(edge[0], edge[1], prob=updated_prob)


def compute_graph_probability(graph, edge_probability_map):
    graph_prob = 0.0

    for pair in combinations(graph.nodes(), r=2):
        key = tuple(sorted(pair))
        edge_prob = edge_probability_map[key]

        if not graph.has_edge(pair[0], pair[1]):
            edge_prob = np.log(1.0 - np.exp(edge_prob, dtype=np.float128), dtype=np.float128)

        graph_prob += edge_prob

    return graph_prob


def sum_graph_probabilities(graph, summed_graph_probabilities, summed_graph_probs_by_edge, edge_probability_map):
    # first get the graph probability
    graph_prob = 0

    for pair in combinations(graph.nodes(), r=2):
        key = tuple(sorted(pair))
        edge_prob = edge_probability_map[key]
        if not graph.has_edge(pair[0], pair[1]):
            edge_prob = np.log((1.0 - np.exp(edge_prob, dtype=np.float128)), dtype=np.float128)

        graph_prob += edge_prob

    # record it for the edges that are in the graph
    for edge in graph.edges():
        key = tuple(sorted(edge))

        if summed_graph_probs_by_edge[key] == -np.inf:
            summed_graph_probs_by_edge[key] = graph_prob
        else:
            summed_graph_probs_by_edge[key] = np.log(np.exp(graph_prob, dtype=np.float128) + np.exp(summed_graph_probs_by_edge[key], dtype=np.float128), dtype=np.float128)

    if summed_graph_probabilities == -np.inf:
        summed_graph_probabilities = graph_prob
    else:
        summed_graph_probabilities = np.log(np.exp(summed_graph_probabilities, dtype=np.float128) + np.exp(graph_prob, dtype=np.float128), dtype=np.float128)

    return summed_graph_probabilities, summed_graph_probs_by_edge


def check_graph_validity(graph):
    graph_is_valid = True

    for edge in graph.edges():
        n1 = edge[0]
        n1_neighbors = graph.neighbors(n1)

        n2 = edge[1]
        n2_neighbors = graph.neighbors(n2)

        diff = (set(n1_neighbors) ^ set(n2_neighbors)) - set([n1]) - set([n2])

        if len(diff) > 0:
            graph_is_valid = False

    return graph_is_valid


def add_edges_to_graph(G, ibd_segments):
    for key in ibd_segments:
        if ibd_segments[key].lod >= const_global_lod_threshold:
            G.add_edge(ibd_segments[key].idv1 + '_' + ibd_segments[key].idv1_hap, ibd_segments[key].idv2 + '_' + ibd_segments[key].idv2_hap, weight=ibd_segments[key].lod, prob=ibd_segments[key].prob)


def assign_ibd_probabilities(samples, ibd_segments, prior_prob):
    # assign probabilities
    for key in ibd_segments:
        posterior_prob = convert_lod_to_prob(prior_prob, ibd_segments[key].lod)

        if posterior_prob == 1:
            posterior_prob = 0.99

        if ibd_segments[key].lod > 0:
            ibd_segments[key].prob = np.log(posterior_prob, dtype=np.float128) 
        else:
            ibd_segments[key].prob = np.log(prior_prob, dtype=np.float128)


def create_edge_probability_map(samples, ibd_segments, prior_prob):
    current_samples = {}
    for key in ibd_segments:
        n1 = ibd_segments[key].idv1 + '_' + ibd_segments[key].idv1_hap
        n2 = ibd_segments[key].idv2 + '_' + ibd_segments[key].idv2_hap

        current_samples[n1] = 1
        current_samples[n2] = 1

    edge_probability_map = {}
    pairs = combinations(current_samples, r=2)
    for pair in pairs:
        hap_key = tuple(sorted(pair))

        edge_probability_map[hap_key] = np.log(prior_prob, dtype=np.float128)

    # assign probabilities
    for key in ibd_segments:
        n1 = ibd_segments[key].idv1 + '_' + ibd_segments[key].idv1_hap
        n2 = ibd_segments[key].idv2 + '_' + ibd_segments[key].idv2_hap
        edge_key = tuple(sorted([n1, n2]))

        edge_probability_map[edge_key] = ibd_segments[key].prob

    return edge_probability_map


def initialize_graph(idv_list):
    samples = []

    i_file = open(idv_list, 'rb')
    for line in i_file:
        samples.append(line.strip())

    i_file.close()

    G = nx.Graph()

    for sample in samples:
        idv_hap1 = sample + '_' + '1'
        idv_hap2 = sample + '_' + '2'

        G.add_node(idv_hap1)
        G.add_node(idv_hap2)

    return G, samples


if __name__ == '__main__':
    main()
