import torch
from sklearn.model_selection import train_test_split
import numpy as np
import os
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
import logging
import time
from pathlib import Path
from utils import (
    get_pos_neg_edges, get_balanced_edges, 
    collated_data_separate)
from torch.utils.data import Dataset
from torch_geometric.data import Batch

class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name, #add
        root, #add
        neg_edge_ratio=1.0,
        to_undirected=True,
        sample_rates=[1.0],
        task_type='classification',
        transform=None, 
        pre_transform=None
    ) -> None:
        """ The SRAM dataset. 
        It can be a combination of several large circuit graphs or millions of sampled subgraphs.
        Args:
            name (str): The name of the dataset.
            root (str): The root directory of the dataset.
            add_target_edges (bool): Whether to add target (labeled) edges to the graph.
            neg_edge_ratio (float): The ratio of negative edges to positive edges.
            to_undirected (bool): Whether to convert the graph to an undirected graph.
            sample_rates (list): The sampling rates of target edges for each dataset.
            task_type (str): The task type. It can be 'classification', 'regression', 'node_classification', 'node_regression'.
        """
        self.name = 'sram'

        ## split the dataset according to '+' in the name
        if '+' in name:
            self.names = name.split('+')
        else:
            self.names = [name]
            
        print(f"SealSramDataset includes {self.names} circuits")

        self.sample_rates = sample_rates
        assert len(self.names) == len(self.sample_rates), \
            f"len of dataset:{len(self.names)}, len of sample_rate: {len(self.sample_rates)}"
        
        self.folder = os.path.join(root, self.name)
        self.neg_edge_ratio = neg_edge_ratio
        self.to_undirected = to_undirected
        ## data_lengths can be the number of total datasets or the number of subgraphs
        self.data_lengths = {}
        ## offset index for each graph
        self.data_offsets = {}

        self.task_type = task_type
        self.max_net_node_feat = torch.ones((1, 17))
        self.max_dev_node_feat = torch.ones((1, 17))
        
        super().__init__(self.folder, transform, pre_transform)
        data_list = []

        for i, name in enumerate(self.names):
            ## If a processed data file exsit, we load it directly
            loaded_data, loaded_slices = torch.load(self.processed_paths[i])
            # print("loaded_data", loaded_data)
            # print("loaded_slices", loaded_slices)

            self.data_offsets[name] = len(data_list)
            self.data_lengths[name] = loaded_data.edge_label.size(0)
            if loaded_slices is not None:
                data_list += collated_data_separate(loaded_data, loaded_slices)#[:data_len]
            else:
                data_list.append(loaded_data)
            
            print(f"load processed {name}, "+
                  f"len(data_list)={self.data_lengths[name]}, "+
                  f"data_offset={self.data_offsets[name]} ")
    
        ## combine multiple graphs into data list
        self.data, self.slices = self.collate(data_list)

    ## This is only used in regression task. Only device and net nodes have circuit statistics.
    def norm_nfeat(self, ntypes):
        if self._data is None or self.slices is None:
            self.data, self.slices = self.collate(self._data_list)
            self._data_list = None

        for ntype in ntypes:
            node_mask = self._data.node_type == ntype
            max_node_feat, _ = self._data.node_attr[node_mask].max(dim=0, keepdim=True)
            max_node_feat[max_node_feat == 0.0] = 1.0

            print(f"normalizing node_attr {ntype}: {max_node_feat} ...")
            self._data.node_attr[node_mask] /= max_node_feat
        
        ## normalize edge_label i.e., coupling capacitance
        self._data.edge_label = torch.log10(self._data.edge_label * 1e21) / 6
        ## mask out edge_label of neg edges
        self._data.edge_label[self._data.edge_label < 0] = 0.0
        print("self._data.edge_label", self._data.edge_label)
        # assert 0
        
    def sram_graph_load(self, name, raw_path):
        """
        In the loaded circuit graph, ground capacitance values are stored in "tar_node_y' attribute of the node. 
        There are to edge sets. 
        The first is the connections existing in circuit topology, attribute names start with "edge". 
        For example, "g.edge_index" and "g.edge_type".
        The second is edges to be predicted, which are parasitic coupling edges. Their attribute names start with "tar_edge". 
        For example, "g.tar_edge_index" and "g.tar_edge_type".
        Coupling capacitance values are stored in the 'tar_edge_y' attribute of the edge.
        Args:
            name (str): The name of the dataset.
            raw_path (str): The path of the raw data file.
        Returns:
            g (torch_geometric.data.Data): The processed homo graph data.
        """
        logging.info(f"raw_path: {raw_path}")
        hg = torch.load(raw_path)
        if isinstance(hg, list):
            hg = hg[0]
        # print("hg", hg)
        power_net_ids = torch.tensor([0, 1])
        
        if name == "sandwich":
            # VDD VSS TTVDD
            power_net_ids = torch.tensor([0, 1, 1422])
        elif name == "ultra8t":
            # VDD VSS SRMVDD
            power_net_ids = torch.tensor([0, 1, 377])
        elif name == "sram_sp_8192w":
            # VSSE VDDCE VDDPE
            power_net_ids = torch.tensor([0, 1, 2])
        elif name == "ssram":
            # VDD VSS VVDD
            power_net_ids = torch.tensor([0, 1, 352])
        elif name == "digtime":
            power_net_ids = torch.tensor([0, 1])
        elif name == "timing_ctrl":
            power_net_ids = torch.tensor([0, 1])
        elif name == "array_128_32_8t":
            power_net_ids = torch.tensor([0, 1])
        
        """ graph transform """ 
        ### remove the power pins
        subset_dict = {}
        for ntype in hg.node_types:
            subset_dict[ntype] = torch.ones(hg[ntype].num_nodes, dtype=torch.bool)
            if ntype == 'net':
                subset_dict[ntype][power_net_ids] = False

        hg = hg.subgraph(subset_dict)
        hg = hg.edge_type_subgraph([
            ## circuit connections in schematics
            ('device', 'device-pin', 'pin'),
            ('pin', 'pin-net', 'net'),
            ## parasitic coupling edges: pin2net, pin2pin, net2net
            ('pin', 'cc_p2n', 'net'),
            ('pin', 'cc_p2p', 'pin'),
            ('net', 'cc_n2n', 'net'),
        ])

        print(hg)

        ### transform hetero g into homo g
        g = hg.to_homogeneous()
        g.name = name
        assert hasattr(g, 'node_type')
        assert hasattr(g, 'edge_type')
        edge_offset = 0
        tar_edge_y = []
        tar_node_y = []
        g._n2type = {}
        node_feat = []
        max_feat_dim = 17
        ## padding y for device nodes (only net and pin nodes has capacitance)
        hg['device'].y = torch.ones((hg['device'].x.shape[0], 1)) * 1e-30
        
        for n, ntype in enumerate(hg.node_types):
            g._n2type[ntype] = n
            feat = hg[ntype].x
            feat = torch.nn.functional.pad(feat, (0, max_feat_dim-feat.size(1)))
            node_feat.append(feat)
            tar_node_y.append(hg[ntype].y)
        
        ## There is 'node_type' attribute after transforming hg to g.
        ## The 'node_type' is used as default node feature, g.x.
        g.x = g.node_type.view(-1, 1)
        ## circuit statistic features
        g.node_attr = torch.cat(node_feat, dim=0)
        ## lumped ground capacitance on net/pin nodes
        g.tar_node_y = torch.cat(tar_node_y, dim=0)

        del g.y
        g._e2type = {}

        for e, (edge, store) in enumerate(hg.edge_items()):
            g._e2type[edge] = e
            if 'cc' in edge[1]:
                tar_edge_y.append(store['y'])
            else:
                # edge_index's shape [2, num_edges]
                edge_offset += store['edge_index'].shape[1]
        
        g._num_ntypes = len(g._n2type)
        g._num_etypes = len(g._e2type)
        logging.info(f"g._n2type {g._n2type}")
        logging.info(f"g._e2type {g._e2type}")

        tar_edge_index = g.edge_index[:, edge_offset:]
        tar_edge_type = g.edge_type[edge_offset:]
        tar_edge_y = torch.cat(tar_edge_y)

        # testing
        # for i in range(tar_edge_type.min(), tar_edge_type.max()+1):
        #     mask = tar_edge_type == i
        #     print("tar_edge_type", tar_edge_type[mask][0], "tar_edge_y", tar_edge_y[mask][0])
        # assert 0

        ## restrict the capcitance value range 
        legel_edge_mask = (tar_edge_y < 1e-15) & (tar_edge_y > 1e-21)
        # tar_edge_src_y = g.tar_node_y[tar_edge_index[0, :]].squeeze()
        # tar_edge_dst_y = g.tar_node_y[tar_edge_index[1, :]].squeeze()
        # legel_node_mask = (tar_edge_src_y < 1e-13) & (tar_edge_src_y > 1e-23)
        # legel_node_mask &= (tar_edge_dst_y < 1e-13) & (tar_edge_dst_y > 1e-23)

        ## remove the target edges with extreme capacitance values
        g.tar_edge_y = tar_edge_y[legel_edge_mask]# & legel_node_mask]
        g.tar_edge_index = tar_edge_index[:, legel_edge_mask]# & legel_node_mask]
        g.tar_edge_type = tar_edge_type[legel_edge_mask]# & legel_node_mask]
        # logging.info(f"we filter out the edges with Cc > 1e-15 and Cc < 1e-21 " + 
        #              f"{legel_edge_mask.size(0)-legel_edge_mask.sum()}")
        # logging.info(f"we filter out the edges with src/dst Cg > 1e-13 and Cg < 1e-23 " +
        #              f"{legel_node_mask.size(0)-legel_node_mask.sum()}")

        ## Calculate target edge type distributions (Cc_p2n : Cc_p2p : Cc_n2n)  
        _, g.tar_edge_dist = g.tar_edge_type.unique(return_counts=True)
        
        ## remove target edges from the original g
        g.edge_type = g.edge_type[0:edge_offset]
        g.edge_index = g.edge_index[:, 0:edge_offset]

        ## convert to undirected edges
        if self.to_undirected:
            g.edge_index, g.edge_type = to_undirected(
                g.edge_index, g.edge_type, g.num_nodes, reduce='mean'
            )
        
        return g

    def single_g_process(self, idx: int):
        logging.info(f"processing dataset {self.names[idx]} "+ 
                     f"with sample_rate {self.sample_rates[idx]}...")
        ## we can load multiple graphs
        graph = self.sram_graph_load(self.names[idx], self.raw_paths[idx])
        logging.info(f"loaded graph {graph}")
        
        ## generate negative edges for the loaded graph
        neg_edge_index, neg_edge_type = get_pos_neg_edges(
            graph, neg_ratio=self.neg_edge_ratio)
        
       
        
        ## sample a portion of pos/neg edges
        (
            pos_edge_index, pos_edge_type, pos_edge_y,
            neg_edge_index, neg_edge_type
        ) = get_balanced_edges(
            graph, neg_edge_index, neg_edge_type, 
            self.neg_edge_ratio, self.sample_rates[idx]
        )

        ## Now we deal with the edge-level task
        if self.task_type == 'classification':
            links = torch.cat([pos_edge_index, neg_edge_index], 1)  # [2, Np + Nn]
            labels = torch.tensor(
                [1]*pos_edge_index.size(1) + [0]*neg_edge_index.size(1), dtype=torch.long
            )
        elif self.task_type == 'regression':
            ## We only consider the positive edges in the regression task.
            links = pos_edge_index  # [2, Np]
            labels = pos_edge_y
        else:
            raise ValueError(f"No defination of task {self.task_type} in this version!")
        
        ## remove the redundant attributes in this version
        del graph.tar_node_y
        del graph.tar_edge_index
        del graph.tar_edge_type

        ## To use LinkNeighborLoader, the target links rename to edge_label_index
        ## target edge labels rename to edge_label
        graph.edge_label_index = links
        graph.edge_label = labels

        torch.save((graph, None), self.processed_paths[idx])
        return graph.edge_label.size(0)

    def process(self):
        data_lens_for_split = []
        p = Path(self.processed_dir)
        ## we can have multiple graphs
        for i, name in enumerate(self.names):
            ## if there is a processed file, we skip the self.single_g_process()
            if os.path.exists(self.processed_paths[i]):
                logging.info(f"Found process file of {name} in {self.processed_paths[i]}, skipping process()")
                continue 
                        
            data_lens_for_split.append(
                self.single_g_process(i)
            )

    @property
    def raw_file_names(self):
        raw_file_names = []
        for name in self.names:
            raw_file_names.append(name+'.pt')
        
        return raw_file_names
    
    @property
    def processed_dir(self) -> str:
        if self.task_type == 'regression' or \
            self.task_type == 'classification':
            return os.path.join(self.root, 'processed', self.task_type)
        elif self.task_type == 'node_regression':
            return os.path.join(self.root, 'processed_for_nodes')
        elif self.task_type == 'node_classification':
            return os.path.join(self.root, 'processed_for_nodes/hop2_w_zero')
        else:
            raise ValueError(f"No defination of task {self.task_type}!")

    @property
    def processed_file_names(self):
        processed_names = []
        for i, name in enumerate(self.names):
            if self.sample_rates[i] < 1.0:
                name += f"_s{self.sample_rates[i]}"

            if self.neg_edge_ratio < 1.0:
                name += f"_nr{self.neg_edge_ratio:.1f}"
            processed_names.append(name+"_processed.pt")
        return processed_names


def performat_SramDataset(dataset_dir, name, 
                          neg_edge_ratio, to_undirected, 
                          sample_rates, task_type):
    start = time.perf_counter()
    num_datasets = len(name.split('+'))

    dataset = SealSramDataset(
        name=name, root=dataset_dir,
        neg_edge_ratio=neg_edge_ratio,
        to_undirected=to_undirected,
        sample_rates=[sample_rates] * num_datasets,
        task_type=task_type,
    )

    elapsed = time.perf_counter() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
            + f'{elapsed:.2f}'[-3:]
    print(f"PID = {os.getpid()}")
    print(f"Building dataset {name} took {timestr}")
    print('Dataloader: Loading success.')

    return dataset