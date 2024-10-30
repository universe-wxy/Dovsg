import numpy as np
import torch
from dovsg.perception.models.myclip import MyClip
from dovsg.memory.view_dataset import ViewDataset
from dovsg.memory.instances.instance_utils import MapObjectList
from typing import List, Dict

class InstanceLocalizer:
    def __init__(
        self,
        view_dataset: ViewDataset,
        instances_objects: MapObjectList,
        device="cuda"
    ):
        print("Initializing Instance Localizer.")
        self.view_dataset = view_dataset
        self.instances_objects = instances_objects
        self.device = device

        ### Initialize the CLIP model ###
        self.myclip = MyClip(device=self.device)

    def calculate_clip_and_st_embeddings_for_queries(self, queries):
        with torch.no_grad():
            if isinstance(queries, str):
                queries = [queries]
            text_feat = self.myclip.get_text_feature(queries)
        return text_feat

    def localize(self, A: str):
        print("A is ", A)
        text_feat = self.calculate_clip_and_st_embeddings_for_queries([A])
        # similarities = F.cosine_similarity(
        #     text_feat, self.objects_clip_feature, dim=-1
        # )
        similarities = self.instances_objects.compute_similarities(text_feat, device=text_feat.device)
        obj_idx = similarities.argmax().item()
        # target = np.mean(np.asarray(self.instances_objects[obj_idx]["indexes"]), axis=0)
        target_indexes = self.instances_objects[obj_idx]["indexes"]
        target = np.mean(np.asarray(
            self.view_dataset.index_to_point(target_indexes)
        ), axis=0)   


    def localize_AonB(self, A, B, k_A = 3, k_B = 5):
        print("A is ", A)
        print("B is ", B)
        if B is None or B == '':
            text_feat = self.calculate_clip_and_st_embeddings_for_queries([A])
            # similarities = F.cosine_similarity(
            #     text_feat, self.objects_clip_feature, dim=-1
            # )
            similarities = self.instances_objects.compute_similarities(text_feat, device=text_feat.device)
            obj_idx = similarities.argmax().item()
            # target = np.mean(np.asarray(self.instances_objects[obj_idx]["indexes"]), axis=0)
            target_indexes = self.instances_objects[obj_idx]["indexes"]
            target = np.mean(np.asarray(
                self.view_dataset.index_to_point(target_indexes)
            ), axis=0)
        else:
            text_feat = self.calculate_clip_and_st_embeddings_for_queries([A, B])
            similarities = self.instances_objects.compute_similarities(text_feat, device=text_feat.device)

            # Find the first k_A instances that are most similar to A
            top_k_A_indices = similarities[0].topk(k=k_A).indices.tolist()
            A_points_list = []
            for idx in top_k_A_indices:
                target_indexes_A = self.instances_objects[idx]["indexes"]
                A_points_list.append(np.mean(self.view_dataset.index_to_point(target_indexes_A), axis=0))

            # Find the first k_B instances that are most similar to B
            top_k_B_indices = similarities[1].topk(k=k_B).indices.tolist()
            B_points_list = []
            for idx in top_k_B_indices:
                target_indexes_B = self.instances_objects[idx]["indexes"]
                B_points_list.append(np.mean(self.view_dataset.index_to_point(target_indexes_B), axis=0))

            # Find the most suitable combination of point A and point B
            target = None
            distances = np.linalg.norm(np.array(A_points_list)[:, None] - np.array(B_points_list)[None, :], axis=-1)

            target = A_points_list[np.argmin(np.min(distances, axis=1))]

        return target