
import os, pickle, torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, SequentialSampler, BatchSampler
from torch.utils.data import Subset as TorchSubset
from torch.utils.data import TensorDataset
from utils.common import Common
from dataset.dataset import _make_eval_train_dataset_from_base
import dataset.dataset as DS

class PrivacyMiaUtils:
    @staticmethod
    def _empty_like_subset(subset):
        x_shape = subset[0][0].shape if len(subset) else (1,1,1)
        return torch.utils.data.TensorDataset(torch.empty(0, *x_shape), torch.empty(0, dtype=torch.long))
    @staticmethod
    def _infer_base_dataset(dl):
        ds = dl.dataset
        # unwrap nested Subset(...) -> dataset
        while isinstance(ds, TorchSubset):
            base_indices = ds.indices  # we keep these elsewhere
            ds = ds.dataset
        return ds
    @staticmethod
    def _indices_from_loader(dl):
        ds = dl.dataset
        while isinstance(ds, TorchSubset):
            idxs = ds.indices
            inner = ds.dataset
            ds = inner
        return idxs
    @staticmethod    
    def get_data_loaders(train_loader_list_or_dataset_path, is_mixed, seed: int, batch_size: int = 10):
        if is_mixed:
            train_loader_list = train_loader_list_or_dataset_path
            if not train_loader_list or len(train_loader_list) == 0:
                raise RuntimeError("'dataset_train_list' is empty.")
            
            Common.set_seed_over_method(seed)

            # 1) Recreate an eval (non-random) training dataset matching the base type
            base_ds = PrivacyMiaUtils._infer_base_dataset(train_loader_list[0])
            eval_train_ds = _make_eval_train_dataset_from_base(base_ds)   # your helper from earlier

            # 2) Indices
            idxs_target = list(PrivacyMiaUtils._indices_from_loader(train_loader_list[0]))  # client 0
            target_len = len(idxs_target)

            # Build round-robin pool from all other clients
            other_lists = [list(PrivacyMiaUtils._indices_from_loader(dl)) for dl in train_loader_list[1:]]
            # deterministically interleave
            mixed = []
            ptrs = [0] * len(other_lists)
            while len(mixed) < target_len and len(other_lists) > 0:
                progressed = False
                for i in range(len(other_lists)):
                    if ptrs[i] < len(other_lists[i]):
                        mixed.append(other_lists[i][ptrs[i]])
                        ptrs[i] += 1
                        progressed = True
                        if len(mixed) == target_len:
                            break
                if not progressed:
                    # ran out of pool (e.g., only one tiny other client) -> stop
                    break

            # 3) Build subsets
            train_subset = Subset(eval_train_ds, idxs_target)
            val_subset = Subset(eval_train_ds, mixed) if len(mixed) > 0 else PrivacyMiaUtils._empty_like_subset(train_subset)

            # 4) Deterministic evaluation loaders: sequential sampling, single worker
            train_loader = DataLoader(
                train_subset,
                batch_sampler=BatchSampler(SequentialSampler(train_subset), batch_size=batch_size, drop_last=False),
                num_workers=0,
            )
            val_loader = DataLoader(
                val_subset,
                batch_sampler=BatchSampler(SequentialSampler(val_subset), batch_size=batch_size, drop_last=False),
                num_workers=0,
            )
            Common.set_seed_over_method(seed)
            return val_loader, train_loader
        else:
            dataset_path = train_loader_list_or_dataset_path
            with open(os.path.join(dataset_path, "dataset_meta.pkl"), "rb") as f:
                dataset_type = pickle.load(f)["type"]

            partitions = []
            i = 0
            while True:
                fpath = os.path.join(dataset_path, f"dataset_node_{i}.ds")
                if not os.path.exists(fpath): break
                with open(fpath, 'rb') as f:
                    partitions.append(pickle.load(f))
                i += 1

            if partitions:
                dataset_train_list, _ = DS.build_loaders_from_partitions(
                    partitions, dataset_type, 10, 10, base_seed=seed, num_workers=0
                )
                
                datasets_to_mix = [
                    (dl.dataset if hasattr(dl, "dataset") else dl)
                    for idx, dl in enumerate(dataset_train_list) if idx != 0
                ]
                if not datasets_to_mix:
                    raise ValueError("No datasets to mix (dataset_train_list has no indices beyond 0).")

                mixed_dataset = ConcatDataset(datasets_to_mix)
                generator = torch.Generator().manual_seed(seed)
                mixed_loader = DataLoader(
                    mixed_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    generator=generator
                )
                return mixed_loader, dataset_train_list[0]
    @staticmethod
    def FedMiaExec(fedmia_attack, global_model, clients_models_tuples, target_model_id, model_class, lr, platform):

            global_model_clone = model_class().to(platform)

            target_model_index = next(
                (i for i, client_state_dict in enumerate(clients_models_tuples) if client_state_dict[0] == target_model_id), 
                None
            )

            global_model_clone.load_state_dict(global_model)

            shadow_models = []
            for i, client_state_dict in enumerate(clients_models_tuples):
                if i != target_model_index:
                    shadow_models.append(client_state_dict[1])

            fedmia_attack.execute(shadow_models, clients_models_tuples[target_model_index][1], global_model_clone, platform, lr)
            return fedmia_attack.get_auc_metrics(platform)

