import argparse
import collections
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from embedding_dataset import EmbDataset
from models.rqvae import RQVAE


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])
    return collision_item_groups


def parse_args():
    parser = argparse.ArgumentParser(description="Generate discrete codes from a trained RQ-VAE model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth).")
    parser.add_argument("--output_codes", type=str, required=True, help="Output .npy path for codes.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_collision_passes", type=int, default=30)
    parser.add_argument(
        "--data_path_override",
        type=str,
        default=None,
        help="Optional embedding path override. If not set, use args.data_path stored in checkpoint.",
    )
    parser.add_argument(
        "--append_resolve_digit",
        action="store_true",
        default=False,
        help="Append one extra digit for duplicate code resolution (same as TIGER behavior).",
    )
    return parser.parse_args()


def main():
    args_setting = parse_args()
    device = torch.device(args_setting.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args_setting.checkpoint, map_location=torch.device("cpu"))
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    data_path = args_setting.data_path_override if args_setting.data_path_override else args.data_path
    data = EmbDataset(data_path)

    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    data_loader = DataLoader(
        data,
        num_workers=args_setting.num_workers,
        batch_size=args_setting.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>"]

    for d in tqdm(data_loader, desc="Initial coding"):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0

    tt = 0
    while True:
        if tt >= args_setting.max_collision_passes or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(collision_item_groups)
        print(len(collision_item_groups))
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    codes = []
    for _, value in all_indices_dict.items():
        code = [int(item.split("_")[1].strip(">")) for item in value]
        codes.append(code)
    codes_array = np.array(codes, dtype=np.int64)

    if args_setting.append_resolve_digit:
        # Keep TIGER's optional behavior: append 1 digit for conflict tie-breaking.
        codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=np.int64)))
        unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
        duplicates = unique_codes[counts > 1]
        if len(duplicates) > 0:
            print("Resolving duplicates in codes...")
            for duplicate in duplicates:
                duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
                for i, idx in enumerate(duplicate_indices):
                    codes_array[idx, -1] = i

        new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
        duplicates = new_unique_codes[new_counts > 1]
        if len(duplicates) > 0:
            print("There still have duplicates:", duplicates)
        else:
            print("There are no duplicates in the codes after resolution.")

    print(f"Saving codes to {args_setting.output_codes}")
    print(f"the first 5 codes: {codes_array[:5]}")
    np.save(args_setting.output_codes, codes_array)


if __name__ == "__main__":
    main()
