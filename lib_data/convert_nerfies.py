import os
import sys
import json
from os.path import join as pjoin
from argparse import ArgumentParser


def process_folder(src_path, tgt_path, img_scale=4):
    print(f'process {src_path} {tgt_path}')
    os.makedirs(tgt_path, exist_ok=True)

    with open(pjoin(src_path, 'dataset.json'), 'r') as f:
        split_info = json.load(f)

    with open(pjoin(src_path, 'metadata.json'), 'r') as f:
        meta_info = json.load(f)

    for src_id, tgt_prefix in zip(['train_ids', 'val_ids'], ['', 'test_']):
        os.makedirs(pjoin(tgt_path, f'{tgt_prefix}cameras'), exist_ok=True)
        os.makedirs(pjoin(tgt_path, f'{tgt_prefix}images'), exist_ok=True)
        for id in split_info[src_id]:
            time_id = meta_info[id].get('time_id', int(id.split('_')[1]))
            camera_id = meta_info[id]['camera_id']
            new_id = f'{camera_id}_{time_id:05d}'

            os.system(f'cp {src_path}/rgb/{img_scale}x/{id}.png {tgt_path}/{tgt_prefix}images/{new_id}.png')

            with open(pjoin(src_path, 'camera', f'{id}.json'), 'r') as f:
                camera = json.load(f)

                def resize(data, scale):
                    if isinstance(data, list):
                        return [resize(item, scale) for item in data]
                    elif isinstance(data, float):
                        return data * scale
                    elif isinstance(data, int):
                        return int(round(data * scale))

                for key in ['focal_length', 'principal_point', 'image_size']:
                    camera[key] = resize(camera[key], 1.0 / img_scale)

            with open(pjoin(tgt_path, f'{tgt_prefix}cameras', f'{new_id}.json'), 'w') as f:
                json.dump(camera, f)



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--src_path', type=str)
    parser.add_argument('--tgt_path', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--batch', action='store_true')

    return parser.parse_args()


def main(args):
    if args.batch:
        for folder in os.listdir(args.src_path):
            if os.path.exists(pjoin(args.src_path, folder, 'metadata.json')):
                process_folder(pjoin(args.src_path, folder), pjoin(args.tgt_path, folder), args.scale)
    else:
        process_folder(args.src_path, args.tgt_path, args.scale)


if __name__ == '__main__':
    args = parse_args()
    main(args)

