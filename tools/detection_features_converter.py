import os
import sys
import csv
import h5py
import json
import base64
import argparse
from tqdm import tqdm
import numpy as np
sys.path.append(os.getcwd())
csv.field_size_limit(sys.maxsize)

import utils.utils as utils
import utils.config as config


FIELDNAMES = ['image_id', 'image_w', 'image_h', \
            'num_boxes', 'boxes', 'features']
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=['trainval', 'test'], default='trainval')
    args = parser.parse_args()

    split_set = ['test'] if args.split == 'test' else ['train', 'val']

    # choose the right bottom up feature file
    prefix = 'test2015' if args.split == 'test' else 'trainval'
    bottom_up_path = os.path.join(config.bottom_up_path,
                    prefix + '_resnet101_faster_rcnn_genome_36.tsv')
    num_images = config.test_num_images if args.split == 'test' \
                else config.trainval_num_images

    # dump indices
    split_indices_path = os.path.join(
        config.ids_path, args.split + '36_imgid2idx.json')

    # load all image ids
    img_ids = []
    for split in split_set:
        split_ids_path = os.path.join(config.ids_path, split + '_ids.json')
        print(split_ids_path)
        if os.path.exists(split_ids_path):
            img_ids += json.load(open(split_ids_path, 'r'))
        else:
            split_year = '2014' if not split == 'test' else '2015'
            split_image_path = os.path.join(
                config.image_path, split + split_year)
            img_ids_dump = utils.load_imageid(split_image_path)
            json.dump(list(img_ids_dump), open(split_ids_path, 'w'))
            img_ids += json.load(open(split_ids_path, 'r'))

    # create h5 files
    h_split = h5py.File(os.path.join(
            config.rcnn_path, args.split + '36.h5'), 'w')
    split_img_features = h_split.create_dataset(
        'image_features', (len(img_ids),
        config.num_fixed_boxes, config.output_features), 'f')
    split_img_bb = h_split.create_dataset(
        'image_bb', (len(img_ids),
        config.num_fixed_boxes, 4), 'f')
    split_spatial_img_features = h_split.create_dataset(
        'spatial_features', (len(img_ids),
        config.num_fixed_boxes, 6), 'f')

    counter, indices = 0, {}
    print("reading tsv...")
    with open(bottom_up_path, 'r') as tsv_in_file:
        reader = csv.DictReader(
                tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in tqdm(reader, total=num_images):
            image_id = int(item['image_id'])
            if image_id in img_ids:
                item['num_boxes'] = int(item['num_boxes'])
                image_w = float(item['image_w'])
                image_h = float(item['image_h'])
                buf = base64.b64decode(item['boxes'])
                bboxes = np.frombuffer(buf,
                    dtype=np.float32).reshape((item['num_boxes'], -1))

                box_width = bboxes[:, 2] - bboxes[:, 0]
                box_height = bboxes[:, 3] - bboxes[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bboxes[:, 0] / image_w
                scaled_y = bboxes[:, 1] / image_h

                box_width = box_width[..., np.newaxis]
                box_height = box_height[..., np.newaxis]
                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]

                spatial_features = np.concatenate(
                    (scaled_x, scaled_y,
                    scaled_x + scaled_width,
                    scaled_y + scaled_height,
                    scaled_width, scaled_height), axis=1)

                img_ids.remove(image_id)
                indices[image_id] = counter
                split_img_bb[counter, :, :] = bboxes
                buf = base64.b64decode(item['features'])
                split_img_features[counter, :, :] = np.frombuffer(buf,
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                split_spatial_img_features[counter, :, :] = spatial_features
                counter += 1

    if len(img_ids) != 0:
        print("Warning: {}_image_ids is not empty".format(args.split))

    print("done!")
    json.dump(indices, open(split_indices_path, 'w'))
    h_split.close()


if __name__ == '__main__':
    main()
