import os
from data.image_folder import make_dataset
from data.pix2pix_dataset import Pix2pixDataset
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
import util.util as util
import torchvision.transforms as transforms
class BoreasDataset(Pix2pixDataset):
    def initialize(self, opt):
        self.opt = opt

        camera_paths, cp_paths, weather_paths = self.get_paths(opt)

        util.natural_sort(camera_paths)
        util.natural_sort(cp_paths)
        util.natural_sort(weather_paths)

        camera_paths = camera_paths[:opt.max_dataset_size]
        cp_paths = cp_paths[:opt.max_dataset_size]
        weather_paths = weather_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(camera_paths, cp_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.camera_paths = camera_paths
        self.cp_paths = cp_paths
        self.weather_paths = weather_paths

        size = len(self.camera_paths)
        self.dataset_size = size
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)
        return parser
    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else opt.phase
        camera_paths = make_dataset(os.path.join(root, '%s_camera' % phase), recursive=False, read_cache=True)
        cp_paths = make_dataset(os.path.join(root, '%s_cp' % phase), recursive=False, read_cache=True)
        weather_paths=make_dataset(os.path.join(root,'%s_weather' % phase),recursive=False,read_cache=True)
        assert len(camera_paths)==len(cp_paths)==len(weather_paths),"not equal images"
        return camera_paths,cp_paths,weather_paths
    def __getitem__(self, index):
        
        # Camera Image
        camera_path = self.camera_paths[index]
        camera = Image.open(camera_path).convert('RGB')
        params = get_params(self.opt, camera.size)
        transform_label = get_transform(self.opt, params)
        camera_tensor = transform_label(camera)
        
        # clouldpoints image
        cp_path = self.cp_paths[index]
        assert self.paths_match(cp_path, camera_path), \
            "The cp_path %s and camera_path %s don't match." % \
            (cp_path, camera_path)
        cp = Image.open(cp_path).convert('RGB')

        transform_image = get_transform(self.opt, params)
        cp_tensor = transform_image(cp)

        # weather image
        weather_path = self.weather_paths[index]
        assert self.paths_match(weather_path, camera_path), \
            "The cp_path %s and camera_path %s don't match." % \
            (cp_path, camera_path)
        weather = Image.open(weather_path).convert('L')

        transform_weather = get_transform(self.opt, params)
        transformList = [transforms.ToTensor()]
        osize = [self.opt.crop_size, self.opt.crop_size]
        transformList.append(transforms.Resize(osize))
        weather_tensor =transforms.Compose( transformList)(weather)
        input_dict = {'camera': camera_tensor,
                      'weather': weather_tensor,
                      'cp': cp_tensor,
                      'path': cp_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict