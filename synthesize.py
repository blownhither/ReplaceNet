import numpy as np
import tensorflow as tf
import neuralgym as ng
from inpaint_model import InpaintCAModel
from load_data import load_parsed_sod
from matplotlib import pyplot as plt
from skimage.transform import resize
import skimage
from skimage import img_as_ubyte

class Synthesizer:

    def __init__(self, patch_size=512, saved_model_path = './model_logs/release_places2_256'):
        '''
        Saved model weights url:
            https://drive.google.com/drive/folders/1y7Irxm3HSHGvp546hZdAZwuNmhLUVcjO
        '''
        self.FLAGS = ng.Config('inpaint.yml')

        self.model = InpaintCAModel()
        self.checkpoint_dir = saved_model_path

        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config= self.sess_config)

        self.model_should_load = True

        self.patch_size = patch_size

    def resize_image(self, image):
        image = resize(image, output_shape=(self.patch_size, self.patch_size, 3))
        return image

    def resize_mask(self, mask):
        mask = resize(mask, output_shape=(self.patch_size, self.patch_size))
        return mask.astype(np.bool)

    def get_background(self, image, mask, reference_mask):
        # image is ground_truth_image
        merged_mask = mask + reference_mask

        mask_3d = np.zeros((merged_mask.shape[0], merged_mask.shape[1], 3))
        mask_3d[:,:,0] = merged_mask
        mask_3d[:,:,1] = merged_mask
        mask_3d[:,:,2] = merged_mask
        mask_3d = mask_3d * 255
        mask_3d = mask_3d.astype(np.uint8)
        image = img_as_ubyte(image)

        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask_3d = mask_3d[:h//grid*grid, :w//grid*grid, :]

        image = np.expand_dims(image, 0)
        mask_3d = np.expand_dims(mask_3d, 0)
        input_image = np.concatenate([image, mask_3d], axis=2)
        input_image = input_image.astype(np.float32)
        
        if self.model_should_load:
            self.model_should_load = False
            res = self.load_model(input_image)
            return res
        else:
            res= self.inpaint(input_image)
            return res

    def inpaint(self, input_image):
        output = self.model.build_server_graph(self.FLAGS, input_image, reuse=True)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        result = self.sess.run(output)
        return result[0][:, :, ::-1]


    def tweak_foreground(self, image):
        """
            tweak foreground by applying random factor
        """
        tweaked =  image * np.random.uniform(0.1, 2)
        new_image = image + tweaked
        new_image *= (1.0/new_image.max())
        return new_image

    def synthesize(self, image, mask, reference_mask):
        image = self.resize_image(image)
        mask = self.resize_mask(mask)
        reference_mask = self.resize_mask(reference_mask)

        inpainted_background_image = self.get_background(image, mask, reference_mask)
        inpainted_background_image = self.resize_image(inpainted_background_image)

        background_image = img_as_ubyte(inpainted_background_image.copy())
        background_image[mask==True] = (0, 0, 0)

        foreground_object = image.copy()
        foreground_object[mask==False] = (0, 0, 0)
        foreground_object = self.tweak_foreground(foreground_object)

        background_image = background_image / 255.0
        synthesized_image = background_image + foreground_object

        return synthesized_image
            

    def load_model(self, input_image):
        output = self.model.build_server_graph(self.FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)

        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(self.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))

        _ = self.sess.run(assign_ops)
        result = self.sess.run(output)
        return result[0][:, :, ::-1]
